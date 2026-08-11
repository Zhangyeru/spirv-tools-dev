// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>

#include "source/opcode.h"
#include "source/opt/basic_block.h"
#include "source/opt/constants.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/function.h"
#include "source/opt/hw_fuse_two_layer_vector_matmul_pass.h"
#include "source/opt/hw_lower_to_standard_pass.h"
#include "source/opt/hw_lower_to_standard_pass_internal.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"
#include "spirv/unified1/GLSL.std.450.h"

namespace spvtools {
namespace opt {

using namespace hw_lower_internal;

bool HwLowerToStandardPass::LowerMatrixMulAdd(Instruction* inst) {
  const uint32_t saved_fast_math_mode = active_fp_fast_math_mode_;
  active_fp_fast_math_mode_ = GetFPFastMathMode(inst->result_id());
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = GetMatrixTypeForValue(a_inst);
  const MatrixTypeInfo* b = GetMatrixTypeForValue(b_inst);
  const MatrixTypeInfo* c = GetMatrixTypeForValue(c_inst);
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddHW");
    active_fp_fast_math_mode_ = saved_fast_math_mode;
    return false;
  }

  const uint64_t mac_count = static_cast<uint64_t>(result->rows) *
                             static_cast<uint64_t>(result->cols) *
                             static_cast<uint64_t>(a->cols);
  bool lowered = mac_count > max_unrolled_matmul_macs_
                     ? LowerMatrixMulAddWithLoop(inst)
                     : (CanUsePackedVec4MatrixMulAdd(*result, *a, *b, *c)
                            ? LowerMatrixMulAddPackedVec4(inst)
                            : LowerMatrixMulAddScalarFallback(inst));
  if (lowered) RemoveFPFastMathMode(inst->result_id());
  active_fp_fast_math_mode_ = saved_fast_math_mode;
  return lowered;
}

bool HwLowerToStandardPass::LowerMatrixMulAddPackedVec4(Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = GetMatrixTypeForValue(a_inst);
  const MatrixTypeInfo* b = GetMatrixTypeForValue(b_inst);
  const MatrixTypeInfo* c = GetMatrixTypeForValue(c_inst);
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddHW");
    return false;
  }

  bool handled = false;
  if (!TryLowerDirectMatrixMulAdd(inst, &handled)) return false;
  if (handled) return true;

  const uint32_t function_id =
      GetOrCreateMatmulPatternFunctionPackedVec4(*result, *a, *b, *c);
  if (function_id == 0) {
    return false;
  }
  RebuildAsFunctionCall(
      inst, result->lowered_type_id, function_id,
      {a_inst->result_id(), b_inst->result_id(), c_inst->result_id()});
  return true;
}

bool HwLowerToStandardPass::LowerMatrixMulAddScalarFallback(Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = GetMatrixTypeForValue(a_inst);
  const MatrixTypeInfo* b = GetMatrixTypeForValue(b_inst);
  const MatrixTypeInfo* c = GetMatrixTypeForValue(c_inst);
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddHW");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> scalar_ids(result->rows * result->cols, 0);

  const uint32_t a_id = a_inst->result_id();
  const uint32_t b_id = b_inst->result_id();
  const uint32_t c_id = c_inst->result_id();
  const uint32_t accumulator_type_id = result->component_type_id;

  for (uint32_t row0 = 0; row0 < result->rows; row0 += kDefaultMatrixTileM) {
    const uint32_t tile_m = std::min(kDefaultMatrixTileM, result->rows - row0);

    for (uint32_t col0 = 0; col0 < result->cols; col0 += kDefaultMatrixTileN) {
      const uint32_t tile_n =
          std::min(kDefaultMatrixTileN, result->cols - col0);
      std::vector<uint32_t> acc(tile_m * tile_n, 0);

      for (uint32_t i = 0; i < tile_m; ++i) {
        for (uint32_t j = 0; j < tile_n; ++j) {
          const uint32_t row = row0 + i;
          const uint32_t col = col0 + j;
          acc[i * tile_n + j] =
              ExtractMatrixScalar(&builder, *c, c_id, row, col);
          if (acc[i * tile_n + j] == 0) return false;
        }
      }

      for (uint32_t k = 0; k < a->cols; ++k) {
        std::vector<uint32_t> a_elems(tile_m, 0);
        std::vector<uint32_t> b_elems(tile_n, 0);

        for (uint32_t i = 0; i < tile_m; ++i) {
          const uint32_t row = row0 + i;
          a_elems[i] = ExtractMatrixScalar(&builder, *a, a_id, row, k);
          if (a_elems[i] == 0) return false;
        }

        for (uint32_t j = 0; j < tile_n; ++j) {
          const uint32_t col = col0 + j;
          b_elems[j] = ExtractMatrixScalar(&builder, *b, b_id, k, col);
          if (b_elems[j] == 0) return false;
        }

        for (uint32_t i = 0; i < tile_m; ++i) {
          for (uint32_t j = 0; j < tile_n; ++j) {
            const uint32_t accumulated = BuildMatmulAccumulate(
                &builder, accumulator_type_id, a->component_type_id, a_elems[i],
                b->component_type_id, b_elems[j], acc[i * tile_n + j]);
            if (accumulated == 0) return false;
            acc[i * tile_n + j] = accumulated;
          }
        }
      }

      for (uint32_t i = 0; i < tile_m; ++i) {
        for (uint32_t j = 0; j < tile_n; ++j) {
          const uint32_t row = row0 + i;
          const uint32_t col = col0 + j;
          scalar_ids[MatrixFlatIndex(*result, row, col)] = acc[i * tile_n + j];
        }
      }
    }
  }

  for (uint32_t id : scalar_ids) {
    if (id == 0) return false;
  }

  if (IsPackedVec4(*result)) {
    std::vector<uint32_t> element_ids(result->rows * result->packed_cols, 0);
    for (uint32_t row = 0; row < result->rows; ++row) {
      for (uint32_t col_pack = 0; col_pack < result->packed_cols; ++col_pack) {
        std::vector<uint32_t> lane_ids;
        lane_ids.reserve(kPackedVec4Width);
        for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
          lane_ids.push_back(scalar_ids[MatrixFlatIndex(
              *result, row, col_pack * kPackedVec4Width + lane)]);
        }
        Instruction* vec = builder.AddCompositeConstruct(
            result->packed_vec4_type_id, lane_ids);
        if (!vec) return false;
        element_ids[MatrixPackedIndex(*result, row, col_pack)] =
            vec->result_id();
      }
    }
    RebuildAsCompositeConstruct(inst, result->lowered_type_id, element_ids);
    return true;
  }

  std::vector<uint32_t> element_ids = scalar_ids;
  RebuildAsCompositeConstruct(inst, result->lowered_type_id, element_ids);
  return true;
}

bool HwLowerToStandardPass::LowerMatrixMulAddWithLoop(Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = GetMatrixTypeForValue(a_inst);
  const MatrixTypeInfo* b = GetMatrixTypeForValue(b_inst);
  const MatrixTypeInfo* c = GetMatrixTypeForValue(c_inst);
  BasicBlock* preheader_block = context()->get_instr_block(inst);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!result || !a || !b || !c || !function) return false;

  const uint32_t a_pointer_type_id =
      GetOrCreatePointerType(a->lowered_type_id, spv::StorageClass::Function);
  const uint32_t b_pointer_type_id =
      GetOrCreatePointerType(b->lowered_type_id, spv::StorageClass::Function);
  const uint32_t c_pointer_type_id =
      GetOrCreatePointerType(c->lowered_type_id, spv::StorageClass::Function);
  const uint32_t result_pointer_type_id = GetOrCreatePointerType(
      result->lowered_type_id, spv::StorageClass::Function);
  const uint32_t accumulator_pointer_type_id = GetOrCreatePointerType(
      result->component_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_pointer_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_id = GetOrCreateUIntConstant(0);
  const uint32_t one_id = GetOrCreateUIntConstant(1);
  const uint32_t output_count_id =
      GetOrCreateUIntConstant(result->rows * result->cols);
  const uint32_t result_cols_id = GetOrCreateUIntConstant(result->cols);
  const uint32_t inner_count_id = GetOrCreateUIntConstant(a->cols);
  const uint32_t result_zero_id = GetOrCreateZero(result->lowered_type_id);
  if (a_pointer_type_id == 0 || b_pointer_type_id == 0 ||
      c_pointer_type_id == 0 || result_pointer_type_id == 0 ||
      accumulator_pointer_type_id == 0 || uint_type_id == 0 ||
      uint_pointer_type_id == 0 || bool_type_id == 0 || zero_id == 0 ||
      one_id == 0 || output_count_id == 0 || result_cols_id == 0 ||
      inner_count_id == 0 || result_zero_id == 0) {
    return false;
  }

  Instruction* a_variable = AddFunctionVariable(function, a_pointer_type_id);
  Instruction* b_variable = AddFunctionVariable(function, b_pointer_type_id);
  Instruction* c_variable = AddFunctionVariable(function, c_pointer_type_id);
  Instruction* result_variable =
      AddFunctionVariable(function, result_pointer_type_id);
  Instruction* output_index_variable =
      AddFunctionVariable(function, uint_pointer_type_id);
  Instruction* k_variable = AddFunctionVariable(function, uint_pointer_type_id);
  Instruction* accumulator_variable =
      AddFunctionVariable(function, accumulator_pointer_type_id);
  if (!a_variable || !b_variable || !c_variable || !result_variable ||
      !output_index_variable || !k_variable || !accumulator_variable) {
    return false;
  }

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() && &*split_iter != inst) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;
  std::unique_ptr<Instruction> enclosing_loop_merge;
  if (Instruction* loop_merge = preheader_block->GetLoopMergeInst()) {
    enclosing_loop_merge.reset(loop_merge->Clone(context()));
  }

  std::array<uint32_t, 8> labels;
  for (uint32_t& label : labels) {
    label = TakeNextId();
    if (label == 0) return false;
  }
  const uint32_t merge_label = labels[0];
  const uint32_t outer_header_label = labels[1];
  const uint32_t outer_body_label = labels[2];
  const uint32_t k_header_label = labels[3];
  const uint32_t k_body_label = labels[4];
  const uint32_t k_continue_label = labels[5];
  const uint32_t k_merge_label = labels[6];
  const uint32_t outer_continue_label = labels[7];
  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label, split_iter);
  BasicBlock* outer_header = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_header_label)),
      preheader_block);
  BasicBlock* outer_body = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_body_label)),
      outer_header);
  BasicBlock* k_header = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_header_label)), outer_body);
  BasicBlock* k_body = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_body_label)), k_header);
  BasicBlock* k_continue = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_continue_label)), k_body);
  BasicBlock* k_merge = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_merge_label)), k_continue);
  BasicBlock* outer_continue = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_continue_label)),
      k_merge);
  if (!merge_block || !outer_header || !outer_body || !k_header || !k_body ||
      !k_continue || !k_merge || !outer_continue) {
    return false;
  }
  if (enclosing_loop_merge) {
    Instruction* moved_loop_merge = merge_block->GetLoopMergeInst();
    if (!moved_loop_merge) return false;
    context()->KillInst(moved_loop_merge);
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(a_variable->result_id(),
                                  a_inst->result_id()) ||
      !preheader_builder.AddStore(b_variable->result_id(),
                                  b_inst->result_id()) ||
      !preheader_builder.AddStore(c_variable->result_id(),
                                  c_inst->result_id()) ||
      !preheader_builder.AddStore(result_variable->result_id(),
                                  result_zero_id) ||
      !preheader_builder.AddStore(output_index_variable->result_id(),
                                  zero_id) ||
      (enclosing_loop_merge &&
       !preheader_builder.AddInstruction(std::move(enclosing_loop_merge))) ||
      !preheader_builder.AddBranch(outer_header_label)) {
    return false;
  }

  InstructionBuilder outer_header_builder(
      context(), outer_header,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* output_index = outer_header_builder.AddLoad(
      uint_type_id, output_index_variable->result_id());
  Instruction* outer_condition =
      output_index ? outer_header_builder.AddBinaryOp(
                         bool_type_id, spv::Op::OpULessThan,
                         output_index->result_id(), output_count_id)
                   : nullptr;
  if (!output_index || !outer_condition ||
      !outer_header_builder.AddLoopMerge(merge_label, outer_continue_label) ||
      !outer_header_builder.AddConditionalBranch(
          outer_condition->result_id(), outer_body_label, merge_label)) {
    return false;
  }

  InstructionBuilder outer_body_builder(
      context(), outer_body,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t initial_accumulator = BuildLogicalAggregateLoad(
      &outer_body_builder, c_variable->result_id(), c->component_type_id,
      c->packed_vec4_type_id, output_index->result_id());
  if (initial_accumulator == 0 ||
      !outer_body_builder.AddStore(accumulator_variable->result_id(),
                                   initial_accumulator) ||
      !outer_body_builder.AddStore(k_variable->result_id(), zero_id) ||
      !outer_body_builder.AddBranch(k_header_label)) {
    return false;
  }

  InstructionBuilder k_header_builder(
      context(), k_header,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* k =
      k_header_builder.AddLoad(uint_type_id, k_variable->result_id());
  Instruction* k_condition =
      k ? k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                       k->result_id(), inner_count_id)
        : nullptr;
  if (!k || !k_condition ||
      !k_header_builder.AddLoopMerge(k_merge_label, k_continue_label) ||
      !k_header_builder.AddConditionalBranch(k_condition->result_id(),
                                             k_body_label, k_merge_label)) {
    return false;
  }

  InstructionBuilder k_body_builder(
      context(), k_body,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* row = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpUDiv, output_index->result_id(), result_cols_id);
  Instruction* col = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpUMod, output_index->result_id(), result_cols_id);
  Instruction* a_row_base =
      row ? k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                       row->result_id(), inner_count_id)
          : nullptr;
  Instruction* a_index =
      a_row_base
          ? k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                       a_row_base->result_id(), k->result_id())
          : nullptr;
  Instruction* b_row_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k->result_id(), result_cols_id);
  Instruction* b_index = b_row_base && col
                             ? k_body_builder.AddBinaryOp(
                                   uint_type_id, spv::Op::OpIAdd,
                                   b_row_base->result_id(), col->result_id())
                             : nullptr;
  if (!row || !col || !a_index || !b_index) return false;
  const uint32_t a_value = BuildLogicalAggregateLoad(
      &k_body_builder, a_variable->result_id(), a->component_type_id,
      a->packed_vec4_type_id, a_index->result_id());
  const uint32_t b_value = BuildLogicalAggregateLoad(
      &k_body_builder, b_variable->result_id(), b->component_type_id,
      b->packed_vec4_type_id, b_index->result_id());
  Instruction* accumulator = k_body_builder.AddLoad(
      result->component_type_id, accumulator_variable->result_id());
  const uint32_t accumulated =
      accumulator ? BuildMatmulAccumulate(
                        &k_body_builder, result->component_type_id,
                        a->component_type_id, a_value, b->component_type_id,
                        b_value, accumulator->result_id())
                  : 0;
  if (accumulated == 0 ||
      !k_body_builder.AddStore(accumulator_variable->result_id(),
                               accumulated) ||
      !k_body_builder.AddBranch(k_continue_label)) {
    return false;
  }

  InstructionBuilder k_continue_builder(
      context(), k_continue,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k->result_id(), one_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_variable->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label)) {
    return false;
  }

  InstructionBuilder k_merge_builder(
      context(), k_merge,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* final_accumulator = k_merge_builder.AddLoad(
      result->component_type_id, accumulator_variable->result_id());
  if (!final_accumulator ||
      !BuildLogicalAggregateStore(
          &k_merge_builder, result_variable->result_id(),
          result->component_type_id, result->packed_vec4_type_id,
          output_index->result_id(), final_accumulator->result_id()) ||
      !k_merge_builder.AddBranch(outer_continue_label)) {
    return false;
  }

  InstructionBuilder outer_continue_builder(
      context(), outer_continue,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_output = outer_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, output_index->result_id(), one_id);
  if (!next_output ||
      !outer_continue_builder.AddStore(output_index_variable->result_id(),
                                       next_output->result_id()) ||
      !outer_continue_builder.AddBranch(outer_header_label)) {
    return false;
  }

  inst->SetOpcode(spv::Op::OpLoad);
  inst->SetResultType(result->lowered_type_id);
  inst->SetInOperands({IdOperand(result_variable->result_id())});
  context()->UpdateDefUse(inst);
  return true;
}

bool HwLowerToStandardPass::LowerMatrixLength(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 1) {
    ReportError(inst, "invalid OpCooperativeMatrixLengthHW");
    return false;
  }
  const MatrixTypeInfo* info = GetMatrixType(inst->GetSingleWordInOperand(0));
  if (!info) {
    ReportError(inst, "invalid OpCooperativeMatrixLengthHW type operand");
    return false;
  }
  const uint32_t length_id =
      GetOrCreateConstant(inst->type_id(), info->rows * info->cols);
  if (length_id == 0) {
    ReportError(inst, "OpCooperativeMatrixLengthHW result type must be int32");
    return false;
  }
  context()->ReplaceAllUsesWith(inst->result_id(), length_id);
  to_kill->push_back(inst);
  return true;
}

bool HwLowerToStandardPass::LowerMatrixReduce(Instruction* inst) {
  const uint32_t saved_fast_math_mode = active_fp_fast_math_mode_;
  active_fp_fast_math_mode_ = inst ? GetFPFastMathMode(inst->result_id()) : 0;
  const bool lowered = LowerMatrixReduceImpl(inst);
  if (lowered && inst) RemoveFPFastMathMode(inst->result_id());
  active_fp_fast_math_mode_ = saved_fast_math_mode;
  return lowered;
}

bool HwLowerToStandardPass::LowerMatrixReduceImpl(Instruction* inst) {
  if (!inst || inst->NumInOperands() != 3) {
    ReportError(inst, "invalid OpCooperativeMatrixReduceHW");
    return false;
  }

  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* input =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  const MatrixTypeInfo* source = GetMatrixTypeForValue(input);
  uint32_t reduce_axis = 0;
  uint32_t combine_op = 0;
  if (!result || !source || !input || result->rows != source->rows ||
      result->cols != source->cols ||
      result->component_type_id != source->component_type_id ||
      !GetConstantU32(inst->GetSingleWordInOperand(1), &reduce_axis) ||
      reduce_axis > 1 ||
      !GetConstantU32(inst->GetSingleWordInOperand(2), &combine_op) ||
      combine_op > 2) {
    ReportError(inst, "invalid OpCooperativeMatrixReduceHW");
    return false;
  }

  if (result->rows * result->cols > max_unrolled_elements_) {
    return LowerMatrixReduceWithLoop(inst, reduce_axis, combine_op);
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> scalar_ids(result->rows * result->cols, 0);

  if (reduce_axis == 0) {
    for (uint32_t row = 0; row < result->rows; ++row) {
      uint32_t reduced =
          ExtractMatrixScalar(&builder, *source, input->result_id(), row, 0);
      if (reduced == 0) return false;
      for (uint32_t col = 1; col < result->cols; ++col) {
        const uint32_t value = ExtractMatrixScalar(
            &builder, *source, input->result_id(), row, col);
        reduced = BuildReduceCombine(&builder, result->component_type_id,
                                     combine_op, reduced, value);
        if (reduced == 0) return false;
      }
      for (uint32_t col = 0; col < result->cols; ++col) {
        scalar_ids[MatrixFlatIndex(*result, row, col)] = reduced;
      }
    }
  } else {
    for (uint32_t col = 0; col < result->cols; ++col) {
      uint32_t reduced =
          ExtractMatrixScalar(&builder, *source, input->result_id(), 0, col);
      if (reduced == 0) return false;
      for (uint32_t row = 1; row < result->rows; ++row) {
        const uint32_t value = ExtractMatrixScalar(
            &builder, *source, input->result_id(), row, col);
        reduced = BuildReduceCombine(&builder, result->component_type_id,
                                     combine_op, reduced, value);
        if (reduced == 0) return false;
      }
      for (uint32_t row = 0; row < result->rows; ++row) {
        scalar_ids[MatrixFlatIndex(*result, row, col)] = reduced;
      }
    }
  }

  return RebuildMatrixFromScalars(inst, *result, scalar_ids);
}

bool HwLowerToStandardPass::LowerMatrixReduceWithLoop(Instruction* inst,
                                                      uint32_t reduce_axis,
                                                      uint32_t combine_op) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* input =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  const MatrixTypeInfo* source = GetMatrixTypeForValue(input);
  BasicBlock* preheader_block = context()->get_instr_block(inst);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!result || !source || !input || !function) return false;

  const uint32_t source_pointer_type_id = GetOrCreatePointerType(
      source->lowered_type_id, spv::StorageClass::Function);
  const uint32_t result_pointer_type_id = GetOrCreatePointerType(
      result->lowered_type_id, spv::StorageClass::Function);
  const uint32_t accumulator_pointer_type_id = GetOrCreatePointerType(
      result->component_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_pointer_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_id = GetOrCreateUIntConstant(0);
  const uint32_t one_id = GetOrCreateUIntConstant(1);
  const uint32_t cols_id = GetOrCreateUIntConstant(result->cols);
  const uint32_t output_count_id =
      GetOrCreateUIntConstant(reduce_axis == 0 ? result->rows : result->cols);
  const uint32_t reduce_count_id =
      GetOrCreateUIntConstant(reduce_axis == 0 ? result->cols : result->rows);
  const uint32_t broadcast_count_id =
      GetOrCreateUIntConstant(reduce_axis == 0 ? result->cols : result->rows);
  const uint32_t result_zero_id = GetOrCreateZero(result->lowered_type_id);
  if (source_pointer_type_id == 0 || result_pointer_type_id == 0 ||
      accumulator_pointer_type_id == 0 || uint_type_id == 0 ||
      uint_pointer_type_id == 0 || bool_type_id == 0 || zero_id == 0 ||
      one_id == 0 || cols_id == 0 || output_count_id == 0 ||
      reduce_count_id == 0 || broadcast_count_id == 0 || result_zero_id == 0) {
    return false;
  }

  Instruction* source_variable =
      AddFunctionVariable(function, source_pointer_type_id);
  Instruction* result_variable =
      AddFunctionVariable(function, result_pointer_type_id);
  Instruction* output_index_variable =
      AddFunctionVariable(function, uint_pointer_type_id);
  Instruction* reduce_index_variable =
      AddFunctionVariable(function, uint_pointer_type_id);
  Instruction* broadcast_index_variable =
      AddFunctionVariable(function, uint_pointer_type_id);
  Instruction* accumulator_variable =
      AddFunctionVariable(function, accumulator_pointer_type_id);
  if (!source_variable || !result_variable || !output_index_variable ||
      !reduce_index_variable || !broadcast_index_variable ||
      !accumulator_variable) {
    return false;
  }

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() && &*split_iter != inst) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;
  std::unique_ptr<Instruction> enclosing_loop_merge;
  if (Instruction* loop_merge = preheader_block->GetLoopMergeInst()) {
    enclosing_loop_merge.reset(loop_merge->Clone(context()));
  }
  std::array<uint32_t, 12> labels;
  for (uint32_t& label : labels) {
    label = TakeNextId();
    if (label == 0) return false;
  }
  const uint32_t merge_label = labels[0];
  const uint32_t outer_header_label = labels[1];
  const uint32_t outer_body_label = labels[2];
  const uint32_t reduce_header_label = labels[3];
  const uint32_t reduce_body_label = labels[4];
  const uint32_t reduce_continue_label = labels[5];
  const uint32_t reduce_merge_label = labels[6];
  const uint32_t broadcast_header_label = labels[7];
  const uint32_t broadcast_body_label = labels[8];
  const uint32_t broadcast_continue_label = labels[9];
  const uint32_t broadcast_merge_label = labels[10];
  const uint32_t outer_continue_label = labels[11];
  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label, split_iter);
  BasicBlock* outer_header = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_header_label)),
      preheader_block);
  BasicBlock* outer_body = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_body_label)),
      outer_header);
  BasicBlock* reduce_header = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(reduce_header_label)),
      outer_body);
  BasicBlock* reduce_body = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(reduce_body_label)),
      reduce_header);
  BasicBlock* reduce_continue = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(reduce_continue_label)),
      reduce_body);
  BasicBlock* reduce_merge = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(reduce_merge_label)),
      reduce_continue);
  BasicBlock* broadcast_header = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(broadcast_header_label)),
      reduce_merge);
  BasicBlock* broadcast_body = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(broadcast_body_label)),
      broadcast_header);
  BasicBlock* broadcast_continue = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(broadcast_continue_label)),
      broadcast_body);
  BasicBlock* broadcast_merge = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(broadcast_merge_label)),
      broadcast_continue);
  BasicBlock* outer_continue = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_continue_label)),
      broadcast_merge);
  if (!merge_block || !outer_header || !outer_body || !reduce_header ||
      !reduce_body || !reduce_continue || !reduce_merge || !broadcast_header ||
      !broadcast_body || !broadcast_continue || !broadcast_merge ||
      !outer_continue) {
    return false;
  }
  if (enclosing_loop_merge) {
    Instruction* moved_loop_merge = merge_block->GetLoopMergeInst();
    if (!moved_loop_merge) return false;
    context()->KillInst(moved_loop_merge);
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(source_variable->result_id(),
                                  input->result_id()) ||
      !preheader_builder.AddStore(result_variable->result_id(),
                                  result_zero_id) ||
      !preheader_builder.AddStore(output_index_variable->result_id(),
                                  zero_id) ||
      (enclosing_loop_merge &&
       !preheader_builder.AddInstruction(std::move(enclosing_loop_merge))) ||
      !preheader_builder.AddBranch(outer_header_label)) {
    return false;
  }

  InstructionBuilder outer_header_builder(
      context(), outer_header,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* output_index = outer_header_builder.AddLoad(
      uint_type_id, output_index_variable->result_id());
  Instruction* outer_condition =
      output_index ? outer_header_builder.AddBinaryOp(
                         bool_type_id, spv::Op::OpULessThan,
                         output_index->result_id(), output_count_id)
                   : nullptr;
  if (!output_index || !outer_condition ||
      !outer_header_builder.AddLoopMerge(merge_label, outer_continue_label) ||
      !outer_header_builder.AddConditionalBranch(
          outer_condition->result_id(), outer_body_label, merge_label)) {
    return false;
  }

  auto build_source_index = [&](InstructionBuilder* builder,
                                uint32_t reduce_index_id) -> uint32_t {
    uint32_t source_row_id =
        reduce_axis == 0 ? output_index->result_id() : reduce_index_id;
    uint32_t source_col_id =
        reduce_axis == 0 ? reduce_index_id : output_index->result_id();
    Instruction* row_base = builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                                 source_row_id, cols_id);
    Instruction* source_index =
        row_base ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                        row_base->result_id(), source_col_id)
                 : nullptr;
    return source_index ? source_index->result_id() : 0;
  };

  InstructionBuilder outer_body_builder(
      context(), outer_body,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t first_source_index =
      build_source_index(&outer_body_builder, zero_id);
  const uint32_t initial_accumulator = BuildLogicalAggregateLoad(
      &outer_body_builder, source_variable->result_id(),
      source->component_type_id, source->packed_vec4_type_id,
      first_source_index);
  if (initial_accumulator == 0 ||
      !outer_body_builder.AddStore(accumulator_variable->result_id(),
                                   initial_accumulator) ||
      !outer_body_builder.AddStore(reduce_index_variable->result_id(),
                                   one_id) ||
      !outer_body_builder.AddBranch(reduce_header_label)) {
    return false;
  }

  InstructionBuilder reduce_header_builder(
      context(), reduce_header,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* reduce_index = reduce_header_builder.AddLoad(
      uint_type_id, reduce_index_variable->result_id());
  Instruction* reduce_condition =
      reduce_index ? reduce_header_builder.AddBinaryOp(
                         bool_type_id, spv::Op::OpULessThan,
                         reduce_index->result_id(), reduce_count_id)
                   : nullptr;
  if (!reduce_index || !reduce_condition ||
      !reduce_header_builder.AddLoopMerge(reduce_merge_label,
                                          reduce_continue_label) ||
      !reduce_header_builder.AddConditionalBranch(reduce_condition->result_id(),
                                                  reduce_body_label,
                                                  reduce_merge_label)) {
    return false;
  }

  InstructionBuilder reduce_body_builder(
      context(), reduce_body,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t source_index =
      build_source_index(&reduce_body_builder, reduce_index->result_id());
  const uint32_t value = BuildLogicalAggregateLoad(
      &reduce_body_builder, source_variable->result_id(),
      source->component_type_id, source->packed_vec4_type_id, source_index);
  Instruction* accumulator = reduce_body_builder.AddLoad(
      result->component_type_id, accumulator_variable->result_id());
  const uint32_t combined =
      accumulator
          ? BuildReduceCombine(&reduce_body_builder, result->component_type_id,
                               combine_op, accumulator->result_id(), value)
          : 0;
  if (combined == 0 ||
      !reduce_body_builder.AddStore(accumulator_variable->result_id(),
                                    combined) ||
      !reduce_body_builder.AddBranch(reduce_continue_label)) {
    return false;
  }

  InstructionBuilder reduce_continue_builder(
      context(), reduce_continue,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_reduce = reduce_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, reduce_index->result_id(), one_id);
  if (!next_reduce ||
      !reduce_continue_builder.AddStore(reduce_index_variable->result_id(),
                                        next_reduce->result_id()) ||
      !reduce_continue_builder.AddBranch(reduce_header_label)) {
    return false;
  }

  InstructionBuilder reduce_merge_builder(
      context(), reduce_merge,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* final_accumulator = reduce_merge_builder.AddLoad(
      result->component_type_id, accumulator_variable->result_id());
  if (!final_accumulator ||
      !reduce_merge_builder.AddStore(broadcast_index_variable->result_id(),
                                     zero_id) ||
      !reduce_merge_builder.AddBranch(broadcast_header_label)) {
    return false;
  }

  InstructionBuilder broadcast_header_builder(
      context(), broadcast_header,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* broadcast_index = broadcast_header_builder.AddLoad(
      uint_type_id, broadcast_index_variable->result_id());
  Instruction* broadcast_condition =
      broadcast_index ? broadcast_header_builder.AddBinaryOp(
                            bool_type_id, spv::Op::OpULessThan,
                            broadcast_index->result_id(), broadcast_count_id)
                      : nullptr;
  if (!broadcast_index || !broadcast_condition ||
      !broadcast_header_builder.AddLoopMerge(broadcast_merge_label,
                                             broadcast_continue_label) ||
      !broadcast_header_builder.AddConditionalBranch(
          broadcast_condition->result_id(), broadcast_body_label,
          broadcast_merge_label)) {
    return false;
  }

  InstructionBuilder broadcast_body_builder(
      context(), broadcast_body,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t destination_row_id = reduce_axis == 0
                                          ? output_index->result_id()
                                          : broadcast_index->result_id();
  const uint32_t destination_col_id = reduce_axis == 0
                                          ? broadcast_index->result_id()
                                          : output_index->result_id();
  Instruction* destination_row_base = broadcast_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, destination_row_id, cols_id);
  Instruction* destination_index =
      destination_row_base
          ? broadcast_body_builder.AddBinaryOp(
                uint_type_id, spv::Op::OpIAdd,
                destination_row_base->result_id(), destination_col_id)
          : nullptr;
  if (!destination_index ||
      !BuildLogicalAggregateStore(
          &broadcast_body_builder, result_variable->result_id(),
          result->component_type_id, result->packed_vec4_type_id,
          destination_index->result_id(), final_accumulator->result_id()) ||
      !broadcast_body_builder.AddBranch(broadcast_continue_label)) {
    return false;
  }

  InstructionBuilder broadcast_continue_builder(
      context(), broadcast_continue,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_broadcast = broadcast_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, broadcast_index->result_id(), one_id);
  if (!next_broadcast ||
      !broadcast_continue_builder.AddStore(
          broadcast_index_variable->result_id(), next_broadcast->result_id()) ||
      !broadcast_continue_builder.AddBranch(broadcast_header_label)) {
    return false;
  }

  InstructionBuilder broadcast_merge_builder(
      context(), broadcast_merge,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!broadcast_merge_builder.AddBranch(outer_continue_label)) return false;

  InstructionBuilder outer_continue_builder(
      context(), outer_continue,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_output = outer_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, output_index->result_id(), one_id);
  if (!next_output ||
      !outer_continue_builder.AddStore(output_index_variable->result_id(),
                                       next_output->result_id()) ||
      !outer_continue_builder.AddBranch(outer_header_label)) {
    return false;
  }

  inst->SetOpcode(spv::Op::OpLoad);
  inst->SetResultType(result->lowered_type_id);
  inst->SetInOperands({IdOperand(result_variable->result_id())});
  context()->UpdateDefUse(inst);
  return true;
}

bool HwLowerToStandardPass::LowerVectorMatrixMul(Instruction* inst,
                                                 bool has_bias) {
  const uint32_t saved_fast_math_mode = active_fp_fast_math_mode_;
  active_fp_fast_math_mode_ = GetFPFastMathMode(inst->result_id());
  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix = GetMatrixTypeForValue(matrix_inst);
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  if (!result || !input || !matrix || (has_bias && !bias)) {
    ReportError(inst, "invalid HW vector matrix multiply");
    active_fp_fast_math_mode_ = saved_fast_math_mode;
    return false;
  }

  const uint64_t mac_count = static_cast<uint64_t>(result->length) *
                             static_cast<uint64_t>(input->length);
  bool lowered =
      mac_count > max_unrolled_matmul_macs_
          ? LowerVectorMatrixMulWithLoop(inst, has_bias)
          : (CanUsePackedVec4VectorMatrixMul(*result, *input, *matrix, bias)
                 ? LowerVectorMatrixMulPackedVec4(inst, has_bias)
                 : LowerVectorMatrixMulScalarFallback(inst, has_bias));
  if (lowered) RemoveFPFastMathMode(inst->result_id());
  active_fp_fast_math_mode_ = saved_fast_math_mode;
  return lowered;
}

bool HwLowerToStandardPass::LowerVectorMatrixMulPackedVec4(Instruction* inst,
                                                           bool has_bias) {
  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix = GetMatrixTypeForValue(matrix_inst);
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  if (!result || !input || !matrix || (has_bias && !bias)) {
    ReportError(inst, "invalid HW vector matrix multiply");
    return false;
  }

  bool handled = false;
  if (!TryLowerDirectVectorMatrixMulPackedVec4(inst, has_bias, &handled)) {
    return false;
  }
  if (handled) return true;

  const uint32_t function_id = GetOrCreateVectorMatmulPatternFunctionPackedVec4(
      *result, *input, *matrix, bias, has_bias);
  std::vector<uint32_t> argument_ids = {input_inst->result_id(),
                                        matrix_inst->result_id()};
  if (has_bias) argument_ids.push_back(bias_inst->result_id());
  if (function_id == 0) return false;
  RebuildAsFunctionCall(inst, result->lowered_type_id, function_id,
                        argument_ids);
  return true;
}

bool HwLowerToStandardPass::LowerVectorMatrixMulScalarFallback(
    Instruction* inst, bool has_bias) {
  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix = GetMatrixTypeForValue(matrix_inst);
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  if (!result || !input || !matrix || (has_bias && !bias)) {
    ReportError(inst, "invalid HW vector matrix multiply");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> scalar_ids(result->length, 0);
  const uint32_t accumulator_type_id = result->component_type_id;
  const uint32_t zero_id = has_bias ? 0 : GetOrCreateZero(accumulator_type_id);
  if (!has_bias && zero_id == 0) return false;

  for (uint32_t col0 = 0; col0 < result->length;
       col0 += kDefaultVectorMatmulTileN) {
    const uint32_t tile_n =
        std::min(kDefaultVectorMatmulTileN, result->length - col0);
    std::vector<uint32_t> acc(tile_n, 0);

    for (uint32_t j = 0; j < tile_n; ++j) {
      const uint32_t col = col0 + j;
      acc[j] = has_bias ? ExtractVectorScalar(&builder, *bias,
                                              bias_inst->result_id(), col)
                        : zero_id;
      if (acc[j] == 0) return false;
    }

    for (uint32_t k = 0; k < input->length; ++k) {
      const uint32_t x_id =
          ExtractVectorScalar(&builder, *input, input_inst->result_id(), k);
      if (x_id == 0) return false;
      for (uint32_t j = 0; j < tile_n; ++j) {
        const uint32_t col = col0 + j;
        const uint32_t w_id = ExtractMatrixScalar(
            &builder, *matrix, matrix_inst->result_id(), k, col);
        if (w_id == 0) return false;
        const uint32_t accumulated = BuildMatmulAccumulate(
            &builder, accumulator_type_id, input->component_type_id, x_id,
            matrix->component_type_id, w_id, acc[j]);
        if (accumulated == 0) return false;
        acc[j] = accumulated;
      }
    }

    for (uint32_t j = 0; j < tile_n; ++j) {
      scalar_ids[col0 + j] = acc[j];
    }
  }

  for (uint32_t id : scalar_ids) {
    if (id == 0) return false;
  }

  if (IsPackedVec4(*result)) {
    std::vector<uint32_t> element_ids(result->packed_length, 0);
    for (uint32_t pack = 0; pack < result->packed_length; ++pack) {
      std::vector<uint32_t> lane_ids;
      lane_ids.reserve(kPackedVec4Width);
      for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
        lane_ids.push_back(scalar_ids[pack * kPackedVec4Width + lane]);
      }
      Instruction* vec =
          builder.AddCompositeConstruct(result->packed_vec4_type_id, lane_ids);
      if (!vec) return false;
      element_ids[pack] = vec->result_id();
    }
    RebuildAsCompositeConstruct(inst, result->lowered_type_id, element_ids);
    return true;
  }

  std::vector<uint32_t> element_ids = scalar_ids;
  RebuildAsCompositeConstruct(inst, result->lowered_type_id, element_ids);
  return true;
}

bool HwLowerToStandardPass::LowerVectorMatrixMulWithLoop(Instruction* inst,
                                                         bool has_bias) {
  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix = GetMatrixTypeForValue(matrix_inst);
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  BasicBlock* preheader_block = context()->get_instr_block(inst);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!result || !input || !matrix || (has_bias && !bias) || !function) {
    return false;
  }

  const uint32_t input_pointer_type_id = GetOrCreatePointerType(
      input->lowered_type_id, spv::StorageClass::Function);
  const uint32_t matrix_pointer_type_id = GetOrCreatePointerType(
      matrix->lowered_type_id, spv::StorageClass::Function);
  const uint32_t bias_pointer_type_id =
      has_bias ? GetOrCreatePointerType(bias->lowered_type_id,
                                        spv::StorageClass::Function)
               : 0;
  const uint32_t result_pointer_type_id = GetOrCreatePointerType(
      result->lowered_type_id, spv::StorageClass::Function);
  const uint32_t accumulator_pointer_type_id = GetOrCreatePointerType(
      result->component_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_pointer_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_index_id = GetOrCreateUIntConstant(0);
  const uint32_t one_id = GetOrCreateUIntConstant(1);
  const uint32_t output_count_id = GetOrCreateUIntConstant(result->length);
  const uint32_t inner_count_id = GetOrCreateUIntConstant(input->length);
  const uint32_t accumulator_zero_id =
      GetOrCreateZero(result->component_type_id);
  const uint32_t result_zero_id = GetOrCreateZero(result->lowered_type_id);
  if (input_pointer_type_id == 0 || matrix_pointer_type_id == 0 ||
      (has_bias && bias_pointer_type_id == 0) || result_pointer_type_id == 0 ||
      accumulator_pointer_type_id == 0 || uint_type_id == 0 ||
      uint_pointer_type_id == 0 || bool_type_id == 0 || zero_index_id == 0 ||
      one_id == 0 || output_count_id == 0 || inner_count_id == 0 ||
      accumulator_zero_id == 0 || result_zero_id == 0) {
    return false;
  }

  Instruction* input_variable =
      AddFunctionVariable(function, input_pointer_type_id);
  Instruction* matrix_variable =
      AddFunctionVariable(function, matrix_pointer_type_id);
  Instruction* bias_variable =
      has_bias ? AddFunctionVariable(function, bias_pointer_type_id) : nullptr;
  Instruction* result_variable =
      AddFunctionVariable(function, result_pointer_type_id);
  Instruction* output_index_variable =
      AddFunctionVariable(function, uint_pointer_type_id);
  Instruction* k_variable = AddFunctionVariable(function, uint_pointer_type_id);
  Instruction* accumulator_variable =
      AddFunctionVariable(function, accumulator_pointer_type_id);
  if (!input_variable || !matrix_variable || (has_bias && !bias_variable) ||
      !result_variable || !output_index_variable || !k_variable ||
      !accumulator_variable) {
    return false;
  }

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() && &*split_iter != inst) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;
  std::unique_ptr<Instruction> enclosing_loop_merge;
  if (Instruction* loop_merge = preheader_block->GetLoopMergeInst()) {
    enclosing_loop_merge.reset(loop_merge->Clone(context()));
  }
  std::array<uint32_t, 8> labels;
  for (uint32_t& label : labels) {
    label = TakeNextId();
    if (label == 0) return false;
  }
  const uint32_t merge_label = labels[0];
  const uint32_t outer_header_label = labels[1];
  const uint32_t outer_body_label = labels[2];
  const uint32_t k_header_label = labels[3];
  const uint32_t k_body_label = labels[4];
  const uint32_t k_continue_label = labels[5];
  const uint32_t k_merge_label = labels[6];
  const uint32_t outer_continue_label = labels[7];
  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label, split_iter);
  BasicBlock* outer_header = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_header_label)),
      preheader_block);
  BasicBlock* outer_body = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_body_label)),
      outer_header);
  BasicBlock* k_header = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_header_label)), outer_body);
  BasicBlock* k_body = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_body_label)), k_header);
  BasicBlock* k_continue = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_continue_label)), k_body);
  BasicBlock* k_merge = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_merge_label)), k_continue);
  BasicBlock* outer_continue = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(outer_continue_label)),
      k_merge);
  if (!merge_block || !outer_header || !outer_body || !k_header || !k_body ||
      !k_continue || !k_merge || !outer_continue) {
    return false;
  }
  if (enclosing_loop_merge) {
    Instruction* moved_loop_merge = merge_block->GetLoopMergeInst();
    if (!moved_loop_merge) return false;
    context()->KillInst(moved_loop_merge);
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(input_variable->result_id(),
                                  input_inst->result_id()) ||
      !preheader_builder.AddStore(matrix_variable->result_id(),
                                  matrix_inst->result_id()) ||
      (has_bias && !preheader_builder.AddStore(bias_variable->result_id(),
                                               bias_inst->result_id())) ||
      !preheader_builder.AddStore(result_variable->result_id(),
                                  result_zero_id) ||
      !preheader_builder.AddStore(output_index_variable->result_id(),
                                  zero_index_id) ||
      (enclosing_loop_merge &&
       !preheader_builder.AddInstruction(std::move(enclosing_loop_merge))) ||
      !preheader_builder.AddBranch(outer_header_label)) {
    return false;
  }

  InstructionBuilder outer_header_builder(
      context(), outer_header,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* output_index = outer_header_builder.AddLoad(
      uint_type_id, output_index_variable->result_id());
  Instruction* outer_condition =
      output_index ? outer_header_builder.AddBinaryOp(
                         bool_type_id, spv::Op::OpULessThan,
                         output_index->result_id(), output_count_id)
                   : nullptr;
  if (!output_index || !outer_condition ||
      !outer_header_builder.AddLoopMerge(merge_label, outer_continue_label) ||
      !outer_header_builder.AddConditionalBranch(
          outer_condition->result_id(), outer_body_label, merge_label)) {
    return false;
  }

  InstructionBuilder outer_body_builder(
      context(), outer_body,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t initial_accumulator =
      has_bias ? BuildLogicalAggregateLoad(
                     &outer_body_builder, bias_variable->result_id(),
                     bias->component_type_id, bias->packed_vec4_type_id,
                     output_index->result_id())
               : accumulator_zero_id;
  if (initial_accumulator == 0 ||
      !outer_body_builder.AddStore(accumulator_variable->result_id(),
                                   initial_accumulator) ||
      !outer_body_builder.AddStore(k_variable->result_id(), zero_index_id) ||
      !outer_body_builder.AddBranch(k_header_label)) {
    return false;
  }

  InstructionBuilder k_header_builder(
      context(), k_header,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* k =
      k_header_builder.AddLoad(uint_type_id, k_variable->result_id());
  Instruction* k_condition =
      k ? k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                       k->result_id(), inner_count_id)
        : nullptr;
  if (!k || !k_condition ||
      !k_header_builder.AddLoopMerge(k_merge_label, k_continue_label) ||
      !k_header_builder.AddConditionalBranch(k_condition->result_id(),
                                             k_body_label, k_merge_label)) {
    return false;
  }

  InstructionBuilder k_body_builder(
      context(), k_body,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* matrix_row_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k->result_id(), output_count_id);
  Instruction* matrix_index =
      matrix_row_base
          ? k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                       matrix_row_base->result_id(),
                                       output_index->result_id())
          : nullptr;
  if (!matrix_index) return false;
  const uint32_t input_value = BuildLogicalAggregateLoad(
      &k_body_builder, input_variable->result_id(), input->component_type_id,
      input->packed_vec4_type_id, k->result_id());
  const uint32_t matrix_value = BuildLogicalAggregateLoad(
      &k_body_builder, matrix_variable->result_id(), matrix->component_type_id,
      matrix->packed_vec4_type_id, matrix_index->result_id());
  Instruction* accumulator = k_body_builder.AddLoad(
      result->component_type_id, accumulator_variable->result_id());
  const uint32_t accumulated =
      accumulator
          ? BuildMatmulAccumulate(&k_body_builder, result->component_type_id,
                                  input->component_type_id, input_value,
                                  matrix->component_type_id, matrix_value,
                                  accumulator->result_id())
          : 0;
  if (accumulated == 0 ||
      !k_body_builder.AddStore(accumulator_variable->result_id(),
                               accumulated) ||
      !k_body_builder.AddBranch(k_continue_label)) {
    return false;
  }

  InstructionBuilder k_continue_builder(
      context(), k_continue,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k->result_id(), one_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_variable->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label)) {
    return false;
  }

  InstructionBuilder k_merge_builder(
      context(), k_merge,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* final_accumulator = k_merge_builder.AddLoad(
      result->component_type_id, accumulator_variable->result_id());
  if (!final_accumulator ||
      !BuildLogicalAggregateStore(
          &k_merge_builder, result_variable->result_id(),
          result->component_type_id, result->packed_vec4_type_id,
          output_index->result_id(), final_accumulator->result_id()) ||
      !k_merge_builder.AddBranch(outer_continue_label)) {
    return false;
  }

  InstructionBuilder outer_continue_builder(
      context(), outer_continue,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_output = outer_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, output_index->result_id(), one_id);
  if (!next_output ||
      !outer_continue_builder.AddStore(output_index_variable->result_id(),
                                       next_output->result_id()) ||
      !outer_continue_builder.AddBranch(outer_header_label)) {
    return false;
  }

  inst->SetOpcode(spv::Op::OpLoad);
  inst->SetResultType(result->lowered_type_id);
  inst->SetInOperands({IdOperand(result_variable->result_id())});
  context()->UpdateDefUse(inst);
  return true;
}

uint32_t HwLowerToStandardPass::BuildVectorTimesScalar(
    InstructionBuilder* builder, spv::Op scale_opcode, uint32_t vec4_type_id,
    uint32_t vector_id, uint32_t scalar_id) {
  const uint32_t scalar_vec_id =
      BuildScalarSplat(builder, vec4_type_id, scalar_id);
  if (scalar_vec_id == 0) return 0;
  Instruction* mul = builder->AddBinaryOp(vec4_type_id, scale_opcode, vector_id,
                                          scalar_vec_id);
  return mul ? mul->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildScalarSplat(InstructionBuilder* builder,
                                                 uint32_t vec4_type_id,
                                                 uint32_t scalar_id) {
  std::vector<uint32_t> lane_ids(kPackedVec4Width, scalar_id);
  Instruction* scalar_vec =
      builder->AddCompositeConstruct(vec4_type_id, lane_ids);
  return scalar_vec ? scalar_vec->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildFma(InstructionBuilder* builder,
                                         uint32_t type_id,
                                         uint32_t multiplicand_id,
                                         uint32_t multiplier_id,
                                         uint32_t addend_id) {
  const uint32_t glsl_std450_id = GetOrCreateGLSLStd450Import();
  if (glsl_std450_id == 0) return 0;
  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::unique_ptr<Instruction> fma = MakeUnique<Instruction>(
      context(), spv::Op::OpExtInst, type_id, result_id,
      std::initializer_list<Operand>{
          IdOperand(glsl_std450_id),
          {SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
           {static_cast<uint32_t>(GLSLstd450Fma)}},
          IdOperand(multiplicand_id),
          IdOperand(multiplier_id),
          IdOperand(addend_id)});
  Instruction* added = builder->AddInstruction(std::move(fma));
  ApplyActiveFPFastMathMode(added);
  return added ? added->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildMatmulAccumulate(
    InstructionBuilder* builder, uint32_t accumulator_type_id,
    uint32_t lhs_type_id, uint32_t lhs_id, uint32_t rhs_type_id,
    uint32_t rhs_id, uint32_t accumulator_id) {
  if (!builder || lhs_id == 0 || rhs_id == 0 || accumulator_id == 0) return 0;
  Instruction* accumulator_type =
      get_def_use_mgr()->GetDef(accumulator_type_id);
  const NumericScalarInfo accumulator =
      DescribeNumericScalarType(accumulator_type);
  const NumericScalarInfo lhs =
      DescribeNumericScalarType(get_def_use_mgr()->GetDef(lhs_type_id));
  const NumericScalarInfo rhs =
      DescribeNumericScalarType(get_def_use_mgr()->GetDef(rhs_type_id));
  if (!accumulator.valid || !lhs.valid || !rhs.valid ||
      accumulator.is_float != lhs.is_float ||
      accumulator.is_float != rhs.is_float || accumulator.width < lhs.width ||
      accumulator.width < rhs.width) {
    return 0;
  }

  auto widen = [&](uint32_t source_type_id, uint32_t value_id,
                   const NumericScalarInfo& source) -> uint32_t {
    if (source_type_id == accumulator_type_id) return value_id;
    if (source.width == accumulator.width) {
      Instruction* converted = builder->AddUnaryOp(
          accumulator_type_id, spv::Op::OpBitcast, value_id);
      return converted ? converted->result_id() : 0;
    }
    if (source.is_float) {
      Instruction* converted = builder->AddUnaryOp(
          accumulator_type_id, spv::Op::OpFConvert, value_id);
      ApplyActiveFPFastMathMode(converted);
      return converted ? converted->result_id() : 0;
    }

    // OpSConvert and OpUConvert require a result type with the matching
    // signedness.  Widen in that domain first, then preserve the resulting bit
    // pattern with OpBitcast when the accumulator has the opposite signedness.
    const uint32_t widened_type_id =
        GetOrCreateIntegerType(accumulator.width, source.is_signed);
    Instruction* widened =
        widened_type_id
            ? builder->AddUnaryOp(
                  widened_type_id,
                  source.is_signed ? spv::Op::OpSConvert : spv::Op::OpUConvert,
                  value_id)
            : nullptr;
    if (!widened) return 0;
    if (widened_type_id == accumulator_type_id) return widened->result_id();
    Instruction* converted = builder->AddUnaryOp(
        accumulator_type_id, spv::Op::OpBitcast, widened->result_id());
    return converted ? converted->result_id() : 0;
  };

  const uint32_t widened_lhs = widen(lhs_type_id, lhs_id, lhs);
  const uint32_t widened_rhs = widen(rhs_type_id, rhs_id, rhs);
  if (widened_lhs == 0 || widened_rhs == 0) return 0;
  if (accumulator.is_float) {
    return BuildFma(builder, accumulator_type_id, widened_lhs, widened_rhs,
                    accumulator_id);
  }

  Instruction* product = builder->AddBinaryOp(
      accumulator_type_id, spv::Op::OpIMul, widened_lhs, widened_rhs);
  Instruction* sum =
      product ? builder->AddBinaryOp(accumulator_type_id, spv::Op::OpIAdd,
                                     product->result_id(), accumulator_id)
              : nullptr;
  return sum ? sum->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildHorizontalReduce(
    InstructionBuilder* builder, uint32_t component_type_id,
    uint32_t vector_id) {
  uint32_t sum =
      ExtractCompositeElement(builder, component_type_id, vector_id, 0);
  if (sum == 0) return 0;
  for (uint32_t lane = 1; lane < kPackedVec4Width; ++lane) {
    const uint32_t value =
        ExtractCompositeElement(builder, component_type_id, vector_id, lane);
    if (value == 0) return 0;
    Instruction* add =
        builder->AddBinaryOp(component_type_id, spv::Op::OpFAdd, sum, value);
    if (!add) return 0;
    ApplyActiveFPFastMathMode(add);
    sum = add->result_id();
  }
  return sum;
}

uint32_t HwLowerToStandardPass::BuildReduceCombine(InstructionBuilder* builder,
                                                   uint32_t component_type_id,
                                                   uint32_t combine_op,
                                                   uint32_t lhs_id,
                                                   uint32_t rhs_id) {
  if (!builder || lhs_id == 0 || rhs_id == 0 || combine_op > 2) return 0;

  Instruction* component_type = get_def_use_mgr()->GetDef(component_type_id);
  if (!component_type) return 0;
  if (combine_op == 0) {
    const spv::Op add_opcode = component_type->opcode() == spv::Op::OpTypeFloat
                                   ? spv::Op::OpFAdd
                                   : spv::Op::OpIAdd;
    Instruction* add =
        builder->AddBinaryOp(component_type_id, add_opcode, lhs_id, rhs_id);
    if (add_opcode == spv::Op::OpFAdd) ApplyActiveFPFastMathMode(add);
    return add ? add->result_id() : 0;
  }

  uint32_t ext_opcode = 0;
  if (component_type->opcode() == spv::Op::OpTypeFloat) {
    ext_opcode = combine_op == 1 ? GLSLstd450FMin : GLSLstd450FMax;
  } else if (component_type->opcode() == spv::Op::OpTypeInt &&
             component_type->NumInOperands() >= 2) {
    const bool is_signed = component_type->GetSingleWordInOperand(1) != 0;
    if (is_signed) {
      ext_opcode = combine_op == 1 ? GLSLstd450SMin : GLSLstd450SMax;
    } else {
      ext_opcode = combine_op == 1 ? GLSLstd450UMin : GLSLstd450UMax;
    }
  } else {
    return 0;
  }

  const uint32_t import_id = GetOrCreateGLSLStd450Import();
  const uint32_t result_id = TakeNextId();
  if (import_id == 0 || result_id == 0) return 0;
  std::unique_ptr<Instruction> ext_inst = MakeUnique<Instruction>(
      context(), spv::Op::OpExtInst, component_type_id, result_id,
      std::initializer_list<Operand>{
          IdOperand(import_id),
          {SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER, {ext_opcode}},
          IdOperand(lhs_id),
          IdOperand(rhs_id)});
  Instruction* added = builder->AddInstruction(std::move(ext_inst));
  ApplyActiveFPFastMathMode(added);
  return added ? added->result_id() : 0;
}

bool HwLowerToStandardPass::BuildVectorMatrixMulPatternPackedVec4(
    InstructionBuilder* builder, const VectorTypeInfo& result,
    const VectorTypeInfo& input, const MatrixTypeInfo& matrix,
    const VectorTypeInfo* /*bias*/, uint32_t input_id, uint32_t matrix_id,
    uint32_t bias_id, bool has_bias, std::vector<uint32_t>* element_ids) {
  element_ids->assign(result.packed_length, 0);

  const uint32_t vec4_type_id = result.packed_vec4_type_id;
  const uint32_t zero4_id = has_bias ? 0 : GetOrCreateZero(vec4_type_id);
  if (!has_bias && zero4_id == 0) return false;

  for (uint32_t out_pack = 0; out_pack < result.packed_length; ++out_pack) {
    uint32_t acc = has_bias ? ExtractCompositeElement(builder, vec4_type_id,
                                                      bias_id, out_pack)
                            : zero4_id;
    if (acc == 0) return false;

    const uint32_t out_col = out_pack * kPackedVec4Width;
    for (uint32_t k = 0; k < input.length; ++k) {
      const uint32_t scalar = ExtractVectorScalar(builder, input, input_id, k);
      const uint32_t weight4 = BuildMatrixRowVector(builder, matrix, matrix_id,
                                                    k, out_col, vec4_type_id);
      if (scalar == 0 || weight4 == 0) return false;

      const uint32_t scalar_vec =
          BuildScalarSplat(builder, vec4_type_id, scalar);
      if (scalar_vec == 0) return false;
      acc = BuildFma(builder, vec4_type_id, weight4, scalar_vec, acc);
      if (acc == 0) return false;
    }

    (*element_ids)[out_pack] = acc;
  }

  for (uint32_t id : *element_ids) {
    if (id == 0) return false;
  }
  return true;
}

bool HwLowerToStandardPass::BuildMatmulPatternPackedVec4(
    InstructionBuilder* builder, const MatrixTypeInfo& result,
    const MatrixTypeInfo& a, const MatrixTypeInfo& b, const MatrixTypeInfo& c,
    uint32_t a_id, uint32_t b_id, uint32_t c_id,
    std::vector<uint32_t>* element_ids) {
  element_ids->assign(result.rows * result.packed_cols, 0);

  const uint32_t vec4_type_id = result.packed_vec4_type_id;
  if (vec4_type_id == 0) return false;
  for (uint32_t row = 0; row < result.rows; ++row) {
    for (uint32_t col_pack = 0; col_pack < result.packed_cols; ++col_pack) {
      const uint32_t col0 = col_pack * kPackedVec4Width;
      uint32_t acc = ExtractCompositeElement(
          builder, vec4_type_id, c_id, MatrixPackedIndex(c, row, col_pack));
      if (acc == 0) return false;
      for (uint32_t k = 0; k < a.cols; ++k) {
        const uint32_t scalar = ExtractMatrixScalar(builder, a, a_id, row, k);
        const uint32_t weights =
            BuildMatrixRowVector(builder, b, b_id, k, col0, vec4_type_id);
        const uint32_t scalar_vec =
            scalar ? BuildScalarSplat(builder, vec4_type_id, scalar) : 0;
        if (weights == 0 || scalar_vec == 0) return false;
        acc = BuildFma(builder, vec4_type_id, scalar_vec, weights, acc);
        if (acc == 0) return false;
      }
      (*element_ids)[MatrixPackedIndex(result, row, col_pack)] = acc;
    }
  }

  for (uint32_t id : *element_ids) {
    if (id == 0) return false;
  }
  return true;
}

}  // namespace opt
}  // namespace spvtools
