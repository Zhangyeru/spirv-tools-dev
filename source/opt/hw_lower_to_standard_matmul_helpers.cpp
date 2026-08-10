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

bool HwLowerToStandardPass::PrepareMatmulPatternFunctions() {
  // Validation only — pattern functions are created lazily via
  // GetOrCreateMatmulPatternFunctionPackedVec4 when needed (by the direct
  // matmul path or the non-fused store path).  This avoids generating dead
  // pattern functions for matmuls that will be handled by the fused
  // matrix-matmul-store path in LowerHwInstructions.
  std::vector<Instruction*> matmul_insts;
  get_module()->ForEachInst([&matmul_insts](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpCooperativeMatrixMulAddHW) {
      matmul_insts.push_back(inst);
    }
  });

  for (Instruction* inst : matmul_insts) {
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
  }

  return true;
}

void HwLowerToStandardPass::AddGeneratedFunction(
    std::unique_ptr<Function> function, uint32_t function_id,
    bool may_write_memory) {
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    auto* def_use_mgr = context()->get_def_use_mgr();
    function->ForEachInst([def_use_mgr](Instruction* inst) {
      def_use_mgr->AnalyzeInstDef(inst);
    });
    function->ForEachInst([def_use_mgr](Instruction* inst) {
      def_use_mgr->AnalyzeInstUse(inst);
    });
  }

  if (context()->AreAnalysesValid(IRContext::kAnalysisInstrToBlockMapping)) {
    for (BasicBlock& basic_block : *function) {
      context()->set_instr_block(basic_block.GetLabelInst(), &basic_block);
      for (Instruction& inst : basic_block) {
        context()->set_instr_block(&inst, &basic_block);
      }
    }
  }

  context()->AddFunction(std::move(function));

  // Instructions built in a detached generated function are deliberately not
  // registered with def-use until the function is complete.  Decorations that
  // target those instructions must therefore be added only after their defs
  // have been registered above; AddDecorationVal immediately analyzes the
  // target use when def-use is valid.
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    auto pending = pending_fp_fast_math_modes_.begin();
    while (pending != pending_fp_fast_math_modes_.end()) {
      if (context()->get_def_use_mgr()->GetDef(pending->first)) {
        context()->get_decoration_mgr()->AddDecorationVal(
            pending->first, uint32_t(spv::Decoration::FPFastMathMode),
            pending->second);
        pending = pending_fp_fast_math_modes_.erase(pending);
      } else {
        ++pending;
      }
    }
  }

  generated_function_ids_.insert(function_id);
  if (!may_write_memory) {
    read_only_generated_function_ids_.insert(function_id);
  }
}

std::string HwLowerToStandardPass::MemoryOperandsKey(
    const std::vector<Operand>& memory_operands) const {
  std::string key;
  for (const Operand& operand : memory_operands) {
    key += std::to_string(static_cast<uint32_t>(operand.type));
    key += ':';
    for (uint32_t word : operand.words) {
      key += std::to_string(word);
      key += ',';
    }
    key += ';';
  }
  return key;
}

uint32_t HwLowerToStandardPass::GetOrCreateFunctionType(
    uint32_t return_type_id, const std::vector<uint32_t>& param_type_ids) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() != spv::Op::OpTypeFunction ||
        inst.NumInOperands() != param_type_ids.size() + 1 ||
        inst.GetSingleWordInOperand(0) != return_type_id) {
      continue;
    }

    bool matches = true;
    for (uint32_t i = 0; i < param_type_ids.size(); ++i) {
      if (inst.GetSingleWordInOperand(i + 1) != param_type_ids[i]) {
        matches = false;
        break;
      }
    }
    if (matches) return inst.result_id();
  }

  const uint32_t type_id = TakeNextId();
  if (type_id == 0) return 0;
  std::vector<Operand> operands;
  operands.reserve(param_type_ids.size() + 1);
  operands.push_back(IdOperand(return_type_id));
  for (uint32_t param_type_id : param_type_ids) {
    operands.push_back(IdOperand(param_type_id));
  }
  context()->AddType(MakeUnique<Instruction>(context(), spv::Op::OpTypeFunction,
                                             0, type_id, operands));
  return type_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateTileWeightFunctionPackedVec4(
    const MatrixTypeInfo& matrix) {
  const std::string key = TileWeightFunctionKey(matrix);
  auto cached = tile_weight_functions_.find(key);
  if (cached != tile_weight_functions_.end()) return cached->second;

  Instruction* insertion_point =
      get_def_use_mgr()->GetDef(matrix.lowered_type_id);
  if (!insertion_point) return 0;
  const uint32_t weight_array_type_id = GetOrCreatePackedArrayType(
      matrix.packed_vec4_type_id, kPackedVec4Width, insertion_point);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t function_type_id = GetOrCreateFunctionType(
      weight_array_type_id,
      {matrix.lowered_type_id, uint_type_id, uint_type_id});
  if (weight_array_type_id == 0 || uint_type_id == 0 || function_type_id == 0) {
    return 0;
  }

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, weight_array_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t matrix_param_id = TakeNextId();
  const uint32_t row_param_id = TakeNextId();
  const uint32_t col_pack_param_id = TakeNextId();
  if (matrix_param_id == 0 || row_param_id == 0 || col_pack_param_id == 0) {
    return 0;
  }
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, matrix.lowered_type_id,
      matrix_param_id, std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, row_param_id,
      std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, col_pack_param_id,
      std::initializer_list<Operand>{}));

  const uint32_t label_id = TakeNextId();
  if (label_id == 0) return 0;
  std::unique_ptr<BasicBlock> block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));

  InstructionBuilder builder(context(), block.get());
  const uint32_t matrix_ptr_type_id = GetOrCreatePointerType(
      matrix.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec4_ptr_type_id = GetOrCreatePointerType(
      matrix.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t packed_cols_id = GetOrCreateUIntConstant(matrix.packed_cols);
  if (matrix_ptr_type_id == 0 || vec4_ptr_type_id == 0 || packed_cols_id == 0) {
    return 0;
  }

  Instruction* matrix_var = builder.AddVariable(
      matrix_ptr_type_id, static_cast<uint32_t>(spv::StorageClass::Function));
  if (!matrix_var) return 0;
  if (!builder.AddStore(matrix_var->result_id(), matrix_param_id)) return 0;

  std::vector<uint32_t> weight_vec_ids;
  weight_vec_ids.reserve(kPackedVec4Width);
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
    const uint32_t row_lane_id = GetOrCreateUIntConstant(row_lane);
    if (row_lane_id == 0) return 0;
    Instruction* row = builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                           row_param_id, row_lane_id);
    if (!row) return 0;
    Instruction* row_offset = builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row->result_id(), packed_cols_id);
    if (!row_offset) return 0;
    Instruction* flat_index =
        builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                            row_offset->result_id(), col_pack_param_id);
    if (!flat_index) return 0;
    Instruction* vec_ptr = builder.AddAccessChain(
        vec4_ptr_type_id, matrix_var->result_id(), {flat_index->result_id()});
    if (!vec_ptr) return 0;
    Instruction* vec =
        builder.AddLoad(matrix.packed_vec4_type_id, vec_ptr->result_id());
    if (!vec) return 0;
    weight_vec_ids.push_back(vec->result_id());
  }

  Instruction* weights =
      builder.AddCompositeConstruct(weight_array_type_id, weight_vec_ids);
  if (!weights) return 0;
  if (!builder.AddUnaryOp(0, spv::Op::OpReturnValue, weights->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);

  tile_weight_functions_[key] = function_id;
  return function_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateMatmulTileWeightFunctionPackedVec4(
    const MatrixTypeInfo& matrix) {
  const std::string key = MatmulTileWeightFunctionKey(matrix);
  auto cached = matmul_tile_weight_functions_.find(key);
  if (cached != matmul_tile_weight_functions_.end()) return cached->second;

  Instruction* insertion_point =
      get_def_use_mgr()->GetDef(matrix.lowered_type_id);
  if (!insertion_point) return 0;
  const uint32_t weight_array_type_id = GetOrCreatePackedArrayType(
      matrix.packed_vec4_type_id, kPackedVec4Width, insertion_point);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t function_type_id = GetOrCreateFunctionType(
      weight_array_type_id,
      {matrix.lowered_type_id, uint_type_id, uint_type_id});
  if (weight_array_type_id == 0 || uint_type_id == 0 || function_type_id == 0) {
    return 0;
  }

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, weight_array_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t matrix_param_id = TakeNextId();
  const uint32_t row_param_id = TakeNextId();
  const uint32_t col_pack_param_id = TakeNextId();
  if (matrix_param_id == 0 || row_param_id == 0 || col_pack_param_id == 0) {
    return 0;
  }
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, matrix.lowered_type_id,
      matrix_param_id, std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, row_param_id,
      std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, col_pack_param_id,
      std::initializer_list<Operand>{}));

  const uint32_t label_id = TakeNextId();
  if (label_id == 0) return 0;
  std::unique_ptr<BasicBlock> block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));

  InstructionBuilder builder(context(), block.get());
  const uint32_t matrix_ptr_type_id = GetOrCreatePointerType(
      matrix.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec4_ptr_type_id = GetOrCreatePointerType(
      matrix.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t packed_cols_id = GetOrCreateUIntConstant(matrix.packed_cols);
  if (matrix_ptr_type_id == 0 || vec4_ptr_type_id == 0 || packed_cols_id == 0) {
    return 0;
  }

  Instruction* matrix_var = builder.AddVariable(
      matrix_ptr_type_id, static_cast<uint32_t>(spv::StorageClass::Function));
  if (!matrix_var) return 0;
  if (!builder.AddStore(matrix_var->result_id(), matrix_param_id)) return 0;

  std::vector<uint32_t> weight_vec_ids;
  weight_vec_ids.reserve(kPackedVec4Width);
  for (uint32_t col_lane = 0; col_lane < kPackedVec4Width; ++col_lane) {
    std::vector<uint32_t> lane_ids;
    lane_ids.reserve(kPackedVec4Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
      const uint32_t row_lane_id = GetOrCreateUIntConstant(row_lane);
      if (row_lane_id == 0) return 0;
      Instruction* row = builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                             row_param_id, row_lane_id);
      if (!row) return 0;
      Instruction* row_offset = builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, row->result_id(), packed_cols_id);
      if (!row_offset) return 0;
      Instruction* flat_index =
          builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                              row_offset->result_id(), col_pack_param_id);
      if (!flat_index) return 0;
      Instruction* vec_ptr = builder.AddAccessChain(
          vec4_ptr_type_id, matrix_var->result_id(), {flat_index->result_id()});
      if (!vec_ptr) return 0;
      Instruction* vec =
          builder.AddLoad(matrix.packed_vec4_type_id, vec_ptr->result_id());
      if (!vec) return 0;
      const uint32_t value_id = ExtractCompositeElement(
          &builder, matrix.component_type_id, vec->result_id(), col_lane);
      if (value_id == 0) return 0;
      lane_ids.push_back(value_id);
    }

    Instruction* weight_vec =
        builder.AddCompositeConstruct(matrix.packed_vec4_type_id, lane_ids);
    if (!weight_vec) return 0;
    weight_vec_ids.push_back(weight_vec->result_id());
  }

  Instruction* weights =
      builder.AddCompositeConstruct(weight_array_type_id, weight_vec_ids);
  if (!weights) return 0;
  if (!builder.AddUnaryOp(0, spv::Op::OpReturnValue, weights->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);

  matmul_tile_weight_functions_[key] = function_id;
  return function_id;
}

uint32_t
HwLowerToStandardPass::GetOrCreateVectorMatmulPatternFunctionPackedVec4(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias) {
  const std::string key =
      VectorMatmulPatternFunctionKey(result, input, matrix, bias, has_bias);
  auto cached = vector_matmul_pattern_functions_.find(key);
  if (cached != vector_matmul_pattern_functions_.end()) return cached->second;

  // The packed loop below groups four K terms and horizontally reduces them.
  // That changes the specified K-order unless reassociation is explicitly
  // allowed.  Small operations use the strict, K-ordered builder here; large
  // operations have already been redirected to the generic structured-loop
  // lowering by LowerVectorMatrixMul.
  const bool use_loop_path = false;

  std::vector<uint32_t> param_type_ids = {input.lowered_type_id,
                                          matrix.lowered_type_id};
  if (has_bias) {
    if (!bias) return 0;
    param_type_ids.push_back(bias->lowered_type_id);
  }
  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, param_type_ids);
  if (function_type_id == 0) return 0;

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, result.lowered_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t input_param_id = TakeNextId();
  const uint32_t matrix_param_id = TakeNextId();
  uint32_t bias_param_id = 0;
  if (input_param_id == 0 || matrix_param_id == 0) return 0;
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, input.lowered_type_id,
      input_param_id, std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, matrix.lowered_type_id,
      matrix_param_id, std::initializer_list<Operand>{}));
  if (has_bias) {
    bias_param_id = TakeNextId();
    if (bias_param_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, bias->lowered_type_id,
        bias_param_id, std::initializer_list<Operand>{}));
  }

  const uint32_t label_id = TakeNextId();
  if (label_id == 0) return 0;
  std::unique_ptr<BasicBlock> block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));

  InstructionBuilder builder(context(), block.get());
  const uint32_t vec4_type_id = result.packed_vec4_type_id;
  const uint32_t zero4_id = GetOrCreateZero(vec4_type_id);
  if (zero4_id == 0) return 0;

  if (!use_loop_path) {
    std::vector<uint32_t> element_ids;
    if (!BuildVectorMatrixMulPatternPackedVec4(
            &builder, result, input, matrix, bias, input_param_id,
            matrix_param_id, bias_param_id, has_bias, &element_ids)) {
      return 0;
    }
    Instruction* result_construct =
        builder.AddCompositeConstruct(result.lowered_type_id, element_ids);
    if (!result_construct) return 0;
    if (!builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                            result_construct->result_id())) {
      return 0;
    }

    function->SetFunctionEnd(
        MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                                std::initializer_list<Operand>{}));
    function->AddBasicBlock(std::move(block));
    AddGeneratedFunction(std::move(function), function_id,
                         /*may_write_memory=*/false);

    vector_matmul_pattern_functions_[key] = function_id;
    return function_id;
  }

  const uint32_t result_function_ptr_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t input_function_ptr_type_id = GetOrCreatePointerType(
      input.lowered_type_id, spv::StorageClass::Function);
  const uint32_t matrix_function_ptr_type_id = GetOrCreatePointerType(
      matrix.lowered_type_id, spv::StorageClass::Function);
  const uint32_t bias_function_ptr_type_id =
      has_bias ? GetOrCreatePointerType(bias->lowered_type_id,
                                        spv::StorageClass::Function)
               : 0;
  const uint32_t vec4_function_ptr_type_id =
      GetOrCreatePointerType(vec4_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t result_packed_length_id =
      GetOrCreateUIntConstant(result.packed_length);
  const uint32_t input_packed_length_id =
      GetOrCreateUIntConstant(input.packed_length);
  const uint32_t matrix_packed_cols_id =
      GetOrCreateUIntConstant(matrix.packed_cols);
  if (result_function_ptr_type_id == 0 || input_function_ptr_type_id == 0 ||
      matrix_function_ptr_type_id == 0 ||
      (has_bias && bias_function_ptr_type_id == 0) ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || four_uint_id == 0 ||
      result_packed_length_id == 0 || input_packed_length_id == 0 ||
      matrix_packed_cols_id == 0) {
    return 0;
  }

  const uint32_t out_header_label_id = TakeNextId();
  const uint32_t out_body_label_id = TakeNextId();
  const uint32_t out_continue_label_id = TakeNextId();
  const uint32_t out_merge_label_id = TakeNextId();
  const uint32_t k_header_label_id = TakeNextId();
  const uint32_t k_body_label_id = TakeNextId();
  const uint32_t k_continue_label_id = TakeNextId();
  const uint32_t k_merge_label_id = TakeNextId();
  if (out_header_label_id == 0 || out_body_label_id == 0 ||
      out_continue_label_id == 0 || out_merge_label_id == 0 ||
      k_header_label_id == 0 || k_body_label_id == 0 ||
      k_continue_label_id == 0 || k_merge_label_id == 0) {
    return 0;
  }
  std::unique_ptr<BasicBlock> out_header_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(out_header_label_id));
  std::unique_ptr<BasicBlock> out_body_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(out_body_label_id));
  std::unique_ptr<BasicBlock> out_continue_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(out_continue_label_id));
  std::unique_ptr<BasicBlock> out_merge_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(out_merge_label_id));
  std::unique_ptr<BasicBlock> k_header_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_header_label_id));
  std::unique_ptr<BasicBlock> k_body_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_body_label_id));
  std::unique_ptr<BasicBlock> k_continue_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_continue_label_id));
  std::unique_ptr<BasicBlock> k_merge_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_merge_label_id));
  if (!out_header_block || !out_body_block || !out_continue_block ||
      !out_merge_block || !k_header_block || !k_body_block ||
      !k_continue_block || !k_merge_block) {
    return 0;
  }

  Instruction* result_var =
      builder.AddVariable(result_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* input_var =
      builder.AddVariable(input_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* matrix_var =
      builder.AddVariable(matrix_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* bias_var =
      has_bias ? builder.AddVariable(
                     bias_function_ptr_type_id,
                     static_cast<uint32_t>(spv::StorageClass::Function))
               : nullptr;
  Instruction* out_pack_var =
      builder.AddVariable(uint_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_pack_var =
      builder.AddVariable(uint_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] =
        builder.AddVariable(vec4_function_ptr_type_id,
                            static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!result_var || !input_var || !matrix_var || (has_bias && !bias_var) ||
      !out_pack_var || !k_pack_var) {
    return 0;
  }
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }
  if (!builder.AddStore(input_var->result_id(), input_param_id) ||
      !builder.AddStore(matrix_var->result_id(), matrix_param_id) ||
      (has_bias && !builder.AddStore(bias_var->result_id(), bias_param_id)) ||
      !builder.AddStore(out_pack_var->result_id(), zero_uint_id) ||
      !builder.AddBranch(out_header_label_id)) {
    return 0;
  }

  InstructionBuilder out_header_builder(context(), out_header_block.get());
  Instruction* out_pack_load =
      out_header_builder.AddLoad(uint_type_id, out_pack_var->result_id());
  if (!out_pack_load) return 0;
  Instruction* out_cond = out_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, out_pack_load->result_id(),
      result_packed_length_id);
  if (!out_cond) return 0;
  if (!out_header_builder.AddLoopMerge(out_merge_label_id,
                                       out_continue_label_id)) {
    return 0;
  }
  if (!out_header_builder.AddConditionalBranch(
          out_cond->result_id(), out_body_label_id, out_merge_label_id)) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!out_body_builder.AddStore(acc_var->result_id(), zero4_id)) return 0;
  }
  if (!out_body_builder.AddStore(k_pack_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k_pack_load =
      k_header_builder.AddLoad(uint_type_id, k_pack_var->result_id());
  if (!k_pack_load) return 0;
  Instruction* k_cond = k_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, k_pack_load->result_id(),
      input_packed_length_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id)) {
    return 0;
  }
  if (!k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* input_vec_ptr = k_body_builder.AddAccessChain(
      vec4_function_ptr_type_id, input_var->result_id(),
      {k_pack_load->result_id()});
  if (!input_vec_ptr) return 0;
  Instruction* input_vec = k_body_builder.AddLoad(input.packed_vec4_type_id,
                                                  input_vec_ptr->result_id());
  if (!input_vec) return 0;
  Instruction* k_element_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_pack_load->result_id(), four_uint_id);
  if (!k_element_base) return 0;
  std::array<uint32_t, kPackedVec4Width> weight_row_ids = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
    const uint32_t lane_id = GetOrCreateUIntConstant(row_lane);
    if (lane_id == 0) return 0;
    Instruction* matrix_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_element_base->result_id(), lane_id);
    if (!matrix_row) return 0;
    Instruction* matrix_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, matrix_row->result_id(),
        matrix_packed_cols_id);
    if (!matrix_row_offset) return 0;
    Instruction* matrix_flat_index = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, matrix_row_offset->result_id(),
        out_pack_load->result_id());
    if (!matrix_flat_index) return 0;
    Instruction* weight_ptr = k_body_builder.AddAccessChain(
        vec4_function_ptr_type_id, matrix_var->result_id(),
        {matrix_flat_index->result_id()});
    if (!weight_ptr) return 0;
    Instruction* weight_row = k_body_builder.AddLoad(matrix.packed_vec4_type_id,
                                                     weight_ptr->result_id());
    if (!weight_row) return 0;
    weight_row_ids[row_lane] = weight_row->result_id();
  }
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    std::vector<uint32_t> weight_lane_ids;
    weight_lane_ids.reserve(kPackedVec4Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
      const uint32_t weight_scalar =
          ExtractCompositeElement(&k_body_builder, result.component_type_id,
                                  weight_row_ids[row_lane], lane);
      if (weight_scalar == 0) return 0;
      weight_lane_ids.push_back(weight_scalar);
    }
    Instruction* weight =
        k_body_builder.AddCompositeConstruct(vec4_type_id, weight_lane_ids);
    Instruction* acc =
        k_body_builder.AddLoad(vec4_type_id, acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, vec4_type_id, input_vec->result_id(),
                 weight->result_id(), acc->result_id());
    if (fma == 0) return 0;
    if (!k_body_builder.AddStore(acc_vars[lane]->result_id(), fma)) return 0;
  }
  if (!k_body_builder.AddBranch(k_continue_label_id)) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue_block.get());
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k_pack_load->result_id(), one_uint_id);
  if (!next_k) return 0;
  if (!k_continue_builder.AddStore(k_pack_var->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc =
        k_merge_builder.AddLoad(vec4_type_id, acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    if (reduced == 0) return 0;
    lane_ids.push_back(reduced);
  }
  Instruction* result_vec =
      k_merge_builder.AddCompositeConstruct(vec4_type_id, lane_ids);
  if (!result_vec) return 0;
  if (has_bias) {
    Instruction* bias_vec_ptr = k_merge_builder.AddAccessChain(
        vec4_function_ptr_type_id, bias_var->result_id(),
        {out_pack_load->result_id()});
    if (!bias_vec_ptr) return 0;
    Instruction* bias_vec = k_merge_builder.AddLoad(bias->packed_vec4_type_id,
                                                    bias_vec_ptr->result_id());
    if (!bias_vec) return 0;
    result_vec = k_merge_builder.AddBinaryOp(vec4_type_id, spv::Op::OpFAdd,
                                             result_vec->result_id(),
                                             bias_vec->result_id());
    if (!result_vec) return 0;
    ApplyActiveFPFastMathMode(result_vec);
  }
  Instruction* result_vec_ptr = k_merge_builder.AddAccessChain(
      vec4_function_ptr_type_id, result_var->result_id(),
      {out_pack_load->result_id()});
  if (!result_vec_ptr) return 0;
  if (!k_merge_builder.AddStore(result_vec_ptr->result_id(),
                                result_vec->result_id()) ||
      !k_merge_builder.AddBranch(out_continue_label_id)) {
    return 0;
  }

  InstructionBuilder out_continue_builder(context(), out_continue_block.get());
  Instruction* next_out = out_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, out_pack_load->result_id(), one_uint_id);
  if (!next_out) return 0;
  if (!out_continue_builder.AddStore(out_pack_var->result_id(),
                                     next_out->result_id()) ||
      !out_continue_builder.AddBranch(out_header_label_id)) {
    return 0;
  }

  InstructionBuilder out_merge_builder(context(), out_merge_block.get());
  Instruction* result_value = out_merge_builder.AddLoad(
      result.lowered_type_id, result_var->result_id());
  if (!result_value) return 0;
  if (!out_merge_builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                    result_value->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(block));
  function->AddBasicBlock(std::move(out_header_block));
  function->AddBasicBlock(std::move(out_body_block));
  function->AddBasicBlock(std::move(k_header_block));
  function->AddBasicBlock(std::move(k_body_block));
  function->AddBasicBlock(std::move(k_continue_block));
  function->AddBasicBlock(std::move(k_merge_block));
  function->AddBasicBlock(std::move(out_continue_block));
  function->AddBasicBlock(std::move(out_merge_block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);

  vector_matmul_pattern_functions_[key] = function_id;
  return function_id;
}

uint32_t
HwLowerToStandardPass::GetOrCreateVectorMatmulPatternPointerFunctionPackedVec4(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
    uint32_t input_pointer_type_id, uint32_t matrix_pointer_type_id,
    uint32_t bias_pointer_type_id) {
  if (!IsPackedVec4(result) || !IsPackedVec4(input) || !IsPackedVec4(matrix) ||
      (has_bias && (!bias || !IsPackedVec4(*bias)))) {
    return 0;
  }

  if (input_pointer_type_id == 0 || matrix_pointer_type_id == 0 ||
      (has_bias && bias_pointer_type_id == 0)) {
    return 0;
  }

  std::string key =
      std::string("ptr|") +
      VectorMatmulPatternFunctionKey(result, input, matrix, bias, has_bias);
  key += "|ip:";
  key += std::to_string(input_pointer_type_id);
  key += "|mp:";
  key += std::to_string(matrix_pointer_type_id);
  key += "|bp:";
  key += std::to_string(bias_pointer_type_id);
  auto cached = vector_matmul_pattern_functions_.find(key);
  if (cached != vector_matmul_pattern_functions_.end()) return cached->second;

  std::vector<uint32_t> param_type_ids = {input_pointer_type_id,
                                          matrix_pointer_type_id};
  if (has_bias) param_type_ids.push_back(bias_pointer_type_id);
  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, param_type_ids);
  if (function_type_id == 0) {
    return 0;
  }

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, result.lowered_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t input_param_id = TakeNextId();
  const uint32_t matrix_param_id = TakeNextId();
  uint32_t bias_param_id = 0;
  if (input_param_id == 0 || matrix_param_id == 0) return 0;
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, input_pointer_type_id,
      input_param_id, std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, matrix_pointer_type_id,
      matrix_param_id, std::initializer_list<Operand>{}));
  if (has_bias) {
    bias_param_id = TakeNextId();
    if (bias_param_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, bias_pointer_type_id,
        bias_param_id, std::initializer_list<Operand>{}));
  }

  const uint32_t label_id = TakeNextId();
  if (label_id == 0) return 0;
  std::unique_ptr<BasicBlock> entry_block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));
  InstructionBuilder builder(context(), entry_block.get());

  const uint32_t vec4_type_id = result.packed_vec4_type_id;
  const uint32_t vec4_function_ptr_type_id =
      GetOrCreatePointerType(vec4_type_id, spv::StorageClass::Function);
  const uint32_t result_function_ptr_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t result_packed_length_id =
      GetOrCreateUIntConstant(result.packed_length);
  const uint32_t input_packed_length_id =
      GetOrCreateUIntConstant(input.packed_length);
  const uint32_t matrix_packed_cols_id =
      GetOrCreateUIntConstant(matrix.packed_cols);
  const uint32_t zero4_id = GetOrCreateZero(vec4_type_id);
  if (vec4_function_ptr_type_id == 0 || result_function_ptr_type_id == 0 ||
      uint_type_id == 0 || uint_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || one_uint_id == 0 ||
      four_uint_id == 0 || result_packed_length_id == 0 ||
      input_packed_length_id == 0 || matrix_packed_cols_id == 0 ||
      zero4_id == 0) {
    return 0;
  }

  const uint32_t out_header_label_id = TakeNextId();
  const uint32_t out_body_label_id = TakeNextId();
  const uint32_t out_continue_label_id = TakeNextId();
  const uint32_t out_merge_label_id = TakeNextId();
  const uint32_t k_header_label_id = TakeNextId();
  const uint32_t k_body_label_id = TakeNextId();
  const uint32_t k_continue_label_id = TakeNextId();
  const uint32_t k_merge_label_id = TakeNextId();
  if (out_header_label_id == 0 || out_body_label_id == 0 ||
      out_continue_label_id == 0 || out_merge_label_id == 0 ||
      k_header_label_id == 0 || k_body_label_id == 0 ||
      k_continue_label_id == 0 || k_merge_label_id == 0) {
    return 0;
  }

  std::unique_ptr<BasicBlock> out_header_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(out_header_label_id));
  std::unique_ptr<BasicBlock> out_body_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(out_body_label_id));
  std::unique_ptr<BasicBlock> out_continue_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(out_continue_label_id));
  std::unique_ptr<BasicBlock> out_merge_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(out_merge_label_id));
  std::unique_ptr<BasicBlock> k_header_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_header_label_id));
  std::unique_ptr<BasicBlock> k_body_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_body_label_id));
  std::unique_ptr<BasicBlock> k_continue_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_continue_label_id));
  std::unique_ptr<BasicBlock> k_merge_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_merge_label_id));
  if (!out_header_block || !out_body_block || !out_continue_block ||
      !out_merge_block || !k_header_block || !k_body_block ||
      !k_continue_block || !k_merge_block) {
    return 0;
  }

  Instruction* result_var =
      builder.AddVariable(result_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* out_pack_var =
      builder.AddVariable(uint_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_pack_var =
      builder.AddVariable(uint_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] =
        builder.AddVariable(vec4_function_ptr_type_id,
                            static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!result_var || !out_pack_var || !k_pack_var) return 0;
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }
  if (!builder.AddStore(out_pack_var->result_id(), zero_uint_id) ||
      !builder.AddBranch(out_header_label_id)) {
    return 0;
  }

  InstructionBuilder out_header_builder(context(), out_header_block.get());
  Instruction* out_pack_load =
      out_header_builder.AddLoad(uint_type_id, out_pack_var->result_id());
  if (!out_pack_load) return 0;
  Instruction* out_cond = out_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, out_pack_load->result_id(),
      result_packed_length_id);
  if (!out_cond) return 0;
  if (!out_header_builder.AddLoopMerge(out_merge_label_id,
                                       out_continue_label_id)) {
    return 0;
  }
  if (!out_header_builder.AddConditionalBranch(
          out_cond->result_id(), out_body_label_id, out_merge_label_id)) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!out_body_builder.AddStore(acc_var->result_id(), zero4_id)) return 0;
  }
  if (!out_body_builder.AddStore(k_pack_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k_pack_load =
      k_header_builder.AddLoad(uint_type_id, k_pack_var->result_id());
  if (!k_pack_load) return 0;
  Instruction* k_cond = k_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, k_pack_load->result_id(),
      input_packed_length_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id)) {
    return 0;
  }
  if (!k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* input_vec_ptr = k_body_builder.AddAccessChain(
      vec4_function_ptr_type_id, input_param_id, {k_pack_load->result_id()});
  if (!input_vec_ptr) return 0;
  Instruction* input_vec = k_body_builder.AddLoad(input.packed_vec4_type_id,
                                                  input_vec_ptr->result_id());
  if (!input_vec) return 0;
  Instruction* out_row_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, out_pack_load->result_id(), four_uint_id);
  if (!out_row_base) return 0;
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    const uint32_t lane_id = GetOrCreateUIntConstant(lane);
    if (lane_id == 0) return 0;
    Instruction* matrix_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, out_row_base->result_id(), lane_id);
    if (!matrix_row) return 0;
    Instruction* matrix_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, matrix_row->result_id(),
        matrix_packed_cols_id);
    if (!matrix_row_offset) return 0;
    Instruction* matrix_flat_index = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, matrix_row_offset->result_id(),
        k_pack_load->result_id());
    if (!matrix_flat_index) return 0;
    Instruction* weight_ptr = k_body_builder.AddAccessChain(
        vec4_function_ptr_type_id, matrix_param_id,
        {matrix_flat_index->result_id()});
    if (!weight_ptr) return 0;
    Instruction* weight = k_body_builder.AddLoad(matrix.packed_vec4_type_id,
                                                 weight_ptr->result_id());
    Instruction* acc =
        k_body_builder.AddLoad(vec4_type_id, acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, vec4_type_id, input_vec->result_id(),
                 weight->result_id(), acc->result_id());
    if (fma == 0) return 0;
    if (!k_body_builder.AddStore(acc_vars[lane]->result_id(), fma)) return 0;
  }
  if (!k_body_builder.AddBranch(k_continue_label_id)) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue_block.get());
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k_pack_load->result_id(), one_uint_id);
  if (!next_k) return 0;
  if (!k_continue_builder.AddStore(k_pack_var->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  Instruction* bias_vec = nullptr;
  if (has_bias) {
    Instruction* bias_vec_ptr = k_merge_builder.AddAccessChain(
        vec4_function_ptr_type_id, bias_param_id, {out_pack_load->result_id()});
    if (!bias_vec_ptr) return 0;
    bias_vec = k_merge_builder.AddLoad(bias->packed_vec4_type_id,
                                       bias_vec_ptr->result_id());
    if (!bias_vec) return 0;
  }
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc =
        k_merge_builder.AddLoad(vec4_type_id, acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    if (reduced == 0) return 0;
    if (has_bias) {
      const uint32_t bias_value =
          ExtractCompositeElement(&k_merge_builder, result.component_type_id,
                                  bias_vec->result_id(), lane);
      if (bias_value == 0) return 0;
      Instruction* add = k_merge_builder.AddBinaryOp(
          result.component_type_id, spv::Op::OpFAdd, reduced, bias_value);
      if (!add) return 0;
      ApplyActiveFPFastMathMode(add);
      reduced = add->result_id();
    }
    lane_ids.push_back(reduced);
  }
  Instruction* result_vec =
      k_merge_builder.AddCompositeConstruct(vec4_type_id, lane_ids);
  if (!result_vec) return 0;
  Instruction* result_vec_ptr = k_merge_builder.AddAccessChain(
      vec4_function_ptr_type_id, result_var->result_id(),
      {out_pack_load->result_id()});
  if (!result_vec_ptr) return 0;
  if (!k_merge_builder.AddStore(result_vec_ptr->result_id(),
                                result_vec->result_id()) ||
      !k_merge_builder.AddBranch(out_continue_label_id)) {
    return 0;
  }

  InstructionBuilder out_continue_builder(context(), out_continue_block.get());
  Instruction* next_out = out_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, out_pack_load->result_id(), one_uint_id);
  if (!next_out) return 0;
  if (!out_continue_builder.AddStore(out_pack_var->result_id(),
                                     next_out->result_id()) ||
      !out_continue_builder.AddBranch(out_header_label_id)) {
    return 0;
  }

  InstructionBuilder out_merge_builder(context(), out_merge_block.get());
  Instruction* result_value = out_merge_builder.AddLoad(
      result.lowered_type_id, result_var->result_id());
  if (!result_value) return 0;
  if (!out_merge_builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                    result_value->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(entry_block));
  function->AddBasicBlock(std::move(out_header_block));
  function->AddBasicBlock(std::move(out_body_block));
  function->AddBasicBlock(std::move(k_header_block));
  function->AddBasicBlock(std::move(k_body_block));
  function->AddBasicBlock(std::move(k_continue_block));
  function->AddBasicBlock(std::move(k_merge_block));
  function->AddBasicBlock(std::move(out_continue_block));
  function->AddBasicBlock(std::move(out_merge_block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);

  vector_matmul_pattern_functions_[key] = function_id;
  return function_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateMatmulPatternFunctionPackedVec4(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c) {
  const std::string key = MatmulPatternFunctionKey(result, a, b, c);
  auto cached = matmul_pattern_functions_.find(key);
  if (cached != matmul_pattern_functions_.end()) return cached->second;

  const uint32_t function_type_id = GetOrCreateFunctionType(
      result.lowered_type_id,
      {a.lowered_type_id, b.lowered_type_id, c.lowered_type_id});
  if (function_type_id == 0) return 0;

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, result.lowered_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t a_param_id = TakeNextId();
  const uint32_t b_param_id = TakeNextId();
  const uint32_t c_param_id = TakeNextId();
  if (a_param_id == 0 || b_param_id == 0 || c_param_id == 0) return 0;
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, a.lowered_type_id, a_param_id,
      std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, b.lowered_type_id, b_param_id,
      std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, c.lowered_type_id, c_param_id,
      std::initializer_list<Operand>{}));

  const uint32_t label_id = TakeNextId();
  if (label_id == 0) return 0;
  std::unique_ptr<BasicBlock> entry_block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));

  // The legacy packed loop below reduces groups of four K terms before adding
  // C, which reassociates the source expression.  Use the strict K-ordered
  // builder for the ordinary helper path.  Large operations are handled by
  // LowerMatrixMulAddWithLoop before reaching this function.
  const bool use_loop_path = false;
  if (!use_loop_path || !IsPackedVec4(a)) {
    InstructionBuilder builder(context(), entry_block.get());
    std::vector<uint32_t> element_ids;
    if (!BuildMatmulPatternPackedVec4(&builder, result, a, b, c, a_param_id,
                                      b_param_id, c_param_id, &element_ids)) {
      return 0;
    }
    Instruction* result_construct =
        builder.AddCompositeConstruct(result.lowered_type_id, element_ids);
    if (!result_construct) return 0;
    if (!builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                            result_construct->result_id())) {
      return 0;
    }

    function->SetFunctionEnd(
        MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                                std::initializer_list<Operand>{}));
    function->AddBasicBlock(std::move(entry_block));
    AddGeneratedFunction(std::move(function), function_id,
                         /*may_write_memory=*/false);
    matmul_pattern_functions_[key] = function_id;
    matmul_pattern_function_ids_.insert(function_id);
    return function_id;
  }

  const uint32_t lowered_function_ptr_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t a_function_ptr_type_id =
      GetOrCreatePointerType(a.lowered_type_id, spv::StorageClass::Function);
  const uint32_t b_function_ptr_type_id =
      GetOrCreatePointerType(b.lowered_type_id, spv::StorageClass::Function);
  const uint32_t c_function_ptr_type_id =
      GetOrCreatePointerType(c.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec4_function_ptr_type_id = GetOrCreatePointerType(
      result.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t row_count_id = GetOrCreateUIntConstant(result.rows);
  const uint32_t result_packed_cols_id =
      GetOrCreateUIntConstant(result.packed_cols);
  const uint32_t a_packed_cols_id = GetOrCreateUIntConstant(a.packed_cols);
  const uint32_t b_packed_cols_id = GetOrCreateUIntConstant(b.packed_cols);
  const uint32_t k_pack_count_id = GetOrCreateUIntConstant(a.packed_cols);
  const uint32_t zero4_id = GetOrCreateZero(result.packed_vec4_type_id);
  if (lowered_function_ptr_type_id == 0 || a_function_ptr_type_id == 0 ||
      b_function_ptr_type_id == 0 || c_function_ptr_type_id == 0 ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || four_uint_id == 0 ||
      row_count_id == 0 || result_packed_cols_id == 0 ||
      a_packed_cols_id == 0 || b_packed_cols_id == 0 || k_pack_count_id == 0 ||
      zero4_id == 0) {
    return 0;
  }

  const uint32_t row_header_label_id = TakeNextId();
  const uint32_t row_body_label_id = TakeNextId();
  const uint32_t row_continue_label_id = TakeNextId();
  const uint32_t row_merge_label_id = TakeNextId();
  const uint32_t col_header_label_id = TakeNextId();
  const uint32_t col_body_label_id = TakeNextId();
  const uint32_t col_continue_label_id = TakeNextId();
  const uint32_t col_merge_label_id = TakeNextId();
  const uint32_t k_header_label_id = TakeNextId();
  const uint32_t k_body_label_id = TakeNextId();
  const uint32_t k_continue_label_id = TakeNextId();
  const uint32_t k_merge_label_id = TakeNextId();
  if (row_header_label_id == 0 || row_body_label_id == 0 ||
      row_continue_label_id == 0 || row_merge_label_id == 0 ||
      col_header_label_id == 0 || col_body_label_id == 0 ||
      col_continue_label_id == 0 || col_merge_label_id == 0 ||
      k_header_label_id == 0 || k_body_label_id == 0 ||
      k_continue_label_id == 0 || k_merge_label_id == 0) {
    return 0;
  }
  std::unique_ptr<BasicBlock> row_header_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(row_header_label_id));
  std::unique_ptr<BasicBlock> row_body_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(row_body_label_id));
  std::unique_ptr<BasicBlock> row_continue_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(row_continue_label_id));
  std::unique_ptr<BasicBlock> row_merge_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(row_merge_label_id));
  std::unique_ptr<BasicBlock> col_header_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(col_header_label_id));
  std::unique_ptr<BasicBlock> col_body_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(col_body_label_id));
  std::unique_ptr<BasicBlock> col_continue_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(col_continue_label_id));
  std::unique_ptr<BasicBlock> col_merge_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(col_merge_label_id));
  std::unique_ptr<BasicBlock> k_header_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_header_label_id));
  std::unique_ptr<BasicBlock> k_body_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_body_label_id));
  std::unique_ptr<BasicBlock> k_continue_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_continue_label_id));
  std::unique_ptr<BasicBlock> k_merge_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(k_merge_label_id));
  if (!row_header_block || !row_body_block || !row_continue_block ||
      !row_merge_block || !col_header_block || !col_body_block ||
      !col_continue_block || !col_merge_block || !k_header_block ||
      !k_body_block || !k_continue_block || !k_merge_block) {
    return 0;
  }

  InstructionBuilder entry_builder(context(), entry_block.get());
  Instruction* result_var = entry_builder.AddVariable(
      lowered_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* a_var = entry_builder.AddVariable(
      a_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* b_var = entry_builder.AddVariable(
      b_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* c_var = entry_builder.AddVariable(
      c_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* row_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* col_pack_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_pack_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] = entry_builder.AddVariable(
        vec4_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!result_var || !a_var || !b_var || !c_var || !row_var || !col_pack_var ||
      !k_pack_var) {
    return 0;
  }
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }
  if (!entry_builder.AddStore(a_var->result_id(), a_param_id) ||
      !entry_builder.AddStore(b_var->result_id(), b_param_id) ||
      !entry_builder.AddStore(c_var->result_id(), c_param_id) ||
      !entry_builder.AddStore(row_var->result_id(), zero_uint_id) ||
      !entry_builder.AddBranch(row_header_label_id)) {
    return 0;
  }

  InstructionBuilder row_header_builder(context(), row_header_block.get());
  Instruction* row_load =
      row_header_builder.AddLoad(uint_type_id, row_var->result_id());
  if (!row_load) return 0;
  Instruction* row_cond = row_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, row_load->result_id(), row_count_id);
  if (!row_cond) return 0;
  if (!row_header_builder.AddLoopMerge(row_merge_label_id,
                                       row_continue_label_id)) {
    return 0;
  }
  if (!row_header_builder.AddConditionalBranch(
          row_cond->result_id(), row_body_label_id, row_merge_label_id)) {
    return 0;
  }

  InstructionBuilder row_body_builder(context(), row_body_block.get());
  if (!row_body_builder.AddStore(col_pack_var->result_id(), zero_uint_id) ||
      !row_body_builder.AddBranch(col_header_label_id)) {
    return 0;
  }

  InstructionBuilder col_header_builder(context(), col_header_block.get());
  Instruction* col_pack_load =
      col_header_builder.AddLoad(uint_type_id, col_pack_var->result_id());
  if (!col_pack_load) return 0;
  Instruction* col_cond = col_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, col_pack_load->result_id(),
      result_packed_cols_id);
  if (!col_cond) return 0;
  if (!col_header_builder.AddLoopMerge(col_merge_label_id,
                                       col_continue_label_id)) {
    return 0;
  }
  if (!col_header_builder.AddConditionalBranch(
          col_cond->result_id(), col_body_label_id, col_merge_label_id)) {
    return 0;
  }

  InstructionBuilder col_body_builder(context(), col_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!col_body_builder.AddStore(acc_var->result_id(), zero4_id)) return 0;
  }
  if (!col_body_builder.AddStore(k_pack_var->result_id(), zero_uint_id) ||
      !col_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k_pack_load =
      k_header_builder.AddLoad(uint_type_id, k_pack_var->result_id());
  if (!k_pack_load) return 0;
  Instruction* k_cond =
      k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                   k_pack_load->result_id(), k_pack_count_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id)) {
    return 0;
  }
  if (!k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* a_row_offset = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, row_load->result_id(), a_packed_cols_id);
  if (!a_row_offset) return 0;
  Instruction* a_flat_index = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, a_row_offset->result_id(),
      k_pack_load->result_id());
  if (!a_flat_index) return 0;
  Instruction* a_vec_ptr = k_body_builder.AddAccessChain(
      vec4_function_ptr_type_id, a_var->result_id(),
      {a_flat_index->result_id()});
  if (!a_vec_ptr) return 0;
  Instruction* a_vec =
      k_body_builder.AddLoad(a.packed_vec4_type_id, a_vec_ptr->result_id());
  if (!a_vec) return 0;
  Instruction* k_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_pack_load->result_id(), four_uint_id);
  if (!k_base) return 0;
  std::array<uint32_t, kPackedVec4Width> b_vecs = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
    const uint32_t row_lane_id = GetOrCreateUIntConstant(row_lane);
    if (row_lane_id == 0) return 0;
    Instruction* b_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_base->result_id(), row_lane_id);
    if (!b_row) return 0;
    Instruction* b_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, b_row->result_id(), b_packed_cols_id);
    if (!b_row_offset) return 0;
    Instruction* b_flat_index = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, b_row_offset->result_id(),
        col_pack_load->result_id());
    if (!b_flat_index) return 0;
    Instruction* b_vec_ptr = k_body_builder.AddAccessChain(
        vec4_function_ptr_type_id, b_var->result_id(),
        {b_flat_index->result_id()});
    if (!b_vec_ptr) return 0;
    Instruction* b_vec =
        k_body_builder.AddLoad(b.packed_vec4_type_id, b_vec_ptr->result_id());
    if (!b_vec) return 0;
    b_vecs[row_lane] = b_vec->result_id();
  }
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    std::vector<uint32_t> weight_lanes;
    weight_lanes.reserve(kPackedVec4Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
      const uint32_t value = ExtractCompositeElement(
          &k_body_builder, result.component_type_id, b_vecs[row_lane], lane);
      if (value == 0) return 0;
      weight_lanes.push_back(value);
    }
    Instruction* weight_vec = k_body_builder.AddCompositeConstruct(
        result.packed_vec4_type_id, weight_lanes);
    Instruction* acc = k_body_builder.AddLoad(result.packed_vec4_type_id,
                                              acc_vars[lane]->result_id());
    if (!weight_vec || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, result.packed_vec4_type_id,
                 a_vec->result_id(), weight_vec->result_id(), acc->result_id());
    if (fma == 0) return 0;
    if (!k_body_builder.AddStore(acc_vars[lane]->result_id(), fma)) return 0;
  }
  if (!k_body_builder.AddBranch(k_continue_label_id)) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue_block.get());
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k_pack_load->result_id(), one_uint_id);
  if (!next_k) return 0;
  if (!k_continue_builder.AddStore(k_pack_var->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  Instruction* result_row_offset =
      k_merge_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                  row_load->result_id(), result_packed_cols_id);
  if (!result_row_offset) return 0;
  Instruction* result_flat_index = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, result_row_offset->result_id(),
      col_pack_load->result_id());
  if (!result_flat_index) return 0;
  Instruction* c_vec_ptr = k_merge_builder.AddAccessChain(
      vec4_function_ptr_type_id, c_var->result_id(),
      {result_flat_index->result_id()});
  if (!c_vec_ptr) return 0;
  Instruction* c_vec =
      k_merge_builder.AddLoad(c.packed_vec4_type_id, c_vec_ptr->result_id());
  if (!c_vec) return 0;
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec4_type_id,
                                               acc_vars[lane]->result_id());
    if (!acc) return 0;
    const uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    const uint32_t bias = ExtractCompositeElement(
        &k_merge_builder, result.component_type_id, c_vec->result_id(), lane);
    if (reduced == 0 || bias == 0) return 0;
    Instruction* add = k_merge_builder.AddBinaryOp(
        result.component_type_id, spv::Op::OpFAdd, bias, reduced);
    if (!add) return 0;
    ApplyActiveFPFastMathMode(add);
    lane_ids.push_back(add->result_id());
  }
  Instruction* result_vec = k_merge_builder.AddCompositeConstruct(
      result.packed_vec4_type_id, lane_ids);
  if (!result_vec) return 0;
  Instruction* result_vec_ptr = k_merge_builder.AddAccessChain(
      vec4_function_ptr_type_id, result_var->result_id(),
      {result_flat_index->result_id()});
  if (!result_vec_ptr) return 0;
  if (!k_merge_builder.AddStore(result_vec_ptr->result_id(),
                                result_vec->result_id()) ||
      !k_merge_builder.AddBranch(col_continue_label_id)) {
    return 0;
  }

  InstructionBuilder col_continue_builder(context(), col_continue_block.get());
  Instruction* next_col = col_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, col_pack_load->result_id(), one_uint_id);
  if (!next_col) return 0;
  if (!col_continue_builder.AddStore(col_pack_var->result_id(),
                                     next_col->result_id()) ||
      !col_continue_builder.AddBranch(col_header_label_id)) {
    return 0;
  }

  InstructionBuilder col_merge_builder(context(), col_merge_block.get());
  if (!col_merge_builder.AddBranch(row_continue_label_id)) return 0;

  InstructionBuilder row_continue_builder(context(), row_continue_block.get());
  Instruction* next_row = row_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, row_load->result_id(), one_uint_id);
  if (!next_row) return 0;
  if (!row_continue_builder.AddStore(row_var->result_id(),
                                     next_row->result_id()) ||
      !row_continue_builder.AddBranch(row_header_label_id)) {
    return 0;
  }

  InstructionBuilder row_merge_builder(context(), row_merge_block.get());
  Instruction* result_value = row_merge_builder.AddLoad(
      result.lowered_type_id, result_var->result_id());
  if (!result_value) return 0;
  if (!row_merge_builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                    result_value->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(entry_block));
  function->AddBasicBlock(std::move(row_header_block));
  function->AddBasicBlock(std::move(row_body_block));
  function->AddBasicBlock(std::move(col_header_block));
  function->AddBasicBlock(std::move(col_body_block));
  function->AddBasicBlock(std::move(k_header_block));
  function->AddBasicBlock(std::move(k_body_block));
  function->AddBasicBlock(std::move(k_continue_block));
  function->AddBasicBlock(std::move(k_merge_block));
  function->AddBasicBlock(std::move(col_continue_block));
  function->AddBasicBlock(std::move(col_merge_block));
  function->AddBasicBlock(std::move(row_continue_block));
  function->AddBasicBlock(std::move(row_merge_block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);
  matmul_pattern_functions_[key] = function_id;
  matmul_pattern_function_ids_.insert(function_id);
  return function_id;
}

std::string HwLowerToStandardPass::TileWeightFunctionKey(
    const MatrixTypeInfo& matrix) const {
  std::string key;
  key.reserve(96);
  key += std::to_string(matrix.component_type_id);
  key += ':';
  key += std::to_string(matrix.rows);
  key += 'x';
  key += std::to_string(matrix.cols);
  key += ':';
  key += std::to_string(matrix.lowered_type_id);
  key += ':';
  key += std::to_string(matrix.packed_vec4_type_id);
  key += ':';
  key += std::to_string(matrix.packed_cols);
  key += ':';
  key += (matrix.packed_f16vec4 ? "h" : (matrix.packed_f32vec4 ? "f" : "s"));
  return key;
}

std::string HwLowerToStandardPass::MatmulTileWeightFunctionKey(
    const MatrixTypeInfo& matrix) const {
  return std::string("matmul|") + TileWeightFunctionKey(matrix);
}

std::string HwLowerToStandardPass::VectorMatmulPatternFunctionKey(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias,
    bool has_bias) const {
  std::string key;
  key.reserve(220);
  auto append_vector = [&key](const VectorTypeInfo& info) {
    key += std::to_string(info.component_type_id);
    key += ':';
    key += std::to_string(info.length);
    key += ':';
    key += std::to_string(info.lowered_type_id);
    key += ':';
    key += std::to_string(info.packed_vec4_type_id);
    key += ':';
    key += std::to_string(info.packed_length);
    key += ':';
    key += (info.packed_f16vec4 ? "h" : (info.packed_f32vec4 ? "f" : "s"));
  };
  auto append_matrix = [&key](const MatrixTypeInfo& info) {
    key += std::to_string(info.component_type_id);
    key += ':';
    key += std::to_string(info.rows);
    key += 'x';
    key += std::to_string(info.cols);
    key += ':';
    key += std::to_string(info.lowered_type_id);
    key += ':';
    key += std::to_string(info.packed_vec4_type_id);
    key += ':';
    key += std::to_string(info.packed_cols);
    key += ':';
    key += (info.packed_f16vec4 ? "h" : (info.packed_f32vec4 ? "f" : "s"));
  };
  append_vector(result);
  key += '|';
  append_vector(input);
  key += '|';
  append_matrix(matrix);
  key += '|';
  key += has_bias ? 'b' : 'n';
  if (has_bias && bias) {
    key += '|';
    append_vector(*bias);
  }
  key += "|fm:";
  key += std::to_string(active_fp_fast_math_mode_);
  return key;
}

std::string HwLowerToStandardPass::MatmulPatternFunctionKey(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c) const {
  std::string key;
  key.reserve(160);
  auto append_matrix = [&key](const MatrixTypeInfo& info) {
    key += std::to_string(info.component_type_id);
    key += ':';
    key += std::to_string(info.rows);
    key += 'x';
    key += std::to_string(info.cols);
    key += ':';
    key += std::to_string(info.lowered_type_id);
    key += ':';
    key += std::to_string(info.packed_vec4_type_id);
    key += ':';
    key += std::to_string(info.packed_cols);
    key += ':';
    key += (info.packed_f16vec4 ? "h" : (info.packed_f32vec4 ? "f" : "s"));
  };
  append_matrix(result);
  key += '|';
  append_matrix(a);
  key += '|';
  append_matrix(b);
  key += '|';
  append_matrix(c);
  key += "|fm:";
  key += std::to_string(active_fp_fast_math_mode_);
  return key;
}

bool HwLowerToStandardPass::IsPackedVec4(const MatrixTypeInfo& info) const {
  return info.packed_f16vec4 || info.packed_f32vec4;
}

bool HwLowerToStandardPass::IsPackedVec4(const VectorTypeInfo& info) const {
  return info.packed_f16vec4 || info.packed_f32vec4;
}

bool HwLowerToStandardPass::IsSamePackedVec4Kind(
    const MatrixTypeInfo& a, const MatrixTypeInfo& b) const {
  return IsPackedVec4(a) && IsPackedVec4(b) &&
         a.packed_f16vec4 == b.packed_f16vec4 &&
         a.packed_f32vec4 == b.packed_f32vec4 &&
         a.component_type_id == b.component_type_id &&
         a.packed_vec4_type_id == b.packed_vec4_type_id;
}

bool HwLowerToStandardPass::IsSamePackedVec4Kind(
    const VectorTypeInfo& a, const VectorTypeInfo& b) const {
  return IsPackedVec4(a) && IsPackedVec4(b) &&
         a.packed_f16vec4 == b.packed_f16vec4 &&
         a.packed_f32vec4 == b.packed_f32vec4 &&
         a.component_type_id == b.component_type_id &&
         a.packed_vec4_type_id == b.packed_vec4_type_id;
}

bool HwLowerToStandardPass::CanUsePackedVec4MatrixMulAdd(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c) const {
  return IsPackedVec4(result) && IsSamePackedVec4Kind(result, a) &&
         IsSamePackedVec4Kind(result, b) && IsSamePackedVec4Kind(result, c);
}

bool HwLowerToStandardPass::CanUsePackedVec4VectorMatrixMul(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias) const {
  if (!IsPackedVec4(result) || !IsPackedVec4(input) || !IsPackedVec4(matrix) ||
      result.component_type_id != input.component_type_id ||
      result.component_type_id != matrix.component_type_id) {
    return false;
  }
  if (result.length != matrix.cols || input.length != matrix.rows) {
    return false;
  }
  return !bias || IsSamePackedVec4Kind(result, *bias);
}

bool HwLowerToStandardPass::CanUseDirectVectorMatrixMul(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias) const {
  if (lowering_mode_ != LoweringMode::kPreferPackedVec4 ||
      result.length != matrix.cols || input.length != matrix.rows ||
      (bias && bias->length != result.length)) {
    return false;
  }

  const bool same_component =
      result.component_type_id == input.component_type_id &&
      result.component_type_id == matrix.component_type_id &&
      (!bias || result.component_type_id == bias->component_type_id) &&
      (IsFloat16Type(result.component_type_id) ||
       IsFloat32Type(result.component_type_id));
  const bool mixed_f16_f32 = IsFloat32Type(result.component_type_id) &&
                             IsFloat16Type(input.component_type_id) &&
                             IsFloat16Type(matrix.component_type_id) &&
                             (!bias || IsFloat32Type(bias->component_type_id));
  return same_component || mixed_f16_f32;
}

bool HwLowerToStandardPass::ShouldUsePackedVec4(uint32_t extent) const {
  return lowering_mode_ == LoweringMode::kPreferPackedVec4 &&
         extent % kPackedVec4Width == 0;
}

uint32_t HwLowerToStandardPass::MatrixFlatIndex(const MatrixTypeInfo& info,
                                                uint32_t row,
                                                uint32_t col) const {
  return row * info.cols + col;
}

uint32_t HwLowerToStandardPass::MatrixPackedIndex(const MatrixTypeInfo& info,
                                                  uint32_t row,
                                                  uint32_t col_pack) const {
  return row * info.packed_cols + col_pack;
}

uint32_t HwLowerToStandardPass::VectorPackedIndex(uint32_t scalar_index) const {
  return scalar_index / kPackedVec4Width;
}

uint32_t HwLowerToStandardPass::PackedLane(uint32_t scalar_index) const {
  return scalar_index % kPackedVec4Width;
}

uint32_t HwLowerToStandardPass::GetOrCreateGLSLStd450Import() {
  uint32_t import_id =
      context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  if (import_id != 0) return import_id;

  import_id = get_module()->GetExtInstImportId("GLSL.std.450");
  if (import_id != 0) return import_id;

  context()->AddExtInstImport("GLSL.std.450");
  return context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
}

uint32_t HwLowerToStandardPass::GetFPFastMathMode(uint32_t result_id) const {
  if (result_id == 0) return 0;
  for (const Instruction* decoration :
       context()->get_decoration_mgr()->GetDecorationsFor(
           result_id, /*include_linkage=*/false)) {
    if (decoration && decoration->NumInOperands() >= 3 &&
        decoration->GetSingleWordInOperand(1) ==
            uint32_t(spv::Decoration::FPFastMathMode)) {
      return decoration->GetSingleWordInOperand(2);
    }
  }
  return 0;
}

void HwLowerToStandardPass::ApplyFPFastMathMode(Instruction* inst,
                                                uint32_t mode) {
  if (!inst || inst->result_id() == 0 || mode == 0) return;
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse) &&
      !context()->get_def_use_mgr()->GetDef(inst->result_id())) {
    pending_fp_fast_math_modes_.emplace_back(inst->result_id(), mode);
    return;
  }
  context()->get_decoration_mgr()->AddDecorationVal(
      inst->result_id(), uint32_t(spv::Decoration::FPFastMathMode), mode);
}

void HwLowerToStandardPass::ApplyActiveFPFastMathMode(Instruction* inst) {
  ApplyFPFastMathMode(inst, active_fp_fast_math_mode_);
}

bool HwLowerToStandardPass::MatmulAllowsReassociation(
    const Instruction* inst) const {
  if (!inst || inst->result_id() == 0) return false;

  const auto* decoration_mgr = context()->get_decoration_mgr();
  if (decoration_mgr->HasDecoration(inst->result_id(),
                                    spv::Decoration::NoContraction)) {
    return false;
  }

  bool has_explicit_mode = false;
  uint32_t mode = 0;
  for (const Instruction* decoration : decoration_mgr->GetDecorationsFor(
           inst->result_id(), /*include_linkage=*/false)) {
    if (decoration && decoration->NumInOperands() >= 3 &&
        decoration->GetSingleWordInOperand(1) ==
            uint32_t(spv::Decoration::FPFastMathMode)) {
      has_explicit_mode = true;
      mode = decoration->GetSingleWordInOperand(2);
      break;
    }
  }

  const uint32_t reassociation_modes =
      uint32_t(spv::FPFastMathModeMask::AllowReassoc) |
      uint32_t(spv::FPFastMathModeMask::Fast);
  if (has_explicit_mode) return (mode & reassociation_modes) != 0;

  uint32_t component_type_id = 0;
  if (const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id())) {
    component_type_id = matrix->component_type_id;
  } else if (const VectorTypeInfo* vector = GetVectorType(inst->type_id())) {
    component_type_id = vector->component_type_id;
  }

  std::unordered_set<uint32_t> default_entry_points;
  std::unordered_set<uint32_t> reassoc_entry_points;
  for (const Instruction& execution_mode : get_module()->execution_modes()) {
    if (execution_mode.opcode() != spv::Op::OpExecutionModeId ||
        execution_mode.NumInOperands() < 4 ||
        execution_mode.GetSingleWordInOperand(1) !=
            uint32_t(spv::ExecutionMode::FPFastMathDefault)) {
      continue;
    }
    const uint32_t entry_point_id = execution_mode.GetSingleWordInOperand(0);
    default_entry_points.insert(entry_point_id);
    if (execution_mode.GetSingleWordInOperand(2) != component_type_id) continue;

    uint32_t default_mode = 0;
    if (!GetConstantU32(execution_mode.GetSingleWordInOperand(3),
                        &default_mode) ||
        (default_mode & uint32_t(spv::FPFastMathModeMask::AllowReassoc)) == 0) {
      return false;
    }
    reassoc_entry_points.insert(entry_point_id);
  }
  for (uint32_t entry_point_id : default_entry_points) {
    if (reassoc_entry_points.count(entry_point_id) == 0) return false;
  }

  // With no explicit per-instruction or module default, this pass deliberately
  // uses its relaxed matmul contract and permits K-direction reassociation.
  return true;
}

void HwLowerToStandardPass::RemoveFPFastMathMode(uint32_t result_id) {
  if (result_id == 0) return;
  context()->get_decoration_mgr()->RemoveDecorationsFrom(
      result_id, [](const Instruction& decoration) {
        return decoration.NumInOperands() >= 2 &&
               decoration.GetSingleWordInOperand(1) ==
                   uint32_t(spv::Decoration::FPFastMathMode);
      });
}

}  // namespace opt
}  // namespace spvtools
