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

uint32_t HwLowerToStandardPass::BuildFusedVectorMatmulStoreFunctionPackedVec2(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, uint32_t input_pointer_id,
    uint32_t input_pointer_type_id,
    const std::vector<Operand>& input_memory_operands,
    uint32_t matrix_pointer_id, uint32_t matrix_pointer_type_id,
    uint32_t matrix_shape_id, uint32_t matrix_offset_id,
    const std::vector<Operand>& matrix_memory_operands,
    uint32_t output_pointer_id, uint32_t output_pointer_type_id,
    const std::vector<Operand>& output_memory_operands) {
  if (!IsPackedVec2(result) || !IsPackedVec2(input) || !IsPackedVec2(matrix) ||
      input.length != matrix.rows || result.length != matrix.cols) {
    return 0;
  }

  const uint32_t input_load_function_id = GetOrCreatePackedLoadChunkFunction(
      input_pointer_id, input_pointer_type_id, input.component_type_id,
      input.packed_vec2_type_id, input_memory_operands);
  const uint32_t matrix_load_function_id = GetOrCreatePackedLoadChunkFunction(
      matrix_pointer_id, matrix_pointer_type_id, matrix.component_type_id,
      matrix.packed_vec2_type_id, matrix_memory_operands);
  const uint32_t output_store_function_id = GetOrCreatePackedStoreChunkFunction(
      output_pointer_id, output_pointer_type_id, result.component_type_id,
      result.packed_vec2_type_id, output_memory_operands);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t function_type_id = GetOrCreateFunctionType(void_type_id, {});
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t vec2_function_ptr_type_id = GetOrCreatePointerType(
      result.packed_vec2_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t result_length_id = GetOrCreateUIntConstant(result.length);
  const uint32_t input_length_id = GetOrCreateUIntConstant(input.length);
  const uint32_t matrix_cols_id = GetOrCreateUIntConstant(matrix.cols);
  const uint32_t zero2_id = GetOrCreateZero(result.packed_vec2_type_id);
  if (input_load_function_id == 0 || matrix_load_function_id == 0 ||
      output_store_function_id == 0 || void_type_id == 0 ||
      function_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || vec2_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || packed_width_uint_id == 0 ||
      result_length_id == 0 || input_length_id == 0 || matrix_cols_id == 0 ||
      zero2_id == 0) {
    return 0;
  }

  auto get_constant_pair_component = [this](uint32_t pair_id, uint32_t index,
                                            uint32_t* value) {
    if (!value || index >= 2) return false;
    Instruction* pair = get_def_use_mgr()->GetDef(pair_id);
    if (!pair) return false;
    if (pair->opcode() == spv::Op::OpConstantNull) {
      *value = 0;
      return true;
    }
    return pair->opcode() == spv::Op::OpConstantComposite &&
           pair->NumInOperands() > index &&
           GetConstantU32(pair->GetSingleWordInOperand(index), value);
  };
  uint32_t shape_cols = 0;
  uint32_t offset_row = 0;
  uint32_t offset_col = 0;
  const bool direct_matrix_index =
      get_constant_pair_component(matrix_shape_id, 1, &shape_cols) &&
      get_constant_pair_component(matrix_offset_id, 0, &offset_row) &&
      get_constant_pair_component(matrix_offset_id, 1, &offset_col) &&
      shape_cols == matrix.cols && offset_row == 0 && offset_col == 0;

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start =
      MakeUnique<Instruction>(context(), spv::Op::OpFunction, void_type_id,
                              function_id, std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t entry_label_id = TakeNextId();
  const uint32_t out_header_label_id = TakeNextId();
  const uint32_t out_body_label_id = TakeNextId();
  const uint32_t out_continue_label_id = TakeNextId();
  const uint32_t out_merge_label_id = TakeNextId();
  const uint32_t k_header_label_id = TakeNextId();
  const uint32_t k_body_label_id = TakeNextId();
  const uint32_t k_continue_label_id = TakeNextId();
  const uint32_t k_merge_label_id = TakeNextId();
  if (entry_label_id == 0 || out_header_label_id == 0 ||
      out_body_label_id == 0 || out_continue_label_id == 0 ||
      out_merge_label_id == 0 || k_header_label_id == 0 ||
      k_body_label_id == 0 || k_continue_label_id == 0 ||
      k_merge_label_id == 0) {
    return 0;
  }

  std::unique_ptr<BasicBlock> entry_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_label_id));
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
  if (!entry_block || !out_header_block || !out_body_block ||
      !out_continue_block || !out_merge_block || !k_header_block ||
      !k_body_block || !k_continue_block || !k_merge_block) {
    return 0;
  }

  InstructionBuilder entry_builder(context(), entry_block.get());
  Instruction* out_pack_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_base_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kPackedVec2Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    acc_vars[lane] = entry_builder.AddVariable(
        vec2_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!out_pack_var || !k_base_var) return 0;
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }
  if (!entry_builder.AddStore(out_pack_var->result_id(), zero_uint_id) ||
      !entry_builder.AddBranch(out_header_label_id)) {
    return 0;
  }

  InstructionBuilder out_header_builder(context(), out_header_block.get());
  Instruction* out_pack_load =
      out_header_builder.AddLoad(uint_type_id, out_pack_var->result_id());
  if (!out_pack_load) return 0;
  Instruction* out_cond = out_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, out_pack_load->result_id(),
      result_length_id);
  if (!out_cond ||
      !out_header_builder.AddLoopMerge(out_merge_label_id,
                                       out_continue_label_id) ||
      !out_header_builder.AddConditionalBranch(
          out_cond->result_id(), out_body_label_id, out_merge_label_id)) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!out_body_builder.AddStore(acc_var->result_id(), zero2_id)) return 0;
  }
  if (!out_body_builder.AddStore(k_base_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k_base_load =
      k_header_builder.AddLoad(uint_type_id, k_base_var->result_id());
  if (!k_base_load) return 0;
  Instruction* k_cond =
      k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                   k_base_load->result_id(), input_length_id);
  if (!k_cond ||
      !k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id) ||
      !k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* input_vec = k_body_builder.AddFunctionCall(
      input.packed_vec2_type_id, input_load_function_id,
      {k_base_load->result_id()});
  if (!input_vec) return 0;
  std::array<uint32_t, kPackedVec2Width> weight_row_ids = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec2Width; ++row_lane) {
    const uint32_t lane_id = GetOrCreateUIntConstant(row_lane);
    if (lane_id == 0) return 0;
    Instruction* matrix_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_base_load->result_id(), lane_id);
    if (!matrix_row) return 0;
    Instruction* matrix_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, matrix_row->result_id(), matrix_cols_id);
    if (!matrix_row_offset) return 0;
    Instruction* matrix_local_base = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, matrix_row_offset->result_id(),
        out_pack_load->result_id());
    if (!matrix_local_base) return 0;
    const uint32_t matrix_memory_base_id =
        direct_matrix_index
            ? matrix_local_base->result_id()
            : BuildRowMajorMatrixMemoryIndex(
                  &k_body_builder, nullptr, matrix_shape_id, matrix_offset_id,
                  matrix.cols, matrix_local_base->result_id());
    if (matrix_memory_base_id == 0) return 0;
    Instruction* weight_row = k_body_builder.AddFunctionCall(
        matrix.packed_vec2_type_id, matrix_load_function_id,
        {matrix_memory_base_id});
    if (!weight_row) return 0;
    weight_row_ids[row_lane] = weight_row->result_id();
  }
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    std::vector<uint32_t> weight_lane_ids;
    weight_lane_ids.reserve(kPackedVec2Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec2Width; ++row_lane) {
      const uint32_t weight_scalar =
          ExtractCompositeElement(&k_body_builder, result.component_type_id,
                                  weight_row_ids[row_lane], lane);
      if (weight_scalar == 0) return 0;
      weight_lane_ids.push_back(weight_scalar);
    }
    Instruction* weight = k_body_builder.AddCompositeConstruct(
        result.packed_vec2_type_id, weight_lane_ids);
    Instruction* acc = k_body_builder.AddLoad(result.packed_vec2_type_id,
                                              acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, result.packed_vec2_type_id,
                 input_vec->result_id(), weight->result_id(), acc->result_id());
    if (fma == 0 ||
        !k_body_builder.AddStore(acc_vars[lane]->result_id(), fma)) {
      return 0;
    }
  }
  if (!k_body_builder.AddBranch(k_continue_label_id)) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue_block.get());
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k_base_load->result_id(),
      packed_width_uint_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_base_var->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec2Width);
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec2_type_id,
                                               acc_vars[lane]->result_id());
    if (!acc) return 0;
    const uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    if (reduced == 0) return 0;
    lane_ids.push_back(reduced);
  }
  Instruction* result_vec = k_merge_builder.AddCompositeConstruct(
      result.packed_vec2_type_id, lane_ids);
  if (!result_vec ||
      !k_merge_builder.AddFunctionCall(
          void_type_id, output_store_function_id,
          {out_pack_load->result_id(), result_vec->result_id()}) ||
      !k_merge_builder.AddBranch(out_continue_label_id)) {
    return 0;
  }

  InstructionBuilder out_continue_builder(context(), out_continue_block.get());
  Instruction* next_out = out_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, out_pack_load->result_id(),
      packed_width_uint_id);
  if (!next_out ||
      !out_continue_builder.AddStore(out_pack_var->result_id(),
                                     next_out->result_id()) ||
      !out_continue_builder.AddBranch(out_header_label_id)) {
    return 0;
  }

  InstructionBuilder out_merge_builder(context(), out_merge_block.get());
  if (!out_merge_builder.AddNullaryOp(0, spv::Op::OpReturn)) return 0;

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
                       /*may_write_memory=*/true);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildConstantPackedVectorSelectFunction(
    const VectorTypeInfo& vector, uint32_t constant_id) {
  if (!IsPackedVec2(vector) || vector.packed_length == 0 ||
      vector.packed_length > kMaxFusedConstantBiasPacks || constant_id == 0) {
    return 0;
  }

  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t function_type_id =
      GetOrCreateFunctionType(vector.packed_vec2_type_id, {uint_type_id});
  if (uint_type_id == 0 || bool_type_id == 0 || function_type_id == 0) {
    return 0;
  }

  const uint32_t function_id = TakeNextId();
  const uint32_t index_param_id = TakeNextId();
  if (function_id == 0 || index_param_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, vector.packed_vec2_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, index_param_id,
      std::initializer_list<Operand>{}));

  std::vector<uint32_t> check_label_ids;
  check_label_ids.reserve(vector.packed_length - 1);
  for (uint32_t pack = 0; pack + 1 < vector.packed_length; ++pack) {
    const uint32_t label_id = TakeNextId();
    if (label_id == 0) return 0;
    check_label_ids.push_back(label_id);
  }
  std::vector<uint32_t> return_label_ids;
  return_label_ids.reserve(vector.packed_length);
  for (uint32_t pack = 0; pack < vector.packed_length; ++pack) {
    const uint32_t label_id = TakeNextId();
    if (label_id == 0) return 0;
    return_label_ids.push_back(label_id);
  }

  std::vector<std::unique_ptr<BasicBlock>> check_blocks;
  check_blocks.reserve(check_label_ids.size());
  for (uint32_t label_id : check_label_ids) {
    check_blocks.emplace_back(MakeBasicBlock(label_id));
    if (!check_blocks.back()) return 0;
  }
  std::vector<std::unique_ptr<BasicBlock>> return_blocks;
  return_blocks.reserve(return_label_ids.size());
  for (uint32_t label_id : return_label_ids) {
    return_blocks.emplace_back(MakeBasicBlock(label_id));
    if (!return_blocks.back()) return 0;
  }

  for (uint32_t pack = 0; pack < check_blocks.size(); ++pack) {
    InstructionBuilder builder(context(), check_blocks[pack].get());
    const uint32_t pack_base_id =
        GetOrCreateUIntConstant(pack * kPackedVec2Width);
    Instruction* is_pack =
        pack_base_id ? builder.AddBinaryOp(bool_type_id, spv::Op::OpIEqual,
                                           index_param_id, pack_base_id)
                     : nullptr;
    const uint32_t next_check_label_id = pack + 1 < check_label_ids.size()
                                             ? check_label_ids[pack + 1]
                                             : return_label_ids.back();
    if (!is_pack || !builder.AddSelectionMerge(next_check_label_id) ||
        !builder.AddConditionalBranch(is_pack->result_id(),
                                      return_label_ids[pack],
                                      next_check_label_id)) {
      return 0;
    }
  }

  Instruction* constant = get_def_use_mgr()->GetDef(constant_id);
  for (uint32_t pack = 0; pack < return_blocks.size(); ++pack) {
    InstructionBuilder builder(context(), return_blocks[pack].get());
    uint32_t value_id = 0;
    if (constant && constant->opcode() == spv::Op::OpConstantComposite &&
        constant->NumInOperands() == vector.packed_length) {
      const uint32_t operand_id = constant->GetSingleWordInOperand(pack);
      Instruction* operand = get_def_use_mgr()->GetDef(operand_id);
      if (operand && operand->type_id() == vector.packed_vec2_type_id) {
        value_id = operand_id;
      }
    }
    if (value_id == 0) {
      value_id = ExtractCompositeElement(&builder, vector.packed_vec2_type_id,
                                         constant_id, pack);
    }
    if (value_id == 0 ||
        !builder.AddUnaryOp(0, spv::Op::OpReturnValue, value_id)) {
      return 0;
    }
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  if (check_blocks.empty()) {
    function->AddBasicBlock(std::move(return_blocks[0]));
  } else {
    for (uint32_t pack = 0; pack < check_blocks.size(); ++pack) {
      function->AddBasicBlock(std::move(check_blocks[pack]));
      function->AddBasicBlock(std::move(return_blocks[pack]));
    }
    function->AddBasicBlock(std::move(return_blocks.back()));
  }
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);
  return function_id;
}

uint32_t
HwLowerToStandardPass::BuildFusedVectorMatmulAddStoreFunctionPackedVec2(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
    uint32_t bias_constant_id, uint32_t input_pointer_id,
    uint32_t input_pointer_type_id,
    const std::vector<Operand>& input_memory_operands,
    uint32_t matrix_pointer_id, uint32_t matrix_pointer_type_id,
    uint32_t matrix_shape_id, uint32_t matrix_offset_id,
    const std::vector<Operand>& matrix_memory_operands,
    uint32_t output_pointer_id, uint32_t output_pointer_type_id,
    const std::vector<Operand>& output_memory_operands) {
  if (!IsPackedVec2(result) || !IsPackedVec2(input) || !IsPackedVec2(matrix) ||
      input.length != matrix.rows || result.length != matrix.cols ||
      (has_bias && (!bias || !IsSamePackedVec2Kind(result, *bias) ||
                    bias_constant_id == 0))) {
    return 0;
  }

  const uint32_t input_load_function_id = GetOrCreatePackedLoadChunkFunction(
      input_pointer_id, input_pointer_type_id, input.component_type_id,
      input.packed_vec2_type_id, input_memory_operands);
  const uint32_t output_store_function_id = GetOrCreatePackedStoreChunkFunction(
      output_pointer_id, output_pointer_type_id, result.component_type_id,
      result.packed_vec2_type_id, output_memory_operands);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t function_type_id = GetOrCreateFunctionType(void_type_id, {});
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t vec2_function_ptr_type_id = GetOrCreatePointerType(
      result.packed_vec2_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t result_length_id = GetOrCreateUIntConstant(result.length);
  const uint32_t input_length_id = GetOrCreateUIntConstant(input.length);
  const uint32_t matrix_cols_id = GetOrCreateUIntConstant(matrix.cols);
  const uint32_t zero2_id = GetOrCreateZero(result.packed_vec2_type_id);
  const uint32_t bias_select_function_id =
      has_bias
          ? BuildConstantPackedVectorSelectFunction(*bias, bias_constant_id)
          : 0;
  if (input_load_function_id == 0 || output_store_function_id == 0 ||
      void_type_id == 0 || function_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || vec2_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || packed_width_uint_id == 0 ||
      result_length_id == 0 || input_length_id == 0 || matrix_cols_id == 0 ||
      zero2_id == 0 || (has_bias && bias_select_function_id == 0)) {
    return 0;
  }

  auto get_constant_pair_component = [this](uint32_t pair_id, uint32_t index,
                                            uint32_t* value) {
    if (!value || index >= 2) return false;
    Instruction* pair = get_def_use_mgr()->GetDef(pair_id);
    if (!pair) return false;
    if (pair->opcode() == spv::Op::OpConstantNull) {
      *value = 0;
      return true;
    }
    return pair->opcode() == spv::Op::OpConstantComposite &&
           pair->NumInOperands() > index &&
           GetConstantU32(pair->GetSingleWordInOperand(index), value);
  };
  uint32_t shape_cols = 0;
  uint32_t offset_row = 0;
  uint32_t offset_col = 0;
  const bool direct_matrix_index =
      get_constant_pair_component(matrix_shape_id, 1, &shape_cols) &&
      get_constant_pair_component(matrix_offset_id, 0, &offset_row) &&
      get_constant_pair_component(matrix_offset_id, 1, &offset_col) &&
      shape_cols == matrix.cols && offset_row == 0 && offset_col == 0;

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start =
      MakeUnique<Instruction>(context(), spv::Op::OpFunction, void_type_id,
                              function_id, std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t entry_label_id = TakeNextId();
  const uint32_t out_header_label_id = TakeNextId();
  const uint32_t out_body_label_id = TakeNextId();
  const uint32_t out_continue_label_id = TakeNextId();
  const uint32_t out_merge_label_id = TakeNextId();
  const uint32_t k_header_label_id = TakeNextId();
  const uint32_t k_body_label_id = TakeNextId();
  const uint32_t k_continue_label_id = TakeNextId();
  const uint32_t k_merge_label_id = TakeNextId();
  if (entry_label_id == 0 || out_header_label_id == 0 ||
      out_body_label_id == 0 || out_continue_label_id == 0 ||
      out_merge_label_id == 0 || k_header_label_id == 0 ||
      k_body_label_id == 0 || k_continue_label_id == 0 ||
      k_merge_label_id == 0) {
    return 0;
  }

  std::unique_ptr<BasicBlock> entry_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_label_id));
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
  if (!entry_block || !out_header_block || !out_body_block ||
      !out_continue_block || !out_merge_block || !k_header_block ||
      !k_body_block || !k_continue_block || !k_merge_block) {
    return 0;
  }

  InstructionBuilder entry_builder(context(), entry_block.get());
  Instruction* output_base_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_base_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* acc_var = entry_builder.AddVariable(
      vec2_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  if (!output_base_var || !k_base_var || !acc_var) return 0;
  const uint32_t captured_matrix_pointer_id =
      BuildCapturedPointer(&entry_builder, matrix_pointer_id);
  if (captured_matrix_pointer_id == 0 ||
      !entry_builder.AddStore(output_base_var->result_id(), zero_uint_id) ||
      !entry_builder.AddBranch(out_header_label_id)) {
    return 0;
  }

  InstructionBuilder out_header_builder(context(), out_header_block.get());
  Instruction* output_base =
      out_header_builder.AddLoad(uint_type_id, output_base_var->result_id());
  if (!output_base) return 0;
  Instruction* out_cond = out_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, output_base->result_id(),
      result_length_id);
  if (!out_cond) return 0;
  if (!out_header_builder.AddLoopMerge(out_merge_label_id,
                                       out_continue_label_id) ||
      !out_header_builder.AddConditionalBranch(
          out_cond->result_id(), out_body_label_id, out_merge_label_id)) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body_block.get());
  if (!out_body_builder.AddStore(acc_var->result_id(), zero2_id) ||
      !out_body_builder.AddStore(k_base_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k_base_load =
      k_header_builder.AddLoad(uint_type_id, k_base_var->result_id());
  if (!k_base_load) return 0;
  Instruction* k_cond =
      k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                   k_base_load->result_id(), input_length_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id) ||
      !k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* input_vec = k_body_builder.AddFunctionCall(
      input.packed_vec2_type_id, input_load_function_id,
      {k_base_load->result_id()});
  if (!input_vec) return 0;
  std::vector<uint32_t> dot_ids;
  dot_ids.reserve(kPackedVec2Width);
  for (uint32_t output_lane = 0; output_lane < kPackedVec2Width;
       ++output_lane) {
    std::vector<uint32_t> weight_column_ids;
    weight_column_ids.reserve(kPackedVec2Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec2Width; ++row_lane) {
      const uint32_t row_lane_id = GetOrCreateUIntConstant(row_lane);
      const uint32_t output_lane_id = GetOrCreateUIntConstant(output_lane);
      if (row_lane_id == 0 || output_lane_id == 0) return 0;
      Instruction* matrix_row = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, k_base_load->result_id(), row_lane_id);
      Instruction* matrix_row_offset =
          matrix_row ? k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                                  matrix_row->result_id(),
                                                  matrix_cols_id)
                     : nullptr;
      Instruction* matrix_output_base =
          matrix_row_offset
              ? k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                           matrix_row_offset->result_id(),
                                           output_base->result_id())
              : nullptr;
      Instruction* matrix_local_index =
          matrix_output_base
              ? k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                           matrix_output_base->result_id(),
                                           output_lane_id)
              : nullptr;
      if (!matrix_local_index) return 0;
      const uint32_t matrix_memory_index_id =
          direct_matrix_index
              ? matrix_local_index->result_id()
              : BuildRowMajorMatrixMemoryIndex(
                    &k_body_builder, nullptr, matrix_shape_id, matrix_offset_id,
                    matrix.cols, matrix_local_index->result_id());
      const uint32_t matrix_element_pointer_id =
          matrix_memory_index_id
              ? BuildElementAccessFromPointerType(
                    &k_body_builder, matrix_pointer_type_id,
                    captured_matrix_pointer_id, matrix.component_type_id,
                    matrix_memory_index_id)
              : 0;
      const uint32_t weight_scalar_id =
          matrix_element_pointer_id
              ? AddLoad(&k_body_builder, matrix.component_type_id,
                        matrix_element_pointer_id, matrix_memory_operands)
              : 0;
      if (weight_scalar_id == 0) return 0;
      weight_column_ids.push_back(weight_scalar_id);
    }
    Instruction* weight_column = k_body_builder.AddCompositeConstruct(
        result.packed_vec2_type_id, weight_column_ids);
    Instruction* dot =
        weight_column ? k_body_builder.AddBinaryOp(
                            result.component_type_id, spv::Op::OpDot,
                            input_vec->result_id(), weight_column->result_id())
                      : nullptr;
    if (!dot) return 0;
    ApplyActiveFPFastMathMode(dot);
    dot_ids.push_back(dot->result_id());
  }
  Instruction* dot_vec =
      k_body_builder.AddCompositeConstruct(result.packed_vec2_type_id, dot_ids);
  Instruction* acc =
      k_body_builder.AddLoad(result.packed_vec2_type_id, acc_var->result_id());
  Instruction* next_acc = dot_vec && acc
                              ? k_body_builder.AddBinaryOp(
                                    result.packed_vec2_type_id, spv::Op::OpFAdd,
                                    acc->result_id(), dot_vec->result_id())
                              : nullptr;
  if (!next_acc ||
      !k_body_builder.AddStore(acc_var->result_id(), next_acc->result_id())) {
    return 0;
  }
  ApplyActiveFPFastMathMode(next_acc);
  if (!k_body_builder.AddBranch(k_continue_label_id)) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue_block.get());
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k_base_load->result_id(),
      packed_width_uint_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_base_var->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  Instruction* result_vec =
      k_merge_builder.AddLoad(result.packed_vec2_type_id, acc_var->result_id());
  if (!result_vec) return 0;
  if (has_bias) {
    Instruction* bias_vec = k_merge_builder.AddFunctionCall(
        bias->packed_vec2_type_id, bias_select_function_id,
        {output_base->result_id()});
    if (!bias_vec) return 0;
    result_vec = k_merge_builder.AddBinaryOp(
        result.packed_vec2_type_id, spv::Op::OpFAdd, result_vec->result_id(),
        bias_vec->result_id());
    if (!result_vec) return 0;
    ApplyActiveFPFastMathMode(result_vec);
  }
  if (!k_merge_builder.AddFunctionCall(
          void_type_id, output_store_function_id,
          {output_base->result_id(), result_vec->result_id()}) ||
      !k_merge_builder.AddBranch(out_continue_label_id)) {
    return 0;
  }

  InstructionBuilder out_continue_builder(context(), out_continue_block.get());
  Instruction* next_out = out_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, output_base->result_id(),
      packed_width_uint_id);
  if (!next_out) return 0;
  if (!out_continue_builder.AddStore(output_base_var->result_id(),
                                     next_out->result_id()) ||
      !out_continue_builder.AddBranch(out_header_label_id)) {
    return 0;
  }

  InstructionBuilder out_merge_builder(context(), out_merge_block.get());
  if (!out_merge_builder.AddNullaryOp(0, spv::Op::OpReturn)) return 0;

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
                       /*may_write_memory=*/true);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildFusedMatrixMatmulStoreFunctionPackedVec2(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c, uint32_t a_pointer_id,
    uint32_t a_pointer_type_id, uint32_t a_shape_id, uint32_t a_offset_id,
    const std::vector<Operand>& a_memory_operands, uint32_t b_pointer_id,
    uint32_t b_pointer_type_id, uint32_t b_shape_id, uint32_t b_offset_id,
    const std::vector<Operand>& b_memory_operands, uint32_t c_pointer_id,
    uint32_t c_pointer_type_id, uint32_t c_shape_id, uint32_t c_offset_id,
    const std::vector<Operand>& c_memory_operands, uint32_t output_pointer_id,
    uint32_t output_pointer_type_id, uint32_t output_shape_id,
    uint32_t output_offset_id,
    const std::vector<Operand>& output_memory_operands) {
  if (!CanUsePackedVec2MatrixMulAdd(result, a, b, c)) return 0;
  if (a_pointer_id == 0 || a_pointer_type_id == 0 || a_shape_id == 0 ||
      a_offset_id == 0 || b_pointer_id == 0 || b_pointer_type_id == 0 ||
      b_shape_id == 0 || b_offset_id == 0 || c_pointer_id == 0 ||
      c_pointer_type_id == 0 || c_shape_id == 0 || c_offset_id == 0 ||
      output_pointer_id == 0 || output_pointer_type_id == 0 ||
      output_shape_id == 0 || output_offset_id == 0) {
    return 0;
  }

  const uint32_t a_load_function_id = GetOrCreatePackedLoadChunkFunction(
      a_pointer_id, a_pointer_type_id, a.component_type_id,
      a.packed_vec2_type_id, a_memory_operands);
  const uint32_t b_load_function_id = GetOrCreatePackedLoadChunkFunction(
      b_pointer_id, b_pointer_type_id, b.component_type_id,
      b.packed_vec2_type_id, b_memory_operands);
  const uint32_t c_load_function_id = GetOrCreatePackedLoadChunkFunction(
      c_pointer_id, c_pointer_type_id, c.component_type_id,
      c.packed_vec2_type_id, c_memory_operands);
  const uint32_t output_store_function_id = GetOrCreatePackedStoreChunkFunction(
      output_pointer_id, output_pointer_type_id, result.component_type_id,
      result.packed_vec2_type_id, output_memory_operands);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t function_type_id = GetOrCreateFunctionType(void_type_id, {});
  const uint32_t vec2_function_ptr_type_id = GetOrCreatePointerType(
      result.packed_vec2_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t row_count_id = GetOrCreateUIntConstant(result.rows);
  const uint32_t result_packed_cols_id =
      GetOrCreateUIntConstant(result.packed_cols);
  const uint32_t a_cols_id = GetOrCreateUIntConstant(a.cols);
  const uint32_t b_cols_id = GetOrCreateUIntConstant(b.cols);
  const uint32_t c_cols_id = GetOrCreateUIntConstant(c.cols);
  const uint32_t result_cols_id = GetOrCreateUIntConstant(result.cols);
  const uint32_t a_packed_cols_id = GetOrCreateUIntConstant(a.packed_cols);
  const uint32_t zero2_id = GetOrCreateZero(result.packed_vec2_type_id);
  if (a_load_function_id == 0 || b_load_function_id == 0 ||
      c_load_function_id == 0 || output_store_function_id == 0 ||
      void_type_id == 0 || function_type_id == 0 ||
      vec2_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || packed_width_uint_id == 0 ||
      row_count_id == 0 || result_packed_cols_id == 0 || a_cols_id == 0 ||
      b_cols_id == 0 || c_cols_id == 0 || result_cols_id == 0 ||
      a_packed_cols_id == 0 || zero2_id == 0) {
    return 0;
  }

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start =
      MakeUnique<Instruction>(context(), spv::Op::OpFunction, void_type_id,
                              function_id, std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t entry_label_id = TakeNextId();
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
  if (entry_label_id == 0 || row_header_label_id == 0 ||
      row_body_label_id == 0 || row_continue_label_id == 0 ||
      row_merge_label_id == 0 || col_header_label_id == 0 ||
      col_body_label_id == 0 || col_continue_label_id == 0 ||
      col_merge_label_id == 0 || k_header_label_id == 0 ||
      k_body_label_id == 0 || k_continue_label_id == 0 ||
      k_merge_label_id == 0) {
    return 0;
  }

  std::unique_ptr<BasicBlock> entry_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_label_id));
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
  if (!entry_block || !row_header_block || !row_body_block ||
      !row_continue_block || !row_merge_block || !col_header_block ||
      !col_body_block || !col_continue_block || !col_merge_block ||
      !k_header_block || !k_body_block || !k_continue_block || !k_merge_block) {
    return 0;
  }

  InstructionBuilder entry_builder(context(), entry_block.get());
  Instruction* row_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* col_pack_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_pack_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kPackedVec2Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    acc_vars[lane] = entry_builder.AddVariable(
        vec2_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!row_var || !col_pack_var || !k_pack_var) return 0;
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }
  if (!entry_builder.AddStore(row_var->result_id(), zero_uint_id) ||
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
                                       row_continue_label_id) ||
      !row_header_builder.AddConditionalBranch(
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
                                       col_continue_label_id) ||
      !col_header_builder.AddConditionalBranch(
          col_cond->result_id(), col_body_label_id, col_merge_label_id)) {
    return 0;
  }

  InstructionBuilder col_body_builder(context(), col_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!col_body_builder.AddStore(acc_var->result_id(), zero2_id)) return 0;
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
                                   k_pack_load->result_id(), a_packed_cols_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id) ||
      !k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* k_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_pack_load->result_id(),
      packed_width_uint_id);
  if (!k_base) return 0;
  Instruction* a_row_offset = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, row_load->result_id(), a_cols_id);
  if (!a_row_offset) return 0;
  Instruction* a_local_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, a_row_offset->result_id(),
      k_base->result_id());
  if (!a_local_base) return 0;
  const uint32_t a_memory_base_id = BuildRowMajorMatrixMemoryIndex(
      &k_body_builder, nullptr, a_shape_id, a_offset_id, a.cols,
      a_local_base->result_id());
  if (a_memory_base_id == 0) return 0;
  Instruction* a_vec = k_body_builder.AddFunctionCall(
      a.packed_vec2_type_id, a_load_function_id, {a_memory_base_id});
  if (!a_vec) return 0;

  std::array<uint32_t, kPackedVec2Width> b_vecs = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec2Width; ++row_lane) {
    const uint32_t row_lane_id = GetOrCreateUIntConstant(row_lane);
    if (row_lane_id == 0) return 0;
    Instruction* b_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_base->result_id(), row_lane_id);
    if (!b_row) return 0;
    Instruction* b_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, b_row->result_id(), b_cols_id);
    if (!b_row_offset) return 0;
    Instruction* b_col_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
        packed_width_uint_id);
    if (!b_col_offset) return 0;
    Instruction* b_local_base = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, b_row_offset->result_id(),
        b_col_offset->result_id());
    if (!b_local_base) return 0;
    const uint32_t b_memory_base_id = BuildRowMajorMatrixMemoryIndex(
        &k_body_builder, nullptr, b_shape_id, b_offset_id, b.cols,
        b_local_base->result_id());
    if (b_memory_base_id == 0) return 0;
    Instruction* b_vec = k_body_builder.AddFunctionCall(
        b.packed_vec2_type_id, b_load_function_id, {b_memory_base_id});
    if (!b_vec) return 0;
    b_vecs[row_lane] = b_vec->result_id();
  }

  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    std::vector<uint32_t> weight_lanes;
    weight_lanes.reserve(kPackedVec2Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec2Width; ++row_lane) {
      const uint32_t value = ExtractCompositeElement(
          &k_body_builder, result.component_type_id, b_vecs[row_lane], lane);
      if (value == 0) return 0;
      weight_lanes.push_back(value);
    }
    Instruction* weight_vec = k_body_builder.AddCompositeConstruct(
        result.packed_vec2_type_id, weight_lanes);
    Instruction* acc = k_body_builder.AddLoad(result.packed_vec2_type_id,
                                              acc_vars[lane]->result_id());
    if (!weight_vec || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, result.packed_vec2_type_id,
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

  // k_merge: finalize the tile — reduce accumulators, add bias from C,
  // and stream the resulting vec2 directly to the output SSBO.
  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  Instruction* c_row_offset = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, row_load->result_id(), c_cols_id);
  if (!c_row_offset) return 0;
  Instruction* c_col_offset = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
      packed_width_uint_id);
  if (!c_col_offset) return 0;
  Instruction* c_local_base = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, c_row_offset->result_id(),
      c_col_offset->result_id());
  if (!c_local_base) return 0;
  const uint32_t c_memory_base_id = BuildRowMajorMatrixMemoryIndex(
      &k_merge_builder, nullptr, c_shape_id, c_offset_id, c.cols,
      c_local_base->result_id());
  if (c_memory_base_id == 0) return 0;
  Instruction* c_vec = k_merge_builder.AddFunctionCall(
      c.packed_vec2_type_id, c_load_function_id, {c_memory_base_id});
  if (!c_vec) return 0;

  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec2Width);
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec2_type_id,
                                               acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    if (reduced == 0) return 0;
    const uint32_t bias_value = ExtractCompositeElement(
        &k_merge_builder, result.component_type_id, c_vec->result_id(), lane);
    if (bias_value == 0) return 0;
    Instruction* add = k_merge_builder.AddBinaryOp(
        result.component_type_id, spv::Op::OpFAdd, bias_value, reduced);
    if (!add) return 0;
    ApplyActiveFPFastMathMode(add);
    lane_ids.push_back(add->result_id());
  }
  Instruction* result_vec = k_merge_builder.AddCompositeConstruct(
      result.packed_vec2_type_id, lane_ids);
  if (!result_vec) return 0;

  // Compute the flat output SSBO element index for the current tile:
  // base = row * result.cols + col_pack * packed width, then apply
  // output_shape/offset.
  Instruction* out_col_offset = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
      packed_width_uint_id);
  if (!out_col_offset) return 0;
  Instruction* out_row_offset = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, row_load->result_id(), result_cols_id);
  if (!out_row_offset) return 0;
  Instruction* out_local_base = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, out_row_offset->result_id(),
      out_col_offset->result_id());
  if (!out_local_base) return 0;
  const uint32_t out_memory_base_id = BuildRowMajorMatrixMemoryIndex(
      &k_merge_builder, nullptr, output_shape_id, output_offset_id, result.cols,
      out_local_base->result_id());
  if (out_memory_base_id == 0) return 0;
  if (!k_merge_builder.AddFunctionCall(
          void_type_id, output_store_function_id,
          {out_memory_base_id, result_vec->result_id()}) ||
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
  if (!row_merge_builder.AddNullaryOp(0, spv::Op::OpReturn)) return 0;

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
                       /*may_write_memory=*/true);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildDirectVectorMatmulFunctionUnrolled(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
    uint32_t input_pointer_id, uint32_t input_pointer_type_id,
    const std::vector<Operand>& input_memory_operands,
    uint32_t input_constant_id, bool input_is_value, uint32_t matrix_pointer_id,
    uint32_t matrix_pointer_type_id, uint32_t matrix_shape_id,
    uint32_t matrix_offset_id,
    const std::vector<Operand>& matrix_memory_operands,
    uint32_t matrix_constant_id, bool matrix_is_value, uint32_t bias_pointer_id,
    uint32_t bias_pointer_type_id,
    const std::vector<Operand>& bias_memory_operands,
    uint32_t bias_source_component_type_id, uint32_t bias_offset,
    uint32_t bias_conversion_fp_fast_math_mode,
    bool bias_conversion_has_explicit_fp_fast_math_mode,
    uint32_t bias_constant_id, bool bias_is_value,
    const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments) {
  const bool same_component =
      result.component_type_id == input.component_type_id &&
      result.component_type_id == matrix.component_type_id &&
      (!has_bias ||
       (bias && result.component_type_id == bias->component_type_id)) &&
      (IsFloat16Type(result.component_type_id) ||
       IsFloat32Type(result.component_type_id));
  const bool mixed_f16_f32 =
      IsFloat32Type(result.component_type_id) &&
      IsFloat16Type(input.component_type_id) &&
      IsFloat16Type(matrix.component_type_id) &&
      (!has_bias || (bias && IsFloat32Type(bias->component_type_id)));
  if ((!same_component && !mixed_f16_f32) || input.length != matrix.rows ||
      result.length != matrix.cols ||
      (has_bias && (!bias || bias->length != result.length))) {
    return 0;
  }
  if (has_bias && !bias_is_value &&
      bias_source_component_type_id != bias->component_type_id &&
      !(IsFloat16Type(bias_source_component_type_id) &&
        IsFloat32Type(bias->component_type_id))) {
    return 0;
  }
  if ((!input_is_value &&
       (input_pointer_id == 0 || input_pointer_type_id == 0)) ||
      (!matrix_is_value &&
       (matrix_pointer_id == 0 || matrix_pointer_type_id == 0 ||
        matrix_shape_id == 0 || matrix_offset_id == 0)) ||
      (has_bias && !bias_is_value &&
       (bias_pointer_id == 0 || bias_pointer_type_id == 0))) {
    return 0;
  }

  std::vector<uint32_t> parameter_type_ids;
  if (input_is_value && input_constant_id == 0) {
    parameter_type_ids.push_back(input.lowered_type_id);
  }
  if (matrix_is_value && matrix_constant_id == 0) {
    parameter_type_ids.push_back(matrix.lowered_type_id);
  }
  if (has_bias && bias_is_value && bias_constant_id == 0) {
    parameter_type_ids.push_back(bias->lowered_type_id);
  }
  if (parameter_type_ids.size() != value_arguments.size()) return 0;
  for (size_t i = 0; i < parameter_type_ids.size(); ++i) {
    if (parameter_type_ids[i] != value_arguments[i].second) return 0;
  }

  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, parameter_type_ids);
  const uint32_t function_id = TakeNextId();
  if (function_type_id == 0 || function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, result.lowered_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  size_t parameter_index = 0;
  auto add_parameter = [&](uint32_t type_id) -> uint32_t {
    if (parameter_index >= parameter_type_ids.size() ||
        parameter_type_ids[parameter_index] != type_id) {
      return 0;
    }
    const uint32_t parameter_id = TakeNextId();
    if (parameter_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, type_id, parameter_id,
        std::initializer_list<Operand>{}));
    ++parameter_index;
    return parameter_id;
  };

  uint32_t input_value_id = input_constant_id;
  if (input_is_value && input_value_id == 0) {
    input_value_id = add_parameter(input.lowered_type_id);
  }
  uint32_t matrix_value_id = matrix_constant_id;
  if (matrix_is_value && matrix_value_id == 0) {
    matrix_value_id = add_parameter(matrix.lowered_type_id);
  }
  uint32_t bias_value_id = bias_constant_id;
  if (has_bias && bias_is_value && bias_value_id == 0) {
    bias_value_id = add_parameter(bias->lowered_type_id);
  }
  if ((input_is_value && input_value_id == 0) ||
      (matrix_is_value && matrix_value_id == 0) ||
      (has_bias && bias_is_value && bias_value_id == 0) ||
      parameter_index != parameter_type_ids.size()) {
    return 0;
  }

  const uint32_t label_id = TakeNextId();
  if (label_id == 0) return 0;
  std::unique_ptr<BasicBlock> block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(label_id));
  if (!block) return 0;
  InstructionBuilder builder(context(), block.get());

  const uint32_t captured_input_pointer_id =
      input_is_value ? 0 : BuildCapturedPointer(&builder, input_pointer_id);
  const uint32_t captured_matrix_pointer_id =
      matrix_is_value ? 0 : BuildCapturedPointer(&builder, matrix_pointer_id);
  const uint32_t captured_bias_pointer_id =
      !has_bias || bias_is_value
          ? 0
          : BuildCapturedPointer(&builder, bias_pointer_id);
  if ((!input_is_value && captured_input_pointer_id == 0) ||
      (!matrix_is_value && captured_matrix_pointer_id == 0) ||
      (has_bias && !bias_is_value && captured_bias_pointer_id == 0)) {
    return 0;
  }

  auto load_buffer_scalar_id =
      [&](uint32_t pointer_type_id, uint32_t pointer_id,
          uint32_t component_type_id, uint32_t index_id,
          const std::vector<Operand>& memory_operands) -> uint32_t {
    const uint32_t element_pointer_id =
        index_id ? BuildElementAccessFromPointerType(
                       &builder, pointer_type_id, pointer_id, component_type_id,
                       index_id)
                 : 0;
    return element_pointer_id ? AddLoad(&builder, component_type_id,
                                        element_pointer_id, memory_operands)
                              : 0;
  };
  auto load_buffer_scalar =
      [&](uint32_t pointer_type_id, uint32_t pointer_id,
          uint32_t component_type_id, uint32_t index,
          const std::vector<Operand>& memory_operands) -> uint32_t {
    const uint32_t index_id = GetOrCreateUIntConstant(index);
    return index_id ? load_buffer_scalar_id(pointer_type_id, pointer_id,
                                            component_type_id, index_id,
                                            memory_operands)
                    : 0;
  };
  auto load_input_scalar = [&](uint32_t index) -> uint32_t {
    return input_is_value
               ? ExtractVectorScalar(&builder, input, input_value_id, index)
               : load_buffer_scalar(
                     input_pointer_type_id, captured_input_pointer_id,
                     input.component_type_id, index, input_memory_operands);
  };
  auto load_matrix_scalar = [&](uint32_t row, uint32_t col) -> uint32_t {
    if (matrix_is_value) {
      return ExtractMatrixScalar(&builder, matrix, matrix_value_id, row, col);
    }
    const uint64_t flat_index = static_cast<uint64_t>(row) * matrix.cols + col;
    if (flat_index > std::numeric_limits<uint32_t>::max()) return 0;
    const uint32_t flat_index_id =
        GetOrCreateUIntConstant(static_cast<uint32_t>(flat_index));
    const uint32_t memory_index_id =
        flat_index_id ? BuildRowMajorMatrixMemoryIndex(
                            &builder, nullptr, matrix_shape_id,
                            matrix_offset_id, matrix.cols, flat_index_id)
                      : 0;
    return memory_index_id
               ? load_buffer_scalar_id(matrix_pointer_type_id,
                                       captured_matrix_pointer_id,
                                       matrix.component_type_id,
                                       memory_index_id, matrix_memory_operands)
               : 0;
  };
  auto load_bias_scalar = [&](uint32_t index) -> uint32_t {
    if (!has_bias || !bias) return 0;
    if (bias_is_value) {
      return ExtractVectorScalar(&builder, *bias, bias_value_id, index);
    }
    const uint64_t source_index = static_cast<uint64_t>(bias_offset) + index;
    if (source_index > std::numeric_limits<uint32_t>::max()) return 0;
    uint32_t value_id = load_buffer_scalar(
        bias_pointer_type_id, captured_bias_pointer_id,
        bias_source_component_type_id, static_cast<uint32_t>(source_index),
        bias_memory_operands);
    if (value_id == 0 ||
        bias_source_component_type_id == bias->component_type_id) {
      return value_id;
    }
    Instruction* converted = builder.AddUnaryOp(bias->component_type_id,
                                                spv::Op::OpFConvert, value_id);
    ApplyFPFastMathMode(converted, bias_conversion_fp_fast_math_mode,
                        bias_conversion_has_explicit_fp_fast_math_mode);
    return converted ? converted->result_id() : 0;
  };
  Instruction* type_insertion_point =
      get_def_use_mgr()->GetDef(result.lowered_type_id);
  if (!type_insertion_point) return 0;
  const uint32_t input_vec2_type_id = GetOrCreateVectorType(
      input.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t result_vec2_type_id = GetOrCreateVectorType(
      result.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t input_zero_id = GetOrCreateZero(input.component_type_id);
  const uint32_t result_zero2_id = GetOrCreateZero(result_vec2_type_id);
  if (input_vec2_type_id == 0 || result_vec2_type_id == 0 ||
      input_zero_id == 0 || result_zero2_id == 0) {
    return 0;
  }
  auto widen_vec = [&](uint32_t value_id) -> uint32_t {
    if (value_id == 0 || input.component_type_id == result.component_type_id) {
      return value_id;
    }
    Instruction* converted =
        builder.AddUnaryOp(result_vec2_type_id, spv::Op::OpFConvert, value_id);
    ApplyActiveFPFastMathMode(converted);
    return converted ? converted->result_id() : 0;
  };

  std::vector<uint32_t> input_scalar_ids(input.length, 0);
  for (uint32_t k = 0; k < input.length; ++k) {
    input_scalar_ids[k] = load_input_scalar(k);
    if (input_scalar_ids[k] == 0) return 0;
  }
  std::vector<uint32_t> matrix_scalar_ids(input.length * result.length, 0);
  for (uint32_t k = 0; k < input.length; ++k) {
    for (uint32_t col = 0; col < result.length; ++col) {
      const uint32_t index = k * result.length + col;
      matrix_scalar_ids[index] = load_matrix_scalar(k, col);
      if (matrix_scalar_ids[index] == 0) return 0;
    }
  }

  std::vector<uint32_t> scalar_ids;
  scalar_ids.reserve(result.length);
  for (uint32_t col = 0; col < result.length; ++col) {
    uint32_t accumulator_vec_id = result_zero2_id;
    for (uint32_t k = 0; k < input.length; k += kPackedVec2Width) {
      std::vector<uint32_t> input_lane_ids;
      std::vector<uint32_t> weight_lane_ids;
      input_lane_ids.reserve(kPackedVec2Width);
      weight_lane_ids.reserve(kPackedVec2Width);
      for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
        const uint32_t input_index = k + lane;
        const uint32_t input_id = input_index < input.length
                                      ? input_scalar_ids[input_index]
                                      : input_zero_id;
        const uint32_t weight_id =
            input_index < input.length
                ? matrix_scalar_ids[input_index * result.length + col]
                : input_zero_id;
        if (input_id == 0 || weight_id == 0) return 0;
        input_lane_ids.push_back(input_id);
        weight_lane_ids.push_back(weight_id);
      }
      Instruction* input_vec =
          builder.AddCompositeConstruct(input_vec2_type_id, input_lane_ids);
      Instruction* weight_vec =
          builder.AddCompositeConstruct(input_vec2_type_id, weight_lane_ids);
      const uint32_t compute_input_id =
          input_vec ? widen_vec(input_vec->result_id()) : 0;
      const uint32_t compute_weight_id =
          weight_vec ? widen_vec(weight_vec->result_id()) : 0;
      if (compute_input_id == 0 || compute_weight_id == 0) return 0;
      accumulator_vec_id =
          BuildFma(&builder, result_vec2_type_id, compute_input_id,
                   compute_weight_id, accumulator_vec_id);
      if (accumulator_vec_id == 0) return 0;
    }
    uint32_t accumulator_id = BuildHorizontalReduce(
        &builder, result.component_type_id, accumulator_vec_id);
    if (accumulator_id == 0) return 0;
    if (has_bias) {
      const uint32_t bias_id = load_bias_scalar(col);
      Instruction* sum = bias_id ? builder.AddBinaryOp(result.component_type_id,
                                                       spv::Op::OpFAdd,
                                                       accumulator_id, bias_id)
                                 : nullptr;
      if (!sum) return 0;
      ApplyActiveFPFastMathMode(sum);
      accumulator_id = sum->result_id();
    }
    scalar_ids.push_back(accumulator_id);
  }

  std::vector<uint32_t> piece_ids;
  if (IsPackedVec2(result)) {
    piece_ids.reserve(result.packed_length);
    for (uint32_t pack = 0; pack < result.packed_length; ++pack) {
      Instruction* piece = builder.AddCompositeConstruct(
          result.packed_vec2_type_id,
          {scalar_ids[pack * kPackedVec2Width],
           scalar_ids[pack * kPackedVec2Width + 1]});
      if (!piece) return 0;
      piece_ids.push_back(piece->result_id());
    }
  } else {
    piece_ids = scalar_ids;
  }
  Instruction* result_value =
      builder.AddCompositeConstruct(result.lowered_type_id, piece_ids);
  if (!result_value || !builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                           result_value->result_id())) {
    return 0;
  }
  function->AddBasicBlock(std::move(block));
  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildDirectVectorMatmulFunctionPackedVec2(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
    uint32_t input_pointer_id, uint32_t input_pointer_type_id,
    const std::vector<Operand>& input_memory_operands,
    uint32_t input_constant_id, bool input_is_value, uint32_t matrix_pointer_id,
    uint32_t matrix_pointer_type_id, uint32_t matrix_shape_id,
    uint32_t matrix_offset_id,
    const std::vector<Operand>& matrix_memory_operands,
    uint32_t matrix_constant_id, bool matrix_is_value, uint32_t bias_pointer_id,
    uint32_t bias_pointer_type_id,
    const std::vector<Operand>& bias_memory_operands,
    uint32_t bias_source_component_type_id, uint32_t bias_offset,
    uint32_t bias_conversion_fp_fast_math_mode,
    bool bias_conversion_has_explicit_fp_fast_math_mode,
    uint32_t bias_constant_id, bool bias_is_value,
    const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments) {
  const uint64_t mac_count = static_cast<uint64_t>(input.length) *
                             static_cast<uint64_t>(result.length);
  if (mac_count <= max_unrolled_matmul_macs_) {
    return BuildDirectVectorMatmulFunctionUnrolled(
        result, input, matrix, bias, has_bias, input_pointer_id,
        input_pointer_type_id, input_memory_operands, input_constant_id,
        input_is_value, matrix_pointer_id, matrix_pointer_type_id,
        matrix_shape_id, matrix_offset_id, matrix_memory_operands,
        matrix_constant_id, matrix_is_value, bias_pointer_id,
        bias_pointer_type_id, bias_memory_operands,
        bias_source_component_type_id, bias_offset,
        bias_conversion_fp_fast_math_mode,
        bias_conversion_has_explicit_fp_fast_math_mode, bias_constant_id,
        bias_is_value, value_arguments);
  }
  const bool same_component =
      result.component_type_id == input.component_type_id &&
      result.component_type_id == matrix.component_type_id &&
      (!has_bias ||
       (bias && result.component_type_id == bias->component_type_id)) &&
      (IsFloat16Type(result.component_type_id) ||
       IsFloat32Type(result.component_type_id));
  const bool mixed_f16_f32 =
      IsFloat32Type(result.component_type_id) &&
      IsFloat16Type(input.component_type_id) &&
      IsFloat16Type(matrix.component_type_id) &&
      (!has_bias || (bias && IsFloat32Type(bias->component_type_id)));
  if ((!same_component && !mixed_f16_f32) || input.length != matrix.rows ||
      result.length != matrix.cols ||
      (has_bias && (!bias || bias->length != result.length))) {
    return 0;
  }
  if (has_bias && !bias_is_value &&
      bias_source_component_type_id != bias->component_type_id &&
      !(IsFloat16Type(bias_source_component_type_id) &&
        IsFloat32Type(bias->component_type_id))) {
    return 0;
  }
  if ((!input_is_value &&
       (input_pointer_id == 0 || input_pointer_type_id == 0)) ||
      (!matrix_is_value &&
       (matrix_pointer_id == 0 || matrix_pointer_type_id == 0 ||
        matrix_shape_id == 0 || matrix_offset_id == 0)) ||
      (has_bias && !bias_is_value &&
       (bias_pointer_id == 0 || bias_pointer_type_id == 0))) {
    return 0;
  }

  const uint32_t full_output_packs = result.length / kPackedVec2Width;
  const uint32_t output_tail_lanes = result.length % kPackedVec2Width;
  const uint32_t full_input_packs = input.length / kPackedVec2Width;
  const uint32_t input_tail_lanes = input.length % kPackedVec2Width;
  const bool has_full_output = full_output_packs != 0;
  const bool has_output_tail = output_tail_lanes != 0;
  const bool has_full_input = full_input_packs != 0;
  const bool has_input_tail = input_tail_lanes != 0;

  Instruction* type_insertion_point =
      get_def_use_mgr()->GetDef(result.lowered_type_id);
  if (!type_insertion_point) return 0;
  const uint32_t input_vec2_type_id = GetOrCreateVectorType(
      input.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t result_vec2_type_id = GetOrCreateVectorType(
      result.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t bias_source_vec2_type_id =
      has_bias && !bias_is_value
          ? GetOrCreateVectorType(bias_source_component_type_id,
                                  kPackedVec2Width, &type_insertion_point)
          : 0;
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t full_output_packs_id =
      has_full_output ? GetOrCreateUIntConstant(full_output_packs) : 0;
  const uint32_t full_input_packs_id =
      has_full_input ? GetOrCreateUIntConstant(full_input_packs) : 0;
  const uint32_t output_tail_base_id =
      has_output_tail
          ? GetOrCreateUIntConstant(full_output_packs * kPackedVec2Width)
          : 0;
  const uint32_t input_tail_base_id =
      has_input_tail
          ? GetOrCreateUIntConstant(full_input_packs * kPackedVec2Width)
          : 0;
  const uint32_t matrix_cols_id = GetOrCreateUIntConstant(matrix.cols);
  const uint32_t matrix_packed_cols_id =
      IsPackedVec2(matrix) ? GetOrCreateUIntConstant(matrix.packed_cols) : 0;
  const uint32_t input_zero_id = GetOrCreateZero(input.component_type_id);
  const uint32_t result_zero2_id = GetOrCreateZero(result_vec2_type_id);
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t result_vec2_function_ptr_type_id =
      GetOrCreatePointerType(result_vec2_type_id, spv::StorageClass::Function);
  const uint32_t input_vec2_function_ptr_type_id =
      GetOrCreatePointerType(input_vec2_type_id, spv::StorageClass::Function);
  const uint32_t input_scalar_function_ptr_type_id = GetOrCreatePointerType(
      input.component_type_id, spv::StorageClass::Function);
  const uint32_t result_scalar_function_ptr_type_id = GetOrCreatePointerType(
      result.component_type_id, spv::StorageClass::Function);
  const uint32_t lowered_result_function_ptr_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  if (input_vec2_type_id == 0 || result_vec2_type_id == 0 ||
      uint_type_id == 0 || bool_type_id == 0 || zero_uint_id == 0 ||
      one_uint_id == 0 || packed_width_uint_id == 0 || matrix_cols_id == 0 ||
      input_zero_id == 0 || result_zero2_id == 0 ||
      uint_function_ptr_type_id == 0 || result_vec2_function_ptr_type_id == 0 ||
      input_vec2_function_ptr_type_id == 0 ||
      input_scalar_function_ptr_type_id == 0 ||
      result_scalar_function_ptr_type_id == 0 ||
      lowered_result_function_ptr_type_id == 0 ||
      (has_bias && !bias_is_value && bias_source_vec2_type_id == 0) ||
      (has_full_output && full_output_packs_id == 0) ||
      (has_full_input && full_input_packs_id == 0) ||
      (has_output_tail && output_tail_base_id == 0) ||
      (has_input_tail && input_tail_base_id == 0) ||
      (IsPackedVec2(matrix) && matrix_packed_cols_id == 0)) {
    return 0;
  }

  const uint32_t input_load_function_id =
      !input_is_value && has_full_input
          ? GetOrCreatePackedLoadChunkFunction(
                input_pointer_id, input_pointer_type_id,
                input.component_type_id, input_vec2_type_id,
                input_memory_operands)
          : 0;
  const uint32_t matrix_load_function_id =
      !matrix_is_value && has_full_output
          ? GetOrCreatePackedLoadChunkFunction(
                matrix_pointer_id, matrix_pointer_type_id,
                matrix.component_type_id, input_vec2_type_id,
                matrix_memory_operands)
          : 0;
  const uint32_t bias_load_function_id =
      has_bias && !bias_is_value && has_full_output
          ? GetOrCreatePackedLoadChunkFunction(
                bias_pointer_id, bias_pointer_type_id,
                bias_source_component_type_id, bias_source_vec2_type_id,
                bias_memory_operands)
          : 0;
  if ((!input_is_value && has_full_input && input_load_function_id == 0) ||
      (!matrix_is_value && has_full_output && matrix_load_function_id == 0) ||
      (has_bias && !bias_is_value && has_full_output &&
       bias_load_function_id == 0)) {
    return 0;
  }

  std::vector<uint32_t> parameter_type_ids;
  if (input_is_value && input_constant_id == 0) {
    parameter_type_ids.push_back(input.lowered_type_id);
  }
  if (matrix_is_value && matrix_constant_id == 0) {
    parameter_type_ids.push_back(matrix.lowered_type_id);
  }
  if (has_bias && bias_is_value && bias_constant_id == 0) {
    parameter_type_ids.push_back(bias->lowered_type_id);
  }
  if (parameter_type_ids.size() != value_arguments.size()) return 0;
  for (size_t i = 0; i < parameter_type_ids.size(); ++i) {
    if (parameter_type_ids[i] != value_arguments[i].second) return 0;
  }

  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, parameter_type_ids);
  const uint32_t function_id = TakeNextId();
  if (function_type_id == 0 || function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, result.lowered_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  size_t parameter_index = 0;
  auto add_parameter = [&](uint32_t type_id) -> uint32_t {
    if (parameter_index >= parameter_type_ids.size() ||
        parameter_type_ids[parameter_index] != type_id) {
      return 0;
    }
    const uint32_t parameter_id = TakeNextId();
    if (parameter_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, type_id, parameter_id,
        std::initializer_list<Operand>{}));
    ++parameter_index;
    return parameter_id;
  };
  uint32_t input_value_id = input_constant_id;
  if (input_is_value && input_value_id == 0) {
    input_value_id = add_parameter(input.lowered_type_id);
  }
  uint32_t matrix_value_id = matrix_constant_id;
  if (matrix_is_value && matrix_value_id == 0) {
    matrix_value_id = add_parameter(matrix.lowered_type_id);
  }
  uint32_t bias_value_id = bias_constant_id;
  if (has_bias && bias_is_value && bias_value_id == 0) {
    bias_value_id = add_parameter(bias->lowered_type_id);
  }
  if ((input_is_value && input_value_id == 0) ||
      (matrix_is_value && matrix_value_id == 0) ||
      (has_bias && bias_is_value && bias_value_id == 0) ||
      parameter_index != parameter_type_ids.size()) {
    return 0;
  }

  const uint32_t entry_label_id = TakeNextId();
  const uint32_t return_label_id = TakeNextId();
  uint32_t out_header_label_id = 0;
  uint32_t out_body_label_id = 0;
  uint32_t out_finalize_label_id = 0;
  uint32_t out_continue_label_id = 0;
  uint32_t out_merge_label_id = 0;
  uint32_t out_k_header_label_id = 0;
  uint32_t out_k_body_label_id = 0;
  uint32_t out_k_continue_label_id = 0;
  uint32_t tail_entry_label_id = 0;
  uint32_t tail_finalize_label_id = 0;
  uint32_t tail_k_header_label_id = 0;
  uint32_t tail_k_body_label_id = 0;
  uint32_t tail_k_continue_label_id = 0;
  if (has_full_output) {
    out_header_label_id = TakeNextId();
    out_body_label_id = TakeNextId();
    out_finalize_label_id = TakeNextId();
    out_continue_label_id = TakeNextId();
    out_merge_label_id = TakeNextId();
    if (has_full_input) {
      out_k_header_label_id = TakeNextId();
      out_k_body_label_id = TakeNextId();
      out_k_continue_label_id = TakeNextId();
    }
  }
  if (has_output_tail) {
    tail_entry_label_id = TakeNextId();
    tail_finalize_label_id = TakeNextId();
    if (has_full_input) {
      tail_k_header_label_id = TakeNextId();
      tail_k_body_label_id = TakeNextId();
      tail_k_continue_label_id = TakeNextId();
    }
  }
  if (entry_label_id == 0 || return_label_id == 0 ||
      (has_full_output &&
       (out_header_label_id == 0 || out_body_label_id == 0 ||
        out_finalize_label_id == 0 || out_continue_label_id == 0 ||
        out_merge_label_id == 0)) ||
      (has_full_output && has_full_input &&
       (out_k_header_label_id == 0 || out_k_body_label_id == 0 ||
        out_k_continue_label_id == 0)) ||
      (has_output_tail &&
       (tail_entry_label_id == 0 || tail_finalize_label_id == 0)) ||
      (has_output_tail && has_full_input &&
       (tail_k_header_label_id == 0 || tail_k_body_label_id == 0 ||
        tail_k_continue_label_id == 0))) {
    return 0;
  }

  auto make_block = [this](uint32_t label_id) {
    return std::unique_ptr<BasicBlock>(MakeBasicBlock(label_id));
  };
  std::unique_ptr<BasicBlock> entry_block = make_block(entry_label_id);
  std::unique_ptr<BasicBlock> return_block = make_block(return_label_id);
  std::unique_ptr<BasicBlock> out_header_block;
  std::unique_ptr<BasicBlock> out_body_block;
  std::unique_ptr<BasicBlock> out_finalize_block;
  std::unique_ptr<BasicBlock> out_continue_block;
  std::unique_ptr<BasicBlock> out_merge_block;
  std::unique_ptr<BasicBlock> out_k_header_block;
  std::unique_ptr<BasicBlock> out_k_body_block;
  std::unique_ptr<BasicBlock> out_k_continue_block;
  std::unique_ptr<BasicBlock> tail_entry_block;
  std::unique_ptr<BasicBlock> tail_finalize_block;
  std::unique_ptr<BasicBlock> tail_k_header_block;
  std::unique_ptr<BasicBlock> tail_k_body_block;
  std::unique_ptr<BasicBlock> tail_k_continue_block;
  if (has_full_output) {
    out_header_block = make_block(out_header_label_id);
    out_body_block = make_block(out_body_label_id);
    out_finalize_block = make_block(out_finalize_label_id);
    out_continue_block = make_block(out_continue_label_id);
    out_merge_block = make_block(out_merge_label_id);
    if (has_full_input) {
      out_k_header_block = make_block(out_k_header_label_id);
      out_k_body_block = make_block(out_k_body_label_id);
      out_k_continue_block = make_block(out_k_continue_label_id);
    }
  }
  if (has_output_tail) {
    tail_entry_block = make_block(tail_entry_label_id);
    tail_finalize_block = make_block(tail_finalize_label_id);
    if (has_full_input) {
      tail_k_header_block = make_block(tail_k_header_label_id);
      tail_k_body_block = make_block(tail_k_body_label_id);
      tail_k_continue_block = make_block(tail_k_continue_label_id);
    }
  }
  if (!entry_block || !return_block ||
      (has_full_output &&
       (!out_header_block || !out_body_block || !out_finalize_block ||
        !out_continue_block || !out_merge_block)) ||
      (has_full_output && has_full_input &&
       (!out_k_header_block || !out_k_body_block || !out_k_continue_block)) ||
      (has_output_tail && (!tail_entry_block || !tail_finalize_block)) ||
      (has_output_tail && has_full_input &&
       (!tail_k_header_block || !tail_k_body_block ||
        !tail_k_continue_block))) {
    return 0;
  }

  const uint32_t input_lowered_function_ptr_type_id =
      input_is_value ? GetOrCreatePointerType(input.lowered_type_id,
                                              spv::StorageClass::Function)
                     : 0;
  const uint32_t matrix_lowered_function_ptr_type_id =
      matrix_is_value ? GetOrCreatePointerType(matrix.lowered_type_id,
                                               spv::StorageClass::Function)
                      : 0;
  const uint32_t bias_lowered_function_ptr_type_id =
      has_bias && bias_is_value
          ? GetOrCreatePointerType(bias->lowered_type_id,
                                   spv::StorageClass::Function)
          : 0;
  if ((input_is_value && input_lowered_function_ptr_type_id == 0) ||
      (matrix_is_value && matrix_lowered_function_ptr_type_id == 0) ||
      (has_bias && bias_is_value && bias_lowered_function_ptr_type_id == 0)) {
    return 0;
  }

  InstructionBuilder entry_builder(context(), entry_block.get());
  Instruction* result_var = entry_builder.AddVariable(
      lowered_result_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* out_pack_var =
      has_full_output ? entry_builder.AddVariable(
                            uint_function_ptr_type_id,
                            static_cast<uint32_t>(spv::StorageClass::Function))
                      : nullptr;
  Instruction* k_pack_var =
      has_full_input ? entry_builder.AddVariable(
                           uint_function_ptr_type_id,
                           static_cast<uint32_t>(spv::StorageClass::Function))
                     : nullptr;
  std::array<Instruction*, kPackedVec2Width> accumulator_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    accumulator_vars[lane] = entry_builder.AddVariable(
        result_vec2_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
  }
  Instruction* input_var =
      input_is_value ? entry_builder.AddVariable(
                           input_lowered_function_ptr_type_id,
                           static_cast<uint32_t>(spv::StorageClass::Function))
                     : nullptr;
  Instruction* matrix_var =
      matrix_is_value ? entry_builder.AddVariable(
                            matrix_lowered_function_ptr_type_id,
                            static_cast<uint32_t>(spv::StorageClass::Function))
                      : nullptr;
  Instruction* bias_var =
      has_bias && bias_is_value
          ? entry_builder.AddVariable(
                bias_lowered_function_ptr_type_id,
                static_cast<uint32_t>(spv::StorageClass::Function))
          : nullptr;
  if (!result_var || (has_full_output && !out_pack_var) ||
      (has_full_input && !k_pack_var) || (input_is_value && !input_var) ||
      (matrix_is_value && !matrix_var) ||
      (has_bias && bias_is_value && !bias_var)) {
    return 0;
  }
  for (Instruction* accumulator_var : accumulator_vars) {
    if (!accumulator_var) return 0;
  }

  const uint32_t captured_input_pointer_id =
      input_is_value ? 0
                     : BuildCapturedPointer(&entry_builder, input_pointer_id);
  const uint32_t captured_matrix_pointer_id =
      matrix_is_value ? 0
                      : BuildCapturedPointer(&entry_builder, matrix_pointer_id);
  const uint32_t captured_bias_pointer_id =
      !has_bias || bias_is_value
          ? 0
          : BuildCapturedPointer(&entry_builder, bias_pointer_id);
  if ((!input_is_value && captured_input_pointer_id == 0) ||
      (!matrix_is_value && captured_matrix_pointer_id == 0) ||
      (has_bias && !bias_is_value && captured_bias_pointer_id == 0)) {
    return 0;
  }
  if ((input_is_value &&
       !entry_builder.AddStore(input_var->result_id(), input_value_id)) ||
      (matrix_is_value &&
       !entry_builder.AddStore(matrix_var->result_id(), matrix_value_id)) ||
      (has_bias && bias_is_value &&
       !entry_builder.AddStore(bias_var->result_id(), bias_value_id)) ||
      (has_full_output &&
       !entry_builder.AddStore(out_pack_var->result_id(), zero_uint_id))) {
    return 0;
  }
  const uint32_t first_label_id = has_full_output   ? out_header_label_id
                                  : has_output_tail ? tail_entry_label_id
                                                    : return_label_id;
  if (!entry_builder.AddBranch(first_label_id)) return 0;

  auto add_offset = [this, uint_type_id](InstructionBuilder* builder,
                                         uint32_t base_id,
                                         uint32_t offset) -> uint32_t {
    if (!builder || base_id == 0) return 0;
    if (offset == 0) return base_id;
    const uint32_t offset_id = GetOrCreateUIntConstant(offset);
    Instruction* sum = offset_id
                           ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                  base_id, offset_id)
                           : nullptr;
    return sum ? sum->result_id() : 0;
  };
  auto load_buffer_scalar =
      [this](InstructionBuilder* builder, uint32_t pointer_type_id,
             uint32_t pointer_id, uint32_t component_type_id,
             uint32_t element_index_id,
             const std::vector<Operand>& memory_operands) -> uint32_t {
    if (!builder || pointer_type_id == 0 || pointer_id == 0 ||
        component_type_id == 0 || element_index_id == 0) {
      return 0;
    }
    const uint32_t element_pointer_id =
        BuildElementAccessFromPointerType(builder, pointer_type_id, pointer_id,
                                          component_type_id, element_index_id);
    return element_pointer_id ? AddLoad(builder, component_type_id,
                                        element_pointer_id, memory_operands)
                              : 0;
  };
  auto load_input_scalar = [&](InstructionBuilder* builder,
                               uint32_t index_id) -> uint32_t {
    if (input_is_value) {
      Instruction* pointer =
          builder->AddAccessChain(input_scalar_function_ptr_type_id,
                                  input_var->result_id(), {index_id});
      Instruction* value = pointer ? builder->AddLoad(input.component_type_id,
                                                      pointer->result_id())
                                   : nullptr;
      return value ? value->result_id() : 0;
    }
    return load_buffer_scalar(
        builder, input_pointer_type_id, captured_input_pointer_id,
        input.component_type_id, index_id, input_memory_operands);
  };
  auto load_matrix_scalar = [&](InstructionBuilder* builder, uint32_t row_id,
                                uint32_t col_id) -> uint32_t {
    Instruction* row_offset = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_id, matrix_cols_id);
    Instruction* flat_index =
        row_offset ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                          row_offset->result_id(), col_id)
                   : nullptr;
    if (!flat_index) return 0;
    if (matrix_is_value) {
      Instruction* pointer = builder->AddAccessChain(
          input_scalar_function_ptr_type_id, matrix_var->result_id(),
          {flat_index->result_id()});
      Instruction* value = pointer ? builder->AddLoad(matrix.component_type_id,
                                                      pointer->result_id())
                                   : nullptr;
      return value ? value->result_id() : 0;
    }
    const uint32_t memory_index_id = BuildRowMajorMatrixMemoryIndex(
        builder, nullptr, matrix_shape_id, matrix_offset_id, matrix.cols,
        flat_index->result_id());
    return memory_index_id
               ? load_buffer_scalar(builder, matrix_pointer_type_id,
                                    captured_matrix_pointer_id,
                                    matrix.component_type_id, memory_index_id,
                                    matrix_memory_operands)
               : 0;
  };
  auto load_bias_scalar = [&](InstructionBuilder* builder,
                              uint32_t index_id) -> uint32_t {
    if (!has_bias || !bias) return 0;
    if (bias_is_value) {
      Instruction* pointer =
          builder->AddAccessChain(result_scalar_function_ptr_type_id,
                                  bias_var->result_id(), {index_id});
      Instruction* value = pointer ? builder->AddLoad(bias->component_type_id,
                                                      pointer->result_id())
                                   : nullptr;
      return value ? value->result_id() : 0;
    }
    const uint32_t source_id = load_buffer_scalar(
        builder, bias_pointer_type_id, captured_bias_pointer_id,
        bias_source_component_type_id,
        add_offset(builder, index_id, bias_offset), bias_memory_operands);
    if (source_id == 0 ||
        bias_source_component_type_id == bias->component_type_id) {
      return source_id;
    }
    Instruction* converted = builder->AddUnaryOp(
        bias->component_type_id, spv::Op::OpFConvert, source_id);
    ApplyFPFastMathMode(converted, bias_conversion_fp_fast_math_mode,
                        bias_conversion_has_explicit_fp_fast_math_mode);
    return converted ? converted->result_id() : 0;
  };

  auto emit_accumulate =
      [&](InstructionBuilder* builder, uint32_t input_base_id,
          uint32_t input_pack_index_id, uint32_t output_base_id,
          uint32_t output_pack_index_id, uint32_t valid_input_lanes,
          uint32_t valid_output_lanes) -> bool {
    if (!builder || input_base_id == 0 || output_base_id == 0 ||
        valid_input_lanes == 0 || valid_input_lanes > kPackedVec2Width ||
        valid_output_lanes == 0 || valid_output_lanes > kPackedVec2Width) {
      return false;
    }

    uint32_t input_vec_id = 0;
    if (valid_input_lanes == kPackedVec2Width) {
      if (input_is_value && IsPackedVec2(input)) {
        Instruction* pointer = builder->AddAccessChain(
            input_vec2_function_ptr_type_id, input_var->result_id(),
            {input_pack_index_id});
        Instruction* value =
            pointer ? builder->AddLoad(input_vec2_type_id, pointer->result_id())
                    : nullptr;
        input_vec_id = value ? value->result_id() : 0;
      } else if (!input_is_value) {
        Instruction* value = builder->AddFunctionCall(
            input_vec2_type_id, input_load_function_id, {input_base_id});
        input_vec_id = value ? value->result_id() : 0;
      }
    }
    if (input_vec_id == 0) {
      std::vector<uint32_t> input_lane_ids;
      input_lane_ids.reserve(kPackedVec2Width);
      for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
        uint32_t value_id = input_zero_id;
        if (lane < valid_input_lanes) {
          const uint32_t index_id = add_offset(builder, input_base_id, lane);
          value_id = index_id ? load_input_scalar(builder, index_id) : 0;
          if (value_id == 0) return false;
        }
        input_lane_ids.push_back(value_id);
      }
      Instruction* input_vec =
          builder->AddCompositeConstruct(input_vec2_type_id, input_lane_ids);
      input_vec_id = input_vec ? input_vec->result_id() : 0;
    }
    uint32_t compute_input_id = input_vec_id;
    if (input.component_type_id != result.component_type_id) {
      Instruction* widened_input =
          input_vec_id ? builder->AddUnaryOp(result_vec2_type_id,
                                             spv::Op::OpFConvert, input_vec_id)
                       : nullptr;
      if (!widened_input) return false;
      ApplyActiveFPFastMathMode(widened_input);
      compute_input_id = widened_input->result_id();
    }
    if (compute_input_id == 0) return false;

    std::array<uint32_t, kPackedVec2Width> matrix_row_vec_ids = {};
    if (valid_output_lanes == kPackedVec2Width) {
      for (uint32_t input_lane = 0; input_lane < valid_input_lanes;
           ++input_lane) {
        const uint32_t row_id = add_offset(builder, input_base_id, input_lane);
        if (row_id == 0) return false;
        if (matrix_is_value && IsPackedVec2(matrix)) {
          Instruction* row_offset = builder->AddBinaryOp(
              uint_type_id, spv::Op::OpIMul, row_id, matrix_packed_cols_id);
          Instruction* packed_index =
              row_offset ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                row_offset->result_id(),
                                                output_pack_index_id)
                         : nullptr;
          Instruction* pointer =
              packed_index
                  ? builder->AddAccessChain(input_vec2_function_ptr_type_id,
                                            matrix_var->result_id(),
                                            {packed_index->result_id()})
                  : nullptr;
          Instruction* value = pointer ? builder->AddLoad(input_vec2_type_id,
                                                          pointer->result_id())
                                       : nullptr;
          matrix_row_vec_ids[input_lane] = value ? value->result_id() : 0;
        } else if (!matrix_is_value) {
          Instruction* row_offset = builder->AddBinaryOp(
              uint_type_id, spv::Op::OpIMul, row_id, matrix_cols_id);
          Instruction* local_index =
              row_offset ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                row_offset->result_id(),
                                                output_base_id)
                         : nullptr;
          const uint32_t memory_index_id =
              local_index
                  ? BuildRowMajorMatrixMemoryIndex(
                        builder, nullptr, matrix_shape_id, matrix_offset_id,
                        matrix.cols, local_index->result_id())
                  : 0;
          Instruction* value =
              memory_index_id ? builder->AddFunctionCall(
                                    input_vec2_type_id, matrix_load_function_id,
                                    {memory_index_id})
                              : nullptr;
          matrix_row_vec_ids[input_lane] = value ? value->result_id() : 0;
        }
        if (matrix_row_vec_ids[input_lane] == 0 &&
            (IsPackedVec2(matrix) || !matrix_is_value)) {
          return false;
        }
      }
    }

    for (uint32_t output_lane = 0; output_lane < valid_output_lanes;
         ++output_lane) {
      std::vector<uint32_t> weight_lane_ids;
      weight_lane_ids.reserve(kPackedVec2Width);
      for (uint32_t input_lane = 0; input_lane < kPackedVec2Width;
           ++input_lane) {
        uint32_t weight_id = input_zero_id;
        if (input_lane < valid_input_lanes) {
          if (matrix_row_vec_ids[input_lane] != 0) {
            weight_id = ExtractCompositeElement(
                builder, matrix.component_type_id,
                matrix_row_vec_ids[input_lane], output_lane);
          } else {
            const uint32_t row_id =
                add_offset(builder, input_base_id, input_lane);
            const uint32_t col_id =
                add_offset(builder, output_base_id, output_lane);
            weight_id = row_id && col_id
                            ? load_matrix_scalar(builder, row_id, col_id)
                            : 0;
          }
          if (weight_id == 0) return false;
        }
        weight_lane_ids.push_back(weight_id);
      }
      Instruction* weight =
          builder->AddCompositeConstruct(input_vec2_type_id, weight_lane_ids);
      uint32_t compute_weight_id = weight ? weight->result_id() : 0;
      Instruction* widened_weight = nullptr;
      if (weight && matrix.component_type_id != result.component_type_id) {
        widened_weight = builder->AddUnaryOp(
            result_vec2_type_id, spv::Op::OpFConvert, weight->result_id());
        compute_weight_id = widened_weight ? widened_weight->result_id() : 0;
      }
      Instruction* accumulator = builder->AddLoad(
          result_vec2_type_id, accumulator_vars[output_lane]->result_id());
      if (compute_weight_id == 0 || !accumulator) return false;
      ApplyActiveFPFastMathMode(widened_weight);
      const uint32_t accumulated =
          BuildFma(builder, result_vec2_type_id, compute_input_id,
                   compute_weight_id, accumulator->result_id());
      if (accumulated == 0 ||
          !builder->AddStore(accumulator_vars[output_lane]->result_id(),
                             accumulated)) {
        return false;
      }
    }
    return true;
  };

  auto emit_full_output = [&](InstructionBuilder* builder,
                              uint32_t output_pack_id,
                              uint32_t output_base_id) -> bool {
    std::vector<uint32_t> lane_ids;
    lane_ids.reserve(kPackedVec2Width);
    for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
      Instruction* accumulator = builder->AddLoad(
          result_vec2_type_id, accumulator_vars[lane]->result_id());
      const uint32_t reduced =
          accumulator ? BuildHorizontalReduce(builder, result.component_type_id,
                                              accumulator->result_id())
                      : 0;
      if (reduced == 0) return false;
      lane_ids.push_back(reduced);
    }
    Instruction* result_vec =
        builder->AddCompositeConstruct(result_vec2_type_id, lane_ids);
    if (!result_vec) return false;

    if (has_bias) {
      uint32_t bias_vec_id = 0;
      if (bias_is_value && IsPackedVec2(*bias)) {
        Instruction* pointer =
            builder->AddAccessChain(result_vec2_function_ptr_type_id,
                                    bias_var->result_id(), {output_pack_id});
        Instruction* value = pointer ? builder->AddLoad(result_vec2_type_id,
                                                        pointer->result_id())
                                     : nullptr;
        bias_vec_id = value ? value->result_id() : 0;
      } else if (!bias_is_value) {
        const uint32_t memory_base_id =
            add_offset(builder, output_base_id, bias_offset);
        Instruction* value =
            memory_base_id ? builder->AddFunctionCall(bias_source_vec2_type_id,
                                                      bias_load_function_id,
                                                      {memory_base_id})
                           : nullptr;
        if (value && bias_source_component_type_id != bias->component_type_id) {
          Instruction* converted = builder->AddUnaryOp(
              result_vec2_type_id, spv::Op::OpFConvert, value->result_id());
          ApplyFPFastMathMode(converted, bias_conversion_fp_fast_math_mode,
                              bias_conversion_has_explicit_fp_fast_math_mode);
          bias_vec_id = converted ? converted->result_id() : 0;
        } else {
          bias_vec_id = value ? value->result_id() : 0;
        }
      } else {
        std::vector<uint32_t> bias_lane_ids;
        bias_lane_ids.reserve(kPackedVec2Width);
        for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
          const uint32_t index_id = add_offset(builder, output_base_id, lane);
          const uint32_t value_id =
              index_id ? load_bias_scalar(builder, index_id) : 0;
          if (value_id == 0) return false;
          bias_lane_ids.push_back(value_id);
        }
        Instruction* value =
            builder->AddCompositeConstruct(result_vec2_type_id, bias_lane_ids);
        bias_vec_id = value ? value->result_id() : 0;
      }
      Instruction* sum = bias_vec_id ? builder->AddBinaryOp(
                                           result_vec2_type_id, spv::Op::OpFAdd,
                                           result_vec->result_id(), bias_vec_id)
                                     : nullptr;
      if (!sum) return false;
      ApplyActiveFPFastMathMode(sum);
      result_vec = sum;
    }

    if (IsPackedVec2(result)) {
      Instruction* pointer =
          builder->AddAccessChain(result_vec2_function_ptr_type_id,
                                  result_var->result_id(), {output_pack_id});
      return pointer &&
             builder->AddStore(pointer->result_id(), result_vec->result_id());
    }
    for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
      const uint32_t index_id = add_offset(builder, output_base_id, lane);
      const uint32_t value_id = ExtractCompositeElement(
          builder, result.component_type_id, result_vec->result_id(), lane);
      Instruction* pointer = index_id ? builder->AddAccessChain(
                                            result_scalar_function_ptr_type_id,
                                            result_var->result_id(), {index_id})
                                      : nullptr;
      if (value_id == 0 || !pointer ||
          !builder->AddStore(pointer->result_id(), value_id)) {
        return false;
      }
    }
    return true;
  };

  auto emit_output_tail = [&](InstructionBuilder* builder) -> bool {
    for (uint32_t lane = 0; lane < output_tail_lanes; ++lane) {
      Instruction* accumulator = builder->AddLoad(
          result_vec2_type_id, accumulator_vars[lane]->result_id());
      uint32_t scalar_id =
          accumulator ? BuildHorizontalReduce(builder, result.component_type_id,
                                              accumulator->result_id())
                      : 0;
      const uint32_t index_id = add_offset(builder, output_tail_base_id, lane);
      if (scalar_id == 0 || index_id == 0) return false;
      if (has_bias) {
        const uint32_t bias_id = load_bias_scalar(builder, index_id);
        Instruction* sum =
            bias_id ? builder->AddBinaryOp(result.component_type_id,
                                           spv::Op::OpFAdd, scalar_id, bias_id)
                    : nullptr;
        if (!sum) return false;
        ApplyActiveFPFastMathMode(sum);
        scalar_id = sum->result_id();
      }
      Instruction* pointer =
          builder->AddAccessChain(result_scalar_function_ptr_type_id,
                                  result_var->result_id(), {index_id});
      if (!pointer || !builder->AddStore(pointer->result_id(), scalar_id)) {
        return false;
      }
    }
    return true;
  };

  auto reset_accumulators = [&](InstructionBuilder* builder,
                                uint32_t lane_count) -> bool {
    for (uint32_t lane = 0; lane < lane_count; ++lane) {
      if (!builder->AddStore(accumulator_vars[lane]->result_id(),
                             result_zero2_id)) {
        return false;
      }
    }
    return true;
  };

  if (has_full_output) {
    InstructionBuilder out_header_builder(context(), out_header_block.get());
    Instruction* out_pack =
        out_header_builder.AddLoad(uint_type_id, out_pack_var->result_id());
    Instruction* out_condition =
        out_pack ? out_header_builder.AddBinaryOp(
                       bool_type_id, spv::Op::OpULessThan,
                       out_pack->result_id(), full_output_packs_id)
                 : nullptr;
    if (!out_condition ||
        !out_header_builder.AddLoopMerge(out_merge_label_id,
                                         out_continue_label_id) ||
        !out_header_builder.AddConditionalBranch(out_condition->result_id(),
                                                 out_body_label_id,
                                                 out_merge_label_id)) {
      return 0;
    }

    InstructionBuilder out_body_builder(context(), out_body_block.get());
    if (!reset_accumulators(&out_body_builder, kPackedVec2Width)) return 0;
    if (has_full_input) {
      if (!out_body_builder.AddStore(k_pack_var->result_id(), zero_uint_id) ||
          !out_body_builder.AddBranch(out_k_header_label_id)) {
        return 0;
      }
    } else if (!out_body_builder.AddBranch(out_finalize_label_id)) {
      return 0;
    }

    if (has_full_input) {
      InstructionBuilder k_header_builder(context(), out_k_header_block.get());
      Instruction* k_pack =
          k_header_builder.AddLoad(uint_type_id, k_pack_var->result_id());
      Instruction* k_condition =
          k_pack ? k_header_builder.AddBinaryOp(
                       bool_type_id, spv::Op::OpULessThan, k_pack->result_id(),
                       full_input_packs_id)
                 : nullptr;
      if (!k_condition ||
          !k_header_builder.AddLoopMerge(out_finalize_label_id,
                                         out_k_continue_label_id) ||
          !k_header_builder.AddConditionalBranch(k_condition->result_id(),
                                                 out_k_body_label_id,
                                                 out_finalize_label_id)) {
        return 0;
      }

      InstructionBuilder k_body_builder(context(), out_k_body_block.get());
      Instruction* input_base =
          k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                     k_pack->result_id(), packed_width_uint_id);
      Instruction* output_base = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, out_pack->result_id(),
          packed_width_uint_id);
      if (!input_base || !output_base ||
          !emit_accumulate(&k_body_builder, input_base->result_id(),
                           k_pack->result_id(), output_base->result_id(),
                           out_pack->result_id(), kPackedVec2Width,
                           kPackedVec2Width) ||
          !k_body_builder.AddBranch(out_k_continue_label_id)) {
        return 0;
      }

      InstructionBuilder k_continue_builder(context(),
                                            out_k_continue_block.get());
      Instruction* next_k = k_continue_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, k_pack->result_id(), one_uint_id);
      if (!next_k ||
          !k_continue_builder.AddStore(k_pack_var->result_id(),
                                       next_k->result_id()) ||
          !k_continue_builder.AddBranch(out_k_header_label_id)) {
        return 0;
      }
    }

    InstructionBuilder out_finalize_builder(context(),
                                            out_finalize_block.get());
    Instruction* output_base = out_finalize_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, out_pack->result_id(),
        packed_width_uint_id);
    if (!output_base) return 0;
    if (has_input_tail) {
      const uint32_t tail_pack_index_id =
          has_full_input ? full_input_packs_id : zero_uint_id;
      if (!emit_accumulate(&out_finalize_builder, input_tail_base_id,
                           tail_pack_index_id, output_base->result_id(),
                           out_pack->result_id(), input_tail_lanes,
                           kPackedVec2Width)) {
        return 0;
      }
    }
    if (!emit_full_output(&out_finalize_builder, out_pack->result_id(),
                          output_base->result_id()) ||
        !out_finalize_builder.AddBranch(out_continue_label_id)) {
      return 0;
    }

    InstructionBuilder out_continue_builder(context(),
                                            out_continue_block.get());
    Instruction* next_out = out_continue_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, out_pack->result_id(), one_uint_id);
    if (!next_out ||
        !out_continue_builder.AddStore(out_pack_var->result_id(),
                                       next_out->result_id()) ||
        !out_continue_builder.AddBranch(out_header_label_id)) {
      return 0;
    }

    InstructionBuilder out_merge_builder(context(), out_merge_block.get());
    if (!out_merge_builder.AddBranch(has_output_tail ? tail_entry_label_id
                                                     : return_label_id)) {
      return 0;
    }
  }

  if (has_output_tail) {
    InstructionBuilder tail_entry_builder(context(), tail_entry_block.get());
    if (!reset_accumulators(&tail_entry_builder, output_tail_lanes)) return 0;
    if (has_full_input) {
      if (!tail_entry_builder.AddStore(k_pack_var->result_id(), zero_uint_id) ||
          !tail_entry_builder.AddBranch(tail_k_header_label_id)) {
        return 0;
      }
    } else if (!tail_entry_builder.AddBranch(tail_finalize_label_id)) {
      return 0;
    }

    if (has_full_input) {
      InstructionBuilder k_header_builder(context(), tail_k_header_block.get());
      Instruction* k_pack =
          k_header_builder.AddLoad(uint_type_id, k_pack_var->result_id());
      Instruction* k_condition =
          k_pack ? k_header_builder.AddBinaryOp(
                       bool_type_id, spv::Op::OpULessThan, k_pack->result_id(),
                       full_input_packs_id)
                 : nullptr;
      if (!k_condition ||
          !k_header_builder.AddLoopMerge(tail_finalize_label_id,
                                         tail_k_continue_label_id) ||
          !k_header_builder.AddConditionalBranch(k_condition->result_id(),
                                                 tail_k_body_label_id,
                                                 tail_finalize_label_id)) {
        return 0;
      }

      InstructionBuilder k_body_builder(context(), tail_k_body_block.get());
      Instruction* input_base =
          k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                     k_pack->result_id(), packed_width_uint_id);
      const uint32_t output_pack_index_id =
          has_full_output ? full_output_packs_id : zero_uint_id;
      if (!input_base ||
          !emit_accumulate(&k_body_builder, input_base->result_id(),
                           k_pack->result_id(), output_tail_base_id,
                           output_pack_index_id, kPackedVec2Width,
                           output_tail_lanes) ||
          !k_body_builder.AddBranch(tail_k_continue_label_id)) {
        return 0;
      }

      InstructionBuilder k_continue_builder(context(),
                                            tail_k_continue_block.get());
      Instruction* next_k = k_continue_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, k_pack->result_id(), one_uint_id);
      if (!next_k ||
          !k_continue_builder.AddStore(k_pack_var->result_id(),
                                       next_k->result_id()) ||
          !k_continue_builder.AddBranch(tail_k_header_label_id)) {
        return 0;
      }
    }

    InstructionBuilder tail_finalize_builder(context(),
                                             tail_finalize_block.get());
    if (has_input_tail) {
      const uint32_t input_pack_index_id =
          has_full_input ? full_input_packs_id : zero_uint_id;
      const uint32_t output_pack_index_id =
          has_full_output ? full_output_packs_id : zero_uint_id;
      if (!emit_accumulate(&tail_finalize_builder, input_tail_base_id,
                           input_pack_index_id, output_tail_base_id,
                           output_pack_index_id, input_tail_lanes,
                           output_tail_lanes)) {
        return 0;
      }
    }
    if (!emit_output_tail(&tail_finalize_builder) ||
        !tail_finalize_builder.AddBranch(return_label_id)) {
      return 0;
    }
  }

  InstructionBuilder return_builder(context(), return_block.get());
  Instruction* result_value =
      return_builder.AddLoad(result.lowered_type_id, result_var->result_id());
  if (!result_value || !return_builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                                  result_value->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(entry_block));
  if (has_full_output) {
    function->AddBasicBlock(std::move(out_header_block));
    function->AddBasicBlock(std::move(out_body_block));
    if (has_full_input) {
      function->AddBasicBlock(std::move(out_k_header_block));
      function->AddBasicBlock(std::move(out_k_body_block));
      function->AddBasicBlock(std::move(out_k_continue_block));
    }
    function->AddBasicBlock(std::move(out_finalize_block));
    function->AddBasicBlock(std::move(out_continue_block));
    function->AddBasicBlock(std::move(out_merge_block));
  }
  if (has_output_tail) {
    function->AddBasicBlock(std::move(tail_entry_block));
    if (has_full_input) {
      function->AddBasicBlock(std::move(tail_k_header_block));
      function->AddBasicBlock(std::move(tail_k_body_block));
      function->AddBasicBlock(std::move(tail_k_continue_block));
    }
    function->AddBasicBlock(std::move(tail_finalize_block));
  }
  function->AddBasicBlock(std::move(return_block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildDirectMatrixMatmulFunctionUnrolled(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c, uint32_t a_pointer_id,
    uint32_t a_pointer_type_id, uint32_t a_shape_id, uint32_t a_offset_id,
    const std::vector<Operand>& a_memory_operands, uint32_t a_constant_id,
    bool a_is_value, uint32_t b_pointer_id, uint32_t b_pointer_type_id,
    uint32_t b_shape_id, uint32_t b_offset_id,
    const std::vector<Operand>& b_memory_operands, uint32_t b_constant_id,
    bool b_is_value, uint32_t c_pointer_id, uint32_t c_pointer_type_id,
    uint32_t c_shape_id, uint32_t c_offset_id,
    const std::vector<Operand>& c_memory_operands, uint32_t c_constant_id,
    bool c_is_value,
    const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments) {
  if (!CanUseDirectMatrixMulAdd(result, a, b, c) || result.rows != a.rows ||
      result.cols != b.cols || a.cols != b.rows || c.rows != result.rows ||
      c.cols != result.cols) {
    return 0;
  }
  if ((!a_is_value && (a_pointer_id == 0 || a_pointer_type_id == 0 ||
                       a_shape_id == 0 || a_offset_id == 0)) ||
      (!b_is_value && (b_pointer_id == 0 || b_pointer_type_id == 0 ||
                       b_shape_id == 0 || b_offset_id == 0)) ||
      (!c_is_value && (c_pointer_id == 0 || c_pointer_type_id == 0 ||
                       c_shape_id == 0 || c_offset_id == 0))) {
    return 0;
  }

  std::vector<uint32_t> parameter_type_ids;
  if (a_is_value && a_constant_id == 0) {
    parameter_type_ids.push_back(a.lowered_type_id);
  }
  if (b_is_value && b_constant_id == 0) {
    parameter_type_ids.push_back(b.lowered_type_id);
  }
  if (c_is_value && c_constant_id == 0) {
    parameter_type_ids.push_back(c.lowered_type_id);
  }
  if (parameter_type_ids.size() != value_arguments.size()) return 0;
  for (size_t i = 0; i < parameter_type_ids.size(); ++i) {
    if (parameter_type_ids[i] != value_arguments[i].second) return 0;
  }

  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, parameter_type_ids);
  const uint32_t function_id = TakeNextId();
  if (function_type_id == 0 || function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, result.lowered_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  size_t parameter_index = 0;
  auto add_parameter = [&](uint32_t type_id) -> uint32_t {
    if (parameter_index >= parameter_type_ids.size() ||
        parameter_type_ids[parameter_index] != type_id) {
      return 0;
    }
    const uint32_t parameter_id = TakeNextId();
    if (parameter_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, type_id, parameter_id,
        std::initializer_list<Operand>{}));
    ++parameter_index;
    return parameter_id;
  };

  uint32_t a_value_id = a_constant_id;
  if (a_is_value && a_value_id == 0) {
    a_value_id = add_parameter(a.lowered_type_id);
  }
  uint32_t b_value_id = b_constant_id;
  if (b_is_value && b_value_id == 0) {
    b_value_id = add_parameter(b.lowered_type_id);
  }
  uint32_t c_value_id = c_constant_id;
  if (c_is_value && c_value_id == 0) {
    c_value_id = add_parameter(c.lowered_type_id);
  }
  if ((a_is_value && a_value_id == 0) || (b_is_value && b_value_id == 0) ||
      (c_is_value && c_value_id == 0) ||
      parameter_index != parameter_type_ids.size()) {
    return 0;
  }

  const uint32_t label_id = TakeNextId();
  if (label_id == 0) return 0;
  std::unique_ptr<BasicBlock> block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(label_id));
  if (!block) return 0;
  InstructionBuilder builder(context(), block.get());

  const uint32_t captured_a_pointer_id =
      a_is_value ? 0 : BuildCapturedPointer(&builder, a_pointer_id);
  const uint32_t captured_b_pointer_id =
      b_is_value ? 0 : BuildCapturedPointer(&builder, b_pointer_id);
  const uint32_t captured_c_pointer_id =
      c_is_value ? 0 : BuildCapturedPointer(&builder, c_pointer_id);
  if ((!a_is_value && captured_a_pointer_id == 0) ||
      (!b_is_value && captured_b_pointer_id == 0) ||
      (!c_is_value && captured_c_pointer_id == 0)) {
    return 0;
  }

  auto load_buffer_scalar_id =
      [&](uint32_t pointer_type_id, uint32_t pointer_id,
          uint32_t component_type_id, uint32_t index_id,
          const std::vector<Operand>& memory_operands) -> uint32_t {
    const uint32_t element_pointer_id =
        index_id ? BuildElementAccessFromPointerType(
                       &builder, pointer_type_id, pointer_id, component_type_id,
                       index_id)
                 : 0;
    return element_pointer_id ? AddLoad(&builder, component_type_id,
                                        element_pointer_id, memory_operands)
                              : 0;
  };
  auto load_matrix_scalar =
      [&](const MatrixTypeInfo& info, bool is_value, uint32_t value_id,
          uint32_t pointer_type_id, uint32_t pointer_id, uint32_t shape_id,
          uint32_t offset_id, const std::vector<Operand>& memory_operands,
          uint32_t row, uint32_t col) -> uint32_t {
    if (is_value) {
      return ExtractMatrixScalar(&builder, info, value_id, row, col);
    }
    const uint64_t flat_index = static_cast<uint64_t>(row) * info.cols + col;
    if (flat_index > std::numeric_limits<uint32_t>::max()) return 0;
    const uint32_t flat_index_id =
        GetOrCreateUIntConstant(static_cast<uint32_t>(flat_index));
    const uint32_t memory_index_id =
        flat_index_id ? BuildRowMajorMatrixMemoryIndex(&builder, nullptr,
                                                       shape_id, offset_id,
                                                       info.cols, flat_index_id)
                      : 0;
    return memory_index_id
               ? load_buffer_scalar_id(pointer_type_id, pointer_id,
                                       info.component_type_id, memory_index_id,
                                       memory_operands)
               : 0;
  };
  Instruction* type_insertion_point =
      get_def_use_mgr()->GetDef(result.lowered_type_id);
  if (!type_insertion_point) return 0;
  const uint32_t operand_vec2_type_id = GetOrCreateVectorType(
      a.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t result_vec2_type_id = GetOrCreateVectorType(
      result.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t operand_zero_id = GetOrCreateZero(a.component_type_id);
  const uint32_t result_zero2_id = GetOrCreateZero(result_vec2_type_id);
  if (operand_vec2_type_id == 0 || result_vec2_type_id == 0 ||
      operand_zero_id == 0 || result_zero2_id == 0) {
    return 0;
  }
  auto widen_vec = [&](uint32_t value_id) -> uint32_t {
    if (value_id == 0 || a.component_type_id == result.component_type_id) {
      return value_id;
    }
    Instruction* converted =
        builder.AddUnaryOp(result_vec2_type_id, spv::Op::OpFConvert, value_id);
    ApplyActiveFPFastMathMode(converted);
    return converted ? converted->result_id() : 0;
  };

  std::vector<uint32_t> a_scalar_ids(a.rows * a.cols, 0);
  for (uint32_t row = 0; row < a.rows; ++row) {
    for (uint32_t col = 0; col < a.cols; ++col) {
      const uint32_t index = MatrixFlatIndex(a, row, col);
      a_scalar_ids[index] = load_matrix_scalar(
          a, a_is_value, a_value_id, a_pointer_type_id, captured_a_pointer_id,
          a_shape_id, a_offset_id, a_memory_operands, row, col);
      if (a_scalar_ids[index] == 0) return 0;
    }
  }
  std::vector<uint32_t> b_scalar_ids(b.rows * b.cols, 0);
  for (uint32_t row = 0; row < b.rows; ++row) {
    for (uint32_t col = 0; col < b.cols; ++col) {
      const uint32_t index = MatrixFlatIndex(b, row, col);
      b_scalar_ids[index] = load_matrix_scalar(
          b, b_is_value, b_value_id, b_pointer_type_id, captured_b_pointer_id,
          b_shape_id, b_offset_id, b_memory_operands, row, col);
      if (b_scalar_ids[index] == 0) return 0;
    }
  }
  std::vector<uint32_t> c_scalar_ids(c.rows * c.cols, 0);
  for (uint32_t row = 0; row < c.rows; ++row) {
    for (uint32_t col = 0; col < c.cols; ++col) {
      const uint32_t index = MatrixFlatIndex(c, row, col);
      c_scalar_ids[index] = load_matrix_scalar(
          c, c_is_value, c_value_id, c_pointer_type_id, captured_c_pointer_id,
          c_shape_id, c_offset_id, c_memory_operands, row, col);
      if (c_scalar_ids[index] == 0) return 0;
    }
  }

  std::vector<uint32_t> scalar_ids(result.rows * result.cols, 0);
  for (uint32_t row = 0; row < result.rows; ++row) {
    for (uint32_t col = 0; col < result.cols; ++col) {
      uint32_t accumulator_vec_id = result_zero2_id;
      for (uint32_t k = 0; k < a.cols; k += kPackedVec2Width) {
        std::vector<uint32_t> lhs_lane_ids;
        std::vector<uint32_t> rhs_lane_ids;
        lhs_lane_ids.reserve(kPackedVec2Width);
        rhs_lane_ids.reserve(kPackedVec2Width);
        for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
          const uint32_t k_index = k + lane;
          const uint32_t lhs_id =
              k_index < a.cols ? a_scalar_ids[MatrixFlatIndex(a, row, k_index)]
                               : operand_zero_id;
          const uint32_t rhs_id =
              k_index < a.cols ? b_scalar_ids[MatrixFlatIndex(b, k_index, col)]
                               : operand_zero_id;
          if (lhs_id == 0 || rhs_id == 0) return 0;
          lhs_lane_ids.push_back(lhs_id);
          rhs_lane_ids.push_back(rhs_id);
        }
        Instruction* lhs_vec =
            builder.AddCompositeConstruct(operand_vec2_type_id, lhs_lane_ids);
        Instruction* rhs_vec =
            builder.AddCompositeConstruct(operand_vec2_type_id, rhs_lane_ids);
        const uint32_t compute_lhs_id =
            lhs_vec ? widen_vec(lhs_vec->result_id()) : 0;
        const uint32_t compute_rhs_id =
            rhs_vec ? widen_vec(rhs_vec->result_id()) : 0;
        if (compute_lhs_id == 0 || compute_rhs_id == 0) return 0;
        accumulator_vec_id =
            BuildFma(&builder, result_vec2_type_id, compute_lhs_id,
                     compute_rhs_id, accumulator_vec_id);
        if (accumulator_vec_id == 0) return 0;
      }
      const uint32_t accumulator_id = BuildHorizontalReduce(
          &builder, result.component_type_id, accumulator_vec_id);
      if (accumulator_id == 0) return 0;
      const uint32_t c_id = c_scalar_ids[MatrixFlatIndex(c, row, col)];
      Instruction* sum =
          c_id ? builder.AddBinaryOp(result.component_type_id, spv::Op::OpFAdd,
                                     accumulator_id, c_id)
               : nullptr;
      if (!sum) return 0;
      ApplyActiveFPFastMathMode(sum);
      scalar_ids[MatrixFlatIndex(result, row, col)] = sum->result_id();
    }
  }

  std::vector<uint32_t> piece_ids;
  if (IsPackedVec2(result)) {
    piece_ids.reserve(result.rows * result.packed_cols);
    for (uint32_t row = 0; row < result.rows; ++row) {
      for (uint32_t pack = 0; pack < result.packed_cols; ++pack) {
        const uint32_t col = pack * kPackedVec2Width;
        Instruction* piece = builder.AddCompositeConstruct(
            result.packed_vec2_type_id,
            {scalar_ids[MatrixFlatIndex(result, row, col)],
             scalar_ids[MatrixFlatIndex(result, row, col + 1)]});
        if (!piece) return 0;
        piece_ids.push_back(piece->result_id());
      }
    }
  } else {
    piece_ids = scalar_ids;
  }
  Instruction* result_value =
      builder.AddCompositeConstruct(result.lowered_type_id, piece_ids);
  if (!result_value || !builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                           result_value->result_id())) {
    return 0;
  }
  function->AddBasicBlock(std::move(block));
  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildDirectMatrixMatmulFunction(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c, uint32_t a_pointer_id,
    uint32_t a_pointer_type_id, uint32_t a_shape_id, uint32_t a_offset_id,
    const std::vector<Operand>& a_memory_operands, uint32_t a_constant_id,
    bool a_is_value, uint32_t b_pointer_id, uint32_t b_pointer_type_id,
    uint32_t b_shape_id, uint32_t b_offset_id,
    const std::vector<Operand>& b_memory_operands, uint32_t b_constant_id,
    bool b_is_value, uint32_t c_pointer_id, uint32_t c_pointer_type_id,
    uint32_t c_shape_id, uint32_t c_offset_id,
    const std::vector<Operand>& c_memory_operands, uint32_t c_constant_id,
    bool c_is_value,
    const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments) {
  if (!CanUseDirectMatrixMulAdd(result, a, b, c)) return 0;
  const uint64_t mac_count = static_cast<uint64_t>(result.rows) *
                             static_cast<uint64_t>(result.cols) *
                             static_cast<uint64_t>(a.cols);
  if (mac_count <= max_unrolled_matmul_macs_) {
    return BuildDirectMatrixMatmulFunctionUnrolled(
        result, a, b, c, a_pointer_id, a_pointer_type_id, a_shape_id,
        a_offset_id, a_memory_operands, a_constant_id, a_is_value, b_pointer_id,
        b_pointer_type_id, b_shape_id, b_offset_id, b_memory_operands,
        b_constant_id, b_is_value, c_pointer_id, c_pointer_type_id, c_shape_id,
        c_offset_id, c_memory_operands, c_constant_id, c_is_value,
        value_arguments);
  }
  if (CanUsePackedVec2MatrixMulAdd(result, a, b, c)) {
    return BuildDirectMatmulFunctionPackedVec2(
        result, a, b, c, a_pointer_id, a_pointer_type_id, a_shape_id,
        a_offset_id, a_memory_operands, a_constant_id, a_is_value, b_pointer_id,
        b_pointer_type_id, b_shape_id, b_offset_id, b_memory_operands,
        b_constant_id, b_is_value, c_pointer_id, c_pointer_type_id, c_shape_id,
        c_offset_id, c_memory_operands, c_constant_id, c_is_value,
        value_arguments);
  }
  if ((!a_is_value && (a_pointer_id == 0 || a_pointer_type_id == 0 ||
                       a_shape_id == 0 || a_offset_id == 0)) ||
      (!b_is_value && (b_pointer_id == 0 || b_pointer_type_id == 0 ||
                       b_shape_id == 0 || b_offset_id == 0)) ||
      (!c_is_value && (c_pointer_id == 0 || c_pointer_type_id == 0 ||
                       c_shape_id == 0 || c_offset_id == 0))) {
    return 0;
  }

  std::vector<uint32_t> parameter_type_ids;
  if (a_is_value && a_constant_id == 0) {
    parameter_type_ids.push_back(a.lowered_type_id);
  }
  if (b_is_value && b_constant_id == 0) {
    parameter_type_ids.push_back(b.lowered_type_id);
  }
  if (c_is_value && c_constant_id == 0) {
    parameter_type_ids.push_back(c.lowered_type_id);
  }
  if (parameter_type_ids.size() != value_arguments.size()) return 0;
  for (size_t i = 0; i < parameter_type_ids.size(); ++i) {
    if (parameter_type_ids[i] != value_arguments[i].second) return 0;
  }

  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, parameter_type_ids);
  const uint32_t function_id = TakeNextId();
  if (function_type_id == 0 || function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, result.lowered_type_id, function_id,
      std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  size_t parameter_index = 0;
  auto add_parameter = [&](uint32_t type_id) -> uint32_t {
    if (parameter_index >= parameter_type_ids.size() ||
        parameter_type_ids[parameter_index] != type_id) {
      return 0;
    }
    const uint32_t parameter_id = TakeNextId();
    if (parameter_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, type_id, parameter_id,
        std::initializer_list<Operand>{}));
    ++parameter_index;
    return parameter_id;
  };
  uint32_t a_value_id = a_constant_id;
  if (a_is_value && a_value_id == 0) {
    a_value_id = add_parameter(a.lowered_type_id);
  }
  uint32_t b_value_id = b_constant_id;
  if (b_is_value && b_value_id == 0) {
    b_value_id = add_parameter(b.lowered_type_id);
  }
  uint32_t c_value_id = c_constant_id;
  if (c_is_value && c_value_id == 0) {
    c_value_id = add_parameter(c.lowered_type_id);
  }
  if ((a_is_value && a_value_id == 0) || (b_is_value && b_value_id == 0) ||
      (c_is_value && c_value_id == 0) ||
      parameter_index != parameter_type_ids.size()) {
    return 0;
  }

  const uint32_t result_pointer_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t a_value_pointer_type_id =
      a_is_value ? GetOrCreatePointerType(a.lowered_type_id,
                                          spv::StorageClass::Function)
                 : 0;
  const uint32_t b_value_pointer_type_id =
      b_is_value ? GetOrCreatePointerType(b.lowered_type_id,
                                          spv::StorageClass::Function)
                 : 0;
  const uint32_t c_value_pointer_type_id =
      c_is_value ? GetOrCreatePointerType(c.lowered_type_id,
                                          spv::StorageClass::Function)
                 : 0;
  const uint32_t accumulator_pointer_type_id = GetOrCreatePointerType(
      result.component_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_pointer_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t output_count_id =
      GetOrCreateUIntConstant(result.rows * result.cols);
  const uint32_t result_cols_id = GetOrCreateUIntConstant(result.cols);
  const uint32_t inner_count_id = GetOrCreateUIntConstant(a.cols);
  const uint32_t result_zero_id = GetOrCreateZero(result.lowered_type_id);
  const uint32_t accumulator_zero_id =
      GetOrCreateZero(result.component_type_id);
  if (result_pointer_type_id == 0 || accumulator_pointer_type_id == 0 ||
      uint_type_id == 0 || uint_pointer_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || output_count_id == 0 ||
      result_cols_id == 0 || inner_count_id == 0 || result_zero_id == 0 ||
      accumulator_zero_id == 0 ||
      (a_is_value && a_value_pointer_type_id == 0) ||
      (b_is_value && b_value_pointer_type_id == 0) ||
      (c_is_value && c_value_pointer_type_id == 0)) {
    return 0;
  }

  std::array<uint32_t, 8> labels = {};
  for (uint32_t& label : labels) {
    label = TakeNextId();
    if (label == 0) return 0;
  }
  const uint32_t entry_label_id = labels[0];
  const uint32_t output_header_label_id = labels[1];
  const uint32_t output_body_label_id = labels[2];
  const uint32_t k_header_label_id = labels[3];
  const uint32_t k_body_label_id = labels[4];
  const uint32_t k_continue_label_id = labels[5];
  const uint32_t k_merge_label_id = labels[6];
  const uint32_t output_continue_label_id = labels[7];
  const uint32_t return_label_id = TakeNextId();
  if (return_label_id == 0) return 0;

  auto make_block = [this](uint32_t label_id) {
    return std::unique_ptr<BasicBlock>(MakeBasicBlock(label_id));
  };
  std::unique_ptr<BasicBlock> entry_block = make_block(entry_label_id);
  std::unique_ptr<BasicBlock> output_header_block =
      make_block(output_header_label_id);
  std::unique_ptr<BasicBlock> output_body_block =
      make_block(output_body_label_id);
  std::unique_ptr<BasicBlock> k_header_block = make_block(k_header_label_id);
  std::unique_ptr<BasicBlock> k_body_block = make_block(k_body_label_id);
  std::unique_ptr<BasicBlock> k_continue_block =
      make_block(k_continue_label_id);
  std::unique_ptr<BasicBlock> k_merge_block = make_block(k_merge_label_id);
  std::unique_ptr<BasicBlock> output_continue_block =
      make_block(output_continue_label_id);
  std::unique_ptr<BasicBlock> return_block = make_block(return_label_id);
  if (!entry_block || !output_header_block || !output_body_block ||
      !k_header_block || !k_body_block || !k_continue_block || !k_merge_block ||
      !output_continue_block || !return_block) {
    return 0;
  }

  InstructionBuilder entry_builder(context(), entry_block.get());
  Instruction* result_var = entry_builder.AddVariable(
      result_pointer_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* output_index_var = entry_builder.AddVariable(
      uint_pointer_type_id, static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_var = entry_builder.AddVariable(
      uint_pointer_type_id, static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* accumulator_var = entry_builder.AddVariable(
      accumulator_pointer_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* a_value_var =
      a_is_value ? entry_builder.AddVariable(
                       a_value_pointer_type_id,
                       static_cast<uint32_t>(spv::StorageClass::Function))
                 : nullptr;
  Instruction* b_value_var =
      b_is_value ? entry_builder.AddVariable(
                       b_value_pointer_type_id,
                       static_cast<uint32_t>(spv::StorageClass::Function))
                 : nullptr;
  Instruction* c_value_var =
      c_is_value ? entry_builder.AddVariable(
                       c_value_pointer_type_id,
                       static_cast<uint32_t>(spv::StorageClass::Function))
                 : nullptr;
  if (!result_var || !output_index_var || !k_var || !accumulator_var ||
      (a_is_value && !a_value_var) || (b_is_value && !b_value_var) ||
      (c_is_value && !c_value_var)) {
    return 0;
  }

  const uint32_t captured_a_pointer_id =
      a_is_value ? 0 : BuildCapturedPointer(&entry_builder, a_pointer_id);
  const uint32_t captured_b_pointer_id =
      b_is_value ? 0 : BuildCapturedPointer(&entry_builder, b_pointer_id);
  const uint32_t captured_c_pointer_id =
      c_is_value ? 0 : BuildCapturedPointer(&entry_builder, c_pointer_id);
  if ((!a_is_value && captured_a_pointer_id == 0) ||
      (!b_is_value && captured_b_pointer_id == 0) ||
      (!c_is_value && captured_c_pointer_id == 0)) {
    return 0;
  }

  if ((a_is_value &&
       !entry_builder.AddStore(a_value_var->result_id(), a_value_id)) ||
      (b_is_value &&
       !entry_builder.AddStore(b_value_var->result_id(), b_value_id)) ||
      (c_is_value &&
       !entry_builder.AddStore(c_value_var->result_id(), c_value_id)) ||
      !entry_builder.AddStore(result_var->result_id(), result_zero_id) ||
      !entry_builder.AddStore(output_index_var->result_id(), zero_uint_id) ||
      !entry_builder.AddBranch(output_header_label_id)) {
    return 0;
  }

  auto load_matrix_scalar =
      [&](InstructionBuilder* builder, const MatrixTypeInfo& info,
          bool is_value, Instruction* value_var, uint32_t pointer_type_id,
          uint32_t captured_pointer_id, uint32_t shape_id, uint32_t offset_id,
          const std::vector<Operand>& memory_operands,
          uint32_t logical_index_id) -> uint32_t {
    if (!builder || logical_index_id == 0) return 0;
    if (is_value) {
      return value_var
                 ? BuildLogicalAggregateLoad(
                       builder, value_var->result_id(), info.component_type_id,
                       info.packed_vec2_type_id, logical_index_id)
                 : 0;
    }
    const uint32_t memory_index_id = BuildRowMajorMatrixMemoryIndex(
        builder, nullptr, shape_id, offset_id, info.cols, logical_index_id);
    const uint32_t element_pointer_id =
        memory_index_id ? BuildElementAccessFromPointerType(
                              builder, pointer_type_id, captured_pointer_id,
                              info.component_type_id, memory_index_id)
                        : 0;
    return element_pointer_id ? AddLoad(builder, info.component_type_id,
                                        element_pointer_id, memory_operands)
                              : 0;
  };

  InstructionBuilder output_header_builder(context(),
                                           output_header_block.get());
  Instruction* output_index = output_header_builder.AddLoad(
      uint_type_id, output_index_var->result_id());
  Instruction* output_condition =
      output_index ? output_header_builder.AddBinaryOp(
                         bool_type_id, spv::Op::OpULessThan,
                         output_index->result_id(), output_count_id)
                   : nullptr;
  if (!output_condition ||
      !output_header_builder.AddLoopMerge(return_label_id,
                                          output_continue_label_id) ||
      !output_header_builder.AddConditionalBranch(output_condition->result_id(),
                                                  output_body_label_id,
                                                  return_label_id)) {
    return 0;
  }

  InstructionBuilder output_body_builder(context(), output_body_block.get());
  if (!output_body_builder.AddStore(accumulator_var->result_id(),
                                    accumulator_zero_id) ||
      !output_body_builder.AddStore(k_var->result_id(), zero_uint_id) ||
      !output_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k = k_header_builder.AddLoad(uint_type_id, k_var->result_id());
  Instruction* k_condition =
      k ? k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                       k->result_id(), inner_count_id)
        : nullptr;
  if (!k_condition ||
      !k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id) ||
      !k_header_builder.AddConditionalBranch(
          k_condition->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
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
  const uint32_t a_scalar =
      a_index ? load_matrix_scalar(&k_body_builder, a, a_is_value, a_value_var,
                                   a_pointer_type_id, captured_a_pointer_id,
                                   a_shape_id, a_offset_id, a_memory_operands,
                                   a_index->result_id())
              : 0;
  const uint32_t b_scalar =
      b_index ? load_matrix_scalar(&k_body_builder, b, b_is_value, b_value_var,
                                   b_pointer_type_id, captured_b_pointer_id,
                                   b_shape_id, b_offset_id, b_memory_operands,
                                   b_index->result_id())
              : 0;
  Instruction* accumulator = k_body_builder.AddLoad(
      result.component_type_id, accumulator_var->result_id());
  const uint32_t accumulated =
      accumulator && a_scalar != 0 && b_scalar != 0
          ? BuildMatmulAccumulate(&k_body_builder, result.component_type_id,
                                  a.component_type_id, a_scalar,
                                  b.component_type_id, b_scalar,
                                  accumulator->result_id())
          : 0;
  if (!row || !col || !a_index || !b_index || accumulated == 0 ||
      !k_body_builder.AddStore(accumulator_var->result_id(), accumulated) ||
      !k_body_builder.AddBranch(k_continue_label_id)) {
    return 0;
  }

  InstructionBuilder k_continue_builder(context(), k_continue_block.get());
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k->result_id(), one_uint_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_var->result_id(), next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  Instruction* product = k_merge_builder.AddLoad(result.component_type_id,
                                                 accumulator_var->result_id());
  const uint32_t c_scalar = load_matrix_scalar(
      &k_merge_builder, c, c_is_value, c_value_var, c_pointer_type_id,
      captured_c_pointer_id, c_shape_id, c_offset_id, c_memory_operands,
      output_index->result_id());
  Instruction* sum = product && c_scalar != 0
                         ? k_merge_builder.AddBinaryOp(
                               result.component_type_id, spv::Op::OpFAdd,
                               product->result_id(), c_scalar)
                         : nullptr;
  ApplyActiveFPFastMathMode(sum);
  if (!sum ||
      !BuildLogicalAggregateStore(
          &k_merge_builder, result_var->result_id(), result.component_type_id,
          result.packed_vec2_type_id, output_index->result_id(),
          sum->result_id()) ||
      !k_merge_builder.AddBranch(output_continue_label_id)) {
    return 0;
  }

  InstructionBuilder output_continue_builder(context(),
                                             output_continue_block.get());
  Instruction* next_output = output_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, output_index->result_id(), one_uint_id);
  if (!next_output ||
      !output_continue_builder.AddStore(output_index_var->result_id(),
                                        next_output->result_id()) ||
      !output_continue_builder.AddBranch(output_header_label_id)) {
    return 0;
  }

  InstructionBuilder return_builder(context(), return_block.get());
  Instruction* result_value =
      return_builder.AddLoad(result.lowered_type_id, result_var->result_id());
  if (!result_value || !return_builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                                  result_value->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(entry_block));
  function->AddBasicBlock(std::move(output_header_block));
  function->AddBasicBlock(std::move(output_body_block));
  function->AddBasicBlock(std::move(k_header_block));
  function->AddBasicBlock(std::move(k_body_block));
  function->AddBasicBlock(std::move(k_continue_block));
  function->AddBasicBlock(std::move(k_merge_block));
  function->AddBasicBlock(std::move(output_continue_block));
  function->AddBasicBlock(std::move(return_block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildDirectMatmulFunctionPackedVec2(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c, uint32_t a_pointer_id,
    uint32_t a_pointer_type_id, uint32_t a_shape_id, uint32_t a_offset_id,
    const std::vector<Operand>& a_memory_operands, uint32_t a_constant_id,
    bool a_is_value, uint32_t b_pointer_id, uint32_t b_pointer_type_id,
    uint32_t b_shape_id, uint32_t b_offset_id,
    const std::vector<Operand>& b_memory_operands, uint32_t b_constant_id,
    bool b_is_value, uint32_t c_pointer_id, uint32_t c_pointer_type_id,
    uint32_t c_shape_id, uint32_t c_offset_id,
    const std::vector<Operand>& c_memory_operands, uint32_t c_constant_id,
    bool c_is_value,
    const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments) {
  if (!CanUsePackedVec2MatrixMulAdd(result, a, b, c)) return 0;
  if ((!a_is_value && (a_pointer_id == 0 || a_pointer_type_id == 0 ||
                       a_shape_id == 0 || a_offset_id == 0)) ||
      (!b_is_value && (b_pointer_id == 0 || b_pointer_type_id == 0 ||
                       b_shape_id == 0 || b_offset_id == 0)) ||
      (!c_is_value && (c_pointer_id == 0 || c_pointer_type_id == 0 ||
                       c_shape_id == 0 || c_offset_id == 0))) {
    return 0;
  }

  const uint32_t a_load_function_id =
      a_is_value ? 0
                 : GetOrCreatePackedLoadChunkFunction(
                       a_pointer_id, a_pointer_type_id, a.component_type_id,
                       a.packed_vec2_type_id, a_memory_operands);
  const uint32_t b_load_function_id =
      b_is_value ? 0
                 : GetOrCreatePackedLoadChunkFunction(
                       b_pointer_id, b_pointer_type_id, b.component_type_id,
                       b.packed_vec2_type_id, b_memory_operands);
  const uint32_t c_load_function_id =
      c_is_value ? 0
                 : GetOrCreatePackedLoadChunkFunction(
                       c_pointer_id, c_pointer_type_id, c.component_type_id,
                       c.packed_vec2_type_id, c_memory_operands);
  std::vector<uint32_t> param_type_ids;
  for (const auto& arg : value_arguments) {
    param_type_ids.push_back(arg.second);
  }
  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, param_type_ids);
  const uint32_t lowered_function_ptr_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec2_function_ptr_type_id = GetOrCreatePointerType(
      result.packed_vec2_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t row_count_id = GetOrCreateUIntConstant(result.rows);
  const uint32_t result_packed_cols_id =
      GetOrCreateUIntConstant(result.packed_cols);
  const uint32_t a_cols_id = GetOrCreateUIntConstant(a.cols);
  const uint32_t b_cols_id = GetOrCreateUIntConstant(b.cols);
  const uint32_t c_cols_id = GetOrCreateUIntConstant(c.cols);
  const uint32_t a_packed_cols_id = GetOrCreateUIntConstant(a.packed_cols);
  const uint32_t zero2_id = GetOrCreateZero(result.packed_vec2_type_id);
  if ((!a_is_value && a_load_function_id == 0) ||
      (!b_is_value && b_load_function_id == 0) ||
      (!c_is_value && c_load_function_id == 0) || function_type_id == 0 ||
      lowered_function_ptr_type_id == 0 || vec2_function_ptr_type_id == 0 ||
      uint_type_id == 0 || uint_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || one_uint_id == 0 ||
      packed_width_uint_id == 0 || row_count_id == 0 ||
      result_packed_cols_id == 0 || a_cols_id == 0 || b_cols_id == 0 ||
      c_cols_id == 0 || a_packed_cols_id == 0 || zero2_id == 0) {
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

  uint32_t a_param_id = 0;
  if (a_is_value && a_constant_id == 0) {
    a_param_id = TakeNextId();
    if (a_param_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, a.lowered_type_id, a_param_id,
        std::initializer_list<Operand>{}));
  }
  uint32_t b_param_id = 0;
  if (b_is_value && b_constant_id == 0) {
    b_param_id = TakeNextId();
    if (b_param_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, b.lowered_type_id, b_param_id,
        std::initializer_list<Operand>{}));
  }
  uint32_t c_param_id = 0;
  if (c_is_value && c_constant_id == 0) {
    c_param_id = TakeNextId();
    if (c_param_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, c.lowered_type_id, c_param_id,
        std::initializer_list<Operand>{}));
  }

  const uint32_t entry_label_id = TakeNextId();
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
  if (entry_label_id == 0 || row_header_label_id == 0 ||
      row_body_label_id == 0 || row_continue_label_id == 0 ||
      row_merge_label_id == 0 || col_header_label_id == 0 ||
      col_body_label_id == 0 || col_continue_label_id == 0 ||
      col_merge_label_id == 0 || k_header_label_id == 0 ||
      k_body_label_id == 0 || k_continue_label_id == 0 ||
      k_merge_label_id == 0) {
    return 0;
  }

  std::unique_ptr<BasicBlock> entry_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_label_id));
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
  if (!entry_block || !row_header_block || !row_body_block ||
      !row_continue_block || !row_merge_block || !col_header_block ||
      !col_body_block || !col_continue_block || !col_merge_block ||
      !k_header_block || !k_body_block || !k_continue_block || !k_merge_block) {
    return 0;
  }

  InstructionBuilder entry_builder(context(), entry_block.get());
  Instruction* result_var = entry_builder.AddVariable(
      lowered_function_ptr_type_id,
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
  std::array<Instruction*, kPackedVec2Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    acc_vars[lane] = entry_builder.AddVariable(
        vec2_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!result_var || !row_var || !col_pack_var || !k_pack_var) return 0;
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }

  // Get pointer types for constant operands (before creating any variables)
  uint32_t a_function_ptr_type_id = 0;
  if (a_is_value) {
    a_function_ptr_type_id =
        GetOrCreatePointerType(a.lowered_type_id, spv::StorageClass::Function);
    if (a_function_ptr_type_id == 0) return 0;
  }
  uint32_t b_function_ptr_type_id = 0;
  if (b_is_value) {
    b_function_ptr_type_id =
        GetOrCreatePointerType(b.lowered_type_id, spv::StorageClass::Function);
    if (b_function_ptr_type_id == 0) return 0;
  }
  uint32_t c_function_ptr_type_id = 0;
  if (c_is_value) {
    c_function_ptr_type_id =
        GetOrCreatePointerType(c.lowered_type_id, spv::StorageClass::Function);
    if (c_function_ptr_type_id == 0) return 0;
  }

  // Create ALL function-local variables FIRST (SPIR-V requirement)
  Instruction* a_var = nullptr;
  if (a_is_value) {
    a_var = entry_builder.AddVariable(
        a_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
    if (!a_var) return 0;
  }
  Instruction* b_var = nullptr;
  if (b_is_value) {
    b_var = entry_builder.AddVariable(
        b_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
    if (!b_var) return 0;
  }
  Instruction* c_var = nullptr;
  if (c_is_value) {
    c_var = entry_builder.AddVariable(
        c_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
    if (!c_var) return 0;
  }

  // Now do all stores AFTER all variables are declared
  if (a_is_value) {
    const uint32_t store_src = a_constant_id != 0 ? a_constant_id : a_param_id;
    if (store_src == 0 ||
        !entry_builder.AddStore(a_var->result_id(), store_src)) {
      return 0;
    }
  }
  if (b_is_value) {
    const uint32_t store_src = b_constant_id != 0 ? b_constant_id : b_param_id;
    if (store_src == 0 ||
        !entry_builder.AddStore(b_var->result_id(), store_src)) {
      return 0;
    }
  }
  if (c_is_value) {
    const uint32_t store_src = c_constant_id != 0 ? c_constant_id : c_param_id;
    if (store_src == 0 ||
        !entry_builder.AddStore(c_var->result_id(), store_src)) {
      return 0;
    }
  }

  if (!entry_builder.AddStore(row_var->result_id(), zero_uint_id) ||
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
                                       row_continue_label_id) ||
      !row_header_builder.AddConditionalBranch(
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
                                       col_continue_label_id) ||
      !col_header_builder.AddConditionalBranch(
          col_cond->result_id(), col_body_label_id, col_merge_label_id)) {
    return 0;
  }

  InstructionBuilder col_body_builder(context(), col_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!col_body_builder.AddStore(acc_var->result_id(), zero2_id)) return 0;
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
                                   k_pack_load->result_id(), a_packed_cols_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id) ||
      !k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* k_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_pack_load->result_id(),
      packed_width_uint_id);
  if (!k_base) return 0;

  // Load A matrix tile: from buffer or from constant
  uint32_t a_vec_id = 0;
  if (a_is_value) {
    // Extract from constant: index = row * a.packed_cols + k_pack
    const uint32_t a_packed_cols_const_id =
        GetOrCreateUIntConstant(a.packed_cols);
    if (a_packed_cols_const_id == 0) return 0;
    const uint32_t a_vec2_ptr_type_id = GetOrCreatePointerType(
        a.packed_vec2_type_id, spv::StorageClass::Function);
    if (a_vec2_ptr_type_id == 0) return 0;
    Instruction* a_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_load->result_id(),
        a_packed_cols_const_id);
    if (!a_row_offset) return 0;
    Instruction* a_packed_idx = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, a_row_offset->result_id(),
        k_pack_load->result_id());
    if (!a_packed_idx) return 0;
    Instruction* a_vec_ptr = k_body_builder.AddAccessChain(
        a_vec2_ptr_type_id, a_var->result_id(), {a_packed_idx->result_id()});
    if (!a_vec_ptr) return 0;
    Instruction* a_vec =
        k_body_builder.AddLoad(a.packed_vec2_type_id, a_vec_ptr->result_id());
    if (!a_vec) return 0;
    a_vec_id = a_vec->result_id();
  } else {
    // Load from buffer via helper function
    Instruction* a_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_load->result_id(), a_cols_id);
    if (!a_row_offset) return 0;
    Instruction* a_local_base = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, a_row_offset->result_id(),
        k_base->result_id());
    if (!a_local_base) return 0;
    const uint32_t a_memory_base_id = BuildRowMajorMatrixMemoryIndex(
        &k_body_builder, nullptr, a_shape_id, a_offset_id, a.cols,
        a_local_base->result_id());
    if (a_memory_base_id == 0) return 0;
    Instruction* a_vec = k_body_builder.AddFunctionCall(
        a.packed_vec2_type_id, a_load_function_id, {a_memory_base_id});
    if (!a_vec) return 0;
    a_vec_id = a_vec->result_id();
  }

  std::array<uint32_t, kPackedVec2Width> b_vecs = {};
  const uint32_t b_packed_cols_const_id =
      b_is_value ? GetOrCreateUIntConstant(b.packed_cols) : 0;
  if (b_is_value && b_packed_cols_const_id == 0) return 0;
  const uint32_t b_vec2_ptr_type_id =
      b_is_value ? GetOrCreatePointerType(b.packed_vec2_type_id,
                                          spv::StorageClass::Function)
                 : 0;
  if (b_is_value && b_vec2_ptr_type_id == 0) return 0;
  for (uint32_t row_lane = 0; row_lane < kPackedVec2Width; ++row_lane) {
    const uint32_t row_lane_id = GetOrCreateUIntConstant(row_lane);
    if (row_lane_id == 0) return 0;
    Instruction* b_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_base->result_id(), row_lane_id);
    if (!b_row) return 0;

    if (b_is_value) {
      // Extract from constant: index = b_row * b.packed_cols + col_pack
      Instruction* b_row_offset = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, b_row->result_id(),
          b_packed_cols_const_id);
      if (!b_row_offset) return 0;
      Instruction* b_packed_idx = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, b_row_offset->result_id(),
          col_pack_load->result_id());
      if (!b_packed_idx) return 0;
      Instruction* b_vec_ptr = k_body_builder.AddAccessChain(
          b_vec2_ptr_type_id, b_var->result_id(), {b_packed_idx->result_id()});
      if (!b_vec_ptr) return 0;
      Instruction* b_vec =
          k_body_builder.AddLoad(b.packed_vec2_type_id, b_vec_ptr->result_id());
      if (!b_vec) return 0;
      b_vecs[row_lane] = b_vec->result_id();
    } else {
      // Load from buffer via helper function
      Instruction* b_row_offset = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, b_row->result_id(), b_cols_id);
      if (!b_row_offset) return 0;
      Instruction* b_col_offset = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
          packed_width_uint_id);
      if (!b_col_offset) return 0;
      Instruction* b_local_base = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, b_row_offset->result_id(),
          b_col_offset->result_id());
      if (!b_local_base) return 0;
      const uint32_t b_memory_base_id = BuildRowMajorMatrixMemoryIndex(
          &k_body_builder, nullptr, b_shape_id, b_offset_id, b.cols,
          b_local_base->result_id());
      if (b_memory_base_id == 0) return 0;
      Instruction* b_vec = k_body_builder.AddFunctionCall(
          b.packed_vec2_type_id, b_load_function_id, {b_memory_base_id});
      if (!b_vec) return 0;
      b_vecs[row_lane] = b_vec->result_id();
    }
  }

  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    std::vector<uint32_t> weight_lanes;
    weight_lanes.reserve(kPackedVec2Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec2Width; ++row_lane) {
      const uint32_t value = ExtractCompositeElement(
          &k_body_builder, result.component_type_id, b_vecs[row_lane], lane);
      if (value == 0) return 0;
      weight_lanes.push_back(value);
    }
    Instruction* weight_vec = k_body_builder.AddCompositeConstruct(
        result.packed_vec2_type_id, weight_lanes);
    Instruction* acc = k_body_builder.AddLoad(result.packed_vec2_type_id,
                                              acc_vars[lane]->result_id());
    if (!weight_vec || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, result.packed_vec2_type_id, a_vec_id,
                 weight_vec->result_id(), acc->result_id());
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

  // Load C matrix tile: from buffer or from constant
  uint32_t c_vec_id = 0;
  if (c_is_value) {
    // Extract from constant: index = row * c.packed_cols + col_pack
    const uint32_t c_packed_cols_const_id =
        GetOrCreateUIntConstant(c.packed_cols);
    if (c_packed_cols_const_id == 0) return 0;
    const uint32_t c_vec2_ptr_type_id = GetOrCreatePointerType(
        c.packed_vec2_type_id, spv::StorageClass::Function);
    if (c_vec2_ptr_type_id == 0) return 0;
    Instruction* c_row_offset = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_load->result_id(),
        c_packed_cols_const_id);
    if (!c_row_offset) return 0;
    Instruction* c_packed_idx = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, c_row_offset->result_id(),
        col_pack_load->result_id());
    if (!c_packed_idx) return 0;
    Instruction* c_vec_ptr = k_merge_builder.AddAccessChain(
        c_vec2_ptr_type_id, c_var->result_id(), {c_packed_idx->result_id()});
    if (!c_vec_ptr) return 0;
    Instruction* c_vec =
        k_merge_builder.AddLoad(c.packed_vec2_type_id, c_vec_ptr->result_id());
    if (!c_vec) return 0;
    c_vec_id = c_vec->result_id();
  } else {
    // Load from buffer via helper function
    Instruction* c_row_offset = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_load->result_id(), c_cols_id);
    if (!c_row_offset) return 0;
    Instruction* c_col_offset = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
        packed_width_uint_id);
    if (!c_col_offset) return 0;
    Instruction* c_local_base = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, c_row_offset->result_id(),
        c_col_offset->result_id());
    if (!c_local_base) return 0;
    const uint32_t c_memory_base_id = BuildRowMajorMatrixMemoryIndex(
        &k_merge_builder, nullptr, c_shape_id, c_offset_id, c.cols,
        c_local_base->result_id());
    if (c_memory_base_id == 0) return 0;
    Instruction* c_vec = k_merge_builder.AddFunctionCall(
        c.packed_vec2_type_id, c_load_function_id, {c_memory_base_id});
    if (!c_vec) return 0;
    c_vec_id = c_vec->result_id();
  }

  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec2Width);
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec2_type_id,
                                               acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    if (reduced == 0) return 0;
    const uint32_t bias_value = ExtractCompositeElement(
        &k_merge_builder, result.component_type_id, c_vec_id, lane);
    if (bias_value == 0) return 0;
    Instruction* add = k_merge_builder.AddBinaryOp(
        result.component_type_id, spv::Op::OpFAdd, bias_value, reduced);
    if (!add) return 0;
    ApplyActiveFPFastMathMode(add);
    lane_ids.push_back(add->result_id());
  }
  Instruction* result_vec = k_merge_builder.AddCompositeConstruct(
      result.packed_vec2_type_id, lane_ids);
  if (!result_vec) return 0;
  Instruction* result_vec_ptr = k_merge_builder.AddAccessChain(
      vec2_function_ptr_type_id, result_var->result_id(),
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
  return function_id;
}

}  // namespace opt
}  // namespace spvtools
