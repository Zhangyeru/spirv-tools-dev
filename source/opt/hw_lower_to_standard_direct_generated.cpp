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

#include "source/opt/hw_lower_to_standard_pass.h"

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

uint32_t HwLowerToStandardPass::BuildFusedVectorMatmulStoreFunctionPackedVec4(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, uint32_t input_pointer_id,
    uint32_t input_pointer_type_id,
    const std::vector<Operand>& input_memory_operands,
    uint32_t matrix_pointer_id, uint32_t matrix_pointer_type_id,
    uint32_t matrix_shape_id, uint32_t matrix_offset_id,
    const std::vector<Operand>& matrix_memory_operands,
    uint32_t output_pointer_id, uint32_t output_pointer_type_id,
    const std::vector<Operand>& output_memory_operands) {
  if (!IsPackedVec4(result) || !IsPackedVec4(input) || !IsPackedVec4(matrix) ||
      input.length != matrix.rows || result.length != matrix.cols) {
    return 0;
  }

  const uint32_t input_load_function_id = GetOrCreatePackedLoadChunkFunction(
      input_pointer_id, input_pointer_type_id, input.component_type_id,
      input.packed_vec4_type_id, input_memory_operands);
  const uint32_t matrix_load_function_id = GetOrCreatePackedLoadChunkFunction(
      matrix_pointer_id, matrix_pointer_type_id, matrix.component_type_id,
      matrix.packed_vec4_type_id, matrix_memory_operands);
  const uint32_t output_store_function_id = GetOrCreatePackedStoreChunkFunction(
      output_pointer_id, output_pointer_type_id, result.component_type_id,
      result.packed_vec4_type_id, output_memory_operands);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t function_type_id = GetOrCreateFunctionType(void_type_id, {});
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t vec4_function_ptr_type_id = GetOrCreatePointerType(
      result.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t result_length_id = GetOrCreateUIntConstant(result.length);
  const uint32_t input_length_id = GetOrCreateUIntConstant(input.length);
  const uint32_t matrix_cols_id = GetOrCreateUIntConstant(matrix.cols);
  const uint32_t zero4_id = GetOrCreateZero(result.packed_vec4_type_id);
  if (input_load_function_id == 0 || matrix_load_function_id == 0 ||
      output_store_function_id == 0 || void_type_id == 0 ||
      function_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || vec4_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || four_uint_id == 0 ||
      result_length_id == 0 || input_length_id == 0 || matrix_cols_id == 0 ||
      zero4_id == 0) {
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
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] = entry_builder.AddVariable(
        vec4_function_ptr_type_id,
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
    if (!out_body_builder.AddStore(acc_var->result_id(), zero4_id)) return 0;
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
      input.packed_vec4_type_id, input_load_function_id,
      {k_base_load->result_id()});
  if (!input_vec) return 0;
  std::array<uint32_t, kPackedVec4Width> weight_row_ids = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
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
        matrix.packed_vec4_type_id, matrix_load_function_id,
        {matrix_memory_base_id});
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
    Instruction* weight = k_body_builder.AddCompositeConstruct(
        result.packed_vec4_type_id, weight_lane_ids);
    Instruction* acc = k_body_builder.AddLoad(result.packed_vec4_type_id,
                                              acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, result.packed_vec4_type_id,
                 input_vec->result_id(), weight->result_id(), acc->result_id());
    if (fma == 0 ||
        !k_body_builder.AddStore(acc_vars[lane]->result_id(), fma)) {
      return 0;
    }
  }
  if (!k_body_builder.AddBranch(k_continue_label_id)) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue_block.get());
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k_base_load->result_id(), four_uint_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_base_var->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec4_type_id,
                                               acc_vars[lane]->result_id());
    if (!acc) return 0;
    const uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    if (reduced == 0) return 0;
    lane_ids.push_back(reduced);
  }
  Instruction* result_vec = k_merge_builder.AddCompositeConstruct(
      result.packed_vec4_type_id, lane_ids);
  if (!result_vec ||
      !k_merge_builder.AddFunctionCall(
          void_type_id, output_store_function_id,
          {out_pack_load->result_id(), result_vec->result_id()}) ||
      !k_merge_builder.AddBranch(out_continue_label_id)) {
    return 0;
  }

  InstructionBuilder out_continue_builder(context(), out_continue_block.get());
  Instruction* next_out = out_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, out_pack_load->result_id(), four_uint_id);
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
  if (!IsPackedVec4(vector) || vector.packed_length == 0 ||
      vector.packed_length > kMaxFusedConstantBiasPacks || constant_id == 0) {
    return 0;
  }

  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t function_type_id =
      GetOrCreateFunctionType(vector.packed_vec4_type_id, {uint_type_id});
  if (uint_type_id == 0 || bool_type_id == 0 || function_type_id == 0) {
    return 0;
  }

  const uint32_t function_id = TakeNextId();
  const uint32_t index_param_id = TakeNextId();
  if (function_id == 0 || index_param_id == 0) return 0;
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, vector.packed_vec4_type_id, function_id,
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
        GetOrCreateUIntConstant(pack * kPackedVec4Width);
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
      if (operand && operand->type_id() == vector.packed_vec4_type_id) {
        value_id = operand_id;
      }
    }
    if (value_id == 0) {
      value_id = ExtractCompositeElement(&builder, vector.packed_vec4_type_id,
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
HwLowerToStandardPass::BuildFusedVectorMatmulAddStoreFunctionPackedVec4(
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
  if (!IsPackedVec4(result) || !IsPackedVec4(input) || !IsPackedVec4(matrix) ||
      input.length != matrix.rows || result.length != matrix.cols ||
      (has_bias && (!bias || !IsSamePackedVec4Kind(result, *bias) ||
                    bias_constant_id == 0))) {
    return 0;
  }

  const uint32_t input_load_function_id = GetOrCreatePackedLoadChunkFunction(
      input_pointer_id, input_pointer_type_id, input.component_type_id,
      input.packed_vec4_type_id, input_memory_operands);
  const uint32_t output_store_function_id = GetOrCreatePackedStoreChunkFunction(
      output_pointer_id, output_pointer_type_id, result.component_type_id,
      result.packed_vec4_type_id, output_memory_operands);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t function_type_id = GetOrCreateFunctionType(void_type_id, {});
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t vec4_function_ptr_type_id = GetOrCreatePointerType(
      result.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t result_length_id = GetOrCreateUIntConstant(result.length);
  const uint32_t input_length_id = GetOrCreateUIntConstant(input.length);
  const uint32_t matrix_cols_id = GetOrCreateUIntConstant(matrix.cols);
  const uint32_t zero4_id = GetOrCreateZero(result.packed_vec4_type_id);
  const uint32_t bias_select_function_id =
      has_bias
          ? BuildConstantPackedVectorSelectFunction(*bias, bias_constant_id)
          : 0;
  if (input_load_function_id == 0 || output_store_function_id == 0 ||
      void_type_id == 0 || function_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || vec4_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || four_uint_id == 0 ||
      result_length_id == 0 || input_length_id == 0 || matrix_cols_id == 0 ||
      zero4_id == 0 || (has_bias && bias_select_function_id == 0)) {
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
      vec4_function_ptr_type_id,
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
  if (!out_body_builder.AddStore(acc_var->result_id(), zero4_id) ||
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
      input.packed_vec4_type_id, input_load_function_id,
      {k_base_load->result_id()});
  if (!input_vec) return 0;
  std::vector<uint32_t> dot_ids;
  dot_ids.reserve(kPackedVec4Width);
  for (uint32_t output_lane = 0; output_lane < kPackedVec4Width;
       ++output_lane) {
    std::vector<uint32_t> weight_column_ids;
    weight_column_ids.reserve(kPackedVec4Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
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
        result.packed_vec4_type_id, weight_column_ids);
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
      k_body_builder.AddCompositeConstruct(result.packed_vec4_type_id, dot_ids);
  Instruction* acc =
      k_body_builder.AddLoad(result.packed_vec4_type_id, acc_var->result_id());
  Instruction* next_acc = dot_vec && acc
                              ? k_body_builder.AddBinaryOp(
                                    result.packed_vec4_type_id, spv::Op::OpFAdd,
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
      uint_type_id, spv::Op::OpIAdd, k_base_load->result_id(), four_uint_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_base_var->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  Instruction* result_vec =
      k_merge_builder.AddLoad(result.packed_vec4_type_id, acc_var->result_id());
  if (!result_vec) return 0;
  if (has_bias) {
    Instruction* bias_vec = k_merge_builder.AddFunctionCall(
        bias->packed_vec4_type_id, bias_select_function_id,
        {output_base->result_id()});
    if (!bias_vec) return 0;
    result_vec = k_merge_builder.AddBinaryOp(
        result.packed_vec4_type_id, spv::Op::OpFAdd, result_vec->result_id(),
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
      uint_type_id, spv::Op::OpIAdd, output_base->result_id(), four_uint_id);
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

uint32_t HwLowerToStandardPass::BuildFusedMatrixMatmulStoreFunctionPackedVec4(
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
  if (!CanUsePackedVec4MatrixMulAdd(result, a, b, c)) return 0;
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
      a.packed_vec4_type_id, a_memory_operands);
  const uint32_t b_load_function_id = GetOrCreatePackedLoadChunkFunction(
      b_pointer_id, b_pointer_type_id, b.component_type_id,
      b.packed_vec4_type_id, b_memory_operands);
  const uint32_t c_load_function_id = GetOrCreatePackedLoadChunkFunction(
      c_pointer_id, c_pointer_type_id, c.component_type_id,
      c.packed_vec4_type_id, c_memory_operands);
  const uint32_t output_store_function_id = GetOrCreatePackedStoreChunkFunction(
      output_pointer_id, output_pointer_type_id, result.component_type_id,
      result.packed_vec4_type_id, output_memory_operands);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t function_type_id = GetOrCreateFunctionType(void_type_id, {});
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
  const uint32_t a_cols_id = GetOrCreateUIntConstant(a.cols);
  const uint32_t b_cols_id = GetOrCreateUIntConstant(b.cols);
  const uint32_t c_cols_id = GetOrCreateUIntConstant(c.cols);
  const uint32_t result_cols_id = GetOrCreateUIntConstant(result.cols);
  const uint32_t a_packed_cols_id = GetOrCreateUIntConstant(a.packed_cols);
  const uint32_t zero4_id = GetOrCreateZero(result.packed_vec4_type_id);
  if (a_load_function_id == 0 || b_load_function_id == 0 ||
      c_load_function_id == 0 || output_store_function_id == 0 ||
      void_type_id == 0 || function_type_id == 0 ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || four_uint_id == 0 ||
      row_count_id == 0 || result_packed_cols_id == 0 || a_cols_id == 0 ||
      b_cols_id == 0 || c_cols_id == 0 || result_cols_id == 0 ||
      a_packed_cols_id == 0 || zero4_id == 0) {
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
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] = entry_builder.AddVariable(
        vec4_function_ptr_type_id,
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
                                   k_pack_load->result_id(), a_packed_cols_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id) ||
      !k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* k_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_pack_load->result_id(), four_uint_id);
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
      a.packed_vec4_type_id, a_load_function_id, {a_memory_base_id});
  if (!a_vec) return 0;

  std::array<uint32_t, kPackedVec4Width> b_vecs = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
    const uint32_t row_lane_id = GetOrCreateUIntConstant(row_lane);
    if (row_lane_id == 0) return 0;
    Instruction* b_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_base->result_id(), row_lane_id);
    if (!b_row) return 0;
    Instruction* b_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, b_row->result_id(), b_cols_id);
    if (!b_row_offset) return 0;
    Instruction* b_col_offset =
        k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                   col_pack_load->result_id(), four_uint_id);
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
        b.packed_vec4_type_id, b_load_function_id, {b_memory_base_id});
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

  // k_merge: finalize the tile — reduce accumulators, add bias from C,
  // and stream the resulting vec4 directly to the output SSBO.
  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  Instruction* c_row_offset = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, row_load->result_id(), c_cols_id);
  if (!c_row_offset) return 0;
  Instruction* c_col_offset = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(), four_uint_id);
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
      c.packed_vec4_type_id, c_load_function_id, {c_memory_base_id});
  if (!c_vec) return 0;

  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec4_type_id,
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
      result.packed_vec4_type_id, lane_ids);
  if (!result_vec) return 0;

  // Compute the flat output SSBO element index for the current tile:
  // base = row * result.cols + col_pack * 4, then apply output_shape/offset.
  Instruction* out_col_offset = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(), four_uint_id);
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

uint32_t HwLowerToStandardPass::BuildDirectVectorMatmulFunctionPackedVec4(
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
    const std::vector<Operand>& bias_memory_operands, uint32_t bias_constant_id,
    bool bias_is_value,
    const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments) {
  if (!IsPackedVec4(result) || !IsPackedVec4(input) || !IsPackedVec4(matrix) ||
      input.length != matrix.rows || result.length != matrix.cols ||
      (has_bias && (!bias || !IsPackedVec4(*bias)))) {
    return 0;
  }
  // For value operands: constant_id may be 0 if passed as function parameter.
  if ((!input_is_value &&
       (input_pointer_id == 0 || input_pointer_type_id == 0)) ||
      (!matrix_is_value &&
       (matrix_pointer_id == 0 || matrix_pointer_type_id == 0 ||
        matrix_shape_id == 0 || matrix_offset_id == 0)) ||
      (has_bias && !bias_is_value &&
       (bias_pointer_id == 0 || bias_pointer_type_id == 0))) {
    return 0;
  }

  const uint32_t input_load_function_id =
      input_is_value ? 0
                     : GetOrCreatePackedLoadChunkFunction(
                           input_pointer_id, input_pointer_type_id,
                           input.component_type_id, input.packed_vec4_type_id,
                           input_memory_operands);
  const uint32_t matrix_load_function_id =
      matrix_is_value ? 0
                      : GetOrCreatePackedLoadChunkFunction(
                            matrix_pointer_id, matrix_pointer_type_id,
                            matrix.component_type_id,
                            matrix.packed_vec4_type_id, matrix_memory_operands);
  const uint32_t bias_load_function_id =
      (has_bias && !bias_is_value)
          ? GetOrCreatePackedLoadChunkFunction(
                bias_pointer_id, bias_pointer_type_id, bias->component_type_id,
                bias->packed_vec4_type_id, bias_memory_operands)
          : 0;
  // Build function type: return type + parameter types for value operands
  // that are passed as arguments (not constants).
  std::vector<uint32_t> param_type_ids;
  for (const auto& arg : value_arguments) {
    param_type_ids.push_back(arg.second);
  }
  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, param_type_ids);
  const uint32_t vec4_function_ptr_type_id = GetOrCreatePointerType(
      result.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t lowered_function_ptr_type_id = GetOrCreatePointerType(
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
  const uint32_t matrix_cols_id = GetOrCreateUIntConstant(matrix.cols);
  const uint32_t zero4_id = GetOrCreateZero(result.packed_vec4_type_id);
  if ((!input_is_value && input_load_function_id == 0) ||
      (!matrix_is_value && matrix_load_function_id == 0) ||
      (has_bias && !bias_is_value && bias_load_function_id == 0) ||
      function_type_id == 0 || vec4_function_ptr_type_id == 0 ||
      lowered_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || four_uint_id == 0 ||
      result_packed_length_id == 0 || input_packed_length_id == 0 ||
      matrix_cols_id == 0 || zero4_id == 0) {
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

  // Add function parameters for value operands passed as arguments.
  uint32_t input_param_id = 0;
  uint32_t matrix_param_id = 0;
  uint32_t bias_param_id = 0;
  {
    size_t arg_idx = 0;
    if (input_is_value && input_constant_id == 0) {
      input_param_id = TakeNextId();
      function->AddParameter(MakeUnique<Instruction>(
          context(), spv::Op::OpFunctionParameter, input.lowered_type_id,
          input_param_id, std::initializer_list<Operand>{}));
      ++arg_idx;
    }
    if (matrix_is_value && matrix_constant_id == 0) {
      matrix_param_id = TakeNextId();
      function->AddParameter(MakeUnique<Instruction>(
          context(), spv::Op::OpFunctionParameter, matrix.lowered_type_id,
          matrix_param_id, std::initializer_list<Operand>{}));
      ++arg_idx;
    }
    if (has_bias && bias_is_value && bias_constant_id == 0) {
      bias_param_id = TakeNextId();
      function->AddParameter(MakeUnique<Instruction>(
          context(), spv::Op::OpFunctionParameter, bias->lowered_type_id,
          bias_param_id, std::initializer_list<Operand>{}));
      ++arg_idx;
    }
  }

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
  Instruction* result_var = entry_builder.AddVariable(
      lowered_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* out_pack_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_base_var = entry_builder.AddVariable(
      uint_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] = entry_builder.AddVariable(
        vec4_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!result_var || !out_pack_var || !k_base_var) return 0;
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }

  // Get pointer types for constant operands (before creating any variables)
  uint32_t input_function_ptr_type_id = 0;
  if (input_is_value) {
    input_function_ptr_type_id = GetOrCreatePointerType(
        input.lowered_type_id, spv::StorageClass::Function);
    if (input_function_ptr_type_id == 0) return 0;
  }
  uint32_t matrix_function_ptr_type_id = 0;
  if (matrix_is_value) {
    matrix_function_ptr_type_id = GetOrCreatePointerType(
        matrix.lowered_type_id, spv::StorageClass::Function);
    if (matrix_function_ptr_type_id == 0) return 0;
  }
  uint32_t bias_function_ptr_type_id = 0;
  if (has_bias && bias_is_value) {
    bias_function_ptr_type_id = GetOrCreatePointerType(
        bias->lowered_type_id, spv::StorageClass::Function);
    if (bias_function_ptr_type_id == 0) return 0;
  }

  // Create ALL function-local variables FIRST (SPIR-V requirement)
  Instruction* input_var = nullptr;
  if (input_is_value) {
    input_var = entry_builder.AddVariable(
        input_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
    if (!input_var) return 0;
  }
  Instruction* matrix_var = nullptr;
  if (matrix_is_value) {
    matrix_var = entry_builder.AddVariable(
        matrix_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
    if (!matrix_var) return 0;
  }
  Instruction* bias_var = nullptr;
  if (has_bias && bias_is_value) {
    bias_var = entry_builder.AddVariable(
        bias_function_ptr_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
    if (!bias_var) return 0;
  }

  // Now do all stores AFTER all variables are declared.
  // For value operands: use constant_id if it's a module constant, or the
  // function parameter ID if it's passed as an argument.
  if (input_is_value) {
    const uint32_t store_src =
        input_constant_id != 0 ? input_constant_id : input_param_id;
    if (!entry_builder.AddStore(input_var->result_id(), store_src)) return 0;
  }
  if (matrix_is_value) {
    const uint32_t store_src =
        matrix_constant_id != 0 ? matrix_constant_id : matrix_param_id;
    if (!entry_builder.AddStore(matrix_var->result_id(), store_src)) return 0;
  }
  if (has_bias && bias_is_value) {
    const uint32_t store_src =
        bias_constant_id != 0 ? bias_constant_id : bias_param_id;
    if (!entry_builder.AddStore(bias_var->result_id(), store_src)) return 0;
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
      result_packed_length_id);
  if (!out_cond) return 0;
  if (!out_header_builder.AddLoopMerge(out_merge_label_id,
                                       out_continue_label_id) ||
      !out_header_builder.AddConditionalBranch(
          out_cond->result_id(), out_body_label_id, out_merge_label_id)) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!out_body_builder.AddStore(acc_var->result_id(), zero4_id)) return 0;
  }
  if (!out_body_builder.AddStore(k_base_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k_base_load =
      k_header_builder.AddLoad(uint_type_id, k_base_var->result_id());
  if (!k_base_load) return 0;
  Instruction* k_cond = k_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, k_base_load->result_id(),
      input_packed_length_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id) ||
      !k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* k_element_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_base_load->result_id(), four_uint_id);
  if (!k_element_base) return 0;

  // Load input vector: from buffer or from constant
  uint32_t input_vec_id = 0;
  if (input_is_value) {
    // Extract from constant via OpAccessChain using packed index
    const uint32_t input_vec4_ptr_type_id = GetOrCreatePointerType(
        input.packed_vec4_type_id, spv::StorageClass::Function);
    if (input_vec4_ptr_type_id == 0) return 0;
    Instruction* input_vec_ptr = k_body_builder.AddAccessChain(
        input_vec4_ptr_type_id, input_var->result_id(),
        {k_base_load->result_id()});
    if (!input_vec_ptr) return 0;
    Instruction* input_vec = k_body_builder.AddLoad(input.packed_vec4_type_id,
                                                    input_vec_ptr->result_id());
    if (!input_vec) return 0;
    input_vec_id = input_vec->result_id();
  } else {
    // Load from buffer via helper function
    Instruction* input_vec = k_body_builder.AddFunctionCall(
        input.packed_vec4_type_id, input_load_function_id,
        {k_element_base->result_id()});
    if (!input_vec) return 0;
    input_vec_id = input_vec->result_id();
  }
  Instruction* output_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, out_pack_load->result_id(), four_uint_id);
  if (!output_base) return 0;
  const uint32_t matrix_packed_cols_id =
      GetOrCreateUIntConstant(matrix.packed_cols);
  if (matrix_packed_cols_id == 0) return 0;
  const uint32_t matrix_vec4_ptr_type_id =
      matrix_is_value ? GetOrCreatePointerType(matrix.packed_vec4_type_id,
                                               spv::StorageClass::Function)
                      : 0;
  if (matrix_is_value && matrix_vec4_ptr_type_id == 0) return 0;
  std::array<uint32_t, kPackedVec4Width> weight_row_ids = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
    const uint32_t lane_id = GetOrCreateUIntConstant(row_lane);
    if (lane_id == 0) return 0;
    Instruction* matrix_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_element_base->result_id(), lane_id);
    if (!matrix_row) return 0;

    uint32_t weight_row_id = 0;
    if (matrix_is_value) {
      // Extract from constant: index = row * packed_cols + col_pack
      Instruction* row_offset = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, matrix_row->result_id(),
          matrix_packed_cols_id);
      if (!row_offset) return 0;
      Instruction* packed_idx = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, row_offset->result_id(),
          out_pack_load->result_id());
      if (!packed_idx) return 0;
      Instruction* weight_row_ptr = k_body_builder.AddAccessChain(
          matrix_vec4_ptr_type_id, matrix_var->result_id(),
          {packed_idx->result_id()});
      if (!weight_row_ptr) return 0;
      Instruction* weight_row = k_body_builder.AddLoad(
          matrix.packed_vec4_type_id, weight_row_ptr->result_id());
      if (!weight_row) return 0;
      weight_row_id = weight_row->result_id();
    } else {
      // Load from buffer via helper function
      Instruction* matrix_row_offset =
          k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                     matrix_row->result_id(), matrix_cols_id);
      if (!matrix_row_offset) return 0;
      Instruction* matrix_local_base = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, matrix_row_offset->result_id(),
          output_base->result_id());
      if (!matrix_local_base) return 0;
      const uint32_t matrix_memory_base_id = BuildRowMajorMatrixMemoryIndex(
          &k_body_builder, nullptr, matrix_shape_id, matrix_offset_id,
          matrix.cols, matrix_local_base->result_id());
      if (matrix_memory_base_id == 0) return 0;
      Instruction* weight_row = k_body_builder.AddFunctionCall(
          matrix.packed_vec4_type_id, matrix_load_function_id,
          {matrix_memory_base_id});
      if (!weight_row) return 0;
      weight_row_id = weight_row->result_id();
    }
    weight_row_ids[row_lane] = weight_row_id;
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
    Instruction* weight = k_body_builder.AddCompositeConstruct(
        result.packed_vec4_type_id, weight_lane_ids);
    Instruction* acc = k_body_builder.AddLoad(result.packed_vec4_type_id,
                                              acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, result.packed_vec4_type_id, input_vec_id,
                 weight->result_id(), acc->result_id());
    if (fma == 0) return 0;
    if (!k_body_builder.AddStore(acc_vars[lane]->result_id(), fma)) return 0;
  }
  if (!k_body_builder.AddBranch(k_continue_label_id)) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue_block.get());
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k_base_load->result_id(), one_uint_id);
  if (!next_k) return 0;
  if (!k_continue_builder.AddStore(k_base_var->result_id(),
                                   next_k->result_id()) ||
      !k_continue_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec4_type_id,
                                               acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    if (reduced == 0) return 0;
    lane_ids.push_back(reduced);
  }
  Instruction* result_vec = k_merge_builder.AddCompositeConstruct(
      result.packed_vec4_type_id, lane_ids);
  if (!result_vec) return 0;
  if (has_bias) {
    uint32_t bias_vec_id = 0;
    if (bias_is_value) {
      // Extract bias pack from the function parameter via local variable
      Instruction* bias_vec_ptr = k_merge_builder.AddAccessChain(
          vec4_function_ptr_type_id, bias_var->result_id(),
          {out_pack_load->result_id()});
      if (!bias_vec_ptr) return 0;
      Instruction* bias_vec = k_merge_builder.AddLoad(
          bias->packed_vec4_type_id, bias_vec_ptr->result_id());
      if (!bias_vec) return 0;
      bias_vec_id = bias_vec->result_id();
    } else {
      // Load bias pack from buffer via load helper function
      Instruction* bias_base =
          k_merge_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                      out_pack_load->result_id(), four_uint_id);
      if (!bias_base) return 0;
      Instruction* bias_vec = k_merge_builder.AddFunctionCall(
          bias->packed_vec4_type_id, bias_load_function_id,
          {bias_base->result_id()});
      if (!bias_vec) return 0;
      bias_vec_id = bias_vec->result_id();
    }
    result_vec =
        k_merge_builder.AddBinaryOp(result.packed_vec4_type_id, spv::Op::OpFAdd,
                                    result_vec->result_id(), bias_vec_id);
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
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildDirectMatmulFunctionPackedVec4(
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
  if (!CanUsePackedVec4MatrixMulAdd(result, a, b, c)) return 0;
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
                       a.packed_vec4_type_id, a_memory_operands);
  const uint32_t b_load_function_id =
      b_is_value ? 0
                 : GetOrCreatePackedLoadChunkFunction(
                       b_pointer_id, b_pointer_type_id, b.component_type_id,
                       b.packed_vec4_type_id, b_memory_operands);
  const uint32_t c_load_function_id =
      c_is_value ? 0
                 : GetOrCreatePackedLoadChunkFunction(
                       c_pointer_id, c_pointer_type_id, c.component_type_id,
                       c.packed_vec4_type_id, c_memory_operands);
  std::vector<uint32_t> param_type_ids;
  for (const auto& arg : value_arguments) {
    param_type_ids.push_back(arg.second);
  }
  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, param_type_ids);
  const uint32_t lowered_function_ptr_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
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
  const uint32_t a_cols_id = GetOrCreateUIntConstant(a.cols);
  const uint32_t b_cols_id = GetOrCreateUIntConstant(b.cols);
  const uint32_t c_cols_id = GetOrCreateUIntConstant(c.cols);
  const uint32_t a_packed_cols_id = GetOrCreateUIntConstant(a.packed_cols);
  const uint32_t zero4_id = GetOrCreateZero(result.packed_vec4_type_id);
  if ((!a_is_value && a_load_function_id == 0) ||
      (!b_is_value && b_load_function_id == 0) ||
      (!c_is_value && c_load_function_id == 0) || function_type_id == 0 ||
      lowered_function_ptr_type_id == 0 || vec4_function_ptr_type_id == 0 ||
      uint_type_id == 0 || uint_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || one_uint_id == 0 ||
      four_uint_id == 0 || row_count_id == 0 || result_packed_cols_id == 0 ||
      a_cols_id == 0 || b_cols_id == 0 || c_cols_id == 0 ||
      a_packed_cols_id == 0 || zero4_id == 0) {
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
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] = entry_builder.AddVariable(
        vec4_function_ptr_type_id,
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
                                   k_pack_load->result_id(), a_packed_cols_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id) ||
      !k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* k_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_pack_load->result_id(), four_uint_id);
  if (!k_base) return 0;

  // Load A matrix tile: from buffer or from constant
  uint32_t a_vec_id = 0;
  if (a_is_value) {
    // Extract from constant: index = row * a.packed_cols + k_pack
    const uint32_t a_packed_cols_const_id =
        GetOrCreateUIntConstant(a.packed_cols);
    if (a_packed_cols_const_id == 0) return 0;
    const uint32_t a_vec4_ptr_type_id = GetOrCreatePointerType(
        a.packed_vec4_type_id, spv::StorageClass::Function);
    if (a_vec4_ptr_type_id == 0) return 0;
    Instruction* a_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_load->result_id(),
        a_packed_cols_const_id);
    if (!a_row_offset) return 0;
    Instruction* a_packed_idx = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, a_row_offset->result_id(),
        k_pack_load->result_id());
    if (!a_packed_idx) return 0;
    Instruction* a_vec_ptr = k_body_builder.AddAccessChain(
        a_vec4_ptr_type_id, a_var->result_id(), {a_packed_idx->result_id()});
    if (!a_vec_ptr) return 0;
    Instruction* a_vec =
        k_body_builder.AddLoad(a.packed_vec4_type_id, a_vec_ptr->result_id());
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
        a.packed_vec4_type_id, a_load_function_id, {a_memory_base_id});
    if (!a_vec) return 0;
    a_vec_id = a_vec->result_id();
  }

  std::array<uint32_t, kPackedVec4Width> b_vecs = {};
  const uint32_t b_packed_cols_const_id =
      b_is_value ? GetOrCreateUIntConstant(b.packed_cols) : 0;
  if (b_is_value && b_packed_cols_const_id == 0) return 0;
  const uint32_t b_vec4_ptr_type_id =
      b_is_value ? GetOrCreatePointerType(b.packed_vec4_type_id,
                                          spv::StorageClass::Function)
                 : 0;
  if (b_is_value && b_vec4_ptr_type_id == 0) return 0;
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
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
          b_vec4_ptr_type_id, b_var->result_id(), {b_packed_idx->result_id()});
      if (!b_vec_ptr) return 0;
      Instruction* b_vec =
          k_body_builder.AddLoad(b.packed_vec4_type_id, b_vec_ptr->result_id());
      if (!b_vec) return 0;
      b_vecs[row_lane] = b_vec->result_id();
    } else {
      // Load from buffer via helper function
      Instruction* b_row_offset = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, b_row->result_id(), b_cols_id);
      if (!b_row_offset) return 0;
      Instruction* b_col_offset =
          k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                     col_pack_load->result_id(), four_uint_id);
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
          b.packed_vec4_type_id, b_load_function_id, {b_memory_base_id});
      if (!b_vec) return 0;
      b_vecs[row_lane] = b_vec->result_id();
    }
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
        BuildFma(&k_body_builder, result.packed_vec4_type_id, a_vec_id,
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
    const uint32_t c_vec4_ptr_type_id = GetOrCreatePointerType(
        c.packed_vec4_type_id, spv::StorageClass::Function);
    if (c_vec4_ptr_type_id == 0) return 0;
    Instruction* c_row_offset = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_load->result_id(),
        c_packed_cols_const_id);
    if (!c_row_offset) return 0;
    Instruction* c_packed_idx = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, c_row_offset->result_id(),
        col_pack_load->result_id());
    if (!c_packed_idx) return 0;
    Instruction* c_vec_ptr = k_merge_builder.AddAccessChain(
        c_vec4_ptr_type_id, c_var->result_id(), {c_packed_idx->result_id()});
    if (!c_vec_ptr) return 0;
    Instruction* c_vec =
        k_merge_builder.AddLoad(c.packed_vec4_type_id, c_vec_ptr->result_id());
    if (!c_vec) return 0;
    c_vec_id = c_vec->result_id();
  } else {
    // Load from buffer via helper function
    Instruction* c_row_offset = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_load->result_id(), c_cols_id);
    if (!c_row_offset) return 0;
    Instruction* c_col_offset =
        k_merge_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                    col_pack_load->result_id(), four_uint_id);
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
        c.packed_vec4_type_id, c_load_function_id, {c_memory_base_id});
    if (!c_vec) return 0;
    c_vec_id = c_vec->result_id();
  }

  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec4_type_id,
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
  return function_id;
}

}  // namespace opt
}  // namespace spvtools
