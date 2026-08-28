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

namespace {
// Keep f32 unrolled helpers below the point where instruction and register
// pressure dominates, and reuse each A/input scalar across several N packs in
// the rolled helpers.
constexpr uint64_t kDirectF32VectorUnrolledMacLimit = 256;
constexpr uint64_t kDirectF32MatrixUnrolledMacLimit = 2048;
constexpr uint32_t kMaxDirectOutputPacksPerGroup = 8;

uint32_t DirectOutputPacksPerGroup(uint32_t extent,
                                   uint32_t max_packs_per_group) {
  for (uint32_t packs = max_packs_per_group; packs > 1; packs /= 2) {
    if (extent % (packs * kPackedVec2Width) == 0) return packs;
  }
  return 1;
}
}  // namespace

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
      const uint32_t weight_scalar_id =
          matrix_memory_index_id
              ? AddMemoryElementLoad(
                    &k_body_builder, nullptr, captured_matrix_pointer_id,
                    matrix_pointer_type_id, matrix.component_type_id,
                    matrix_memory_index_id, matrix_memory_operands,
                    uint_type_id)
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
  if (!CanUseDirectVectorMatrixMul(result, input, matrix, bias) ||
      (has_bias && !bias)) {
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
  const uint32_t element_index_type_id = GetOrCreateUIntType();
  if (element_index_type_id == 0) return 0;

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
    return index_id ? AddMemoryElementLoad(&builder, nullptr, pointer_id,
                                           pointer_type_id, component_type_id,
                                           index_id, memory_operands,
                                           element_index_type_id)
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
  const uint32_t matrix_vec2_type_id = GetOrCreateVectorType(
      matrix.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t result_vec2_type_id = GetOrCreateVectorType(
      result.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t result_zero_id = GetOrCreateZero(result.component_type_id);
  const uint32_t result_zero2_id = GetOrCreateZero(result_vec2_type_id);
  if (input_vec2_type_id == 0 || matrix_vec2_type_id == 0 ||
      result_vec2_type_id == 0 || result_zero_id == 0 || result_zero2_id == 0) {
    return 0;
  }
  auto widen_vec = [&](uint32_t value_id,
                       uint32_t source_component_type_id) -> uint32_t {
    if (value_id == 0 || source_component_type_id == result.component_type_id) {
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

  std::vector<uint32_t> scalar_ids(result.length, 0);
  const uint32_t full_packs = result.length / kPackedVec2Width;
  std::vector<uint32_t> packed_ids(full_packs, result_zero2_id);
  for (uint32_t pack = 0; pack < full_packs; ++pack) {
    const uint32_t col = pack * kPackedVec2Width;
    if (has_bias) {
      const uint32_t bias0 = load_bias_scalar(col);
      const uint32_t bias1 = load_bias_scalar(col + 1);
      Instruction* bias_vec = bias0 && bias1
                                  ? builder.AddCompositeConstruct(
                                        result_vec2_type_id, {bias0, bias1})
                                  : nullptr;
      if (!bias_vec) return 0;
      packed_ids[pack] = bias_vec->result_id();
    }
  }
  const uint32_t tail_col = full_packs * kPackedVec2Width;
  uint32_t tail_accumulator =
      tail_col < result.length
          ? (has_bias ? load_bias_scalar(tail_col) : result_zero_id)
          : 0;
  if (tail_col < result.length && tail_accumulator == 0) return 0;

  for (uint32_t k = 0; k < input.length; ++k) {
    const uint32_t input_id = input_scalar_ids[k];
    Instruction* input_vec = input_id
                                 ? builder.AddCompositeConstruct(
                                       input_vec2_type_id, {input_id, input_id})
                                 : nullptr;
    const uint32_t compute_input_id =
        input_vec ? widen_vec(input_vec->result_id(), input.component_type_id)
                  : 0;
    if (compute_input_id == 0) return 0;
    for (uint32_t pack = 0; pack < full_packs; ++pack) {
      const uint32_t col = pack * kPackedVec2Width;
      const uint32_t weight0 = matrix_scalar_ids[k * result.length + col];
      const uint32_t weight1 = matrix_scalar_ids[k * result.length + col + 1];
      Instruction* weight_vec =
          weight0 && weight1 ? builder.AddCompositeConstruct(
                                   matrix_vec2_type_id, {weight0, weight1})
                             : nullptr;
      const uint32_t compute_weight_id =
          weight_vec
              ? widen_vec(weight_vec->result_id(), matrix.component_type_id)
              : 0;
      if (compute_weight_id == 0) return 0;
      packed_ids[pack] =
          BuildFma(&builder, result_vec2_type_id, compute_input_id,
                   compute_weight_id, packed_ids[pack]);
      if (packed_ids[pack] == 0) return 0;
    }
    if (tail_col < result.length) {
      tail_accumulator = BuildMatmulAccumulate(
          &builder, result.component_type_id, input.component_type_id, input_id,
          matrix.component_type_id,
          matrix_scalar_ids[k * result.length + tail_col], tail_accumulator);
      if (tail_accumulator == 0) return 0;
    }
  }

  if (!IsPackedVec2(result)) {
    for (uint32_t pack = 0; pack < full_packs; ++pack) {
      const uint32_t col = pack * kPackedVec2Width;
      for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
        scalar_ids[col + lane] = ExtractCompositeElement(
            &builder, result.component_type_id, packed_ids[pack], lane);
        if (scalar_ids[col + lane] == 0) return 0;
      }
    }
    if (tail_col < result.length) scalar_ids[tail_col] = tail_accumulator;
  }

  const std::vector<uint32_t>& piece_ids =
      IsPackedVec2(result) ? packed_ids : scalar_ids;
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
  const uint64_t unrolled_limit =
      IsFloat32Type(result.component_type_id)
          ? std::min(max_unrolled_matmul_macs_,
                     kDirectF32VectorUnrolledMacLimit)
          : max_unrolled_matmul_macs_;
  if (mac_count <= unrolled_limit) {
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
  return BuildDirectVectorMatmulFunctionPackedNLoop(
      result, input, matrix, bias, has_bias, input_pointer_id,
      input_pointer_type_id, input_memory_operands, input_constant_id,
      input_is_value, matrix_pointer_id, matrix_pointer_type_id,
      matrix_shape_id, matrix_offset_id, matrix_memory_operands,
      matrix_constant_id, matrix_is_value, bias_pointer_id,
      bias_pointer_type_id, bias_memory_operands, bias_source_component_type_id,
      bias_offset, bias_conversion_fp_fast_math_mode,
      bias_conversion_has_explicit_fp_fast_math_mode, bias_constant_id,
      bias_is_value, value_arguments);
}

uint32_t HwLowerToStandardPass::BuildDirectVectorMatmulFunctionPackedNLoop(
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
  if (!CanUseDirectVectorMatrixMul(result, input, matrix, bias) ||
      (has_bias && !bias)) {
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
  if (has_bias && !bias_is_value &&
      bias_source_component_type_id != bias->component_type_id &&
      !(IsFloat16Type(bias_source_component_type_id) &&
        IsFloat32Type(bias->component_type_id))) {
    return 0;
  }

  std::vector<uint32_t> parameter_type_ids;
  if (input_is_value && input_constant_id == 0)
    parameter_type_ids.push_back(input.lowered_type_id);
  if (matrix_is_value && matrix_constant_id == 0)
    parameter_type_ids.push_back(matrix.lowered_type_id);
  if (has_bias && bias_is_value && bias_constant_id == 0)
    parameter_type_ids.push_back(bias->lowered_type_id);
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
  if (input_is_value && input_value_id == 0)
    input_value_id = add_parameter(input.lowered_type_id);
  uint32_t matrix_value_id = matrix_constant_id;
  if (matrix_is_value && matrix_value_id == 0)
    matrix_value_id = add_parameter(matrix.lowered_type_id);
  uint32_t bias_value_id = bias_constant_id;
  if (has_bias && bias_is_value && bias_value_id == 0)
    bias_value_id = add_parameter(bias->lowered_type_id);
  if ((input_is_value && input_value_id == 0) ||
      (matrix_is_value && matrix_value_id == 0) ||
      (has_bias && bias_is_value && bias_value_id == 0) ||
      parameter_index != parameter_type_ids.size()) {
    return 0;
  }

  Instruction* insertion_point =
      get_def_use_mgr()->GetDef(result.lowered_type_id);
  if (!insertion_point) return 0;
  const uint32_t input_vec2_type_id = GetOrCreateVectorType(
      input.component_type_id, kPackedVec2Width, &insertion_point);
  const uint32_t matrix_vec2_type_id = GetOrCreateVectorType(
      matrix.component_type_id, kPackedVec2Width, &insertion_point);
  const uint32_t result_vec2_type_id = GetOrCreateVectorType(
      result.component_type_id, kPackedVec2Width, &insertion_point);
  const uint32_t result_pointer_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t result_vec2_pointer_type_id =
      GetOrCreatePointerType(result_vec2_type_id, spv::StorageClass::Function);
  const uint32_t result_scalar_pointer_type_id = GetOrCreatePointerType(
      result.component_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_pointer_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t packs_per_group =
      DirectOutputPacksPerGroup(result.length, kMaxDirectOutputPacksPerGroup);
  const uint32_t group_width = packs_per_group * kPackedVec2Width;
  const uint32_t group_width_id = GetOrCreateUIntConstant(group_width);
  const uint32_t full_packs_id =
      GetOrCreateUIntConstant(result.length / group_width);
  const uint32_t input_length_id = GetOrCreateUIntConstant(input.length);
  const uint32_t matrix_cols_id = GetOrCreateUIntConstant(matrix.cols);
  const uint32_t tail_index_id = GetOrCreateUIntConstant(
      (result.length / kPackedVec2Width) * kPackedVec2Width);
  const uint32_t result_zero_id = GetOrCreateZero(result.component_type_id);
  const uint32_t result_aggregate_zero_id =
      GetOrCreateZero(result.lowered_type_id);
  const uint32_t input_value_pointer_type_id =
      input_is_value ? GetOrCreatePointerType(input.lowered_type_id,
                                              spv::StorageClass::Function)
                     : 0;
  const uint32_t matrix_value_pointer_type_id =
      matrix_is_value ? GetOrCreatePointerType(matrix.lowered_type_id,
                                               spv::StorageClass::Function)
                      : 0;
  const uint32_t bias_value_pointer_type_id =
      has_bias && bias_is_value
          ? GetOrCreatePointerType(bias->lowered_type_id,
                                   spv::StorageClass::Function)
          : 0;
  if (input_vec2_type_id == 0 || matrix_vec2_type_id == 0 ||
      result_vec2_type_id == 0 || result_pointer_type_id == 0 ||
      result_vec2_pointer_type_id == 0 || result_scalar_pointer_type_id == 0 ||
      uint_type_id == 0 || uint_pointer_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || group_width_id == 0 ||
      full_packs_id == 0 || input_length_id == 0 || matrix_cols_id == 0 ||
      tail_index_id == 0 || result_zero_id == 0 ||
      result_aggregate_zero_id == 0 ||
      (input_is_value && input_value_pointer_type_id == 0) ||
      (matrix_is_value && matrix_value_pointer_type_id == 0) ||
      (has_bias && bias_is_value && bias_value_pointer_type_id == 0)) {
    return 0;
  }

  std::array<uint32_t, 10> labels = {};
  for (uint32_t& label : labels) {
    label = TakeNextId();
    if (label == 0) return 0;
  }
  const bool has_tail = result.length % kPackedVec2Width != 0;
  std::array<uint32_t, 5> tail_labels = {};
  if (has_tail) {
    for (uint32_t& label : tail_labels) {
      label = TakeNextId();
      if (label == 0) return 0;
    }
  }
  auto make_block = [this](uint32_t label) {
    return std::unique_ptr<BasicBlock>(MakeBasicBlock(label));
  };
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  blocks.reserve(labels.size() + tail_labels.size());
  for (uint32_t label : labels) blocks.push_back(make_block(label));
  std::vector<std::unique_ptr<BasicBlock>> tail_blocks;
  if (has_tail) {
    for (uint32_t label : tail_labels) tail_blocks.push_back(make_block(label));
  }
  for (const auto& block : blocks)
    if (!block) return 0;
  for (const auto& block : tail_blocks)
    if (!block) return 0;

  BasicBlock* entry = blocks[0].get();
  BasicBlock* out_header = blocks[1].get();
  BasicBlock* out_body = blocks[2].get();
  BasicBlock* k_header = blocks[3].get();
  BasicBlock* k_body = blocks[4].get();
  BasicBlock* k_continue = blocks[5].get();
  BasicBlock* k_merge = blocks[6].get();
  BasicBlock* out_continue = blocks[7].get();
  BasicBlock* out_merge = blocks[8].get();
  BasicBlock* return_block = blocks[9].get();

  InstructionBuilder entry_builder(context(), entry);
  Instruction* result_var = entry_builder.AddVariable(
      result_pointer_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* out_var = entry_builder.AddVariable(
      uint_pointer_type_id, static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_var = entry_builder.AddVariable(
      uint_pointer_type_id, static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kMaxDirectOutputPacksPerGroup> acc_vec_vars = {};
  for (uint32_t pack = 0; pack < packs_per_group; ++pack) {
    acc_vec_vars[pack] = entry_builder.AddVariable(
        result_vec2_pointer_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
  }
  Instruction* acc_scalar_var =
      has_tail ? entry_builder.AddVariable(
                     result_scalar_pointer_type_id,
                     static_cast<uint32_t>(spv::StorageClass::Function))
               : nullptr;
  Instruction* input_var =
      input_is_value ? entry_builder.AddVariable(
                           input_value_pointer_type_id,
                           static_cast<uint32_t>(spv::StorageClass::Function))
                     : nullptr;
  Instruction* matrix_var =
      matrix_is_value ? entry_builder.AddVariable(
                            matrix_value_pointer_type_id,
                            static_cast<uint32_t>(spv::StorageClass::Function))
                      : nullptr;
  Instruction* bias_var =
      has_bias && bias_is_value
          ? entry_builder.AddVariable(
                bias_value_pointer_type_id,
                static_cast<uint32_t>(spv::StorageClass::Function))
          : nullptr;
  bool accumulators_valid = true;
  for (uint32_t pack = 0; pack < packs_per_group; ++pack)
    accumulators_valid &= acc_vec_vars[pack] != nullptr;
  if (!result_var || !out_var || !k_var || !accumulators_valid ||
      (has_tail && !acc_scalar_var) || (input_is_value && !input_var) ||
      (matrix_is_value && !matrix_var) ||
      (has_bias && bias_is_value && !bias_var)) {
    return 0;
  }

  const uint32_t captured_input =
      input_is_value ? 0
                     : BuildCapturedPointer(&entry_builder, input_pointer_id);
  const uint32_t captured_matrix =
      matrix_is_value ? 0
                      : BuildCapturedPointer(&entry_builder, matrix_pointer_id);
  const uint32_t captured_bias =
      !has_bias || bias_is_value
          ? 0
          : BuildCapturedPointer(&entry_builder, bias_pointer_id);
  if ((!input_is_value && captured_input == 0) ||
      (!matrix_is_value && captured_matrix == 0) ||
      (has_bias && !bias_is_value && captured_bias == 0)) {
    return 0;
  }
  const uint32_t matrix_shape_cols =
      matrix_is_value ? 0
                      : BuildPairComponentAsUInt(&entry_builder, nullptr,
                                                 matrix_shape_id, 1);
  const uint32_t matrix_offset_row =
      matrix_is_value ? 0
                      : BuildPairComponentAsUInt(&entry_builder, nullptr,
                                                 matrix_offset_id, 0);
  const uint32_t matrix_offset_col =
      matrix_is_value ? 0
                      : BuildPairComponentAsUInt(&entry_builder, nullptr,
                                                 matrix_offset_id, 1);
  if (!matrix_is_value && (matrix_shape_cols == 0 || matrix_offset_row == 0 ||
                           matrix_offset_col == 0)) {
    return 0;
  }
  if ((input_is_value &&
       !entry_builder.AddStore(input_var->result_id(), input_value_id)) ||
      (matrix_is_value &&
       !entry_builder.AddStore(matrix_var->result_id(), matrix_value_id)) ||
      (has_bias && bias_is_value &&
       !entry_builder.AddStore(bias_var->result_id(), bias_value_id)) ||
      !entry_builder.AddStore(result_var->result_id(),
                              result_aggregate_zero_id) ||
      !entry_builder.AddStore(out_var->result_id(), zero_uint_id) ||
      !entry_builder.AddBranch(labels[1])) {
    return 0;
  }

  auto add_offset = [this, uint_type_id](InstructionBuilder* builder,
                                         uint32_t base, uint32_t offset) {
    if (offset == 0) return base;
    const uint32_t offset_id = GetOrCreateUIntConstant(offset);
    Instruction* sum = offset_id
                           ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                  base, offset_id)
                           : nullptr;
    return sum ? sum->result_id() : 0;
  };
  auto load_buffer_scalar =
      [&](InstructionBuilder* builder, uint32_t pointer_type_id,
          uint32_t pointer_id, uint32_t component_type_id, uint32_t index_id,
          const std::vector<Operand>& memory_operands) -> uint32_t {
    return AddMemoryElementLoad(builder, nullptr, pointer_id, pointer_type_id,
                                component_type_id, index_id, memory_operands,
                                uint_type_id);
  };
  auto load_input = [&](InstructionBuilder* builder, uint32_t index_id) {
    return input_is_value
               ? BuildLogicalAggregateLoad(builder, input_var->result_id(),
                                           input.component_type_id,
                                           input.packed_vec2_type_id, index_id)
               : load_buffer_scalar(builder, input_pointer_type_id,
                                    captured_input, input.component_type_id,
                                    index_id, input_memory_operands);
  };
  auto load_matrix = [&](InstructionBuilder* builder, uint32_t row_id,
                         uint32_t col_id) -> uint32_t {
    if (matrix_is_value) {
      Instruction* row_base = builder->AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, row_id, matrix_cols_id);
      Instruction* logical_index =
          row_base ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                          row_base->result_id(), col_id)
                   : nullptr;
      if (!logical_index) return 0;
      return BuildLogicalAggregateLoad(
          builder, matrix_var->result_id(), matrix.component_type_id,
          matrix.packed_vec2_type_id, logical_index->result_id());
    }
    Instruction* global_row = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, matrix_offset_row, row_id);
    Instruction* global_col = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, matrix_offset_col, col_id);
    Instruction* row_base =
        global_row
            ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                   global_row->result_id(), matrix_shape_cols)
            : nullptr;
    Instruction* memory_index =
        row_base && global_col
            ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                   row_base->result_id(),
                                   global_col->result_id())
            : nullptr;
    return memory_index ? load_buffer_scalar(
                              builder, matrix_pointer_type_id, captured_matrix,
                              matrix.component_type_id,
                              memory_index->result_id(), matrix_memory_operands)
                        : 0;
  };
  auto load_bias = [&](InstructionBuilder* builder,
                       uint32_t index_id) -> uint32_t {
    if (!has_bias) return result_zero_id;
    if (bias_is_value) {
      return BuildLogicalAggregateLoad(builder, bias_var->result_id(),
                                       bias->component_type_id,
                                       bias->packed_vec2_type_id, index_id);
    }
    const uint32_t source_index = add_offset(builder, index_id, bias_offset);
    uint32_t value =
        source_index
            ? load_buffer_scalar(builder, bias_pointer_type_id, captured_bias,
                                 bias_source_component_type_id, source_index,
                                 bias_memory_operands)
            : 0;
    if (value == 0 ||
        bias_source_component_type_id == bias->component_type_id) {
      return value;
    }
    Instruction* converted = builder->AddUnaryOp(bias->component_type_id,
                                                 spv::Op::OpFConvert, value);
    ApplyFPFastMathMode(converted, bias_conversion_fp_fast_math_mode,
                        bias_conversion_has_explicit_fp_fast_math_mode);
    return converted ? converted->result_id() : 0;
  };
  auto widen_vec = [&](InstructionBuilder* builder, uint32_t value_id,
                       uint32_t source_component_type_id) {
    if (source_component_type_id == result.component_type_id) return value_id;
    Instruction* converted =
        builder->AddUnaryOp(result_vec2_type_id, spv::Op::OpFConvert, value_id);
    ApplyActiveFPFastMathMode(converted);
    return converted ? converted->result_id() : 0;
  };

  InstructionBuilder out_header_builder(context(), out_header);
  Instruction* out_pack =
      out_header_builder.AddLoad(uint_type_id, out_var->result_id());
  Instruction* out_cond = out_pack ? out_header_builder.AddBinaryOp(
                                         bool_type_id, spv::Op::OpULessThan,
                                         out_pack->result_id(), full_packs_id)
                                   : nullptr;
  if (!out_cond || !out_header_builder.AddLoopMerge(labels[8], labels[7]) ||
      !out_header_builder.AddConditionalBranch(out_cond->result_id(), labels[2],
                                               labels[8])) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body);
  Instruction* output_base = out_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, out_pack->result_id(), group_width_id);
  if (!output_base) return 0;
  for (uint32_t pack = 0; pack < packs_per_group; ++pack) {
    const uint32_t pack_base = add_offset(
        &out_body_builder, output_base->result_id(), pack * kPackedVec2Width);
    const uint32_t pack_lane1 = add_offset(&out_body_builder, pack_base, 1);
    const uint32_t bias0 = load_bias(&out_body_builder, pack_base);
    const uint32_t bias1 = load_bias(&out_body_builder, pack_lane1);
    Instruction* initial_acc = bias0 && bias1
                                   ? out_body_builder.AddCompositeConstruct(
                                         result_vec2_type_id, {bias0, bias1})
                                   : nullptr;
    if (!initial_acc ||
        !out_body_builder.AddStore(acc_vec_vars[pack]->result_id(),
                                   initial_acc->result_id())) {
      return 0;
    }
  }
  if (!out_body_builder.AddStore(k_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(labels[3]))
    return 0;

  InstructionBuilder k_header_builder(context(), k_header);
  Instruction* k = k_header_builder.AddLoad(uint_type_id, k_var->result_id());
  Instruction* k_cond =
      k ? k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                       k->result_id(), input_length_id)
        : nullptr;
  if (!k_cond || !k_header_builder.AddLoopMerge(labels[6], labels[5]) ||
      !k_header_builder.AddConditionalBranch(k_cond->result_id(), labels[4],
                                             labels[6])) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body);
  const uint32_t x = load_input(&k_body_builder, k->result_id());
  Instruction* x_vec =
      x ? k_body_builder.AddCompositeConstruct(input_vec2_type_id, {x, x})
        : nullptr;
  const uint32_t compute_x =
      x_vec ? widen_vec(&k_body_builder, x_vec->result_id(),
                        input.component_type_id)
            : 0;
  if (compute_x == 0) return 0;
  for (uint32_t pack = 0; pack < packs_per_group; ++pack) {
    const uint32_t pack_base = add_offset(
        &k_body_builder, output_base->result_id(), pack * kPackedVec2Width);
    const uint32_t pack_lane1 = add_offset(&k_body_builder, pack_base, 1);
    const uint32_t w0 = load_matrix(&k_body_builder, k->result_id(), pack_base);
    const uint32_t w1 =
        load_matrix(&k_body_builder, k->result_id(), pack_lane1);
    Instruction* w_vec = w0 && w1 ? k_body_builder.AddCompositeConstruct(
                                        matrix_vec2_type_id, {w0, w1})
                                  : nullptr;
    const uint32_t compute_w =
        w_vec ? widen_vec(&k_body_builder, w_vec->result_id(),
                          matrix.component_type_id)
              : 0;
    Instruction* acc = k_body_builder.AddLoad(result_vec2_type_id,
                                              acc_vec_vars[pack]->result_id());
    const uint32_t accumulated =
        acc && compute_w ? BuildFma(&k_body_builder, result_vec2_type_id,
                                    compute_x, compute_w, acc->result_id())
                         : 0;
    if (accumulated == 0 || !k_body_builder.AddStore(
                                acc_vec_vars[pack]->result_id(), accumulated)) {
      return 0;
    }
  }
  if (!k_body_builder.AddBranch(labels[5])) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue);
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k->result_id(), one_uint_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_var->result_id(), next_k->result_id()) ||
      !k_continue_builder.AddBranch(labels[3])) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge);
  for (uint32_t pack = 0; pack < packs_per_group; ++pack) {
    Instruction* result_vec = k_merge_builder.AddLoad(
        result_vec2_type_id, acc_vec_vars[pack]->result_id());
    if (!result_vec) return 0;
    if (IsPackedVec2(result)) {
      Instruction* packed_base = k_merge_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, out_pack->result_id(),
          GetOrCreateUIntConstant(packs_per_group));
      const uint32_t packed_index =
          packed_base
              ? add_offset(&k_merge_builder, packed_base->result_id(), pack)
              : 0;
      Instruction* pointer = packed_index
                                 ? k_merge_builder.AddAccessChain(
                                       result_vec2_pointer_type_id,
                                       result_var->result_id(), {packed_index})
                                 : nullptr;
      if (!pointer || !k_merge_builder.AddStore(pointer->result_id(),
                                                result_vec->result_id())) {
        return 0;
      }
    } else {
      const uint32_t pack_base = add_offset(
          &k_merge_builder, output_base->result_id(), pack * kPackedVec2Width);
      for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
        const uint32_t index = add_offset(&k_merge_builder, pack_base, lane);
        const uint32_t value =
            ExtractCompositeElement(&k_merge_builder, result.component_type_id,
                                    result_vec->result_id(), lane);
        if (!BuildLogicalAggregateStore(
                &k_merge_builder, result_var->result_id(),
                result.component_type_id, result.packed_vec2_type_id, index,
                value)) {
          return 0;
        }
      }
    }
  }
  if (!k_merge_builder.AddBranch(labels[7])) return 0;

  InstructionBuilder out_continue_builder(context(), out_continue);
  Instruction* next_out = out_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, out_pack->result_id(), one_uint_id);
  if (!next_out ||
      !out_continue_builder.AddStore(out_var->result_id(),
                                     next_out->result_id()) ||
      !out_continue_builder.AddBranch(labels[1])) {
    return 0;
  }

  InstructionBuilder out_merge_builder(context(), out_merge);
  if (!out_merge_builder.AddBranch(has_tail ? tail_labels[0] : labels[9]))
    return 0;

  if (has_tail) {
    BasicBlock* tail_entry = tail_blocks[0].get();
    BasicBlock* tail_k_header = tail_blocks[1].get();
    BasicBlock* tail_k_body = tail_blocks[2].get();
    BasicBlock* tail_k_continue = tail_blocks[3].get();
    BasicBlock* tail_k_merge = tail_blocks[4].get();
    InstructionBuilder tail_entry_builder(context(), tail_entry);
    const uint32_t tail_bias = load_bias(&tail_entry_builder, tail_index_id);
    if (tail_bias == 0 ||
        !tail_entry_builder.AddStore(acc_scalar_var->result_id(), tail_bias) ||
        !tail_entry_builder.AddStore(k_var->result_id(), zero_uint_id) ||
        !tail_entry_builder.AddBranch(tail_labels[1])) {
      return 0;
    }
    InstructionBuilder tail_header_builder(context(), tail_k_header);
    Instruction* tail_k =
        tail_header_builder.AddLoad(uint_type_id, k_var->result_id());
    Instruction* tail_cond = tail_k ? tail_header_builder.AddBinaryOp(
                                          bool_type_id, spv::Op::OpULessThan,
                                          tail_k->result_id(), input_length_id)
                                    : nullptr;
    if (!tail_cond ||
        !tail_header_builder.AddLoopMerge(tail_labels[4], tail_labels[3]) ||
        !tail_header_builder.AddConditionalBranch(
            tail_cond->result_id(), tail_labels[2], tail_labels[4])) {
      return 0;
    }
    InstructionBuilder tail_body_builder(context(), tail_k_body);
    const uint32_t tail_x = load_input(&tail_body_builder, tail_k->result_id());
    const uint32_t tail_w =
        load_matrix(&tail_body_builder, tail_k->result_id(), tail_index_id);
    Instruction* tail_acc = tail_body_builder.AddLoad(
        result.component_type_id, acc_scalar_var->result_id());
    const uint32_t tail_value =
        tail_acc && tail_x && tail_w
            ? BuildMatmulAccumulate(
                  &tail_body_builder, result.component_type_id,
                  input.component_type_id, tail_x, matrix.component_type_id,
                  tail_w, tail_acc->result_id())
            : 0;
    if (tail_value == 0 ||
        !tail_body_builder.AddStore(acc_scalar_var->result_id(), tail_value) ||
        !tail_body_builder.AddBranch(tail_labels[3])) {
      return 0;
    }
    InstructionBuilder tail_continue_builder(context(), tail_k_continue);
    Instruction* tail_next = tail_continue_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, tail_k->result_id(), one_uint_id);
    if (!tail_next ||
        !tail_continue_builder.AddStore(k_var->result_id(),
                                        tail_next->result_id()) ||
        !tail_continue_builder.AddBranch(tail_labels[1])) {
      return 0;
    }
    InstructionBuilder tail_merge_builder(context(), tail_k_merge);
    Instruction* tail_result = tail_merge_builder.AddLoad(
        result.component_type_id, acc_scalar_var->result_id());
    if (!tail_result ||
        !BuildLogicalAggregateStore(
            &tail_merge_builder, result_var->result_id(),
            result.component_type_id, result.packed_vec2_type_id, tail_index_id,
            tail_result->result_id()) ||
        !tail_merge_builder.AddBranch(labels[9])) {
      return 0;
    }
  }

  InstructionBuilder return_builder(context(), return_block);
  Instruction* result_value =
      return_builder.AddLoad(result.lowered_type_id, result_var->result_id());
  if (!result_value || !return_builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                                  result_value->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  for (size_t i = 0; i + 1 < blocks.size(); ++i)
    function->AddBasicBlock(std::move(blocks[i]));
  for (auto& block : tail_blocks) function->AddBasicBlock(std::move(block));
  function->AddBasicBlock(std::move(blocks.back()));
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
  const uint32_t element_index_type_id = GetOrCreateUIntType();
  if (element_index_type_id == 0) return 0;

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
    return index_id ? AddMemoryElementLoad(&builder, nullptr, pointer_id,
                                           pointer_type_id, component_type_id,
                                           index_id, memory_operands,
                                           element_index_type_id)
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
  const uint32_t a_vec2_type_id = GetOrCreateVectorType(
      a.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t b_vec2_type_id = GetOrCreateVectorType(
      b.component_type_id, kPackedVec2Width, &type_insertion_point);
  const uint32_t result_vec2_type_id = GetOrCreateVectorType(
      result.component_type_id, kPackedVec2Width, &type_insertion_point);
  if (a_vec2_type_id == 0 || b_vec2_type_id == 0 || result_vec2_type_id == 0) {
    return 0;
  }
  auto widen_vec = [&](uint32_t value_id,
                       uint32_t source_component_type_id) -> uint32_t {
    if (value_id == 0 || source_component_type_id == result.component_type_id) {
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
  std::vector<uint32_t> packed_ids;
  const uint32_t full_packs = result.cols / kPackedVec2Width;
  packed_ids.reserve(result.rows * full_packs);
  for (uint32_t row = 0; row < result.rows; ++row) {
    std::vector<uint32_t> row_accumulators(full_packs, 0);
    for (uint32_t pack = 0; pack < full_packs; ++pack) {
      const uint32_t col = pack * kPackedVec2Width;
      const uint32_t c0 = c_scalar_ids[MatrixFlatIndex(c, row, col)];
      const uint32_t c1 = c_scalar_ids[MatrixFlatIndex(c, row, col + 1)];
      Instruction* accumulator_vec =
          c0 && c1
              ? builder.AddCompositeConstruct(result_vec2_type_id, {c0, c1})
              : nullptr;
      if (!accumulator_vec) return 0;
      row_accumulators[pack] = accumulator_vec->result_id();
    }
    const uint32_t tail_col = full_packs * kPackedVec2Width;
    uint32_t tail_accumulator =
        tail_col < result.cols ? c_scalar_ids[MatrixFlatIndex(c, row, tail_col)]
                               : 0;
    for (uint32_t k = 0; k < a.cols; ++k) {
      const uint32_t lhs_id = a_scalar_ids[MatrixFlatIndex(a, row, k)];
      Instruction* lhs_vec = lhs_id ? builder.AddCompositeConstruct(
                                          a_vec2_type_id, {lhs_id, lhs_id})
                                    : nullptr;
      const uint32_t compute_lhs_id =
          lhs_vec ? widen_vec(lhs_vec->result_id(), a.component_type_id) : 0;
      if (compute_lhs_id == 0) return 0;
      for (uint32_t pack = 0; pack < full_packs; ++pack) {
        const uint32_t col = pack * kPackedVec2Width;
        const uint32_t rhs0 = b_scalar_ids[MatrixFlatIndex(b, k, col)];
        const uint32_t rhs1 = b_scalar_ids[MatrixFlatIndex(b, k, col + 1)];
        Instruction* rhs_vec = rhs0 && rhs1 ? builder.AddCompositeConstruct(
                                                  b_vec2_type_id, {rhs0, rhs1})
                                            : nullptr;
        const uint32_t compute_rhs_id =
            rhs_vec ? widen_vec(rhs_vec->result_id(), b.component_type_id) : 0;
        if (compute_rhs_id == 0) return 0;
        row_accumulators[pack] =
            BuildFma(&builder, result_vec2_type_id, compute_lhs_id,
                     compute_rhs_id, row_accumulators[pack]);
        if (row_accumulators[pack] == 0) return 0;
      }
      if (tail_col < result.cols) {
        tail_accumulator = BuildMatmulAccumulate(
            &builder, result.component_type_id, a.component_type_id, lhs_id,
            b.component_type_id, b_scalar_ids[MatrixFlatIndex(b, k, tail_col)],
            tail_accumulator);
        if (tail_accumulator == 0) return 0;
      }
    }
    if (IsPackedVec2(result)) {
      packed_ids.insert(packed_ids.end(), row_accumulators.begin(),
                        row_accumulators.end());
    } else {
      for (uint32_t pack = 0; pack < full_packs; ++pack) {
        const uint32_t col = pack * kPackedVec2Width;
        for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
          const uint32_t index = MatrixFlatIndex(result, row, col + lane);
          scalar_ids[index] = ExtractCompositeElement(
              &builder, result.component_type_id, row_accumulators[pack], lane);
          if (scalar_ids[index] == 0) return 0;
        }
      }
      if (tail_col < result.cols) {
        scalar_ids[MatrixFlatIndex(result, row, tail_col)] = tail_accumulator;
      }
    }
  }

  const std::vector<uint32_t>& piece_ids =
      IsPackedVec2(result) ? packed_ids : scalar_ids;
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
  const uint64_t unrolled_limit =
      IsFloat32Type(result.component_type_id)
          ? std::min(max_unrolled_matmul_macs_,
                     kDirectF32MatrixUnrolledMacLimit)
          : max_unrolled_matmul_macs_;
  if (mac_count <= unrolled_limit) {
    return BuildDirectMatrixMatmulFunctionUnrolled(
        result, a, b, c, a_pointer_id, a_pointer_type_id, a_shape_id,
        a_offset_id, a_memory_operands, a_constant_id, a_is_value, b_pointer_id,
        b_pointer_type_id, b_shape_id, b_offset_id, b_memory_operands,
        b_constant_id, b_is_value, c_pointer_id, c_pointer_type_id, c_shape_id,
        c_offset_id, c_memory_operands, c_constant_id, c_is_value,
        value_arguments);
  }
  return BuildDirectMatrixMatmulFunctionPackedNLoop(
      result, a, b, c, a_pointer_id, a_pointer_type_id, a_shape_id, a_offset_id,
      a_memory_operands, a_constant_id, a_is_value, b_pointer_id,
      b_pointer_type_id, b_shape_id, b_offset_id, b_memory_operands,
      b_constant_id, b_is_value, c_pointer_id, c_pointer_type_id, c_shape_id,
      c_offset_id, c_memory_operands, c_constant_id, c_is_value,
      value_arguments);
}

uint32_t HwLowerToStandardPass::BuildDirectMatrixMatmulFunctionPackedNLoop(
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
  if ((!a_is_value && (a_pointer_id == 0 || a_pointer_type_id == 0 ||
                       a_shape_id == 0 || a_offset_id == 0)) ||
      (!b_is_value && (b_pointer_id == 0 || b_pointer_type_id == 0 ||
                       b_shape_id == 0 || b_offset_id == 0)) ||
      (!c_is_value && (c_pointer_id == 0 || c_pointer_type_id == 0 ||
                       c_shape_id == 0 || c_offset_id == 0))) {
    return 0;
  }

  std::vector<uint32_t> parameter_type_ids;
  if (a_is_value && a_constant_id == 0)
    parameter_type_ids.push_back(a.lowered_type_id);
  if (b_is_value && b_constant_id == 0)
    parameter_type_ids.push_back(b.lowered_type_id);
  if (c_is_value && c_constant_id == 0)
    parameter_type_ids.push_back(c.lowered_type_id);
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
  if (a_is_value && a_value_id == 0)
    a_value_id = add_parameter(a.lowered_type_id);
  uint32_t b_value_id = b_constant_id;
  if (b_is_value && b_value_id == 0)
    b_value_id = add_parameter(b.lowered_type_id);
  uint32_t c_value_id = c_constant_id;
  if (c_is_value && c_value_id == 0)
    c_value_id = add_parameter(c.lowered_type_id);
  if ((a_is_value && a_value_id == 0) || (b_is_value && b_value_id == 0) ||
      (c_is_value && c_value_id == 0) ||
      parameter_index != parameter_type_ids.size()) {
    return 0;
  }

  Instruction* insertion_point =
      get_def_use_mgr()->GetDef(result.lowered_type_id);
  if (!insertion_point) return 0;
  const uint32_t a_vec2_type_id = GetOrCreateVectorType(
      a.component_type_id, kPackedVec2Width, &insertion_point);
  const uint32_t b_vec2_type_id = GetOrCreateVectorType(
      b.component_type_id, kPackedVec2Width, &insertion_point);
  const uint32_t result_vec2_type_id = GetOrCreateVectorType(
      result.component_type_id, kPackedVec2Width, &insertion_point);
  const uint32_t result_pointer_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t result_vec2_pointer_type_id =
      GetOrCreatePointerType(result_vec2_type_id, spv::StorageClass::Function);
  const uint32_t result_scalar_pointer_type_id = GetOrCreatePointerType(
      result.component_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_pointer_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t packs_per_group =
      DirectOutputPacksPerGroup(result.cols, kMaxDirectOutputPacksPerGroup);
  const uint32_t group_width = packs_per_group * kPackedVec2Width;
  const uint32_t group_width_id = GetOrCreateUIntConstant(group_width);
  const uint32_t full_packs_per_row = result.cols / group_width;
  const uint32_t pack_divisor_id =
      GetOrCreateUIntConstant(std::max(1u, full_packs_per_row));
  const uint32_t total_full_packs_id =
      GetOrCreateUIntConstant(result.rows * full_packs_per_row);
  const uint32_t row_count_id = GetOrCreateUIntConstant(result.rows);
  const uint32_t result_cols_id = GetOrCreateUIntConstant(result.cols);
  const uint32_t inner_count_id = GetOrCreateUIntConstant(a.cols);
  const uint32_t tail_col_id = GetOrCreateUIntConstant(
      (result.cols / kPackedVec2Width) * kPackedVec2Width);
  const uint32_t result_aggregate_zero_id =
      GetOrCreateZero(result.lowered_type_id);
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
  if (a_vec2_type_id == 0 || b_vec2_type_id == 0 || result_vec2_type_id == 0 ||
      result_pointer_type_id == 0 || result_vec2_pointer_type_id == 0 ||
      result_scalar_pointer_type_id == 0 || uint_type_id == 0 ||
      uint_pointer_type_id == 0 || bool_type_id == 0 || zero_uint_id == 0 ||
      one_uint_id == 0 || group_width_id == 0 || pack_divisor_id == 0 ||
      total_full_packs_id == 0 || row_count_id == 0 || result_cols_id == 0 ||
      inner_count_id == 0 || tail_col_id == 0 ||
      result_aggregate_zero_id == 0 ||
      (a_is_value && a_value_pointer_type_id == 0) ||
      (b_is_value && b_value_pointer_type_id == 0) ||
      (c_is_value && c_value_pointer_type_id == 0)) {
    return 0;
  }

  std::array<uint32_t, 10> labels = {};
  for (uint32_t& label : labels) {
    label = TakeNextId();
    if (label == 0) return 0;
  }
  const bool has_tail = result.cols % kPackedVec2Width != 0;
  std::array<uint32_t, 8> tail_labels = {};
  if (has_tail) {
    for (uint32_t& label : tail_labels) {
      label = TakeNextId();
      if (label == 0) return 0;
    }
  }
  auto make_block = [this](uint32_t label) {
    return std::unique_ptr<BasicBlock>(MakeBasicBlock(label));
  };
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  for (uint32_t label : labels) blocks.push_back(make_block(label));
  std::vector<std::unique_ptr<BasicBlock>> tail_blocks;
  if (has_tail) {
    for (uint32_t label : tail_labels) tail_blocks.push_back(make_block(label));
  }
  for (const auto& block : blocks)
    if (!block) return 0;
  for (const auto& block : tail_blocks)
    if (!block) return 0;

  BasicBlock* entry = blocks[0].get();
  BasicBlock* out_header = blocks[1].get();
  BasicBlock* out_body = blocks[2].get();
  BasicBlock* k_header = blocks[3].get();
  BasicBlock* k_body = blocks[4].get();
  BasicBlock* k_continue = blocks[5].get();
  BasicBlock* k_merge = blocks[6].get();
  BasicBlock* out_continue = blocks[7].get();
  BasicBlock* out_merge = blocks[8].get();
  BasicBlock* return_block = blocks[9].get();

  InstructionBuilder entry_builder(context(), entry);
  Instruction* result_var = entry_builder.AddVariable(
      result_pointer_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* out_var = entry_builder.AddVariable(
      uint_pointer_type_id, static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_var = entry_builder.AddVariable(
      uint_pointer_type_id, static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kMaxDirectOutputPacksPerGroup> acc_vec_vars = {};
  for (uint32_t pack = 0; pack < packs_per_group; ++pack) {
    acc_vec_vars[pack] = entry_builder.AddVariable(
        result_vec2_pointer_type_id,
        static_cast<uint32_t>(spv::StorageClass::Function));
  }
  Instruction* tail_row_var =
      has_tail ? entry_builder.AddVariable(
                     uint_pointer_type_id,
                     static_cast<uint32_t>(spv::StorageClass::Function))
               : nullptr;
  Instruction* acc_scalar_var =
      has_tail ? entry_builder.AddVariable(
                     result_scalar_pointer_type_id,
                     static_cast<uint32_t>(spv::StorageClass::Function))
               : nullptr;
  Instruction* a_var =
      a_is_value ? entry_builder.AddVariable(
                       a_value_pointer_type_id,
                       static_cast<uint32_t>(spv::StorageClass::Function))
                 : nullptr;
  Instruction* b_var =
      b_is_value ? entry_builder.AddVariable(
                       b_value_pointer_type_id,
                       static_cast<uint32_t>(spv::StorageClass::Function))
                 : nullptr;
  Instruction* c_var =
      c_is_value ? entry_builder.AddVariable(
                       c_value_pointer_type_id,
                       static_cast<uint32_t>(spv::StorageClass::Function))
                 : nullptr;
  bool accumulators_valid = true;
  for (uint32_t pack = 0; pack < packs_per_group; ++pack)
    accumulators_valid &= acc_vec_vars[pack] != nullptr;
  if (!result_var || !out_var || !k_var || !accumulators_valid ||
      (has_tail && (!tail_row_var || !acc_scalar_var)) ||
      (a_is_value && !a_var) || (b_is_value && !b_var) ||
      (c_is_value && !c_var)) {
    return 0;
  }

  const uint32_t captured_a =
      a_is_value ? 0 : BuildCapturedPointer(&entry_builder, a_pointer_id);
  const uint32_t captured_b =
      b_is_value ? 0 : BuildCapturedPointer(&entry_builder, b_pointer_id);
  const uint32_t captured_c =
      c_is_value ? 0 : BuildCapturedPointer(&entry_builder, c_pointer_id);
  if ((!a_is_value && captured_a == 0) || (!b_is_value && captured_b == 0) ||
      (!c_is_value && captured_c == 0)) {
    return 0;
  }
  struct MatrixIndexComponents {
    uint32_t shape_cols = 0;
    uint32_t offset_row = 0;
    uint32_t offset_col = 0;
  };
  auto get_index_components = [&](bool is_value, uint32_t shape_id,
                                  uint32_t offset_id) {
    MatrixIndexComponents components;
    if (!is_value) {
      components.shape_cols =
          BuildPairComponentAsUInt(&entry_builder, nullptr, shape_id, 1);
      components.offset_row =
          BuildPairComponentAsUInt(&entry_builder, nullptr, offset_id, 0);
      components.offset_col =
          BuildPairComponentAsUInt(&entry_builder, nullptr, offset_id, 1);
    }
    return components;
  };
  const MatrixIndexComponents a_index_components =
      get_index_components(a_is_value, a_shape_id, a_offset_id);
  const MatrixIndexComponents b_index_components =
      get_index_components(b_is_value, b_shape_id, b_offset_id);
  const MatrixIndexComponents c_index_components =
      get_index_components(c_is_value, c_shape_id, c_offset_id);
  auto index_components_valid = [](bool is_value,
                                   const MatrixIndexComponents& components) {
    return is_value ||
           (components.shape_cols != 0 && components.offset_row != 0 &&
            components.offset_col != 0);
  };
  if (!index_components_valid(a_is_value, a_index_components) ||
      !index_components_valid(b_is_value, b_index_components) ||
      !index_components_valid(c_is_value, c_index_components)) {
    return 0;
  }
  if ((a_is_value && !entry_builder.AddStore(a_var->result_id(), a_value_id)) ||
      (b_is_value && !entry_builder.AddStore(b_var->result_id(), b_value_id)) ||
      (c_is_value && !entry_builder.AddStore(c_var->result_id(), c_value_id)) ||
      !entry_builder.AddStore(result_var->result_id(),
                              result_aggregate_zero_id) ||
      !entry_builder.AddStore(out_var->result_id(), zero_uint_id) ||
      !entry_builder.AddBranch(labels[1])) {
    return 0;
  }

  auto add_offset = [this, uint_type_id](InstructionBuilder* builder,
                                         uint32_t base, uint32_t offset) {
    if (offset == 0) return base;
    const uint32_t offset_id = GetOrCreateUIntConstant(offset);
    Instruction* sum = offset_id
                           ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                  base, offset_id)
                           : nullptr;
    return sum ? sum->result_id() : 0;
  };
  auto load_buffer_scalar =
      [&](InstructionBuilder* builder, uint32_t pointer_type_id,
          uint32_t pointer_id, uint32_t component_type_id, uint32_t index_id,
          const std::vector<Operand>& memory_operands) -> uint32_t {
    return AddMemoryElementLoad(builder, nullptr, pointer_id, pointer_type_id,
                                component_type_id, index_id, memory_operands,
                                uint_type_id);
  };
  auto load_matrix_scalar =
      [&](InstructionBuilder* builder, const MatrixTypeInfo& info,
          bool is_value, Instruction* value_var, uint32_t pointer_type_id,
          uint32_t captured_pointer, const MatrixIndexComponents& components,
          const std::vector<Operand>& memory_operands, uint32_t logical_index,
          uint32_t row_id, uint32_t col_id) -> uint32_t {
    if (is_value) {
      return BuildLogicalAggregateLoad(builder, value_var->result_id(),
                                       info.component_type_id,
                                       info.packed_vec2_type_id, logical_index);
    }
    Instruction* global_row = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, components.offset_row, row_id);
    Instruction* global_col = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, components.offset_col, col_id);
    Instruction* row_base =
        global_row ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                          global_row->result_id(),
                                          components.shape_cols)
                   : nullptr;
    Instruction* memory_index =
        row_base && global_col
            ? builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                   row_base->result_id(),
                                   global_col->result_id())
            : nullptr;
    return memory_index
               ? load_buffer_scalar(builder, pointer_type_id, captured_pointer,
                                    info.component_type_id,
                                    memory_index->result_id(), memory_operands)
               : 0;
  };
  auto widen_vec = [&](InstructionBuilder* builder, uint32_t value_id,
                       uint32_t source_component_type_id) {
    if (source_component_type_id == result.component_type_id) return value_id;
    Instruction* converted =
        builder->AddUnaryOp(result_vec2_type_id, spv::Op::OpFConvert, value_id);
    ApplyActiveFPFastMathMode(converted);
    return converted ? converted->result_id() : 0;
  };

  InstructionBuilder out_header_builder(context(), out_header);
  Instruction* out_index =
      out_header_builder.AddLoad(uint_type_id, out_var->result_id());
  Instruction* out_cond = out_index
                              ? out_header_builder.AddBinaryOp(
                                    bool_type_id, spv::Op::OpULessThan,
                                    out_index->result_id(), total_full_packs_id)
                              : nullptr;
  if (!out_cond || !out_header_builder.AddLoopMerge(labels[8], labels[7]) ||
      !out_header_builder.AddConditionalBranch(out_cond->result_id(), labels[2],
                                               labels[8])) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body);
  Instruction* row = out_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpUDiv, out_index->result_id(), pack_divisor_id);
  Instruction* pack_col = out_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpUMod, out_index->result_id(), pack_divisor_id);
  Instruction* col = pack_col ? out_body_builder.AddBinaryOp(
                                    uint_type_id, spv::Op::OpIMul,
                                    pack_col->result_id(), group_width_id)
                              : nullptr;
  Instruction* c_row_base =
      row ? out_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                         row->result_id(), result_cols_id)
          : nullptr;
  if (!row || !col || !c_row_base) return 0;
  for (uint32_t pack = 0; pack < packs_per_group; ++pack) {
    const uint32_t pack_col_id = add_offset(&out_body_builder, col->result_id(),
                                            pack * kPackedVec2Width);
    const uint32_t pack_col1_id = add_offset(&out_body_builder, pack_col_id, 1);
    Instruction* c_index0 = out_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, c_row_base->result_id(), pack_col_id);
    const uint32_t c_index1 =
        c_index0 ? add_offset(&out_body_builder, c_index0->result_id(), 1) : 0;
    const uint32_t c0 =
        c_index0
            ? load_matrix_scalar(
                  &out_body_builder, c, c_is_value, c_var, c_pointer_type_id,
                  captured_c, c_index_components, c_memory_operands,
                  c_index0->result_id(), row->result_id(), pack_col_id)
            : 0;
    const uint32_t c1 =
        c_index1 ? load_matrix_scalar(&out_body_builder, c, c_is_value, c_var,
                                      c_pointer_type_id, captured_c,
                                      c_index_components, c_memory_operands,
                                      c_index1, row->result_id(), pack_col1_id)
                 : 0;
    Instruction* initial_acc = c0 && c1
                                   ? out_body_builder.AddCompositeConstruct(
                                         result_vec2_type_id, {c0, c1})
                                   : nullptr;
    if (!initial_acc ||
        !out_body_builder.AddStore(acc_vec_vars[pack]->result_id(),
                                   initial_acc->result_id())) {
      return 0;
    }
  }
  if (!out_body_builder.AddStore(k_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(labels[3]))
    return 0;

  InstructionBuilder k_header_builder(context(), k_header);
  Instruction* k = k_header_builder.AddLoad(uint_type_id, k_var->result_id());
  Instruction* k_cond =
      k ? k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                       k->result_id(), inner_count_id)
        : nullptr;
  if (!k_cond || !k_header_builder.AddLoopMerge(labels[6], labels[5]) ||
      !k_header_builder.AddConditionalBranch(k_cond->result_id(), labels[4],
                                             labels[6])) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body);
  Instruction* a_row_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, row->result_id(), inner_count_id);
  Instruction* a_index =
      a_row_base
          ? k_body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                       a_row_base->result_id(), k->result_id())
          : nullptr;
  const uint32_t a_scalar =
      a_index ? load_matrix_scalar(
                    &k_body_builder, a, a_is_value, a_var, a_pointer_type_id,
                    captured_a, a_index_components, a_memory_operands,
                    a_index->result_id(), row->result_id(), k->result_id())
              : 0;
  Instruction* a_vec = a_scalar ? k_body_builder.AddCompositeConstruct(
                                      a_vec2_type_id, {a_scalar, a_scalar})
                                : nullptr;
  const uint32_t compute_a =
      a_vec
          ? widen_vec(&k_body_builder, a_vec->result_id(), a.component_type_id)
          : 0;
  Instruction* b_row_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k->result_id(), result_cols_id);
  if (compute_a == 0 || !b_row_base) return 0;
  for (uint32_t pack = 0; pack < packs_per_group; ++pack) {
    const uint32_t pack_col_id =
        add_offset(&k_body_builder, col->result_id(), pack * kPackedVec2Width);
    const uint32_t pack_col1_id = add_offset(&k_body_builder, pack_col_id, 1);
    Instruction* b_index0 = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, b_row_base->result_id(), pack_col_id);
    const uint32_t b_index1 =
        b_index0 ? add_offset(&k_body_builder, b_index0->result_id(), 1) : 0;
    const uint32_t b0 =
        b_index0 ? load_matrix_scalar(
                       &k_body_builder, b, b_is_value, b_var, b_pointer_type_id,
                       captured_b, b_index_components, b_memory_operands,
                       b_index0->result_id(), k->result_id(), pack_col_id)
                 : 0;
    const uint32_t b1 =
        b_index1 ? load_matrix_scalar(&k_body_builder, b, b_is_value, b_var,
                                      b_pointer_type_id, captured_b,
                                      b_index_components, b_memory_operands,
                                      b_index1, k->result_id(), pack_col1_id)
                 : 0;
    Instruction* b_vec = b0 && b1 ? k_body_builder.AddCompositeConstruct(
                                        b_vec2_type_id, {b0, b1})
                                  : nullptr;
    const uint32_t compute_b =
        b_vec ? widen_vec(&k_body_builder, b_vec->result_id(),
                          b.component_type_id)
              : 0;
    Instruction* acc = k_body_builder.AddLoad(result_vec2_type_id,
                                              acc_vec_vars[pack]->result_id());
    const uint32_t accumulated =
        acc && compute_b ? BuildFma(&k_body_builder, result_vec2_type_id,
                                    compute_a, compute_b, acc->result_id())
                         : 0;
    if (accumulated == 0 ||
        !k_body_builder.AddStore(acc_vec_vars[pack]->result_id(), accumulated))
      return 0;
  }
  if (!k_body_builder.AddBranch(labels[5])) return 0;

  InstructionBuilder k_continue_builder(context(), k_continue);
  Instruction* next_k = k_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, k->result_id(), one_uint_id);
  if (!next_k ||
      !k_continue_builder.AddStore(k_var->result_id(), next_k->result_id()) ||
      !k_continue_builder.AddBranch(labels[3])) {
    return 0;
  }

  InstructionBuilder k_merge_builder(context(), k_merge);
  for (uint32_t pack = 0; pack < packs_per_group; ++pack) {
    Instruction* result_vec = k_merge_builder.AddLoad(
        result_vec2_type_id, acc_vec_vars[pack]->result_id());
    if (!result_vec) return 0;
    if (IsPackedVec2(result)) {
      Instruction* packed_base = k_merge_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, out_index->result_id(),
          GetOrCreateUIntConstant(packs_per_group));
      const uint32_t packed_index =
          packed_base
              ? add_offset(&k_merge_builder, packed_base->result_id(), pack)
              : 0;
      Instruction* pointer = packed_index
                                 ? k_merge_builder.AddAccessChain(
                                       result_vec2_pointer_type_id,
                                       result_var->result_id(), {packed_index})
                                 : nullptr;
      if (!pointer || !k_merge_builder.AddStore(pointer->result_id(),
                                                result_vec->result_id()))
        return 0;
    } else {
      Instruction* row_base = k_merge_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, row->result_id(), result_cols_id);
      const uint32_t pack_col_id = add_offset(
          &k_merge_builder, col->result_id(), pack * kPackedVec2Width);
      Instruction* logical_base =
          row_base
              ? k_merge_builder.AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                            row_base->result_id(), pack_col_id)
              : nullptr;
      if (!logical_base) return 0;
      for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
        const uint32_t logical_index =
            add_offset(&k_merge_builder, logical_base->result_id(), lane);
        const uint32_t value =
            ExtractCompositeElement(&k_merge_builder, result.component_type_id,
                                    result_vec->result_id(), lane);
        if (!BuildLogicalAggregateStore(
                &k_merge_builder, result_var->result_id(),
                result.component_type_id, result.packed_vec2_type_id,
                logical_index, value))
          return 0;
      }
    }
  }
  if (!k_merge_builder.AddBranch(labels[7])) return 0;

  InstructionBuilder out_continue_builder(context(), out_continue);
  Instruction* next_out = out_continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, out_index->result_id(), one_uint_id);
  if (!next_out ||
      !out_continue_builder.AddStore(out_var->result_id(),
                                     next_out->result_id()) ||
      !out_continue_builder.AddBranch(labels[1])) {
    return 0;
  }

  InstructionBuilder out_merge_builder(context(), out_merge);
  if (has_tail) {
    if (!out_merge_builder.AddStore(tail_row_var->result_id(), zero_uint_id) ||
        !out_merge_builder.AddBranch(tail_labels[0])) {
      return 0;
    }
  } else if (!out_merge_builder.AddBranch(labels[9])) {
    return 0;
  }

  if (has_tail) {
    BasicBlock* tail_header = tail_blocks[0].get();
    BasicBlock* tail_body = tail_blocks[1].get();
    BasicBlock* tail_k_header = tail_blocks[2].get();
    BasicBlock* tail_k_body = tail_blocks[3].get();
    BasicBlock* tail_k_continue = tail_blocks[4].get();
    BasicBlock* tail_k_merge = tail_blocks[5].get();
    BasicBlock* tail_continue = tail_blocks[6].get();
    BasicBlock* tail_merge = tail_blocks[7].get();
    InstructionBuilder tail_header_builder(context(), tail_header);
    Instruction* tail_row =
        tail_header_builder.AddLoad(uint_type_id, tail_row_var->result_id());
    Instruction* tail_row_cond = tail_row
                                     ? tail_header_builder.AddBinaryOp(
                                           bool_type_id, spv::Op::OpULessThan,
                                           tail_row->result_id(), row_count_id)
                                     : nullptr;
    if (!tail_row_cond ||
        !tail_header_builder.AddLoopMerge(tail_labels[7], tail_labels[6]) ||
        !tail_header_builder.AddConditionalBranch(
            tail_row_cond->result_id(), tail_labels[1], tail_labels[7])) {
      return 0;
    }
    InstructionBuilder tail_body_builder(context(), tail_body);
    Instruction* tail_row_base = tail_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, tail_row->result_id(), result_cols_id);
    Instruction* tail_result_index =
        tail_row_base ? tail_body_builder.AddBinaryOp(
                            uint_type_id, spv::Op::OpIAdd,
                            tail_row_base->result_id(), tail_col_id)
                      : nullptr;
    const uint32_t tail_c =
        tail_result_index
            ? load_matrix_scalar(&tail_body_builder, c, c_is_value, c_var,
                                 c_pointer_type_id, captured_c,
                                 c_index_components, c_memory_operands,
                                 tail_result_index->result_id(),
                                 tail_row->result_id(), tail_col_id)
            : 0;
    if (tail_c == 0 ||
        !tail_body_builder.AddStore(acc_scalar_var->result_id(), tail_c) ||
        !tail_body_builder.AddStore(k_var->result_id(), zero_uint_id) ||
        !tail_body_builder.AddBranch(tail_labels[2])) {
      return 0;
    }
    InstructionBuilder tail_k_header_builder(context(), tail_k_header);
    Instruction* tail_k =
        tail_k_header_builder.AddLoad(uint_type_id, k_var->result_id());
    Instruction* tail_k_cond = tail_k ? tail_k_header_builder.AddBinaryOp(
                                            bool_type_id, spv::Op::OpULessThan,
                                            tail_k->result_id(), inner_count_id)
                                      : nullptr;
    if (!tail_k_cond ||
        !tail_k_header_builder.AddLoopMerge(tail_labels[5], tail_labels[4]) ||
        !tail_k_header_builder.AddConditionalBranch(
            tail_k_cond->result_id(), tail_labels[3], tail_labels[5])) {
      return 0;
    }
    InstructionBuilder tail_k_body_builder(context(), tail_k_body);
    Instruction* tail_a_row_base = tail_k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, tail_row->result_id(), inner_count_id);
    Instruction* tail_a_index =
        tail_a_row_base ? tail_k_body_builder.AddBinaryOp(
                              uint_type_id, spv::Op::OpIAdd,
                              tail_a_row_base->result_id(), tail_k->result_id())
                        : nullptr;
    Instruction* tail_b_row_base = tail_k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, tail_k->result_id(), result_cols_id);
    Instruction* tail_b_index =
        tail_b_row_base ? tail_k_body_builder.AddBinaryOp(
                              uint_type_id, spv::Op::OpIAdd,
                              tail_b_row_base->result_id(), tail_col_id)
                        : nullptr;
    const uint32_t tail_a =
        tail_a_index ? load_matrix_scalar(
                           &tail_k_body_builder, a, a_is_value, a_var,
                           a_pointer_type_id, captured_a, a_index_components,
                           a_memory_operands, tail_a_index->result_id(),
                           tail_row->result_id(), tail_k->result_id())
                     : 0;
    const uint32_t tail_b =
        tail_b_index
            ? load_matrix_scalar(
                  &tail_k_body_builder, b, b_is_value, b_var, b_pointer_type_id,
                  captured_b, b_index_components, b_memory_operands,
                  tail_b_index->result_id(), tail_k->result_id(), tail_col_id)
            : 0;
    Instruction* tail_acc = tail_k_body_builder.AddLoad(
        result.component_type_id, acc_scalar_var->result_id());
    const uint32_t tail_accumulated =
        tail_acc && tail_a && tail_b
            ? BuildMatmulAccumulate(
                  &tail_k_body_builder, result.component_type_id,
                  a.component_type_id, tail_a, b.component_type_id, tail_b,
                  tail_acc->result_id())
            : 0;
    if (tail_accumulated == 0 ||
        !tail_k_body_builder.AddStore(acc_scalar_var->result_id(),
                                      tail_accumulated) ||
        !tail_k_body_builder.AddBranch(tail_labels[4])) {
      return 0;
    }
    InstructionBuilder tail_k_continue_builder(context(), tail_k_continue);
    Instruction* tail_next_k = tail_k_continue_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, tail_k->result_id(), one_uint_id);
    if (!tail_next_k ||
        !tail_k_continue_builder.AddStore(k_var->result_id(),
                                          tail_next_k->result_id()) ||
        !tail_k_continue_builder.AddBranch(tail_labels[2])) {
      return 0;
    }
    InstructionBuilder tail_k_merge_builder(context(), tail_k_merge);
    Instruction* tail_value = tail_k_merge_builder.AddLoad(
        result.component_type_id, acc_scalar_var->result_id());
    if (!tail_value ||
        !BuildLogicalAggregateStore(
            &tail_k_merge_builder, result_var->result_id(),
            result.component_type_id, result.packed_vec2_type_id,
            tail_result_index->result_id(), tail_value->result_id()) ||
        !tail_k_merge_builder.AddBranch(tail_labels[6])) {
      return 0;
    }
    InstructionBuilder tail_continue_builder(context(), tail_continue);
    Instruction* next_tail_row = tail_continue_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, tail_row->result_id(), one_uint_id);
    if (!next_tail_row ||
        !tail_continue_builder.AddStore(tail_row_var->result_id(),
                                        next_tail_row->result_id()) ||
        !tail_continue_builder.AddBranch(tail_labels[0])) {
      return 0;
    }
    InstructionBuilder tail_merge_builder(context(), tail_merge);
    if (!tail_merge_builder.AddBranch(labels[9])) return 0;
  }

  InstructionBuilder return_builder(context(), return_block);
  Instruction* result_value =
      return_builder.AddLoad(result.lowered_type_id, result_var->result_id());
  if (!result_value || !return_builder.AddUnaryOp(0, spv::Op::OpReturnValue,
                                                  result_value->result_id())) {
    return 0;
  }
  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  for (size_t i = 0; i + 1 < blocks.size(); ++i)
    function->AddBasicBlock(std::move(blocks[i]));
  for (auto& block : tail_blocks) function->AddBasicBlock(std::move(block));
  function->AddBasicBlock(std::move(blocks.back()));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);
  return function_id;
}

}  // namespace opt
}  // namespace spvtools
