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
#include <functional>
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

bool HwLowerToStandardPass::TryLowerFusedVectorMatmulStore(Instruction* inst,
                                                           bool* handled) {
  if (handled) *handled = false;
  if (!handled || !inst || inst->NumInOperands() < 2) return false;
  if (!MemoryAccessOperandsAreMovable(inst,
                                      kHwVectorStoreMemoryOperandsInIdx)) {
    return true;
  }
  uint32_t output_offset = 0;
  if (!GetConstantU32(inst->GetSingleWordInOperand(kHwVectorStoreOffsetInIdx),
                      &output_offset) ||
      output_offset != 0) {
    return true;
  }

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorStoreObjectInIdx));
  std::vector<Instruction*> object_chain;
  Instruction* matmul = TraceFunctionValueSource(object, inst, &object_chain);
  if (!matmul ||
      (matmul->opcode() != spv::Op::OpCooperativeVectorMatrixMulHW &&
       matmul->opcode() != spv::Op::OpCooperativeVectorMatrixMulAddHW)) {
    return true;
  }
  if (!MatmulAllowsReassociation(matmul)) return true;
  const bool has_bias =
      matmul->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddHW;

  Instruction* input_value = get_def_use_mgr()->GetDef(
      matmul->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_value = get_def_use_mgr()->GetDef(
      matmul->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  std::vector<Instruction*> input_chain;
  std::vector<Instruction*> matrix_chain;
  Instruction* input_load =
      TraceFunctionValueSource(input_value, matmul, &input_chain);
  Instruction* matrix_load =
      TraceFunctionValueSource(matrix_value, matmul, &matrix_chain);
  if (!input_load || !matrix_load ||
      input_load->opcode() != spv::Op::OpCooperativeVectorLoadHW ||
      matrix_load->opcode() != spv::Op::OpCooperativeMatrixLoadHW)
    return true;
  uint32_t input_offset = 0;
  if (!GetConstantU32(
          input_load->GetSingleWordInOperand(kHwVectorLoadOffsetInIdx),
          &input_offset) ||
      input_offset != 0) {
    return true;
  }

  const VectorTypeInfo* result = GetVectorType(matmul->type_id());
  const VectorTypeInfo* input = GetVectorType(input_load->type_id());
  const MatrixTypeInfo* matrix = GetMatrixTypeForValue(matrix_load);
  const VectorTypeInfo* bias = nullptr;
  Instruction* bias_source = nullptr;
  std::vector<Instruction*> bias_chain;
  uint32_t bias_constant_id = 0;
  if (has_bias) {
    Instruction* bias_value = get_def_use_mgr()->GetDef(
        matmul->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_value ? GetVectorType(bias_value->type_id()) : nullptr;
    bias_source = TraceFunctionValueSource(bias_value, matmul, &bias_chain);
    if (!bias || !bias_source ||
        bias_source->opcode() != spv::Op::OpConstantComposite ||
        bias_source->type_id() != bias_value->type_id()) {
      return true;
    }
    if (bias->packed_length > kMaxFusedConstantBiasPacks) return true;
    bias_constant_id = bias_source->result_id();
  }
  if (!result || !input || !matrix ||
      !CanUsePackedVec4VectorMatrixMul(*result, *input, *matrix, bias)) {
    return true;
  }

  uint32_t layout = 0;
  if (!GetConstantU32(
          matrix_load->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
          &layout) ||
      layout !=
          static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    return true;
  }

  const uint32_t input_pointer_id =
      input_load->GetSingleWordInOperand(kHwVectorLoadPointerInIdx);
  const uint32_t matrix_pointer_id =
      matrix_load->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
  const uint32_t matrix_shape_id =
      matrix_load->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
  const uint32_t matrix_offset_id =
      matrix_load->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
  const uint32_t output_pointer_id =
      inst->GetSingleWordInOperand(kHwVectorStorePointerInIdx);
  const uint32_t input_root_id = GetRootModulePointerId(input_pointer_id);
  const uint32_t matrix_root_id = GetRootModulePointerId(matrix_pointer_id);
  const uint32_t output_root_id = GetRootModulePointerId(output_pointer_id);
  auto root_may_alias = [this](uint32_t root_id) {
    auto* decoration_mgr = context()->get_decoration_mgr();
    return decoration_mgr &&
           (decoration_mgr->HasDecoration(root_id,
                                          uint32_t(spv::Decoration::Aliased)) ||
            decoration_mgr->HasDecoration(
                root_id, uint32_t(spv::Decoration::AliasedPointer)));
  };
  if (input_root_id == 0 || matrix_root_id == 0 || output_root_id == 0 ||
      output_root_id == input_root_id || output_root_id == matrix_root_id) {
    return true;
  }
  if (root_may_alias(input_root_id) || root_may_alias(matrix_root_id) ||
      root_may_alias(output_root_id)) {
    return true;
  }
  const uint32_t input_pointer_type_id = GetPointerTypeId(input_pointer_id);
  const uint32_t matrix_pointer_type_id = GetPointerTypeId(matrix_pointer_id);
  const uint32_t output_pointer_type_id = GetPointerTypeId(output_pointer_id);
  if (input_pointer_type_id == 0 || matrix_pointer_type_id == 0 ||
      output_pointer_type_id == 0 || !CanCapturePointer(input_pointer_id) ||
      !CanCapturePointer(matrix_pointer_id) ||
      !CanCapturePointer(output_pointer_id) ||
      !IsModuleVisibleValue(matrix_shape_id) ||
      !IsModuleVisibleValue(matrix_offset_id)) {
    return true;
  }
  if (!CanMoveLoadToUse(input_load, inst, /*function_memory=*/false,
                        kHwVectorLoadMemoryOperandsInIdx) ||
      !CanMoveLoadToUse(matrix_load, inst, /*function_memory=*/false,
                        kHwMatrixLoadMemoryOperandsInIdx)) {
    return true;
  }

  auto is_ignorable_user = [](Instruction* user) {
    return user &&
           (user->opcode() == spv::Op::OpName ||
            user->opcode() == spv::Op::OpMemberName || user->IsDecoration() ||
            user->IsNonSemanticInstruction() || user->IsDebugLineInst());
  };

  std::vector<Instruction*> kill_list;
  std::unordered_set<Instruction*> kill_set;
  auto add_kill = [&kill_list, &kill_set](Instruction* kill) {
    if (kill && kill_set.insert(kill).second) kill_list.push_back(kill);
  };
  auto add_dead_function_store_users = [this, inst, &is_ignorable_user,
                                        &add_kill,
                                        &kill_set](Instruction* value) {
    if (!value || value->result_id() == 0) return false;
    bool ok = true;
    get_def_use_mgr()->ForEachUser(value, [&](Instruction* user) {
      if (!ok || !user || user == inst || is_ignorable_user(user) ||
          kill_set.find(user) != kill_set.end()) {
        return;
      }
      if (user->opcode() != spv::Op::OpStore || user->NumInOperands() < 2 ||
          user->GetSingleWordInOperand(1) != value->result_id()) {
        ok = false;
        return;
      }
      const uint32_t pointer_id = user->GetSingleWordInOperand(0);
      if (!IsFunctionPointer(pointer_id)) {
        ok = false;
        return;
      }
      Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
      if (!pointer) {
        ok = false;
        return;
      }
      bool only_dead_users = true;
      get_def_use_mgr()->ForEachUser(pointer, [&](Instruction* pointer_user) {
        if (pointer_user == user || is_ignorable_user(pointer_user) ||
            kill_set.find(pointer_user) != kill_set.end()) {
          return;
        }
        only_dead_users = false;
      });
      if (!only_dead_users) {
        ok = false;
        return;
      }
      add_kill(user);
      add_kill(pointer);
    });
    return ok;
  };
  add_kill(inst);
  for (Instruction* kill : object_chain) add_kill(kill);
  for (Instruction* kill : input_chain) add_kill(kill);
  for (Instruction* kill : matrix_chain) add_kill(kill);
  for (Instruction* kill : bias_chain) add_kill(kill);
  add_kill(matmul);
  add_kill(input_load);
  add_kill(matrix_load);
  if (!add_dead_function_store_users(matmul) ||
      !add_dead_function_store_users(input_load) ||
      !add_dead_function_store_users(matrix_load) ||
      (has_bias && !add_dead_function_store_users(bias_source))) {
    return true;
  }

  for (Instruction* kill : kill_list) {
    if (!kill) continue;
    if (kill->result_id() != 0) {
      bool only_killed_users = true;
      get_def_use_mgr()->ForEachUser(kill, [&kill_set, &only_killed_users,
                                            &is_ignorable_user](
                                               Instruction* user) {
        if (!is_ignorable_user(user) && kill_set.find(user) == kill_set.end()) {
          only_killed_users = false;
        }
      });
      if (!only_killed_users) return true;
    }
    if (kill->opcode() == spv::Op::OpStore && kill->NumInOperands() >= 1) {
      const uint32_t pointer_id = kill->GetSingleWordInOperand(0);
      if (!IsFunctionPointer(pointer_id)) continue;
      Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
      if (!pointer) return true;
      bool only_killed_users = true;
      get_def_use_mgr()->ForEachUser(pointer, [&kill_set, &only_killed_users,
                                               &is_ignorable_user](
                                                  Instruction* user) {
        if (!is_ignorable_user(user) && kill_set.find(user) == kill_set.end()) {
          only_killed_users = false;
        }
      });
      if (!only_killed_users) return true;
    }
  }

  const std::vector<Operand> input_memory_operands =
      CopyMemoryOperands(input_load, kHwVectorLoadMemoryOperandsInIdx);
  const std::vector<Operand> matrix_memory_operands =
      CopyMemoryOperands(matrix_load, kHwMatrixLoadMemoryOperandsInIdx);
  const std::vector<Operand> output_memory_operands =
      CopyMemoryOperands(inst, kHwVectorStoreMemoryOperandsInIdx);
  const uint32_t function_id =
      has_bias
          ? BuildFusedVectorMatmulAddStoreFunctionPackedVec4(
                *result, *input, *matrix, bias, has_bias, bias_constant_id,
                input_pointer_id, input_pointer_type_id, input_memory_operands,
                matrix_pointer_id, matrix_pointer_type_id, matrix_shape_id,
                matrix_offset_id, matrix_memory_operands, output_pointer_id,
                output_pointer_type_id, output_memory_operands)
          : BuildFusedVectorMatmulStoreFunctionPackedVec4(
                *result, *input, *matrix, input_pointer_id,
                input_pointer_type_id, input_memory_operands, matrix_pointer_id,
                matrix_pointer_type_id, matrix_shape_id, matrix_offset_id,
                matrix_memory_operands, output_pointer_id,
                output_pointer_type_id, output_memory_operands);
  if (function_id == 0) return false;

  const uint32_t void_type_id = GetOrCreateVoidType();
  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!builder.AddFunctionCall(void_type_id, function_id, {})) return false;

  for (Instruction* kill : kill_list) {
    if (!kill->IsNop()) context()->KillInst(kill);
  }
  *handled = true;
  return true;
}

bool HwLowerToStandardPass::TryLowerFusedMatrixMatmulStore(Instruction* inst,
                                                           bool* handled) {
  if (handled) *handled = false;
  if (!handled || !inst ||
      inst->opcode() != spv::Op::OpCooperativeMatrixStoreHW ||
      inst->NumInOperands() < 5) {
    return false;
  }
  if (!MemoryAccessOperandsAreMovable(inst,
                                      kHwMatrixStoreMemoryOperandsInIdx)) {
    return true;
  }

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixStoreObjectInIdx));
  std::vector<Instruction*> object_chain;
  Instruction* matmul = TraceFunctionValueSource(object, inst, &object_chain);
  if (!matmul || matmul->opcode() != spv::Op::OpCooperativeMatrixMulAddHW) {
    return true;
  }
  if (!MatmulAllowsReassociation(matmul)) return true;

  Instruction* a_value = get_def_use_mgr()->GetDef(
      matmul->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_value = get_def_use_mgr()->GetDef(
      matmul->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_value = get_def_use_mgr()->GetDef(
      matmul->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  std::vector<Instruction*> a_chain;
  std::vector<Instruction*> b_chain;
  std::vector<Instruction*> c_chain;
  Instruction* a_load = TraceFunctionValueSource(a_value, matmul, &a_chain);
  Instruction* b_load = TraceFunctionValueSource(b_value, matmul, &b_chain);
  Instruction* c_load = TraceFunctionValueSource(c_value, matmul, &c_chain);
  if (!a_load || !b_load || !c_load ||
      a_load->opcode() != spv::Op::OpCooperativeMatrixLoadHW ||
      b_load->opcode() != spv::Op::OpCooperativeMatrixLoadHW ||
      c_load->opcode() != spv::Op::OpCooperativeMatrixLoadHW) {
    return true;
  }

  const MatrixTypeInfo* result = GetMatrixTypeForValue(matmul);
  const MatrixTypeInfo* a = GetMatrixTypeForValue(a_load);
  const MatrixTypeInfo* b = GetMatrixTypeForValue(b_load);
  const MatrixTypeInfo* c = GetMatrixTypeForValue(c_load);
  if (!result || !a || !b || !c ||
      !CanUsePackedVec4MatrixMulAdd(*result, *a, *b, *c)) {
    return true;
  }

  uint32_t a_layout = 0, b_layout = 0, c_layout = 0, output_layout = 0;
  if (!GetConstantU32(a_load->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
                      &a_layout) ||
      !GetConstantU32(b_load->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
                      &b_layout) ||
      !GetConstantU32(c_load->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
                      &c_layout) ||
      !GetConstantU32(inst->GetSingleWordInOperand(kHwMatrixStoreLayoutInIdx),
                      &output_layout) ||
      a_layout !=
          static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR) ||
      b_layout !=
          static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR) ||
      c_layout !=
          static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR) ||
      output_layout !=
          static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    return true;
  }

  const uint32_t a_pointer_id =
      a_load->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
  const uint32_t b_pointer_id =
      b_load->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
  const uint32_t c_pointer_id =
      c_load->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
  const uint32_t output_pointer_id =
      inst->GetSingleWordInOperand(kHwMatrixStorePointerInIdx);
  const uint32_t a_shape_id =
      a_load->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
  const uint32_t a_offset_id =
      a_load->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
  const uint32_t b_shape_id =
      b_load->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
  const uint32_t b_offset_id =
      b_load->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
  const uint32_t c_shape_id =
      c_load->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
  const uint32_t c_offset_id =
      c_load->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
  const uint32_t output_shape_id =
      inst->GetSingleWordInOperand(kHwMatrixStoreShapeInIdx);
  const uint32_t output_offset_id =
      inst->GetSingleWordInOperand(kHwMatrixStoreOffsetInIdx);
  const uint32_t a_pointer_type_id = GetPointerTypeId(a_pointer_id);
  const uint32_t b_pointer_type_id = GetPointerTypeId(b_pointer_id);
  const uint32_t c_pointer_type_id = GetPointerTypeId(c_pointer_id);
  const uint32_t output_pointer_type_id = GetPointerTypeId(output_pointer_id);
  if (a_pointer_type_id == 0 || b_pointer_type_id == 0 ||
      c_pointer_type_id == 0 || output_pointer_type_id == 0 ||
      !CanCapturePointer(a_pointer_id) || !CanCapturePointer(b_pointer_id) ||
      !CanCapturePointer(c_pointer_id) ||
      !CanCapturePointer(output_pointer_id) ||
      !IsModuleVisibleValue(a_shape_id) || !IsModuleVisibleValue(a_offset_id) ||
      !IsModuleVisibleValue(b_shape_id) || !IsModuleVisibleValue(b_offset_id) ||
      !IsModuleVisibleValue(c_shape_id) || !IsModuleVisibleValue(c_offset_id) ||
      !IsModuleVisibleValue(output_shape_id) ||
      !IsModuleVisibleValue(output_offset_id)) {
    return true;
  }
  if (!CanMoveLoadToUse(a_load, inst, /*function_memory=*/false,
                        kHwMatrixLoadMemoryOperandsInIdx) ||
      !CanMoveLoadToUse(b_load, inst, /*function_memory=*/false,
                        kHwMatrixLoadMemoryOperandsInIdx) ||
      !CanMoveLoadToUse(c_load, inst, /*function_memory=*/false,
                        kHwMatrixLoadMemoryOperandsInIdx)) {
    return true;
  }

  auto is_ignorable_user = [](Instruction* user) {
    return user &&
           (user->opcode() == spv::Op::OpName ||
            user->opcode() == spv::Op::OpMemberName || user->IsDecoration() ||
            user->IsNonSemanticInstruction() || user->IsDebugLineInst());
  };

  std::vector<Instruction*> kill_list;
  std::unordered_set<Instruction*> kill_set;
  auto add_kill = [&kill_list, &kill_set](Instruction* kill) {
    if (kill && kill_set.insert(kill).second) kill_list.push_back(kill);
  };
  auto add_dead_function_store_users = [this, inst, &is_ignorable_user,
                                        &add_kill,
                                        &kill_set](Instruction* value) {
    if (!value || value->result_id() == 0) return false;
    bool ok = true;
    get_def_use_mgr()->ForEachUser(value, [&](Instruction* user) {
      if (!ok || !user || user == inst || is_ignorable_user(user) ||
          kill_set.find(user) != kill_set.end()) {
        return;
      }
      if (user->opcode() != spv::Op::OpStore || user->NumInOperands() < 2 ||
          user->GetSingleWordInOperand(1) != value->result_id()) {
        ok = false;
        return;
      }
      const uint32_t pointer_id = user->GetSingleWordInOperand(0);
      if (!IsFunctionPointer(pointer_id)) {
        ok = false;
        return;
      }
      Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
      if (!pointer) {
        ok = false;
        return;
      }
      bool only_dead_users = true;
      get_def_use_mgr()->ForEachUser(pointer, [&](Instruction* pointer_user) {
        if (pointer_user == user || is_ignorable_user(pointer_user) ||
            kill_set.find(pointer_user) != kill_set.end()) {
          return;
        }
        only_dead_users = false;
      });
      if (!only_dead_users) {
        ok = false;
        return;
      }
      add_kill(user);
      add_kill(pointer);
    });
    return ok;
  };
  add_kill(inst);
  for (Instruction* kill : object_chain) add_kill(kill);
  for (Instruction* kill : a_chain) add_kill(kill);
  for (Instruction* kill : b_chain) add_kill(kill);
  for (Instruction* kill : c_chain) add_kill(kill);
  add_kill(matmul);
  add_kill(a_load);
  add_kill(b_load);
  add_kill(c_load);
  if (!add_dead_function_store_users(matmul) ||
      !add_dead_function_store_users(a_load) ||
      !add_dead_function_store_users(b_load) ||
      !add_dead_function_store_users(c_load)) {
    return true;
  }

  for (Instruction* kill : kill_list) {
    if (!kill) continue;
    if (kill->result_id() != 0) {
      bool only_killed_users = true;
      get_def_use_mgr()->ForEachUser(kill, [&kill_set, &only_killed_users,
                                            &is_ignorable_user](
                                               Instruction* user) {
        if (!is_ignorable_user(user) && kill_set.find(user) == kill_set.end()) {
          only_killed_users = false;
        }
      });
      if (!only_killed_users) return true;
    }
    if (kill->opcode() == spv::Op::OpStore && kill->NumInOperands() >= 1) {
      const uint32_t pointer_id = kill->GetSingleWordInOperand(0);
      if (!IsFunctionPointer(pointer_id)) continue;
      Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
      if (!pointer) return true;
      bool only_killed_users = true;
      get_def_use_mgr()->ForEachUser(pointer, [&kill_set, &only_killed_users,
                                               &is_ignorable_user](
                                                  Instruction* user) {
        if (!is_ignorable_user(user) && kill_set.find(user) == kill_set.end()) {
          only_killed_users = false;
        }
      });
      if (!only_killed_users) return true;
    }
  }

  const uint32_t function_id = BuildFusedMatrixMatmulStoreFunctionPackedVec4(
      *result, *a, *b, *c, a_pointer_id, a_pointer_type_id, a_shape_id,
      a_offset_id, CopyMemoryOperands(a_load, kHwMatrixLoadMemoryOperandsInIdx),
      b_pointer_id, b_pointer_type_id, b_shape_id, b_offset_id,
      CopyMemoryOperands(b_load, kHwMatrixLoadMemoryOperandsInIdx),
      c_pointer_id, c_pointer_type_id, c_shape_id, c_offset_id,
      CopyMemoryOperands(c_load, kHwMatrixLoadMemoryOperandsInIdx),
      output_pointer_id, output_pointer_type_id, output_shape_id,
      output_offset_id,
      CopyMemoryOperands(inst, kHwMatrixStoreMemoryOperandsInIdx));
  if (function_id == 0) return false;

  const uint32_t void_type_id = GetOrCreateVoidType();
  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!builder.AddFunctionCall(void_type_id, function_id, {})) return false;

  for (Instruction* kill : kill_list) {
    if (!kill->IsNop()) context()->KillInst(kill);
  }
  *handled = true;
  return true;
}

bool HwLowerToStandardPass::ResolveDirectVectorLoad(
    Instruction* value_inst, Instruction* use, DirectLoadSource* source) const {
  if (!source) return false;
  *source = {};

  Instruction* load = TraceFunctionValueSource(value_inst, use, &source->chain);
  if (!load || load->opcode() != spv::Op::OpCooperativeVectorLoadHW) {
    return false;
  }

  source->offset_id = load->GetSingleWordInOperand(kHwVectorLoadOffsetInIdx);
  uint32_t offset = 0;
  if (!GetConstantU32(source->offset_id, &offset) || offset != 0) {
    return false;
  }

  source->source_load = load;
  source->pointer_id = load->GetSingleWordInOperand(kHwVectorLoadPointerInIdx);
  source->pointer_type_id = GetPointerTypeId(source->pointer_id);
  source->memory_operands =
      CopyMemoryOperands(load, kHwVectorLoadMemoryOperandsInIdx);
  return source->pointer_type_id != 0 &&
         CanCapturePointer(source->pointer_id) &&
         CanMoveLoadToUse(load, use, /*function_memory=*/false,
                          kHwVectorLoadMemoryOperandsInIdx);
}

bool HwLowerToStandardPass::ResolveDirectMatrixLoad(
    Instruction* value_inst, Instruction* use, DirectLoadSource* source) const {
  if (!source) return false;
  *source = {};

  Instruction* load = TraceFunctionValueSource(value_inst, use, &source->chain);
  if (!load || load->opcode() != spv::Op::OpCooperativeMatrixLoadHW) {
    return false;
  }
  if (!GetConstantU32(load->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
                      &source->layout) ||
      source->layout !=
          static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    return false;
  }

  source->source_load = load;
  source->pointer_id = load->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
  source->pointer_type_id = GetPointerTypeId(source->pointer_id);
  source->shape_id = load->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
  source->offset_id = load->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
  source->memory_operands =
      CopyMemoryOperands(load, kHwMatrixLoadMemoryOperandsInIdx);
  return source->pointer_type_id != 0 &&
         CanCapturePointer(source->pointer_id) &&
         IsModuleVisibleValue(source->shape_id) &&
         IsModuleVisibleValue(source->offset_id) &&
         CanMoveLoadToUse(load, use, /*function_memory=*/false,
                          kHwMatrixLoadMemoryOperandsInIdx);
}

bool HwLowerToStandardPass::DirectKillListUsersAreClosed(
    Instruction* current_inst,
    const std::vector<Instruction*>& kill_list) const {
  std::unordered_set<Instruction*> allowed(kill_list.begin(), kill_list.end());
  allowed.insert(current_inst);
  for (Instruction* kill : kill_list) {
    if (!kill) continue;
    if (kill->result_id() != 0) {
      bool only_allowed_users = true;
      get_def_use_mgr()->ForEachUser(
          kill, [this, kill, &allowed, &only_allowed_users](Instruction* user) {
            if (IsFunctionPointer(kill->result_id()) &&
                user->opcode() == spv::Op::OpStore &&
                user->NumInOperands() >= 1 &&
                user->GetSingleWordInOperand(0) == kill->result_id()) {
              return;
            }
            if (!IsIgnorableDirectUser(user) &&
                allowed.find(user) == allowed.end()) {
              only_allowed_users = false;
            }
          });
      if (!only_allowed_users) return false;
    }
    if (kill->opcode() != spv::Op::OpStore || kill->NumInOperands() < 1) {
      continue;
    }

    const uint32_t pointer_id = kill->GetSingleWordInOperand(0);
    if (!IsFunctionPointer(pointer_id)) continue;
    Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
    if (!pointer) return false;
    bool only_allowed_users = true;
    get_def_use_mgr()->ForEachUser(pointer, [this, pointer, &allowed,
                                             &only_allowed_users](
                                                Instruction* user) {
      if (user->opcode() == spv::Op::OpStore && user->NumInOperands() >= 1 &&
          user->GetSingleWordInOperand(0) == pointer->result_id()) {
        return;
      }
      if (!IsIgnorableDirectUser(user) && allowed.find(user) == allowed.end()) {
        only_allowed_users = false;
      }
    });
    if (!only_allowed_users) return false;
  }
  return true;
}

bool HwLowerToStandardPass::TryLowerDirectMatrixMulAddPackedVec4(
    Instruction* inst, bool* handled) {
  if (handled) *handled = false;
  if (!handled || !inst ||
      inst->opcode() != spv::Op::OpCooperativeMatrixMulAddHW) {
    return false;
  }
  if (!MatmulAllowsReassociation(inst)) return true;

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
  if (!result || !a || !b || !c ||
      !CanUsePackedVec4MatrixMulAdd(*result, *a, *b, *c)) {
    return true;
  }

  DirectLoadSource direct_a;
  DirectLoadSource direct_b;
  DirectLoadSource direct_c;
  std::vector<std::pair<uint32_t, uint32_t>> value_arguments;

  // Resolve A: buffer load or constant
  bool a_is_value = false;
  uint32_t a_constant_id = 0;
  uint32_t a_value_id = a_inst ? a_inst->result_id() : 0;
  if (ResolveDirectMatrixLoad(a_inst, inst, &direct_a)) {
    a_is_value = false;
  } else {
    std::vector<Instruction*> a_chain;
    Instruction* a_source = TraceFunctionValueSource(a_inst, inst, &a_chain);
    if (a_source && a_source->opcode() == spv::Op::OpConstantComposite) {
      a_is_value = true;
      a_constant_id = a_source->result_id();
      a_inst = a_source;
    } else if (a_source &&
               a_source->opcode() == spv::Op::OpCompositeConstruct) {
      // Try to convert to module constant if all operands are constants
      const uint32_t const_id =
          GetOrCreateModuleConstantFromCompositeConstruct(a_source);
      if (const_id != 0) {
        a_is_value = true;
        a_constant_id = const_id;
        // Keep a_inst as the original for type info, but use const_id for
        // access
      } else {
        // Not all constant operands — pass as value argument.
        a_is_value = true;
        Instruction* a_value = a_source;
        if (!a_value || a_value->result_id() == 0) return true;
        a_value_id = a_value->result_id();
        value_arguments.push_back({a_value_id, a->lowered_type_id});
      }
    } else {
      a_is_value = true;
      Instruction* a_value = a_source ? a_source : a_inst;
      if (!a_value || a_value->result_id() == 0) return true;
      a_value_id = a_value->result_id();
      value_arguments.push_back({a_value_id, a->lowered_type_id});
    }
  }

  // Resolve B: buffer load or constant
  bool b_is_value = false;
  uint32_t b_constant_id = 0;
  uint32_t b_value_id = b_inst ? b_inst->result_id() : 0;
  if (ResolveDirectMatrixLoad(b_inst, inst, &direct_b)) {
    b_is_value = false;
  } else {
    std::vector<Instruction*> b_chain;
    Instruction* b_source = TraceFunctionValueSource(b_inst, inst, &b_chain);
    if (b_source && b_source->opcode() == spv::Op::OpConstantComposite) {
      b_is_value = true;
      b_constant_id = b_source->result_id();
      b_inst = b_source;
    } else if (b_source &&
               b_source->opcode() == spv::Op::OpCompositeConstruct) {
      const uint32_t const_id =
          GetOrCreateModuleConstantFromCompositeConstruct(b_source);
      if (const_id != 0) {
        b_is_value = true;
        b_constant_id = const_id;
      } else {
        b_is_value = true;
        Instruction* b_value = b_source;
        if (!b_value || b_value->result_id() == 0) return true;
        b_value_id = b_value->result_id();
        value_arguments.push_back({b_value_id, b->lowered_type_id});
      }
    } else {
      b_is_value = true;
      Instruction* b_value = b_source ? b_source : b_inst;
      if (!b_value || b_value->result_id() == 0) return true;
      b_value_id = b_value->result_id();
      value_arguments.push_back({b_value_id, b->lowered_type_id});
    }
  }

  // Resolve C: buffer load or constant
  bool c_is_value = false;
  uint32_t c_constant_id = 0;
  uint32_t c_value_id = c_inst ? c_inst->result_id() : 0;
  if (ResolveDirectMatrixLoad(c_inst, inst, &direct_c)) {
    c_is_value = false;
  } else {
    std::vector<Instruction*> c_chain;
    Instruction* c_source = TraceFunctionValueSource(c_inst, inst, &c_chain);
    if (c_source && c_source->opcode() == spv::Op::OpConstantComposite) {
      c_is_value = true;
      c_constant_id = c_source->result_id();
      c_inst = c_source;
    } else if (c_source &&
               c_source->opcode() == spv::Op::OpCompositeConstruct) {
      const uint32_t const_id =
          GetOrCreateModuleConstantFromCompositeConstruct(c_source);
      if (const_id != 0) {
        c_is_value = true;
        c_constant_id = const_id;
      } else {
        c_is_value = true;
        Instruction* c_value = c_source;
        if (!c_value || c_value->result_id() == 0) return true;
        c_value_id = c_value->result_id();
        value_arguments.push_back({c_value_id, c->lowered_type_id});
      }
    } else {
      c_is_value = true;
      Instruction* c_value = c_source ? c_source : c_inst;
      if (!c_value || c_value->result_id() == 0) return true;
      c_value_id = c_value->result_id();
      value_arguments.push_back({c_value_id, c->lowered_type_id});
    }
  }

  std::vector<Instruction*> kill_list;
  std::unordered_set<Instruction*> kill_set;
  auto add_kill = [&kill_list, &kill_set](Instruction* kill) {
    if (kill && kill_set.insert(kill).second) kill_list.push_back(kill);
  };
  auto remove_kill = [&kill_list, &kill_set](Instruction* kill) {
    if (!kill) return;
    kill_set.erase(kill);
    kill_list.erase(std::remove(kill_list.begin(), kill_list.end(), kill),
                    kill_list.end());
  };
  auto keep_direct_chain_alive =
      [this, &remove_kill](const std::vector<Instruction*>& chain) {
        for (Instruction* chain_inst : chain) {
          remove_kill(chain_inst);
          if (chain_inst->opcode() != spv::Op::OpStore ||
              chain_inst->NumInOperands() < 1) {
            continue;
          }
          const uint32_t pointer_id = chain_inst->GetSingleWordInOperand(0);
          if (!IsFunctionPointer(pointer_id)) continue;
          remove_kill(get_def_use_mgr()->GetDef(pointer_id));
        }
      };
  auto has_live_users_outside_kill = [this, inst,
                                      &kill_set](Instruction* source_load) {
    if (!source_load || source_load->result_id() == 0) return false;
    bool has_live_users = false;
    get_def_use_mgr()->ForEachUser(source_load, [&](Instruction* user) {
      if (!user || user == inst || IsIgnorableDirectUser(user) ||
          kill_set.find(user) != kill_set.end()) {
        return;
      }
      has_live_users = true;
    });
    return has_live_users;
  };
  // Only add kills for non-constant operands
  if (!a_is_value) {
    for (Instruction* kill : direct_a.chain) add_kill(kill);
    add_kill(direct_a.source_load);
    if (!AddSharedDirectSourceKillsOrCheckSafe(inst, direct_a.source_load,
                                               &kill_list, &kill_set)) {
      return true;
    }
  }
  if (!b_is_value) {
    for (Instruction* kill : direct_b.chain) add_kill(kill);
    add_kill(direct_b.source_load);
    if (!AddSharedDirectSourceKillsOrCheckSafe(inst, direct_b.source_load,
                                               &kill_list, &kill_set)) {
      return true;
    }
  }
  if (!c_is_value) {
    for (Instruction* kill : direct_c.chain) add_kill(kill);
    add_kill(direct_c.source_load);
    if (!AddSharedDirectSourceKillsOrCheckSafe(inst, direct_c.source_load,
                                               &kill_list, &kill_set)) {
      return true;
    }
  }
  if (!a_is_value && HasLiveSafeSharedDirectSourceUsers(
                         inst, direct_a.source_load, kill_set)) {
    remove_kill(direct_a.source_load);
    KeepSharedDirectSourceAlive(inst, direct_a.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_a.chain);
  }
  if (!b_is_value && HasLiveSafeSharedDirectSourceUsers(
                         inst, direct_b.source_load, kill_set)) {
    remove_kill(direct_b.source_load);
    KeepSharedDirectSourceAlive(inst, direct_b.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_b.chain);
  }
  if (!c_is_value && HasLiveSafeSharedDirectSourceUsers(
                         inst, direct_c.source_load, kill_set)) {
    remove_kill(direct_c.source_load);
    KeepSharedDirectSourceAlive(inst, direct_c.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_c.chain);
  }
  if (!a_is_value && has_live_users_outside_kill(direct_a.source_load)) {
    remove_kill(direct_a.source_load);
  }
  if (!b_is_value && has_live_users_outside_kill(direct_b.source_load)) {
    remove_kill(direct_b.source_load);
  }
  if (!c_is_value && has_live_users_outside_kill(direct_c.source_load)) {
    remove_kill(direct_c.source_load);
  }
  if (!DirectKillListUsersAreClosed(inst, kill_list)) return true;

  const uint32_t function_id = BuildDirectMatmulFunctionPackedVec4(
      *result, *a, *b, *c, a_is_value ? 0 : direct_a.pointer_id,
      a_is_value ? 0 : direct_a.pointer_type_id,
      a_is_value ? 0 : direct_a.shape_id, a_is_value ? 0 : direct_a.offset_id,
      a_is_value ? std::vector<Operand>{} : direct_a.memory_operands,
      a_constant_id, a_is_value, b_is_value ? 0 : direct_b.pointer_id,
      b_is_value ? 0 : direct_b.pointer_type_id,
      b_is_value ? 0 : direct_b.shape_id, b_is_value ? 0 : direct_b.offset_id,
      b_is_value ? std::vector<Operand>{} : direct_b.memory_operands,
      b_constant_id, b_is_value, c_is_value ? 0 : direct_c.pointer_id,
      c_is_value ? 0 : direct_c.pointer_type_id,
      c_is_value ? 0 : direct_c.shape_id, c_is_value ? 0 : direct_c.offset_id,
      c_is_value ? std::vector<Operand>{} : direct_c.memory_operands,
      c_constant_id, c_is_value, value_arguments);
  if (function_id == 0) return false;

  std::vector<uint32_t> call_args;
  call_args.reserve(value_arguments.size());
  for (const auto& arg : value_arguments) call_args.push_back(arg.first);
  RebuildAsFunctionCall(inst, result->lowered_type_id, function_id, call_args);
  for (Instruction* kill : kill_list) {
    if (!kill->IsNop()) context()->KillInst(kill);
  }
  *handled = true;
  return true;
}

bool HwLowerToStandardPass::TryLowerDirectVectorMatrixMulPackedVec4(
    Instruction* inst, bool has_bias, bool* handled) {
  if (handled) *handled = false;
  if (!handled || !inst ||
      (inst->opcode() != spv::Op::OpCooperativeVectorMatrixMulHW &&
       inst->opcode() != spv::Op::OpCooperativeVectorMatrixMulAddHW)) {
    return false;
  }
  if (!MatmulAllowsReassociation(inst)) return true;

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
  if (!result || !input || !matrix || (has_bias && !bias) ||
      !CanUseDirectVectorMatrixMul(*result, *input, *matrix, bias)) {
    return true;
  }

  auto is_compatible_matrix_use_bitcast = [this](Instruction* bitcast) {
    if (!bitcast || bitcast->opcode() != spv::Op::OpBitcast ||
        bitcast->NumInOperands() < 1) {
      return false;
    }
    Instruction* source =
        get_def_use_mgr()->GetDef(bitcast->GetSingleWordInOperand(0));
    const MatrixTypeInfo* source_matrix = GetMatrixTypeForValue(source);
    const MatrixTypeInfo* result_matrix = GetMatrixTypeForValue(bitcast);
    return source_matrix && result_matrix && source_matrix->has_matrix_use &&
           result_matrix->has_matrix_use &&
           source_matrix->component_type_id ==
               result_matrix->component_type_id &&
           source_matrix->rows == result_matrix->rows &&
           source_matrix->cols == result_matrix->cols &&
           source_matrix->matrix_use ==
               spv::CooperativeMatrixUseHW::MatrixUseAHW &&
           result_matrix->matrix_use ==
               spv::CooperativeMatrixUseHW::MatrixUseBHW;
  };
  auto trace_types_are_compatible = [this, inst,
                                     &is_compatible_matrix_use_bitcast](
                                        Instruction* value_inst) {
    if (!value_inst) return false;
    std::vector<Instruction*> chain;
    Instruction* source = TraceFunctionValueSource(value_inst, inst, &chain);
    bool saw_compatible_type_change = false;
    for (Instruction* chain_inst : chain) {
      if (!chain_inst || chain_inst->opcode() != spv::Op::OpBitcast ||
          chain_inst->NumInOperands() < 1) {
        continue;
      }

      Instruction* bitcast_source =
          get_def_use_mgr()->GetDef(chain_inst->GetSingleWordInOperand(0));
      if (!bitcast_source) return false;
      const MatrixTypeInfo* source_matrix =
          GetMatrixTypeForValue(bitcast_source);
      const MatrixTypeInfo* result_matrix = GetMatrixTypeForValue(chain_inst);
      if (source_matrix && result_matrix &&
          source_matrix->type_id == result_matrix->type_id) {
        continue;
      }
      if (!is_compatible_matrix_use_bitcast(chain_inst)) {
        return false;
      }
      saw_compatible_type_change = true;
    }

    if (!source) return true;
    const MatrixTypeInfo* source_matrix = GetMatrixTypeForValue(source);
    const MatrixTypeInfo* result_matrix = GetMatrixTypeForValue(value_inst);
    if (source_matrix && result_matrix &&
        source_matrix->type_id != result_matrix->type_id) {
      return saw_compatible_type_change && source_matrix->has_matrix_use &&
             result_matrix->has_matrix_use &&
             source_matrix->component_type_id ==
                 result_matrix->component_type_id &&
             source_matrix->rows == result_matrix->rows &&
             source_matrix->cols == result_matrix->cols &&
             source_matrix->matrix_use ==
                 spv::CooperativeMatrixUseHW::MatrixUseAHW &&
             result_matrix->matrix_use ==
                 spv::CooperativeMatrixUseHW::MatrixUseBHW;
    }
    return source->type_id() == value_inst->type_id();
  };
  if (!trace_types_are_compatible(input_inst) ||
      !trace_types_are_compatible(matrix_inst) ||
      (has_bias && !trace_types_are_compatible(bias_inst))) {
    return true;
  }

  DirectLoadSource direct_input;
  DirectLoadSource direct_matrix;
  DirectLoadSource direct_bias;

  // Track operands passed as function parameters (hybrid direct path).
  // Each pair: (operand_id, lowered_type_id).
  std::vector<std::pair<uint32_t, uint32_t>> value_arguments;

  // Resolve input: buffer load, constant, or parameter
  bool input_is_value = false;
  uint32_t input_constant_id = 0;
  uint32_t input_value_id = input_inst->result_id();
  if (ResolveDirectVectorLoad(input_inst, inst, &direct_input)) {
    input_is_value = false;
  } else {
    std::vector<Instruction*> input_chain;
    Instruction* input_source =
        TraceFunctionValueSource(input_inst, inst, &input_chain);
    if (input_source &&
        input_source->opcode() == spv::Op::OpConstantComposite) {
      input_is_value = true;
      input_constant_id = input_source->result_id();
    } else if (input_source &&
               input_source->opcode() == spv::Op::OpCompositeConstruct) {
      const uint32_t const_id =
          GetOrCreateModuleConstantFromCompositeConstruct(input_source);
      if (const_id != 0) {
        input_is_value = true;
        input_constant_id = const_id;
      } else {
        input_is_value = true;
        Instruction* input_value = input_source;
        if (!input_value || input_value->result_id() == 0) return true;
        input_value_id = input_value->result_id();
        value_arguments.push_back({input_value_id, input->lowered_type_id});
      }
    } else {
      // Cannot resolve to SSBO load or constant — pass as function parameter.
      input_is_value = true;
      Instruction* input_value = input_source ? input_source : input_inst;
      if (!input_value || input_value->result_id() == 0) return true;
      input_value_id = input_value->result_id();
      value_arguments.push_back({input_value_id, input->lowered_type_id});
    }
  }

  // Resolve matrix: buffer load or constant
  bool matrix_is_value = false;
  uint32_t matrix_constant_id = 0;
  uint32_t matrix_value_id = matrix_inst->result_id();
  if (ResolveDirectMatrixLoad(matrix_inst, inst, &direct_matrix)) {
    matrix_is_value = false;
  } else {
    std::vector<Instruction*> matrix_chain;
    Instruction* matrix_source =
        TraceFunctionValueSource(matrix_inst, inst, &matrix_chain);
    if (matrix_source &&
        matrix_source->opcode() == spv::Op::OpConstantComposite) {
      matrix_is_value = true;
      matrix_constant_id = matrix_source->result_id();
    } else if (matrix_source &&
               matrix_source->opcode() == spv::Op::OpCompositeConstruct) {
      const uint32_t const_id =
          GetOrCreateModuleConstantFromCompositeConstruct(matrix_source);
      if (const_id != 0) {
        matrix_is_value = true;
        matrix_constant_id = const_id;
      } else {
        matrix_is_value = true;
        Instruction* matrix_value = matrix_source;
        if (!matrix_value || matrix_value->result_id() == 0) return true;
        matrix_value_id = matrix_value->result_id();
        value_arguments.push_back({matrix_value_id, matrix->lowered_type_id});
      }
    } else {
      // Cannot resolve to SSBO load or constant — pass as function parameter.
      matrix_is_value = true;
      Instruction* matrix_value = matrix_source ? matrix_source : matrix_inst;
      if (!matrix_value || matrix_value->result_id() == 0) return true;
      matrix_value_id = matrix_value->result_id();
      value_arguments.push_back({matrix_value_id, matrix->lowered_type_id});
    }
  }

  bool bias_is_value = false;
  uint32_t bias_constant_id = 0;
  uint32_t bias_value_id = bias_inst ? bias_inst->result_id() : 0;
  if (has_bias) {
    if (ResolveDirectVectorLoad(bias_inst, inst, &direct_bias)) {
      // Bias comes from a buffer load (existing path)
      bias_is_value = false;
    } else {
      // Try to trace back to find the actual source
      std::vector<Instruction*> bias_chain;
      Instruction* bias_source =
          TraceFunctionValueSource(bias_inst, inst, &bias_chain);
      if (bias_source &&
          bias_source->opcode() == spv::Op::OpConstantComposite) {
        bias_is_value = true;
        bias_constant_id = bias_source->result_id();
      } else if (bias_source &&
                 bias_source->opcode() == spv::Op::OpCompositeConstruct) {
        const uint32_t const_id =
            GetOrCreateModuleConstantFromCompositeConstruct(bias_source);
        if (const_id != 0) {
          bias_is_value = true;
          bias_constant_id = const_id;
        } else {
          bias_is_value = true;
          Instruction* bias_value = bias_source;
          if (!bias_value || bias_value->result_id() == 0) return true;
          bias_value_id = bias_value->result_id();
          value_arguments.push_back({bias_value_id, bias->lowered_type_id});
        }
      } else {
        // Cannot resolve to SSBO load or constant — pass as function parameter.
        bias_is_value = true;
        Instruction* bias_value = bias_source ? bias_source : bias_inst;
        if (!bias_value || bias_value->result_id() == 0) return true;
        bias_value_id = bias_value->result_id();
        value_arguments.push_back({bias_value_id, bias->lowered_type_id});
      }
    }
  }

  std::vector<Instruction*> kill_list;
  std::unordered_set<Instruction*> kill_set;
  auto add_kill = [&kill_list, &kill_set](Instruction* kill) {
    if (kill && kill_set.insert(kill).second) kill_list.push_back(kill);
  };
  auto remove_kill = [&kill_list, &kill_set](Instruction* kill) {
    if (!kill) return;
    kill_set.erase(kill);
    kill_list.erase(std::remove(kill_list.begin(), kill_list.end(), kill),
                    kill_list.end());
  };
  auto keep_direct_chain_alive =
      [this, &remove_kill](const std::vector<Instruction*>& chain) {
        for (Instruction* chain_inst : chain) {
          remove_kill(chain_inst);
          if (chain_inst->opcode() != spv::Op::OpStore ||
              chain_inst->NumInOperands() < 1) {
            continue;
          }
          const uint32_t pointer_id = chain_inst->GetSingleWordInOperand(0);
          if (!IsFunctionPointer(pointer_id)) continue;
          remove_kill(get_def_use_mgr()->GetDef(pointer_id));
        }
      };
  auto has_live_users_outside_kill = [this, inst,
                                      &kill_set](Instruction* source_load) {
    if (!source_load || source_load->result_id() == 0) return false;
    bool has_live_users = false;
    get_def_use_mgr()->ForEachUser(source_load, [&](Instruction* user) {
      if (!user || user == inst || IsIgnorableDirectUser(user) ||
          kill_set.find(user) != kill_set.end()) {
        return;
      }
      has_live_users = true;
    });
    return has_live_users;
  };
  // Only add input kills if not a constant
  if (!input_is_value) {
    for (Instruction* kill : direct_input.chain) add_kill(kill);
    add_kill(direct_input.source_load);
    if (!AddSharedDirectSourceKillsOrCheckSafe(inst, direct_input.source_load,
                                               &kill_list, &kill_set)) {
      return true;
    }
  }
  // Only add matrix kills if not a constant
  if (!matrix_is_value) {
    for (Instruction* kill : direct_matrix.chain) add_kill(kill);
    add_kill(direct_matrix.source_load);
    if (!AddSharedDirectSourceKillsOrCheckSafe(inst, direct_matrix.source_load,
                                               &kill_list, &kill_set)) {
      return true;
    }
  }
  if (has_bias && !bias_is_value) {
    for (Instruction* kill : direct_bias.chain) add_kill(kill);
    add_kill(direct_bias.source_load);
    if (!AddSharedDirectSourceKillsOrCheckSafe(inst, direct_bias.source_load,
                                               &kill_list, &kill_set)) {
      return true;
    }
  }
  if (!input_is_value && HasLiveSafeSharedDirectSourceUsers(
                             inst, direct_input.source_load, kill_set)) {
    remove_kill(direct_input.source_load);
    KeepSharedDirectSourceAlive(inst, direct_input.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_input.chain);
  }
  if (!matrix_is_value && HasLiveSafeSharedDirectSourceUsers(
                              inst, direct_matrix.source_load, kill_set)) {
    remove_kill(direct_matrix.source_load);
    KeepSharedDirectSourceAlive(inst, direct_matrix.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_matrix.chain);
  }
  if (has_bias && !bias_is_value &&
      HasLiveSafeSharedDirectSourceUsers(inst, direct_bias.source_load,
                                         kill_set)) {
    remove_kill(direct_bias.source_load);
    KeepSharedDirectSourceAlive(inst, direct_bias.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_bias.chain);
  }
  if (!input_is_value &&
      has_live_users_outside_kill(direct_input.source_load)) {
    remove_kill(direct_input.source_load);
  }
  if (!matrix_is_value &&
      has_live_users_outside_kill(direct_matrix.source_load)) {
    remove_kill(direct_matrix.source_load);
  }
  if (has_bias && !bias_is_value &&
      has_live_users_outside_kill(direct_bias.source_load)) {
    remove_kill(direct_bias.source_load);
  }

  // EliminateHwFunctionVariables rewrites the matmul operand to its original
  // value, but deliberately leaves dead forward copies in place.  Frontends
  // commonly emit those copies after a MatrixUseA-to-MatrixUseB bitcast.  Add
  // an entire copy chain to the kill list only when every pointer is a
  // function-local variable with a single store and every loaded value is
  // consumed solely by another such copy or by the current matmul.
  auto add_dead_forward_copy_chain = [this, inst, &add_kill,
                                      &kill_set](Instruction* seed) {
    if (!seed || seed->result_id() == 0 ||
        IsFunctionPointer(seed->result_id())) {
      return;
    }

    std::unordered_set<Instruction*> candidates;
    std::unordered_set<const Instruction*> active_values;
    std::unordered_set<const Instruction*> proven_values;
    std::unordered_set<const Instruction*> active_pointers;
    std::unordered_set<const Instruction*> proven_pointers;
    std::unordered_map<const Instruction*, const Instruction*> pointer_stores;
    std::function<bool(Instruction*)> prove_value;
    std::function<bool(Instruction*, Instruction*)> prove_pointer;

    prove_pointer = [&](Instruction* pointer, Instruction* originating_store) {
      if (!pointer || pointer->opcode() != spv::Op::OpVariable ||
          !IsFunctionPointer(pointer->result_id())) {
        return false;
      }
      const auto* decoration_mgr = context()->get_decoration_mgr();
      if (decoration_mgr &&
          decoration_mgr->HasDecoration(pointer->result_id(),
                                        uint32_t(spv::Decoration::Volatile))) {
        return false;
      }
      auto store_it = pointer_stores.find(pointer);
      if (store_it != pointer_stores.end() &&
          store_it->second != originating_store) {
        return false;
      }
      pointer_stores[pointer] = originating_store;
      if (proven_pointers.find(pointer) != proven_pointers.end()) {
        return true;
      }
      if (!active_pointers.insert(pointer).second) return false;

      bool safe = true;
      get_def_use_mgr()->ForEachUser(pointer, [&](Instruction* pointer_user) {
        if (!safe || !pointer_user || IsIgnorableDirectUser(pointer_user) ||
            kill_set.find(pointer_user) != kill_set.end() ||
            candidates.find(pointer_user) != candidates.end()) {
          return;
        }
        if (pointer_user == originating_store) {
          candidates.insert(pointer_user);
          return;
        }
        if (pointer_user->opcode() != spv::Op::OpLoad ||
            pointer_user->NumInOperands() != 1 ||
            pointer_user->GetSingleWordInOperand(0) != pointer->result_id() ||
            !prove_value(pointer_user)) {
          safe = false;
          return;
        }
        candidates.insert(pointer_user);
      });

      active_pointers.erase(pointer);
      if (!safe) return false;
      candidates.insert(originating_store);
      candidates.insert(pointer);
      proven_pointers.insert(pointer);
      return true;
    };

    prove_value = [&](Instruction* value) {
      if (!value || value->result_id() == 0) return false;
      if (proven_values.find(value) != proven_values.end()) return true;
      if (!active_values.insert(value).second) return false;

      bool safe = true;
      get_def_use_mgr()->ForEachUser(value, [&](Instruction* user) {
        if (!safe || !user || user == inst || IsIgnorableDirectUser(user) ||
            kill_set.find(user) != kill_set.end() ||
            candidates.find(user) != candidates.end()) {
          return;
        }
        if (user->opcode() != spv::Op::OpStore || user->NumInOperands() != 2 ||
            user->GetSingleWordInOperand(1) != value->result_id()) {
          safe = false;
          return;
        }
        Instruction* pointer =
            get_def_use_mgr()->GetDef(user->GetSingleWordInOperand(0));
        if (!prove_pointer(pointer, user)) safe = false;
      });

      active_values.erase(value);
      if (!safe) return false;
      proven_values.insert(value);
      return true;
    };

    if (!prove_value(seed)) return;
    for (Instruction* candidate : candidates) add_kill(candidate);
  };

  for (Instruction* seed : direct_matrix.chain) {
    if (kill_set.find(seed) != kill_set.end() &&
        is_compatible_matrix_use_bitcast(seed)) {
      add_dead_forward_copy_chain(seed);
    }
  }
  if (!DirectKillListUsersAreClosed(inst, kill_list)) return true;

  const uint32_t function_id = BuildDirectVectorMatmulFunctionPackedVec4(
      *result, *input, *matrix, bias, has_bias,
      input_is_value ? 0 : direct_input.pointer_id,
      input_is_value ? 0 : direct_input.pointer_type_id,
      input_is_value ? std::vector<Operand>{} : direct_input.memory_operands,
      input_constant_id, input_is_value,
      matrix_is_value ? 0 : direct_matrix.pointer_id,
      matrix_is_value ? 0 : direct_matrix.pointer_type_id,
      matrix_is_value ? 0 : direct_matrix.shape_id,
      matrix_is_value ? 0 : direct_matrix.offset_id,
      matrix_is_value ? std::vector<Operand>{} : direct_matrix.memory_operands,
      matrix_constant_id, matrix_is_value,
      (has_bias && !bias_is_value) ? direct_bias.pointer_id : 0,
      (has_bias && !bias_is_value) ? direct_bias.pointer_type_id : 0,
      (has_bias && !bias_is_value) ? direct_bias.memory_operands
                                   : std::vector<Operand>{},
      bias_constant_id, bias_is_value, value_arguments);
  if (function_id == 0) return false;

  std::vector<uint32_t> call_args;
  call_args.reserve(value_arguments.size());
  for (const auto& arg : value_arguments) call_args.push_back(arg.first);
  RebuildAsFunctionCall(inst, result->lowered_type_id, function_id, call_args);
  for (Instruction* kill : kill_list) {
    if (!kill->IsNop()) context()->KillInst(kill);
  }
  *handled = true;
  return true;
}

uint32_t HwLowerToStandardPass::GetFunctionPointerOperandForLoad(
    Instruction* inst, uint32_t original_pointee_type_id,
    uint32_t lowered_pointee_type_id, uint32_t* pointer_type_id) const {
  if (!inst || inst->opcode() != spv::Op::OpLoad || inst->NumInOperands() < 1) {
    return 0;
  }
  if (pointer_type_id) *pointer_type_id = 0;
  const uint32_t pointer_id = inst->GetSingleWordInOperand(0);
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer || pointer->type_id() == 0) return 0;
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer->type_id());
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    return 0;
  }
  const uint32_t pointee_type_id = pointer_type->GetSingleWordInOperand(1);
  if (pointer_type->GetSingleWordInOperand(0) !=
          uint32_t(spv::StorageClass::Function) ||
      (pointee_type_id != original_pointee_type_id &&
       pointee_type_id != lowered_pointee_type_id)) {
    return 0;
  }
  if (pointer_type_id) *pointer_type_id = pointer->type_id();
  return pointer_id;
}

bool HwLowerToStandardPass::IsIgnorableDirectUser(
    const Instruction* user) const {
  return user &&
         (user->opcode() == spv::Op::OpName ||
          user->opcode() == spv::Op::OpMemberName || user->IsDecoration() ||
          user->IsNonSemanticInstruction() || user->IsDebugLineInst());
}

bool HwLowerToStandardPass::CanMoveLoadToUse(
    Instruction* load, Instruction* use, bool function_memory,
    uint32_t first_memory_operand) const {
  if (!load || !use) return false;
  if (!MemoryAccessOperandsAreMovable(load, first_memory_operand)) return false;
  return !HasUnsafeMemoryInstructionBetween(load, use, function_memory);
}

bool HwLowerToStandardPass::IsDirectSafeSharedValueUser(
    Instruction* current_inst, Instruction* user,
    const std::unordered_set<Instruction*>& kill_set) const {
  if (!user || user == current_inst || IsIgnorableDirectUser(user) ||
      kill_set.find(user) != kill_set.end()) {
    return true;
  }
  if (IsHwOpcode(user->opcode())) return true;
  if (user->opcode() != spv::Op::OpBitcast) return false;

  bool ok = true;
  get_def_use_mgr()->ForEachUser(user, [&](Instruction* bitcast_user) {
    if (!ok) return;
    ok = IsDirectSafeSharedValueUser(current_inst, bitcast_user, kill_set);
  });
  return ok;
}

bool HwLowerToStandardPass::IsDirectSafeSharedFunctionPointerUser(
    Instruction* current_inst, Instruction* pointer_user,
    const std::unordered_set<Instruction*>& kill_set) const {
  if (!pointer_user || pointer_user == current_inst ||
      IsIgnorableDirectUser(pointer_user) ||
      kill_set.find(pointer_user) != kill_set.end()) {
    return true;
  }
  if (pointer_user->opcode() != spv::Op::OpLoad ||
      pointer_user->NumInOperands() < 1) {
    return false;
  }

  bool ok = true;
  get_def_use_mgr()->ForEachUser(pointer_user, [&](Instruction* load_user) {
    if (!ok) return;
    ok = IsDirectSafeSharedValueUser(current_inst, load_user, kill_set);
  });
  return ok;
}

HwLowerToStandardPass::SharedDirectUserState
HwLowerToStandardPass::AnalyzeSharedDirectValueUses(
    Instruction* current_inst, Instruction* value,
    const std::unordered_set<Instruction*>& kill_set,
    std::unordered_set<Instruction*>* keep_alive,
    std::unordered_set<const Instruction*>* visited_values,
    std::unordered_set<const Instruction*>* visited_pointers) const {
  if (!value || value->result_id() == 0 || !visited_values ||
      !visited_pointers) {
    return SharedDirectUserState::kUnsafe;
  }
  if (!visited_values->insert(value).second) {
    return SharedDirectUserState::kNone;
  }

  SharedDirectUserState state = SharedDirectUserState::kNone;
  get_def_use_mgr()->ForEachUser(value, [&](Instruction* user) {
    if (state == SharedDirectUserState::kUnsafe || !user ||
        user == current_inst || IsIgnorableDirectUser(user)) {
      return;
    }

    SharedDirectUserState user_state = SharedDirectUserState::kUnsafe;
    if (IsHwOpcode(user->opcode())) {
      user_state = kill_set.find(user) == kill_set.end()
                       ? SharedDirectUserState::kSafeLive
                       : SharedDirectUserState::kNone;
    } else if (user->opcode() == spv::Op::OpBitcast) {
      user_state =
          AnalyzeSharedDirectValueUses(current_inst, user, kill_set, keep_alive,
                                       visited_values, visited_pointers);
      if (user_state == SharedDirectUserState::kSafeLive && keep_alive) {
        keep_alive->insert(user);
      }
    } else if (user->opcode() == spv::Op::OpStore &&
               user->NumInOperands() >= 2 &&
               user->GetSingleWordInOperand(1) == value->result_id()) {
      const uint32_t pointer_id = user->GetSingleWordInOperand(0);
      if (IsFunctionPointer(pointer_id)) {
        Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
        if (pointer) {
          user_state = AnalyzeSharedDirectPointerUses(
              current_inst, pointer, user, kill_set, keep_alive, visited_values,
              visited_pointers);
          if (user_state == SharedDirectUserState::kSafeLive && keep_alive) {
            keep_alive->insert(user);
            keep_alive->insert(pointer);
          }
        }
      }
    }

    if (user_state == SharedDirectUserState::kUnsafe) {
      state = SharedDirectUserState::kUnsafe;
      return;
    }
    if (user_state == SharedDirectUserState::kSafeLive) {
      state = SharedDirectUserState::kSafeLive;
    }
  });

  return state;
}

HwLowerToStandardPass::SharedDirectUserState
HwLowerToStandardPass::AnalyzeSharedDirectPointerUses(
    Instruction* current_inst, Instruction* pointer,
    Instruction* originating_store,
    const std::unordered_set<Instruction*>& kill_set,
    std::unordered_set<Instruction*>* keep_alive,
    std::unordered_set<const Instruction*>* visited_values,
    std::unordered_set<const Instruction*>* visited_pointers) const {
  if (!pointer || !visited_values || !visited_pointers) {
    return SharedDirectUserState::kUnsafe;
  }
  if (!visited_pointers->insert(pointer).second) {
    return SharedDirectUserState::kNone;
  }

  SharedDirectUserState state = SharedDirectUserState::kNone;
  get_def_use_mgr()->ForEachUser(pointer, [&](Instruction* pointer_user) {
    if (state == SharedDirectUserState::kUnsafe || !pointer_user ||
        pointer_user == originating_store ||
        IsIgnorableDirectUser(pointer_user)) {
      return;
    }
    if (pointer_user->opcode() == spv::Op::OpStore &&
        pointer_user->NumInOperands() >= 1 &&
        pointer_user->GetSingleWordInOperand(0) == pointer->result_id()) {
      return;
    }
    if (pointer_user->opcode() != spv::Op::OpLoad ||
        pointer_user->NumInOperands() < 1) {
      state = SharedDirectUserState::kUnsafe;
      return;
    }

    SharedDirectUserState load_state = AnalyzeSharedDirectValueUses(
        current_inst, pointer_user, kill_set, keep_alive, visited_values,
        visited_pointers);
    if (load_state == SharedDirectUserState::kUnsafe) {
      state = SharedDirectUserState::kUnsafe;
      return;
    }
    if (load_state == SharedDirectUserState::kSafeLive) {
      if (keep_alive) keep_alive->insert(pointer_user);
      state = SharedDirectUserState::kSafeLive;
    }
  });

  return state;
}

bool HwLowerToStandardPass::AnalyzeSharedDirectSourceUsers(
    Instruction* current_inst, Instruction* source_load,
    const std::unordered_set<Instruction*>& kill_set,
    bool* has_live_shared_users,
    std::unordered_set<Instruction*>* keep_alive) const {
  if (has_live_shared_users) *has_live_shared_users = false;
  if (!current_inst || !source_load || source_load->result_id() == 0) {
    return false;
  }

  std::unordered_set<const Instruction*> visited_values;
  std::unordered_set<const Instruction*> visited_pointers;
  SharedDirectUserState state = AnalyzeSharedDirectValueUses(
      current_inst, source_load, kill_set, keep_alive, &visited_values,
      &visited_pointers);
  if (state == SharedDirectUserState::kUnsafe) return false;
  if (state == SharedDirectUserState::kSafeLive) {
    if (has_live_shared_users) *has_live_shared_users = true;
    if (keep_alive) keep_alive->insert(source_load);
  }
  return true;
}

bool HwLowerToStandardPass::AddSharedDirectSourceKillsOrCheckSafe(
    Instruction* current_inst, Instruction* source_load,
    std::vector<Instruction*>* kill_list,
    std::unordered_set<Instruction*>* kill_set) const {
  if (!current_inst || !source_load || source_load->result_id() == 0 ||
      !kill_list || !kill_set) {
    return false;
  }

  auto add_kill = [kill_list, kill_set](Instruction* kill) {
    if (kill && kill_set->insert(kill).second) kill_list->push_back(kill);
  };

  bool has_live_shared_users = false;
  if (!AnalyzeSharedDirectSourceUsers(current_inst, source_load, *kill_set,
                                      &has_live_shared_users, nullptr)) {
    return false;
  }
  if (has_live_shared_users) return true;

  get_def_use_mgr()->ForEachUser(source_load, [&](Instruction* user) {
    if (!user || user == current_inst || IsIgnorableDirectUser(user) ||
        kill_set->find(user) != kill_set->end()) {
      return;
    }
    if (user->opcode() != spv::Op::OpStore || user->NumInOperands() < 2 ||
        user->GetSingleWordInOperand(1) != source_load->result_id()) {
      return;
    }
    const uint32_t pointer_id = user->GetSingleWordInOperand(0);
    if (!IsFunctionPointer(pointer_id)) return;
    Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
    if (!pointer) return;
    bool only_dead_users = true;
    bool has_later_store_to_pointer = false;
    get_def_use_mgr()->ForEachUser(pointer, [&](Instruction* pointer_user) {
      if (pointer_user == user || IsIgnorableDirectUser(pointer_user) ||
          kill_set->find(pointer_user) != kill_set->end()) {
        return;
      }
      if (pointer_user->opcode() == spv::Op::OpStore &&
          pointer_user->NumInOperands() >= 1 &&
          pointer_user->GetSingleWordInOperand(0) == pointer->result_id()) {
        has_later_store_to_pointer = true;
        return;
      }
      only_dead_users = false;
    });
    if (!only_dead_users) return;
    add_kill(user);
    if (!has_later_store_to_pointer) add_kill(pointer);
  });

  return true;
}

bool HwLowerToStandardPass::HasLiveSafeSharedDirectSourceUsers(
    Instruction* current_inst, Instruction* source_load,
    const std::unordered_set<Instruction*>& kill_set) const {
  bool has_live_shared_users = false;
  if (!AnalyzeSharedDirectSourceUsers(current_inst, source_load, kill_set,
                                      &has_live_shared_users, nullptr)) {
    return false;
  }
  return has_live_shared_users;
}

void HwLowerToStandardPass::KeepSharedDirectSourceAlive(
    Instruction* current_inst, Instruction* source_load,
    std::vector<Instruction*>* kill_list,
    std::unordered_set<Instruction*>* kill_set) const {
  if (!current_inst || !source_load || !kill_list || !kill_set) return;

  auto remove_from_kill = [kill_list, kill_set](Instruction* inst) {
    kill_set->erase(inst);
    kill_list->erase(std::remove(kill_list->begin(), kill_list->end(), inst),
                     kill_list->end());
  };

  bool has_live_shared_users = false;
  std::unordered_set<Instruction*> keep_alive;
  if (!AnalyzeSharedDirectSourceUsers(current_inst, source_load, *kill_set,
                                      &has_live_shared_users, &keep_alive) ||
      !has_live_shared_users) {
    return;
  }
  for (Instruction* inst : keep_alive) {
    remove_from_kill(inst);
  }
}

bool HwLowerToStandardPass::HasUnsafeMemoryInstructionBetween(
    Instruction* start, Instruction* end, bool function_memory) const {
  if (!start || !end) return true;
  BasicBlock* start_block = context()->get_instr_block(start);
  BasicBlock* end_block = context()->get_instr_block(end);
  if (!start_block || start_block != end_block) return true;
  const uint32_t load_pointer_id = GetMemoryPointerOperandId(start);

  bool after_start = false;
  for (Instruction& inst : *start_block) {
    if (&inst == start) {
      after_start = true;
      continue;
    }
    if (&inst == end) return false;
    if (!after_start) continue;
    if (InstructionMayWriteOrOrderMemory(&inst, function_memory)) {
      if (!function_memory &&
          IsDisjointModuleMemoryWrite(load_pointer_id, &inst)) {
        continue;
      }
      return true;
    }
  }
  return true;
}

bool HwLowerToStandardPass::InstructionMayWriteOrOrderMemory(
    const Instruction* inst, bool function_memory) const {
  if (!inst) return true;
  if (inst->IsAtomicOp()) return true;

  auto pointer_write_matches = [this, function_memory](uint32_t pointer_id) {
    uint32_t storage_class = 0;
    if (!GetPointerStorageClass(pointer_id, &storage_class)) return true;
    const bool is_function =
        storage_class == uint32_t(spv::StorageClass::Function);
    return function_memory ? is_function : !is_function;
  };

  switch (inst->opcode()) {
    case spv::Op::OpStore:
      return inst->NumInOperands() >= 1 &&
             pointer_write_matches(inst->GetSingleWordInOperand(0));
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
      return inst->NumInOperands() >= 1 &&
             pointer_write_matches(inst->GetSingleWordInOperand(0));
    case spv::Op::OpCooperativeMatrixStoreHW:
    case spv::Op::OpCooperativeVectorStoreHW:
      return inst->NumInOperands() >= 1 &&
             pointer_write_matches(inst->GetSingleWordInOperand(0));
    case spv::Op::OpFunctionCall:
      // Local helper writes cannot alias memory in the calling function.
      if (inst->NumInOperands() >= 1 && read_only_generated_function_ids_.count(
                                            inst->GetSingleWordInOperand(0))) {
        return false;
      }
      return true;
    case spv::Op::OpControlBarrier:
    case spv::Op::OpMemoryBarrier:
    case spv::Op::OpImageWrite:
      return true;
    default:
      return false;
  }
}

uint32_t HwLowerToStandardPass::GetMemoryPointerOperandId(
    const Instruction* inst) const {
  if (!inst || inst->NumInOperands() < 1) return 0;
  switch (inst->opcode()) {
    case spv::Op::OpLoad:
    case spv::Op::OpStore:
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
    case spv::Op::OpCooperativeMatrixLoadHW:
    case spv::Op::OpCooperativeMatrixStoreHW:
    case spv::Op::OpCooperativeVectorLoadHW:
    case spv::Op::OpCooperativeVectorStoreHW:
      return inst->GetSingleWordInOperand(0);
    default:
      return 0;
  }
}

uint32_t HwLowerToStandardPass::GetRootModulePointerId(
    uint32_t pointer_id) const {
  if (pointer_id == 0) return 0;
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer) return 0;

  if (pointer->opcode() == spv::Op::OpVariable &&
      context()->get_instr_block(pointer) == nullptr) {
    return pointer_id;
  }

  if (pointer->opcode() == spv::Op::OpLoad && pointer->NumInOperands() >= 1) {
    const uint32_t function_pointer_id = pointer->GetSingleWordInOperand(0);
    if (!IsFunctionPointer(function_pointer_id)) return 0;
    Instruction* store =
        FindLastStoreToFunctionPointer(function_pointer_id, pointer);
    if (!store || store->NumInOperands() < 2) return 0;
    return GetRootModulePointerId(store->GetSingleWordInOperand(1));
  }

  switch (pointer->opcode()) {
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
    case spv::Op::OpCopyObject:
    case spv::Op::OpBitcast:
      return pointer->NumInOperands() >= 1
                 ? GetRootModulePointerId(pointer->GetSingleWordInOperand(0))
                 : 0;
    default:
      return 0;
  }
}

bool HwLowerToStandardPass::IsDisjointModuleMemoryWrite(
    uint32_t load_pointer_id, const Instruction* inst) const {
  if (load_pointer_id == 0 || !inst) return false;

  uint32_t storage_class = 0;
  if (!GetPointerStorageClass(load_pointer_id, &storage_class) ||
      storage_class == uint32_t(spv::StorageClass::Function)) {
    return false;
  }

  const uint32_t write_pointer_id = GetMemoryPointerOperandId(inst);
  if (write_pointer_id == 0 ||
      !GetPointerStorageClass(write_pointer_id, &storage_class) ||
      storage_class == uint32_t(spv::StorageClass::Function)) {
    return false;
  }

  const uint32_t load_root_id = GetRootModulePointerId(load_pointer_id);
  const uint32_t write_root_id = GetRootModulePointerId(write_pointer_id);
  return load_root_id != 0 && write_root_id != 0 &&
         load_root_id != write_root_id;
}

bool HwLowerToStandardPass::MemoryAccessOperandsAreMovable(
    const Instruction* inst, uint32_t first_in_operand) const {
  if (!inst || inst->NumInOperands() <= first_in_operand) return true;
  const Operand& access = inst->GetInOperand(first_in_operand);
  if (access.type != SPV_OPERAND_TYPE_MEMORY_ACCESS ||
      access.words.size() != 1) {
    return false;
  }
  const uint32_t mask = access.words[0];
  const uint32_t aligned = uint32_t(spv::MemoryAccessMask::Aligned);
  const uint32_t alias_scope =
      uint32_t(spv::MemoryAccessMask::AliasScopeINTELMask);
  const uint32_t no_alias = uint32_t(spv::MemoryAccessMask::NoAliasINTELMask);
  const uint32_t allowed = aligned |
                           uint32_t(spv::MemoryAccessMask::Nontemporal) |
                           uint32_t(spv::MemoryAccessMask::NonPrivatePointer) |
                           alias_scope | no_alias;
  if ((mask & ~allowed) != 0) return false;

  uint32_t parameter_count = 0;
  uint32_t parameter_index = first_in_operand + 1;
  if ((mask & aligned) != 0) {
    ++parameter_count;
    ++parameter_index;
  }
  if ((mask & alias_scope) != 0) {
    if (parameter_index >= inst->NumInOperands() ||
        !IsModuleVisibleValue(inst->GetSingleWordInOperand(parameter_index))) {
      return false;
    }
    ++parameter_count;
    ++parameter_index;
  }
  if ((mask & no_alias) != 0) {
    if (parameter_index >= inst->NumInOperands() ||
        !IsModuleVisibleValue(inst->GetSingleWordInOperand(parameter_index))) {
      return false;
    }
    ++parameter_count;
  }
  return first_in_operand + 1 + parameter_count == inst->NumInOperands();
}

Instruction* HwLowerToStandardPass::TraceFunctionValueSource(
    Instruction* value_inst, Instruction* before,
    std::vector<Instruction*>* chain, uint32_t depth) const {
  if (!value_inst || !before || !chain || depth > 8) return nullptr;

  // Trace through OpBitcast - common when converting between HW matrix types
  if (value_inst->opcode() == spv::Op::OpBitcast &&
      value_inst->NumInOperands() >= 1) {
    chain->push_back(value_inst);
    Instruction* source =
        get_def_use_mgr()->GetDef(value_inst->GetSingleWordInOperand(0));
    return TraceFunctionValueSource(source, before, chain, depth + 1);
  }

  if (value_inst->opcode() != spv::Op::OpLoad ||
      value_inst->NumInOperands() < 1) {
    return value_inst;
  }

  const uint32_t pointer_id = value_inst->GetSingleWordInOperand(0);
  if (!IsFunctionPointer(pointer_id)) return value_inst;

  Instruction* store = FindLastStoreToFunctionPointer(pointer_id, value_inst);
  if (!store || store->NumInOperands() < 2) return nullptr;

  chain->push_back(value_inst);
  chain->push_back(store);
  Instruction* stored_value =
      get_def_use_mgr()->GetDef(store->GetSingleWordInOperand(1));
  return TraceFunctionValueSource(stored_value, store, chain, depth + 1);
}

Instruction* HwLowerToStandardPass::FindLastStoreToFunctionPointer(
    uint32_t pointer_id, Instruction* before) const {
  BasicBlock* block = context()->get_instr_block(before);
  if (!block) return nullptr;

  Instruction* last_store = nullptr;
  for (Instruction& inst : *block) {
    if (&inst == before) break;
    if (inst.opcode() == spv::Op::OpStore && inst.NumInOperands() >= 2 &&
        inst.GetSingleWordInOperand(0) == pointer_id) {
      last_store = &inst;
    }
  }
  return last_store;
}

bool HwLowerToStandardPass::IsFunctionPointer(uint32_t pointer_id) const {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer || pointer->type_id() == 0) return false;
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer->type_id());
  return pointer_type && pointer_type->opcode() == spv::Op::OpTypePointer &&
         pointer_type->GetSingleWordInOperand(0) ==
             uint32_t(spv::StorageClass::Function);
}

bool HwLowerToStandardPass::GetPointerStorageClass(
    uint32_t pointer_id, uint32_t* storage_class) const {
  if (storage_class) *storage_class = 0;
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer || pointer->type_id() == 0) return false;
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer->type_id());
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer ||
      pointer_type->NumInOperands() < 2) {
    return false;
  }
  if (storage_class) *storage_class = pointer_type->GetSingleWordInOperand(0);
  return true;
}

}  // namespace opt
}  // namespace spvtools
