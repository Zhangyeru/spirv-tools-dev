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

bool HwLowerToStandardPass::EliminateHwFunctionVariables() {
  // Collect all function variables whose pointee type contains HW types.
  std::vector<Instruction*> hw_vars;
  get_module()->ForEachInst([this, &hw_vars](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpVariable &&
        inst->GetSingleWordInOperand(0) ==
            static_cast<uint32_t>(spv::StorageClass::Function) &&
        TypeContainsHw(inst->type_id())) {
      hw_vars.push_back(inst);
    }
  });
  if (hw_vars.empty()) return true;

  // For each variable, collect stores and loads in program order.
  // For each load that follows a store, record the replacement.
  std::unordered_map<uint32_t, uint32_t> load_replacements;
  std::unordered_set<Instruction*> elim_loads;
  std::unordered_set<Instruction*> elim_stores;

  auto is_ignorable_user = [](Instruction* user) {
    return user &&
           (user->opcode() == spv::Op::OpName ||
            user->opcode() == spv::Op::OpMemberName || user->IsDecoration() ||
            user->IsNonSemanticInstruction() || user->IsDebugLineInst());
  };

  for (Instruction* var : hw_vars) {
    const uint32_t var_id = var->result_id();

    // Walk the block in program order, tracking the last stored value.
    // Every load from this variable is replaced with the most recent store.
    std::vector<Instruction*> var_stores;
    std::vector<Instruction*> var_loads;
    BasicBlock* block = nullptr;
    bool spans_multiple_blocks = false;
    bool has_non_forwardable_user = false;

    get_def_use_mgr()->ForEachUser(var, [&](Instruction* user) {
      if (is_ignorable_user(user)) return;
      BasicBlock* user_block = context()->get_instr_block(user);
      if (!user_block) {
        has_non_forwardable_user = true;
        return;
      }
      if (!block) {
        block = user_block;
      } else if (block != user_block) {
        spans_multiple_blocks = true;
      }
      if (user->opcode() == spv::Op::OpStore && user->NumInOperands() == 2 &&
          user->GetSingleWordInOperand(0) == var_id) {
        var_stores.push_back(user);
      } else if (user->opcode() == spv::Op::OpLoad &&
                 user->NumInOperands() == 1 &&
                 user->GetSingleWordInOperand(0) == var_id) {
        var_loads.push_back(user);
      } else {
        // Derived pointers may alias a later direct load, and memory operands
        // such as Volatile must not be dropped by value forwarding.  Only a
        // variable whose complete non-metadata use set is plain direct
        // stores/loads is eligible.
        has_non_forwardable_user = true;
      }
    });

    // This local forwarding is intentionally limited to one basic block.
    // Across blocks (especially loop headers/backedges), program order is not
    // dominance order and replacing a load with the textually preceding store
    // would break loop-carried cooperative values.
    if (has_non_forwardable_user || var_stores.empty() || var_loads.empty() ||
        !block || spans_multiple_blocks) {
      continue;
    }

    // Walk block in order: track the last stored value and record
    // replacements for each load.
    uint32_t last_stored_value = 0;
    for (Instruction& inst : *block) {
      if (inst.opcode() == spv::Op::OpStore && inst.NumInOperands() >= 2 &&
          inst.GetSingleWordInOperand(0) == var_id) {
        last_stored_value = inst.GetSingleWordInOperand(1);
      } else if (inst.opcode() == spv::Op::OpLoad &&
                 inst.NumInOperands() >= 1 &&
                 inst.GetSingleWordInOperand(0) == var_id &&
                 last_stored_value != 0) {
        load_replacements[inst.result_id()] = last_stored_value;
        elim_loads.insert(&inst);
      }
    }
  }

  if (load_replacements.empty()) return true;

  // Resolve chains: a replaced load's ID may appear as a stored value for
  // another variable.  Follow chains to their final (non-load) value.
  std::unordered_map<uint32_t, uint32_t> final_replacements;
  for (const auto& pair : load_replacements) {
    uint32_t current = pair.second;
    std::unordered_set<uint32_t> visited;
    visited.insert(pair.first);
    while (load_replacements.count(current) && !visited.count(current)) {
      visited.insert(current);
      current = load_replacements[current];
    }
    final_replacements[pair.first] = current;
  }

  // Apply all replacements.
  for (const auto& pair : final_replacements) {
    context()->ReplaceAllUsesWith(pair.first, pair.second);
  }

  // Kill eliminated loads.
  for (Instruction* load : elim_loads) {
    context()->KillInst(load);
  }

  // Try to eliminate stores and their variables.  A store can be killed if
  // every user of both the store and the variable is being eliminated or is
  // ignorable.
  for (Instruction* var : hw_vars) {
    const uint32_t var_id = var->result_id();
    std::vector<Instruction*> stores;
    get_def_use_mgr()->ForEachUser(var, [&](Instruction* user) {
      if (user->opcode() == spv::Op::OpStore &&
          user->GetSingleWordInOperand(0) == var_id) {
        stores.push_back(user);
      }
    });

    bool all_safe = true;
    for (Instruction* store : stores) {
      if (!elim_stores.count(store)) {
        bool store_safe = true;
        get_def_use_mgr()->ForEachUser(store, [&](Instruction* user) {
          if (!is_ignorable_user(user)) store_safe = false;
        });
        if (!store_safe) {
          all_safe = false;
          break;
        }
      }
      // Verify all users of the stored value are handled.
      if (store->NumInOperands() >= 2) {
        const uint32_t stored_id = store->GetSingleWordInOperand(1);
        Instruction* stored = get_def_use_mgr()->GetDef(stored_id);
        if (stored) {
          bool value_safe = true;
          get_def_use_mgr()->ForEachUser(stored, [&](Instruction* user) {
            if (user != store && !is_ignorable_user(user) &&
                !elim_loads.count(user) && !elim_stores.count(user)) {
              value_safe = false;
            }
          });
          if (!value_safe) {
            all_safe = false;
            break;
          }
        }
      }
    }
    if (!all_safe) continue;

    // Check all users of the variable itself.
    bool var_safe = true;
    get_def_use_mgr()->ForEachUser(var, [&](Instruction* user) {
      if (is_ignorable_user(user)) return;
      if (user->opcode() == spv::Op::OpStore &&
          user->GetSingleWordInOperand(0) == var_id) {
        return;  // Will be killed below.
      }
      if (user->opcode() == spv::Op::OpLoad &&
          user->GetSingleWordInOperand(0) == var_id && elim_loads.count(user)) {
        return;
      }
      var_safe = false;
    });
    if (!var_safe) continue;

    for (Instruction* store : stores) {
      elim_stores.insert(store);
    }
  }

  for (Instruction* store : elim_stores) {
    context()->KillInst(store);
  }
  for (Instruction* var : hw_vars) {
    bool safe = true;
    get_def_use_mgr()->ForEachUser(var, [&](Instruction* user) {
      if (!is_ignorable_user(user)) safe = false;
    });
    if (safe) context()->KillInst(var);
  }

  return true;
}

bool HwLowerToStandardPass::LowerCompositeConstruct(Instruction* inst) {
  const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
  const VectorTypeInfo* vector = GetVectorType(inst->type_id());
  if (!matrix && !vector) {
    ReportError(inst, "invalid HW OpCompositeConstruct result type");
    return false;
  }

  const uint32_t element_count =
      matrix ? matrix->rows * matrix->cols : vector->length;
  if (inst->NumInOperands() == 1 && element_count > max_unrolled_elements_) {
    Instruction* operand =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
    const uint32_t component_type_id =
        matrix ? matrix->component_type_id : vector->component_type_id;
    if (operand && operand->type_id() == component_type_id) {
      return LowerElementwiseWithLoop(inst, ElementwiseLoopKind::kBroadcast);
    }
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> scalar_ids;

  if (matrix) {
    const uint32_t expected_operands = matrix->rows * matrix->cols;
    Instruction* scalar =
        inst->NumInOperands() == 1
            ? get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0))
            : nullptr;
    if (!scalar || scalar->type_id() != matrix->component_type_id) {
      ReportError(inst,
                  "HW matrix OpCompositeConstruct requires one scalar "
                  "constituent");
      return false;
    }
    scalar_ids.assign(expected_operands, scalar->result_id());
    return RebuildMatrixFromScalars(inst, *matrix, scalar_ids);
  }

  if (inst->opcode() == spv::Op::OpCompositeConstructReplicateEXT) {
    Instruction* scalar =
        inst->NumInOperands() == 1
            ? get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0))
            : nullptr;
    if (!scalar || scalar->type_id() != vector->component_type_id) {
      ReportError(inst,
                  "HW OpCompositeConstructReplicateEXT operand is invalid");
      return false;
    }
    scalar_ids.assign(vector->length, scalar->result_id());
    return RebuildVectorFromScalars(inst, *vector, scalar_ids);
  }

  scalar_ids.reserve(vector->length);
  for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
    Instruction* constituent =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(i));
    if (!constituent) {
      ReportError(inst, "invalid HW vector OpCompositeConstruct constituent");
      return false;
    }
    if (constituent->type_id() == vector->component_type_id) {
      scalar_ids.push_back(constituent->result_id());
      continue;
    }
    Instruction* type = get_def_use_mgr()->GetDef(constituent->type_id());
    if (!type || type->opcode() != spv::Op::OpTypeVector ||
        type->NumInOperands() < 2 ||
        type->GetSingleWordInOperand(0) != vector->component_type_id) {
      ReportError(inst, "invalid HW vector OpCompositeConstruct constituent");
      return false;
    }
    const uint32_t count = type->GetSingleWordInOperand(1);
    for (uint32_t lane = 0; lane < count; ++lane) {
      Instruction* scalar = builder.AddCompositeExtract(
          vector->component_type_id, constituent->result_id(), {lane});
      if (!scalar) return false;
      scalar_ids.push_back(scalar->result_id());
    }
  }
  if (scalar_ids.size() != vector->length) {
    ReportError(inst,
                "HW vector OpCompositeConstruct operand count is invalid");
    return false;
  }
  return RebuildVectorFromScalars(inst, *vector, scalar_ids);
}

bool HwLowerToStandardPass::LowerConstantComposite(Instruction* inst) {
  const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
  const VectorTypeInfo* vector = GetVectorType(inst->type_id());
  if (!matrix && !vector) {
    ReportError(inst, "invalid HW OpConstantComposite result type");
    return false;
  }
  const bool is_replicate =
      inst->opcode() == spv::Op::OpConstantCompositeReplicateEXT ||
      inst->opcode() == spv::Op::OpSpecConstantCompositeReplicateEXT;
  const bool is_spec =
      inst->opcode() == spv::Op::OpSpecConstantComposite ||
      inst->opcode() == spv::Op::OpSpecConstantCompositeReplicateEXT;
  const spv::Op lowered_constituent_opcode =
      is_spec ? spv::Op::OpSpecConstantComposite : spv::Op::OpConstantComposite;

  const uint32_t component_type_id =
      matrix ? matrix->component_type_id : vector->component_type_id;
  const uint32_t expected_operands =
      matrix ? matrix->rows * matrix->cols : vector->length;

  std::vector<uint32_t> scalar_ids;
  for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
    Instruction* operand =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(i));
    if (!operand || operand->type_id() != component_type_id) {
      ReportError(inst, "unsupported HW OpConstantComposite operand");
      return false;
    }
    scalar_ids.push_back(operand->result_id());
  }

  if ((matrix || is_replicate) && scalar_ids.size() == 1 &&
      expected_operands > 1) {
    scalar_ids.resize(expected_operands, scalar_ids[0]);
  }
  if (scalar_ids.size() != expected_operands) {
    ReportError(inst, "HW OpConstantComposite operand count is invalid");
    return false;
  }

  if ((matrix && !IsPackedVec4(*matrix)) ||
      (vector && !IsPackedVec4(*vector))) {
    if (is_replicate) inst->SetOpcode(lowered_constituent_opcode);
    std::vector<Operand> operands;
    operands.reserve(scalar_ids.size());
    for (uint32_t id : scalar_ids) operands.push_back(IdOperand(id));
    inst->SetResultType(matrix ? matrix->lowered_type_id
                               : vector->lowered_type_id);
    inst->SetInOperands(std::move(operands));
    context()->UpdateDefUse(inst);
    return true;
  }

  // Find insertion point that is after both the scalar constants AND the
  // packed_vec4_type_id (which may have been created after the scalars).
  // Track which appears last in the module order.
  Instruction* insert_after = nullptr;
  const uint32_t packed_vec4_type_id =
      matrix ? matrix->packed_vec4_type_id : vector->packed_vec4_type_id;
  for (Instruction& candidate : get_module()->types_values()) {
    const uint32_t candidate_id = candidate.result_id();
    if (candidate_id == 0) continue;
    // Update insert_after whenever we see a scalar or the packed vec4 type.
    // Since we iterate in module order, the last match will be the later one.
    if (candidate_id == packed_vec4_type_id ||
        std::find(scalar_ids.begin(), scalar_ids.end(), candidate_id) !=
            scalar_ids.end()) {
      insert_after = &candidate;
    }
  }
  if (!insert_after) {
    ReportError(inst, "invalid HW OpConstantComposite constituent");
    return false;
  }

  std::vector<uint32_t> element_ids;
  if (matrix) {
    element_ids.resize(matrix->rows * matrix->packed_cols, 0);
    for (uint32_t row = 0; row < matrix->rows; ++row) {
      for (uint32_t col_pack = 0; col_pack < matrix->packed_cols; ++col_pack) {
        std::vector<uint32_t> lane_ids;
        lane_ids.reserve(kPackedVec4Width);
        for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
          lane_ids.push_back(scalar_ids[MatrixFlatIndex(
              *matrix, row, col_pack * kPackedVec4Width + lane)]);
        }
        const uint32_t element_id = GetOrCreateCompositeConstant(
            matrix->packed_vec4_type_id, lane_ids, &insert_after,
            lowered_constituent_opcode);
        if (element_id == 0) return false;
        element_ids[MatrixPackedIndex(*matrix, row, col_pack)] = element_id;
      }
    }
  } else {
    element_ids.resize(vector->packed_length, 0);
    for (uint32_t pack = 0; pack < vector->packed_length; ++pack) {
      std::vector<uint32_t> lane_ids;
      lane_ids.reserve(kPackedVec4Width);
      for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
        lane_ids.push_back(scalar_ids[pack * kPackedVec4Width + lane]);
      }
      element_ids[pack] = GetOrCreateCompositeConstant(
          vector->packed_vec4_type_id, lane_ids, &insert_after,
          lowered_constituent_opcode);
      if (element_ids[pack] == 0) return false;
    }
  }

  std::vector<Operand> operands;
  operands.reserve(element_ids.size());
  for (uint32_t id : element_ids) operands.push_back(IdOperand(id));
  if (is_replicate) inst->SetOpcode(lowered_constituent_opcode);
  inst->SetResultType(matrix ? matrix->lowered_type_id
                             : vector->lowered_type_id);
  inst->SetInOperands(std::move(operands));
  context()->UpdateDefUse(inst);
  return true;
}

bool HwLowerToStandardPass::LowerCompositeExtract(Instruction* inst) {
  return RemapCompositeIndices(inst, 0, 1);
}

bool HwLowerToStandardPass::LowerCompositeInsert(Instruction* inst) {
  return RemapCompositeIndices(inst, 1, 2);
}

bool HwLowerToStandardPass::RemapCompositeIndices(
    Instruction* inst, uint32_t composite_in_operand,
    uint32_t first_index_in_operand) {
  if (!inst || inst->NumInOperands() <= composite_in_operand ||
      inst->NumInOperands() <= first_index_in_operand) {
    ReportError(inst, "invalid HW composite indexing instruction");
    return false;
  }

  Instruction* composite = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(composite_in_operand));
  if (!composite) return false;
  uint32_t current_type_id = composite->type_id();
  auto original = original_hw_value_types_.find(composite->result_id());
  if (original != original_hw_value_types_.end()) {
    current_type_id = original->second;
  }

  std::vector<Operand> operands;
  operands.reserve(inst->NumInOperands() + 1);
  for (uint32_t i = 0; i < first_index_in_operand; ++i) {
    operands.push_back(inst->GetInOperand(i));
  }

  bool remapped = false;
  for (uint32_t i = first_index_in_operand; i < inst->NumInOperands(); ++i) {
    const uint32_t index = inst->GetSingleWordInOperand(i);
    if (const MatrixTypeInfo* matrix = GetMatrixType(current_type_id)) {
      if (i + 1 >= inst->NumInOperands()) {
        ReportError(inst,
                    "HW matrix composite indexing requires row and column");
        return false;
      }
      const uint32_t row = index;
      const uint32_t col = inst->GetSingleWordInOperand(++i);
      if (row >= matrix->rows || col >= matrix->cols) {
        ReportError(inst, "HW matrix composite index is out of range");
        return false;
      }
      if (IsPackedVec4(*matrix)) {
        operands.push_back(
            {SPV_OPERAND_TYPE_LITERAL_INTEGER,
             {MatrixPackedIndex(*matrix, row, VectorPackedIndex(col))}});
        operands.push_back(
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {PackedLane(col)}});
      } else {
        operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                            {MatrixFlatIndex(*matrix, row, col)}});
      }
      current_type_id = matrix->component_type_id;
      remapped = true;
      continue;
    }
    if (const VectorTypeInfo* vector = GetVectorType(current_type_id)) {
      if (index >= vector->length) {
        ReportError(inst, "HW vector composite index is out of range");
        return false;
      }
      if (IsPackedVec4(*vector)) {
        operands.push_back(
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {VectorPackedIndex(index)}});
        operands.push_back(
            {SPV_OPERAND_TYPE_LITERAL_INTEGER, {PackedLane(index)}});
      } else {
        operands.push_back(inst->GetInOperand(i));
      }
      current_type_id = vector->component_type_id;
      remapped = true;
      continue;
    }

    Instruction* type = get_def_use_mgr()->GetDef(current_type_id);
    if (!type) return false;
    operands.push_back(inst->GetInOperand(i));
    switch (type->opcode()) {
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeMatrix:
        current_type_id = type->GetSingleWordInOperand(0);
        break;
      case spv::Op::OpTypeStruct:
        if (index >= type->NumInOperands()) {
          ReportError(inst, "nested HW composite struct index is out of range");
          return false;
        }
        current_type_id = type->GetSingleWordInOperand(index);
        break;
      default:
        ReportError(inst, "composite index path reaches a non-composite type");
        return false;
    }
  }

  if (remapped) {
    inst->SetInOperands(std::move(operands));
    context()->UpdateDefUse(inst);
  }
  return true;
}

bool HwLowerToStandardPass::LowerAccessChain(Instruction* inst) {
  if (!inst || inst->NumInOperands() < 2) {
    ReportError(inst, "invalid HW access chain");
    return false;
  }
  Instruction* base =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  Instruction* pointer_type =
      base ? get_def_use_mgr()->GetDef(base->type_id()) : nullptr;
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer ||
      pointer_type->NumInOperands() < 2) {
    return true;
  }

  uint32_t current_type_id = pointer_type->GetSingleWordInOperand(1);
  const bool is_ptr_chain = inst->opcode() == spv::Op::OpPtrAccessChain ||
                            inst->opcode() == spv::Op::OpInBoundsPtrAccessChain;
  const uint32_t first_index = is_ptr_chain ? 2 : 1;
  if (inst->NumInOperands() <= first_index) return true;

  std::vector<Operand> operands;
  operands.reserve(inst->NumInOperands() + 1);
  for (uint32_t i = 0; i < first_index; ++i) {
    operands.push_back(inst->GetInOperand(i));
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  bool remapped = false;
  for (uint32_t i = first_index; i < inst->NumInOperands(); ++i) {
    const uint32_t index_id = inst->GetSingleWordInOperand(i);
    const MatrixTypeInfo* matrix = GetMatrixType(current_type_id);
    const VectorTypeInfo* vector = GetVectorType(current_type_id);
    if (matrix || vector) {
      const bool packed =
          matrix ? IsPackedVec4(*matrix) : IsPackedVec4(*vector);
      const uint32_t component_type_id =
          matrix ? matrix->component_type_id : vector->component_type_id;
      if (!packed) {
        operands.push_back(inst->GetInOperand(i));
      } else {
        Instruction* index = get_def_use_mgr()->GetDef(index_id);
        Instruction* index_type =
            index ? get_def_use_mgr()->GetDef(index->type_id()) : nullptr;
        if (!index_type || index_type->opcode() != spv::Op::OpTypeInt ||
            index_type->NumInOperands() < 2 ||
            index_type->GetSingleWordInOperand(0) == 0 ||
            index_type->GetSingleWordInOperand(0) > 64) {
          ReportError(inst,
                      "packed HW access chain requires an integer index of at "
                      "most 64 bits");
          return false;
        }
        const uint32_t four_id = GetOrCreateConstant(index->type_id(), 4);
        const bool is_signed = index_type->GetSingleWordInOperand(1) != 0;
        Instruction* piece = builder.AddBinaryOp(
            index->type_id(), is_signed ? spv::Op::OpSDiv : spv::Op::OpUDiv,
            index_id, four_id);
        Instruction* lane = builder.AddBinaryOp(
            index->type_id(), is_signed ? spv::Op::OpSRem : spv::Op::OpUMod,
            index_id, four_id);
        if (!piece || !lane) return false;
        operands.push_back(IdOperand(piece->result_id()));
        operands.push_back(IdOperand(lane->result_id()));
      }
      current_type_id = component_type_id;
      remapped = true;
      continue;
    }

    Instruction* type = get_def_use_mgr()->GetDef(current_type_id);
    if (!type) return false;
    operands.push_back(inst->GetInOperand(i));
    switch (type->opcode()) {
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
      case spv::Op::OpTypeMatrix:
        current_type_id = type->GetSingleWordInOperand(0);
        break;
      case spv::Op::OpTypeStruct: {
        uint32_t member = 0;
        if (!GetConstantU32(index_id, &member) ||
            member >= type->NumInOperands()) {
          ReportError(inst, "nested HW access-chain struct index is invalid");
          return false;
        }
        current_type_id = type->GetSingleWordInOperand(member);
        break;
      }
      default:
        ReportError(inst, "HW access-chain path reaches a non-composite type");
        return false;
    }
  }

  if (remapped) {
    inst->SetInOperands(std::move(operands));
    context()->UpdateDefUse(inst);
  }
  return true;
}

bool HwLowerToStandardPass::LowerSelect(Instruction* inst) {
  const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
  const VectorTypeInfo* vector = GetVectorType(inst->type_id());
  if ((!matrix && !vector) || !inst || inst->NumInOperands() != 3) {
    ReportError(inst, "invalid HW OpSelect");
    return false;
  }

  const uint32_t condition_id = inst->GetSingleWordInOperand(0);
  const uint32_t true_id = inst->GetSingleWordInOperand(1);
  const uint32_t false_id = inst->GetSingleWordInOperand(2);
  Instruction* condition = get_def_use_mgr()->GetDef(condition_id);
  Instruction* condition_type =
      condition ? get_def_use_mgr()->GetDef(condition->type_id()) : nullptr;
  if (!condition_type || condition_type->opcode() != spv::Op::OpTypeBool) {
    ReportError(inst, "HW OpSelect requires a scalar bool condition");
    return false;
  }

  const uint32_t element_count =
      matrix ? matrix->rows * matrix->cols : vector->length;
  if (element_count > max_unrolled_elements_) {
    return LowerElementwiseWithLoop(inst, ElementwiseLoopKind::kSelect);
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (matrix) {
    std::vector<uint32_t> scalar_ids(matrix->rows * matrix->cols, 0);
    for (uint32_t row = 0; row < matrix->rows; ++row) {
      for (uint32_t col = 0; col < matrix->cols; ++col) {
        const uint32_t lhs =
            ExtractMatrixScalar(&builder, *matrix, true_id, row, col);
        const uint32_t rhs =
            ExtractMatrixScalar(&builder, *matrix, false_id, row, col);
        Instruction* select =
            builder.AddTernaryOp(matrix->component_type_id, spv::Op::OpSelect,
                                 condition_id, lhs, rhs);
        if (!select) return false;
        scalar_ids[MatrixFlatIndex(*matrix, row, col)] = select->result_id();
      }
    }
    return RebuildMatrixFromScalars(inst, *matrix, scalar_ids);
  }

  std::vector<uint32_t> scalar_ids(vector->length, 0);
  for (uint32_t i = 0; i < vector->length; ++i) {
    const uint32_t lhs = ExtractVectorScalar(&builder, *vector, true_id, i);
    const uint32_t rhs = ExtractVectorScalar(&builder, *vector, false_id, i);
    Instruction* select = builder.AddTernaryOp(
        vector->component_type_id, spv::Op::OpSelect, condition_id, lhs, rhs);
    if (!select) return false;
    scalar_ids[i] = select->result_id();
  }
  return RebuildVectorFromScalars(inst, *vector, scalar_ids);
}

bool HwLowerToStandardPass::LowerVectorExtractDynamic(Instruction* inst) {
  if (!inst || inst->NumInOperands() != 2) {
    ReportError(inst, "invalid HW OpVectorExtractDynamic");
    return false;
  }
  Instruction* object =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  const VectorTypeInfo* vector =
      object ? GetVectorType(object->type_id()) : nullptr;
  if (!vector || inst->type_id() != vector->component_type_id) {
    ReportError(inst, "invalid HW OpVectorExtractDynamic object");
    return false;
  }

  BasicBlock* block = context()->get_instr_block(inst);
  Function* function = block ? block->GetParent() : nullptr;
  const uint32_t value_ptr_type = GetOrCreatePointerType(
      vector->lowered_type_id, spv::StorageClass::Function);
  if (!function || value_ptr_type == 0) return false;
  Instruction* variable = AddFunctionVariable(function, value_ptr_type);
  if (!variable) return false;

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!AddStore(&builder, variable->result_id(), object->result_id(), {})) {
    return false;
  }
  uint32_t index_id = inst->GetSingleWordInOperand(1);
  uint32_t piece_index_id = index_id;
  uint32_t lane_id = 0;
  uint32_t piece_type_id = vector->component_type_id;
  if (IsPackedVec4(*vector)) {
    Instruction* index = get_def_use_mgr()->GetDef(index_id);
    Instruction* index_type =
        index ? get_def_use_mgr()->GetDef(index->type_id()) : nullptr;
    if (!index || !index_type || index_type->opcode() != spv::Op::OpTypeInt ||
        index_type->NumInOperands() < 2) {
      return false;
    }
    const uint32_t four_id = GetOrCreateConstant(index->type_id(), 4);
    const bool is_signed = index_type->GetSingleWordInOperand(1) != 0;
    Instruction* piece_index = builder.AddBinaryOp(
        index->type_id(), is_signed ? spv::Op::OpSDiv : spv::Op::OpUDiv,
        index_id, four_id);
    Instruction* lane = builder.AddBinaryOp(
        index->type_id(), is_signed ? spv::Op::OpSRem : spv::Op::OpUMod,
        index_id, four_id);
    if (!piece_index || !lane) return false;
    piece_index_id = piece_index->result_id();
    lane_id = lane->result_id();
    piece_type_id = vector->packed_vec4_type_id;
  }

  const uint32_t piece_ptr_type =
      GetOrCreatePointerType(piece_type_id, spv::StorageClass::Function);
  Instruction* pointer = builder.AddAccessChain(
      piece_ptr_type, variable->result_id(), {piece_index_id});
  if (!pointer) return false;
  const uint32_t piece =
      AddLoad(&builder, piece_type_id, pointer->result_id(), {});
  if (piece == 0) return false;
  uint32_t result_id = piece;
  if (IsPackedVec4(*vector)) {
    Instruction* extract =
        builder.AddBinaryOp(vector->component_type_id,
                            spv::Op::OpVectorExtractDynamic, piece, lane_id);
    if (!extract) return false;
    result_id = extract->result_id();
  }
  inst->SetOpcode(spv::Op::OpCopyObject);
  inst->SetInOperands({IdOperand(result_id)});
  context()->UpdateDefUse(inst);
  return true;
}

bool HwLowerToStandardPass::LowerVectorInsertDynamic(Instruction* inst) {
  const VectorTypeInfo* vector =
      inst ? GetVectorType(inst->type_id()) : nullptr;
  if (!vector || inst->NumInOperands() != 3) {
    ReportError(inst, "invalid HW OpVectorInsertDynamic");
    return false;
  }
  const uint32_t vector_id = inst->GetSingleWordInOperand(0);
  const uint32_t object_id = inst->GetSingleWordInOperand(1);
  const uint32_t index_id = inst->GetSingleWordInOperand(2);
  BasicBlock* block = context()->get_instr_block(inst);
  Function* function = block ? block->GetParent() : nullptr;
  const uint32_t value_ptr_type = GetOrCreatePointerType(
      vector->lowered_type_id, spv::StorageClass::Function);
  if (!function || value_ptr_type == 0) return false;
  Instruction* variable = AddFunctionVariable(function, value_ptr_type);
  if (!variable) return false;

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!AddStore(&builder, variable->result_id(), vector_id, {})) return false;
  uint32_t piece_index_id = index_id;
  uint32_t lane_id = 0;
  uint32_t piece_type_id = vector->component_type_id;
  if (IsPackedVec4(*vector)) {
    Instruction* index = get_def_use_mgr()->GetDef(index_id);
    Instruction* index_type =
        index ? get_def_use_mgr()->GetDef(index->type_id()) : nullptr;
    if (!index || !index_type || index_type->opcode() != spv::Op::OpTypeInt ||
        index_type->NumInOperands() < 2) {
      return false;
    }
    const uint32_t four_id = GetOrCreateConstant(index->type_id(), 4);
    const bool is_signed = index_type->GetSingleWordInOperand(1) != 0;
    Instruction* piece_index = builder.AddBinaryOp(
        index->type_id(), is_signed ? spv::Op::OpSDiv : spv::Op::OpUDiv,
        index_id, four_id);
    Instruction* lane = builder.AddBinaryOp(
        index->type_id(), is_signed ? spv::Op::OpSRem : spv::Op::OpUMod,
        index_id, four_id);
    if (!piece_index || !lane) return false;
    piece_index_id = piece_index->result_id();
    lane_id = lane->result_id();
    piece_type_id = vector->packed_vec4_type_id;
  }

  const uint32_t piece_ptr_type =
      GetOrCreatePointerType(piece_type_id, spv::StorageClass::Function);
  Instruction* pointer = builder.AddAccessChain(
      piece_ptr_type, variable->result_id(), {piece_index_id});
  if (!pointer) return false;
  uint32_t stored_id = object_id;
  if (IsPackedVec4(*vector)) {
    const uint32_t old_piece =
        AddLoad(&builder, piece_type_id, pointer->result_id(), {});
    Instruction* inserted =
        builder.AddTernaryOp(piece_type_id, spv::Op::OpVectorInsertDynamic,
                             old_piece, object_id, lane_id);
    if (!inserted) return false;
    stored_id = inserted->result_id();
  }
  if (!AddStore(&builder, pointer->result_id(), stored_id, {})) return false;
  const uint32_t result_id =
      AddLoad(&builder, vector->lowered_type_id, variable->result_id(), {});
  if (result_id == 0) return false;
  inst->SetOpcode(spv::Op::OpCopyObject);
  inst->SetResultType(vector->lowered_type_id);
  inst->SetInOperands({IdOperand(result_id)});
  context()->UpdateDefUse(inst);
  return true;
}

bool HwLowerToStandardPass::LowerNullOrUndef(Instruction* inst) {
  const uint32_t lowered_type_id = GetLoweredType(inst->type_id());
  if (lowered_type_id == 0) {
    ReportError(inst, "invalid HW null/undef result type");
    return false;
  }
  inst->SetResultType(lowered_type_id);
  context()->UpdateDefUse(inst);
  return true;
}

uint32_t HwLowerToStandardPass::ExtractVectorScalar(InstructionBuilder* builder,
                                                    const VectorTypeInfo& info,
                                                    uint32_t vector_id,
                                                    uint32_t index) {
  if (!IsPackedVec4(info)) {
    return ExtractCompositeElement(builder, info.component_type_id, vector_id,
                                   index);
  }

  const uint32_t vec_id = ExtractCompositeElement(
      builder, info.packed_vec4_type_id, vector_id, VectorPackedIndex(index));
  if (vec_id == 0) return 0;
  return ExtractCompositeElement(builder, info.component_type_id, vec_id,
                                 PackedLane(index));
}

uint32_t HwLowerToStandardPass::ExtractMatrixScalar(InstructionBuilder* builder,
                                                    const MatrixTypeInfo& info,
                                                    uint32_t matrix_id,
                                                    uint32_t row,
                                                    uint32_t col) {
  if (!IsPackedVec4(info)) {
    return ExtractCompositeElement(builder, info.component_type_id, matrix_id,
                                   MatrixFlatIndex(info, row, col));
  }

  const uint32_t vec_id = ExtractCompositeElement(
      builder, info.packed_vec4_type_id, matrix_id,
      MatrixPackedIndex(info, row, VectorPackedIndex(col)));
  if (vec_id == 0) return 0;
  return ExtractCompositeElement(builder, info.component_type_id, vec_id,
                                 PackedLane(col));
}

bool HwLowerToStandardPass::DescribeVectorValue(uint32_t value_id,
                                                uint32_t expected_length,
                                                ValueLayout* layout) const {
  if (!layout) return false;
  *layout = {};

  Instruction* value = get_def_use_mgr()->GetDef(value_id);
  if (!value) return false;

  if (const VectorTypeInfo* info = GetVectorType(value->type_id())) {
    if (info->length != expected_length) return false;
    layout->component_type_id = info->component_type_id;
    layout->piece_type_id = IsPackedVec4(*info) ? info->packed_vec4_type_id
                                                : info->component_type_id;
    layout->piece_count =
        IsPackedVec4(*info) ? info->packed_length : info->length;
    layout->packed_vec4 = IsPackedVec4(*info);
    return true;
  }

  Instruction* type = get_def_use_mgr()->GetDef(value->type_id());
  if (!type || type->opcode() != spv::Op::OpTypeArray ||
      type->NumInOperands() < 2) {
    return false;
  }

  uint32_t array_length = 0;
  if (!GetConstantU32(type->GetSingleWordInOperand(1), &array_length)) {
    return false;
  }

  Instruction* element_type =
      get_def_use_mgr()->GetDef(type->GetSingleWordInOperand(0));
  if (!element_type) return false;

  if (element_type->opcode() == spv::Op::OpTypeVector &&
      element_type->NumInOperands() >= 2 &&
      element_type->GetSingleWordInOperand(1) == kPackedVec4Width) {
    if (array_length * kPackedVec4Width != expected_length) return false;
    layout->component_type_id = element_type->GetSingleWordInOperand(0);
    layout->piece_type_id = element_type->result_id();
    layout->piece_count = array_length;
    layout->packed_vec4 = true;
    return true;
  }

  if (!IsNumericScalarType(element_type) || array_length != expected_length) {
    return false;
  }
  layout->component_type_id = element_type->result_id();
  layout->piece_type_id = element_type->result_id();
  layout->piece_count = array_length;
  layout->packed_vec4 = false;
  return true;
}

bool HwLowerToStandardPass::DescribeMatrixValue(uint32_t value_id,
                                                uint32_t expected_rows,
                                                uint32_t expected_cols,
                                                ValueLayout* layout) const {
  if (!layout) return false;
  *layout = {};

  Instruction* value = get_def_use_mgr()->GetDef(value_id);
  if (!value) return false;

  if (const MatrixTypeInfo* info = GetMatrixTypeForValue(value)) {
    if (info->rows != expected_rows || info->cols != expected_cols) {
      return false;
    }
    layout->component_type_id = info->component_type_id;
    layout->piece_type_id = IsPackedVec4(*info) ? info->packed_vec4_type_id
                                                : info->component_type_id;
    layout->piece_count = IsPackedVec4(*info) ? info->rows * info->packed_cols
                                              : info->rows * info->cols;
    layout->packed_vec4 = IsPackedVec4(*info);
    return true;
  }

  Instruction* type = get_def_use_mgr()->GetDef(value->type_id());
  if (!type || type->opcode() != spv::Op::OpTypeArray ||
      type->NumInOperands() < 2) {
    return false;
  }

  uint32_t array_length = 0;
  if (!GetConstantU32(type->GetSingleWordInOperand(1), &array_length)) {
    return false;
  }

  Instruction* element_type =
      get_def_use_mgr()->GetDef(type->GetSingleWordInOperand(0));
  if (!element_type) return false;

  if (element_type->opcode() == spv::Op::OpTypeVector &&
      element_type->NumInOperands() >= 2 &&
      element_type->GetSingleWordInOperand(1) == kPackedVec4Width) {
    if (expected_cols % kPackedVec4Width != 0 ||
        array_length * kPackedVec4Width != expected_rows * expected_cols) {
      return false;
    }
    layout->component_type_id = element_type->GetSingleWordInOperand(0);
    layout->piece_type_id = element_type->result_id();
    layout->piece_count = array_length;
    layout->packed_vec4 = true;
    return true;
  }

  if (!IsNumericScalarType(element_type) ||
      array_length != expected_rows * expected_cols) {
    return false;
  }
  layout->component_type_id = element_type->result_id();
  layout->piece_type_id = element_type->result_id();
  layout->piece_count = array_length;
  layout->packed_vec4 = false;
  return true;
}

uint32_t HwLowerToStandardPass::ExtractValuePiece(InstructionBuilder* builder,
                                                  const ValueLayout& layout,
                                                  uint32_t value_id,
                                                  uint32_t piece_index) {
  return ExtractCompositeElement(builder, layout.piece_type_id, value_id,
                                 piece_index);
}

uint32_t HwLowerToStandardPass::ExtractVectorValueScalar(
    InstructionBuilder* builder, const ValueLayout& layout, uint32_t value_id,
    uint32_t index) {
  if (!layout.packed_vec4) {
    return ExtractValuePiece(builder, layout, value_id, index);
  }

  const uint32_t piece_id =
      ExtractValuePiece(builder, layout, value_id, VectorPackedIndex(index));
  if (piece_id == 0) return 0;
  return ExtractCompositeElement(builder, layout.component_type_id, piece_id,
                                 PackedLane(index));
}

uint32_t HwLowerToStandardPass::ExtractMatrixValueScalar(
    InstructionBuilder* builder, const ValueLayout& layout, uint32_t value_id,
    uint32_t cols, uint32_t row, uint32_t col) {
  if (!layout.packed_vec4) {
    return ExtractValuePiece(builder, layout, value_id, row * cols + col);
  }

  const uint32_t packed_cols = cols / kPackedVec4Width;
  const uint32_t piece_id = ExtractValuePiece(
      builder, layout, value_id, row * packed_cols + VectorPackedIndex(col));
  if (piece_id == 0) return 0;
  return ExtractCompositeElement(builder, layout.component_type_id, piece_id,
                                 PackedLane(col));
}

bool HwLowerToStandardPass::RebuildVectorFromScalars(
    Instruction* inst, const VectorTypeInfo& info,
    const std::vector<uint32_t>& scalar_ids) {
  return RebuildAggregateFromScalars(
      inst, info.lowered_type_id,
      IsPackedVec4(info) ? info.packed_vec4_type_id : 0,
      IsPackedVec4(info) ? info.packed_length : 0, info.length, scalar_ids,
      "invalid HW vector scalar rebuild");
}

bool HwLowerToStandardPass::RebuildMatrixFromScalars(
    Instruction* inst, const MatrixTypeInfo& info,
    const std::vector<uint32_t>& scalar_ids) {
  return RebuildAggregateFromScalars(
      inst, info.lowered_type_id,
      IsPackedVec4(info) ? info.packed_vec4_type_id : 0,
      IsPackedVec4(info) ? info.rows * info.packed_cols : 0,
      info.rows * info.cols, scalar_ids, "invalid HW matrix scalar rebuild");
}

bool HwLowerToStandardPass::RebuildAggregateFromScalars(
    Instruction* inst, uint32_t lowered_type_id, uint32_t packed_vec4_type_id,
    uint32_t packed_piece_count, uint32_t expected_scalar_count,
    const std::vector<uint32_t>& scalar_ids, const char* error_message) {
  if (scalar_ids.size() != expected_scalar_count ||
      (packed_vec4_type_id != 0 &&
       packed_piece_count * kPackedVec4Width != expected_scalar_count)) {
    ReportError(inst, error_message);
    return false;
  }

  if (packed_vec4_type_id == 0) {
    RebuildAsCompositeConstruct(inst, lowered_type_id, scalar_ids);
    return true;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> piece_ids;
  piece_ids.reserve(packed_piece_count);
  for (uint32_t piece = 0; piece < packed_piece_count; ++piece) {
    std::vector<uint32_t> lane_ids;
    lane_ids.reserve(kPackedVec4Width);
    for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
      lane_ids.push_back(scalar_ids[piece * kPackedVec4Width + lane]);
    }
    Instruction* vec =
        builder.AddCompositeConstruct(packed_vec4_type_id, lane_ids);
    if (!vec) return false;
    piece_ids.push_back(vec->result_id());
  }
  RebuildAsCompositeConstruct(inst, lowered_type_id, piece_ids);
  return true;
}

uint32_t HwLowerToStandardPass::BuildMatrixRowVector(
    InstructionBuilder* builder, const MatrixTypeInfo& info, uint32_t matrix_id,
    uint32_t row, uint32_t col_start, uint32_t vec4_type_id) {
  if (IsPackedVec4(info) && col_start % kPackedVec4Width == 0 &&
      col_start + kPackedVec4Width <= info.cols &&
      info.packed_vec4_type_id == vec4_type_id) {
    return ExtractCompositeElement(
        builder, vec4_type_id, matrix_id,
        MatrixPackedIndex(info, row, col_start / kPackedVec4Width));
  }

  const uint32_t zero_id = GetOrCreateZero(info.component_type_id);
  if (zero_id == 0) return 0;
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    const uint32_t col = col_start + lane;
    const uint32_t value_id =
        col < info.cols
            ? ExtractMatrixScalar(builder, info, matrix_id, row, col)
            : zero_id;
    if (value_id == 0) return 0;
    lane_ids.push_back(value_id);
  }
  Instruction* vec = builder->AddCompositeConstruct(vec4_type_id, lane_ids);
  return vec ? vec->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildMatrixColumnVector(
    InstructionBuilder* builder, const MatrixTypeInfo& info, uint32_t matrix_id,
    uint32_t row_start, uint32_t col, uint32_t vec4_type_id) {
  const uint32_t zero_id = GetOrCreateZero(info.component_type_id);
  if (zero_id == 0) return 0;

  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    const uint32_t row = row_start + lane;
    const uint32_t value_id =
        row < info.rows && col < info.cols
            ? ExtractMatrixScalar(builder, info, matrix_id, row, col)
            : zero_id;
    if (value_id == 0) return 0;
    lane_ids.push_back(value_id);
  }
  Instruction* vec = builder->AddCompositeConstruct(vec4_type_id, lane_ids);
  return vec ? vec->result_id() : 0;
}

void HwLowerToStandardPass::RebuildAsCompositeConstruct(
    Instruction* inst, uint32_t type_id,
    const std::vector<uint32_t>& element_ids) {
  std::vector<Operand> operands;
  operands.reserve(element_ids.size());
  for (uint32_t id : element_ids) operands.push_back(IdOperand(id));
  inst->SetOpcode(spv::Op::OpCompositeConstruct);
  inst->SetResultType(type_id);
  inst->SetInOperands(std::move(operands));
  context()->UpdateDefUse(inst);
}

void HwLowerToStandardPass::RebuildAsFunctionCall(
    Instruction* inst, uint32_t type_id, uint32_t function_id,
    const std::vector<uint32_t>& argument_ids) {
  std::vector<Operand> operands;
  operands.reserve(argument_ids.size() + 1);
  operands.push_back(IdOperand(function_id));
  for (uint32_t id : argument_ids) operands.push_back(IdOperand(id));
  inst->SetOpcode(spv::Op::OpFunctionCall);
  inst->SetResultType(type_id);
  inst->SetInOperands(std::move(operands));
  context()->UpdateDefUse(inst);
}

uint32_t HwLowerToStandardPass::GetOrCreateModuleConstantFromCompositeConstruct(
    Instruction* composite_construct) {
  if (!composite_construct ||
      composite_construct->opcode() != spv::Op::OpCompositeConstruct) {
    return 0;
  }

  // Check if all operands are module-scope constants
  std::vector<Operand> const_operands;
  for (uint32_t i = 0; i < composite_construct->NumInOperands(); ++i) {
    const uint32_t operand_id = composite_construct->GetSingleWordInOperand(i);
    Instruction* operand_def = get_def_use_mgr()->GetDef(operand_id);
    if (!operand_def) return 0;

    // Check if operand is a module-scope constant
    const spv::Op op = operand_def->opcode();
    if (op != spv::Op::OpConstant && op != spv::Op::OpConstantNull &&
        op != spv::Op::OpConstantComposite && op != spv::Op::OpConstantTrue &&
        op != spv::Op::OpConstantFalse) {
      return 0;  // Not all operands are constants
    }
    const_operands.push_back(IdOperand(operand_id));
  }

  // Create a module-level OpConstantComposite
  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;

  std::unique_ptr<Instruction> new_const = MakeUnique<Instruction>(
      context(), spv::Op::OpConstantComposite, composite_construct->type_id(),
      result_id, std::move(const_operands));

  // Add as global value
  context()->AddGlobalValue(std::move(new_const));
  Instruction* inserted = &*(--context()->types_values_end());
  if (!inserted) return 0;

  get_def_use_mgr()->AnalyzeInstDefUse(inserted);
  return result_id;
}

}  // namespace opt
}  // namespace spvtools
