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

#include "source/opt/azd_fix_cooperative_matrix_use_pass.h"

#include <string>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {

bool IsAzdMatrixMulAdd(const Instruction* inst) {
  return inst->opcode() == spv::Op::OpCooperativeMatrixMulAddAZD;
}

Instruction* AddTypeOrConstantBeforeGlobalValues(
    IRContext* context, std::unique_ptr<Instruction>&& inst) {
  auto insert_pos = context->types_values_begin();
  for (; insert_pos != context->types_values_end(); ++insert_pos) {
    if (!IsTypeInst(insert_pos->opcode()) &&
        !IsConstantInst(insert_pos->opcode())) {
      break;
    }
  }

  if (insert_pos == context->types_values_end()) {
    const bool is_type = IsTypeInst(inst->opcode());
    if (is_type) {
      context->AddType(std::move(inst));
    } else {
      context->AddGlobalValue(std::move(inst));
    }
    return &*(--context->types_values_end());
  }

  Instruction* added = &*insert_pos.InsertBefore(std::move(inst));
  if (context->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse)) {
    context->get_def_use_mgr()->AnalyzeInstDefUse(added);
  }
  return added;
}

}  // namespace

Pass::Status AzdFixCooperativeMatrixUsePass::Process() {
  value_use_stats_.clear();
  pointer_use_stats_.clear();
  pointer_preferred_pointees_.clear();
  processed_value_types_.clear();
  processed_pointer_pointees_.clear();

  get_module()->ForEachInst([this](Instruction* inst) {
    if (!IsAzdMatrixMulAdd(inst) || inst->NumInOperands() < 2) return;

    AddValueUseStat(inst->GetSingleWordInOperand(0),
                    &MatrixUseStat::left_count);
    AddValueUseStat(inst->GetSingleWordInOperand(1),
                    &MatrixUseStat::right_count);
    if (inst->NumInOperands() > 2) {
      AddValueUseStat(inst->GetSingleWordInOperand(2),
                      &MatrixUseStat::accumulator_count);
    }
    AddValueUseStat(inst->result_id(), &MatrixUseStat::accumulator_count);
  });

  for (const auto& id_and_stat : value_use_stats_) {
    if (HasRoleConflict(id_and_stat.second)) {
      ReportRoleConflict(id_and_stat.first);
      return Status::Failure;
    }
  }

  for (const auto& id_and_stat : value_use_stats_) {
    Instruction* value_inst = get_def_use_mgr()->GetDef(id_and_stat.first);
    if (!value_inst || value_inst->opcode() != spv::Op::OpLoad ||
        value_inst->NumInOperands() == 0) {
      continue;
    }

    const uint32_t pointer_id = value_inst->GetSingleWordInOperand(0);
    const uint32_t pointee_type_id = GetPointerPointeeType(pointer_id);
    if (!IsAzdCooperativeMatrixType(pointee_type_id)) continue;

    AddUseStats(&pointer_use_stats_[pointer_id], id_and_stat.second);
  }

  for (const auto& id_and_stat : pointer_use_stats_) {
    const uint32_t pointer_id = id_and_stat.first;
    const uint32_t pointee_type_id = GetPointerPointeeType(pointer_id);
    if (!IsAzdCooperativeMatrixType(pointee_type_id)) continue;
    if (HasRoleConflict(id_and_stat.second)) {
      ReportRoleConflict(pointer_id);
      return Status::Failure;
    }

    const uint32_t new_type_id = GetOrCreateAzdCooperativeMatrixTypeWithUse(
        pointee_type_id, InferUse(id_and_stat.second));
    if (new_type_id == 0) return Status::Failure;

    pointer_preferred_pointees_[pointer_id] = new_type_id;
  }

  bool modified = false;
  for (const auto& id_and_stat : value_use_stats_) {
    const uint32_t value_id = id_and_stat.first;
    Instruction* value_inst = get_def_use_mgr()->GetDef(value_id);
    if (!value_inst || !IsAzdCooperativeMatrixType(value_inst->type_id())) {
      continue;
    }

    const uint32_t new_type_id = GetOrCreateAzdCooperativeMatrixTypeWithUse(
        value_inst->type_id(), InferUse(id_and_stat.second));
    if (new_type_id == 0) return Status::Failure;

    uint32_t rewrite_type_id = new_type_id;
    if (value_inst->opcode() == spv::Op::OpLoad &&
        value_inst->NumInOperands() > 0) {
      const uint32_t pointer_id = value_inst->GetSingleWordInOperand(0);
      auto preferred = pointer_preferred_pointees_.find(pointer_id);
      if (preferred != pointer_preferred_pointees_.end()) {
        rewrite_type_id = preferred->second;
      }
    }

    modified |= RewriteValueType(value_id, rewrite_type_id);
  }

  modified |= FixCopyObjectTypeMismatches();
  modified |= FixStoreTypeMismatches();

  return modified ? Status::SuccessWithChange
                  : Status::SuccessWithoutChange;
}

bool AzdFixCooperativeMatrixUsePass::IsAzdCooperativeMatrixType(
    uint32_t type_id) const {
  Instruction* type_inst = get_def_use_mgr()->GetDef(type_id);
  return type_inst &&
         type_inst->opcode() == spv::Op::OpTypeCooperativeMatrixAZD;
}

bool AzdFixCooperativeMatrixUsePass::HasRoleConflict(
    const MatrixUseStat& stat) const {
  return stat.accumulator_count > 0 &&
         (stat.left_count > 0 || stat.right_count > 0);
}

void AzdFixCooperativeMatrixUsePass::ReportRoleConflict(uint32_t id) const {
  if (!consumer()) return;

  const std::string message =
      "AZD cooperative matrix id " + std::to_string(id) +
      " has both OperandAB and Accumulator uses";
  consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
}

void AzdFixCooperativeMatrixUsePass::AddValueUseStat(
    uint32_t value_id, uint32_t MatrixUseStat::*field) {
  if (value_id == 0) return;

  Instruction* value_inst = get_def_use_mgr()->GetDef(value_id);
  if (!value_inst || !IsAzdCooperativeMatrixType(value_inst->type_id())) {
    return;
  }

  value_use_stats_[value_id].*field += 1;
}

void AzdFixCooperativeMatrixUsePass::AddUseStats(
    MatrixUseStat* target, const MatrixUseStat& source) const {
  target->left_count += source.left_count;
  target->right_count += source.right_count;
  target->accumulator_count += source.accumulator_count;
}

spv::CooperativeMatrixUseAZD AzdFixCooperativeMatrixUsePass::InferUse(
    const MatrixUseStat& stat) const {
  if (stat.accumulator_count > 0) {
    return spv::CooperativeMatrixUseAZD::MatrixAccumulatorAZD;
  }
  return stat.left_count >= stat.right_count
             ? spv::CooperativeMatrixUseAZD::MatrixUseAAZD
             : spv::CooperativeMatrixUseAZD::MatrixUseBAZD;
}

uint32_t
AzdFixCooperativeMatrixUsePass::GetOrCreateAzdCooperativeMatrixTypeWithUse(
    uint32_t old_type_id, spv::CooperativeMatrixUseAZD use) {
  Instruction* old_type_inst = get_def_use_mgr()->GetDef(old_type_id);
  if (!old_type_inst ||
      old_type_inst->opcode() != spv::Op::OpTypeCooperativeMatrixAZD) {
    return 0;
  }

  std::vector<uint32_t> operands;
  const uint32_t old_operand_count = old_type_inst->NumInOperands();
  const bool old_has_use_operand = old_operand_count > 3;
  const uint32_t operands_to_copy =
      old_has_use_operand ? old_operand_count - 1 : old_operand_count;
  for (uint32_t i = 0; i < operands_to_copy; ++i) {
    operands.push_back(old_type_inst->GetSingleWordInOperand(i));
  }
  operands.push_back(static_cast<uint32_t>(use));

  for (Instruction* type_inst : get_module()->GetTypes()) {
    if (type_inst->opcode() != spv::Op::OpTypeCooperativeMatrixAZD ||
        type_inst->NumInOperands() != operands.size()) {
      continue;
    }

    bool same_type = true;
    for (uint32_t i = 0; i < operands.size(); ++i) {
      if (type_inst->GetSingleWordInOperand(i) != operands[i]) {
        same_type = false;
        break;
      }
    }
    if (same_type) return type_inst->result_id();
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;

  Instruction::OperandList in_operands;
  in_operands.reserve(operands.size());
  for (uint32_t i = 0; i < operands.size(); ++i) {
    const spv_operand_type_t operand_type =
        i + 1 == operands.size() ? SPV_OPERAND_TYPE_COOPERATIVE_MATRIX_USE_AZD
                                 : SPV_OPERAND_TYPE_ID;
    in_operands.push_back({operand_type, {operands[i]}});
  }
  AddTypeOrConstantBeforeGlobalValues(context(), MakeUnique<Instruction>(
      context(), spv::Op::OpTypeCooperativeMatrixAZD, 0, result_id,
      in_operands));
  return result_id;
}

uint32_t AzdFixCooperativeMatrixUsePass::GetOrCreatePointerType(
    uint32_t old_pointer_type_id, uint32_t pointee_type_id) {
  Instruction* old_pointer_type = get_def_use_mgr()->GetDef(old_pointer_type_id);
  if (!old_pointer_type ||
      old_pointer_type->opcode() != spv::Op::OpTypePointer) {
    return 0;
  }

  const uint32_t storage_class = old_pointer_type->GetSingleWordInOperand(0);
  for (Instruction* type_inst : get_module()->GetTypes()) {
    if (type_inst->opcode() == spv::Op::OpTypePointer &&
        type_inst->GetSingleWordInOperand(0) == storage_class &&
        type_inst->GetSingleWordInOperand(1) == pointee_type_id) {
      return type_inst->result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;

  AddTypeOrConstantBeforeGlobalValues(context(), MakeUnique<Instruction>(
      context(), spv::Op::OpTypePointer, 0, result_id,
      Instruction::OperandList{
          {SPV_OPERAND_TYPE_STORAGE_CLASS, {storage_class}},
          {SPV_OPERAND_TYPE_ID, {pointee_type_id}},
      }));
  return result_id;
}

uint32_t AzdFixCooperativeMatrixUsePass::GetPointerPointeeType(
    uint32_t pointer_id) const {
  Instruction* pointer_inst = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer_inst || pointer_inst->type_id() == 0) return 0;

  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer_inst->type_id());
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer ||
      pointer_type->NumInOperands() < 2) {
    return 0;
  }

  return pointer_type->GetSingleWordInOperand(1);
}

uint32_t AzdFixCooperativeMatrixUsePass::GetStorePointerPointeeType(
    uint32_t pointer_id, uint32_t fallback_type_id) const {
  auto preferred = pointer_preferred_pointees_.find(pointer_id);
  return preferred == pointer_preferred_pointees_.end() ? fallback_type_id
                                                       : preferred->second;
}

bool AzdFixCooperativeMatrixUsePass::ChangeResultType(Instruction* inst,
                                                      uint32_t new_type_id) {
  if (!inst || inst->type_id() == 0 || inst->type_id() == new_type_id) {
    return false;
  }

  context()->ForgetUses(inst);
  inst->SetResultType(new_type_id);
  context()->AnalyzeUses(inst);
  return true;
}

bool AzdFixCooperativeMatrixUsePass::RewriteValueType(uint32_t value_id,
                                                      uint32_t new_type_id) {
  if (value_id == 0 || new_type_id == 0) return false;

  auto processed = processed_value_types_.find(value_id);
  if (processed != processed_value_types_.end()) {
    return false;
  }
  processed_value_types_[value_id] = new_type_id;

  Instruction* value_inst = get_def_use_mgr()->GetDef(value_id);
  if (!value_inst || value_inst->type_id() == 0) return false;
  if (value_inst->type_id() != new_type_id &&
      !IsAzdCooperativeMatrixType(value_inst->type_id())) {
    return false;
  }

  bool modified = ChangeResultType(value_inst, new_type_id);

  switch (value_inst->opcode()) {
    case spv::Op::OpLoad:
      modified |= RewritePointerPointeeType(
          value_inst->GetSingleWordInOperand(0),
          GetStorePointerPointeeType(value_inst->GetSingleWordInOperand(0),
                                     new_type_id));
      break;
    case spv::Op::OpPhi:
      modified |= FixPhiIncomingTypes(value_inst, new_type_id);
      break;
    case spv::Op::OpSelect:
      modified |= FixSelectOperandTypes(value_inst, new_type_id);
      break;
    default:
      break;
  }

  std::vector<std::pair<Instruction*, uint32_t>> uses;
  get_def_use_mgr()->ForEachUse(value_inst,
                                [&uses](Instruction* use, uint32_t index) {
                                  uses.push_back({use, index});
                                });

  for (const auto& use : uses) {
    Instruction* user = use.first;
    switch (user->opcode()) {
      case spv::Op::OpStore:
        if (user->NumInOperands() > 1 &&
            user->GetSingleWordInOperand(1) == value_id) {
          modified |= RewritePointerPointeeType(
              user->GetSingleWordInOperand(0),
              GetStorePointerPointeeType(user->GetSingleWordInOperand(0),
                                         new_type_id));
        }
        break;
      default:
        break;
    }
  }

  return modified;
}

bool AzdFixCooperativeMatrixUsePass::RewritePointerPointeeType(
    uint32_t pointer_id, uint32_t pointee_type_id) {
  if (pointer_id == 0 || pointee_type_id == 0) return false;

  auto processed = processed_pointer_pointees_.find(pointer_id);
  if (processed != processed_pointer_pointees_.end()) {
    return false;
  }
  processed_pointer_pointees_[pointer_id] = pointee_type_id;

  Instruction* pointer_inst = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer_inst || pointer_inst->type_id() == 0) return false;

  Instruction* old_pointer_type =
      get_def_use_mgr()->GetDef(pointer_inst->type_id());
  if (!old_pointer_type ||
      old_pointer_type->opcode() != spv::Op::OpTypePointer) {
    return false;
  }

  const uint32_t new_pointer_type_id =
      GetOrCreatePointerType(pointer_inst->type_id(), pointee_type_id);
  if (new_pointer_type_id == 0) return false;

  bool modified = ChangeResultType(pointer_inst, new_pointer_type_id);

  switch (pointer_inst->opcode()) {
    case spv::Op::OpCopyObject:
      modified |= RewritePointerPointeeType(
          pointer_inst->GetSingleWordInOperand(0), pointee_type_id);
      break;
    case spv::Op::OpPhi:
      for (uint32_t i = 0; i < pointer_inst->NumInOperands(); i += 2) {
        modified |= RewritePointerPointeeType(
            pointer_inst->GetSingleWordInOperand(i), pointee_type_id);
      }
      break;
    case spv::Op::OpSelect:
      if (pointer_inst->NumInOperands() > 2) {
        modified |= RewritePointerPointeeType(
            pointer_inst->GetSingleWordInOperand(1), pointee_type_id);
        modified |= RewritePointerPointeeType(
            pointer_inst->GetSingleWordInOperand(2), pointee_type_id);
      }
      break;
    default:
      break;
  }

  std::vector<std::pair<Instruction*, uint32_t>> uses;
  get_def_use_mgr()->ForEachUse(pointer_inst,
                                [&uses](Instruction* use, uint32_t index) {
                                  uses.push_back({use, index});
                                });

  for (const auto& use : uses) {
    Instruction* user = use.first;
    switch (user->opcode()) {
      case spv::Op::OpLoad:
        if (user->NumInOperands() > 0 &&
            user->GetSingleWordInOperand(0) == pointer_id) {
          modified |= RewriteValueType(user->result_id(), pointee_type_id);
        }
        break;
      case spv::Op::OpStore:
        // OpStore is a conversion boundary for AZD cooperative matrices.  The
        // stored object is fixed up after all preferred pointer roles are known.
        break;
      case spv::Op::OpCopyObject:
        if (user->NumInOperands() > 0 &&
            user->GetSingleWordInOperand(0) == pointer_id) {
          modified |= RewritePointerPointeeType(user->result_id(),
                                                pointee_type_id);
        }
        break;
      case spv::Op::OpPhi:
        for (uint32_t i = 0; i < user->NumInOperands(); i += 2) {
          if (user->GetSingleWordInOperand(i) == pointer_id) {
            modified |= RewritePointerPointeeType(user->result_id(),
                                                  pointee_type_id);
            break;
          }
        }
        break;
      case spv::Op::OpSelect:
        if (user->NumInOperands() > 2 &&
            (user->GetSingleWordInOperand(1) == pointer_id ||
             user->GetSingleWordInOperand(2) == pointer_id)) {
          modified |= RewritePointerPointeeType(user->result_id(),
                                                pointee_type_id);
        }
        break;
      default:
        break;
    }
  }

  return modified;
}

bool AzdFixCooperativeMatrixUsePass::FixCopyObjectTypeMismatches() {
  bool modified = false;
  get_module()->ForEachInst([this, &modified](Instruction* inst) {
    if (inst->opcode() != spv::Op::OpCopyObject || inst->NumInOperands() == 0 ||
        !IsAzdCooperativeMatrixType(inst->type_id())) {
      return;
    }

    Instruction* object_inst =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
    if (!object_inst || !IsAzdCooperativeMatrixType(object_inst->type_id()) ||
        object_inst->type_id() == inst->type_id()) {
      return;
    }

    inst->SetOpcode(spv::Op::OpBitcast);
    modified = true;
  });
  return modified;
}

bool AzdFixCooperativeMatrixUsePass::FixStoreTypeMismatches() {
  std::vector<Instruction*> stores;
  get_module()->ForEachInst([&stores](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpStore && inst->NumInOperands() > 1) {
      stores.push_back(inst);
    }
  });

  bool modified = false;
  for (Instruction* store : stores) {
    const uint32_t pointer_id = store->GetSingleWordInOperand(0);
    const uint32_t object_id = store->GetSingleWordInOperand(1);
    const uint32_t pointee_type_id = GetPointerPointeeType(pointer_id);

    Instruction* object_inst = get_def_use_mgr()->GetDef(object_id);
    if (!object_inst || !IsAzdCooperativeMatrixType(pointee_type_id) ||
        !IsAzdCooperativeMatrixType(object_inst->type_id()) ||
        pointee_type_id == object_inst->type_id()) {
      continue;
    }

    const uint32_t converted_id =
        InsertBitcastBefore(store, pointee_type_id, object_id);
    if (converted_id == 0) continue;

    context()->ForgetUses(store);
    store->SetInOperand(1, {converted_id});
    context()->AnalyzeUses(store);
    modified = true;
  }

  return modified;
}

uint32_t AzdFixCooperativeMatrixUsePass::EnsureValueTypeBefore(
    Instruction* insert_before, uint32_t value_id, uint32_t target_type_id) {
  if (!insert_before || value_id == 0 || target_type_id == 0) return 0;

  Instruction* value_inst = get_def_use_mgr()->GetDef(value_id);
  if (!value_inst || value_inst->type_id() == 0) return 0;
  if (value_inst->type_id() == target_type_id) return value_id;

  if (IsAzdCooperativeMatrixType(value_inst->type_id()) &&
      IsAzdCooperativeMatrixType(target_type_id)) {
    return InsertBitcastBefore(insert_before, target_type_id, value_id);
  }

  return value_id;
}

uint32_t AzdFixCooperativeMatrixUsePass::InsertBitcastOnPhiEdge(
    Instruction* phi, uint32_t incoming_operand_index,
    uint32_t result_type_id) {
  if (!phi || phi->opcode() != spv::Op::OpPhi ||
      incoming_operand_index + 1 >= phi->NumInOperands()) {
    return 0;
  }

  const uint32_t incoming_value_id =
      phi->GetSingleWordInOperand(incoming_operand_index);
  const uint32_t predecessor_label_id =
      phi->GetSingleWordInOperand(incoming_operand_index + 1);
  BasicBlock* predecessor_block = context()->get_instr_block(predecessor_label_id);
  if (!predecessor_block || !predecessor_block->terminator()) return 0;

  return EnsureValueTypeBefore(predecessor_block->terminator(),
                               incoming_value_id, result_type_id);
}

bool AzdFixCooperativeMatrixUsePass::FixPhiIncomingTypes(
    Instruction* phi, uint32_t result_type_id) {
  if (!phi || phi->opcode() != spv::Op::OpPhi || result_type_id == 0) {
    return false;
  }

  bool modified = false;
  for (uint32_t i = 0; i < phi->NumInOperands(); i += 2) {
    const uint32_t incoming_value_id = phi->GetSingleWordInOperand(i);
    const uint32_t converted_id =
        InsertBitcastOnPhiEdge(phi, i, result_type_id);
    if (converted_id != 0 && converted_id != incoming_value_id) {
      phi->SetInOperand(i, {converted_id});
      modified = true;
    }
  }

  if (modified) context()->UpdateDefUse(phi);
  return modified;
}

bool AzdFixCooperativeMatrixUsePass::FixSelectOperandTypes(
    Instruction* select, uint32_t result_type_id) {
  if (!select || select->opcode() != spv::Op::OpSelect ||
      select->NumInOperands() <= 2 || result_type_id == 0) {
    return false;
  }

  bool modified = false;
  for (uint32_t i = 1; i <= 2; ++i) {
    const uint32_t operand_id = select->GetSingleWordInOperand(i);
    const uint32_t converted_id =
        EnsureValueTypeBefore(select, operand_id, result_type_id);
    if (converted_id != 0 && converted_id != operand_id) {
      select->SetInOperand(i, {converted_id});
      modified = true;
    }
  }

  if (modified) context()->UpdateDefUse(select);
  return modified;
}

uint32_t AzdFixCooperativeMatrixUsePass::InsertBitcastBefore(
    Instruction* insert_before, uint32_t result_type_id, uint32_t object_id) {
  if (!insert_before || result_type_id == 0 || object_id == 0) return 0;

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;

  Instruction* bitcast = insert_before->InsertBefore(MakeUnique<Instruction>(
      context(), spv::Op::OpBitcast, result_type_id, result_id,
      Instruction::OperandList{{SPV_OPERAND_TYPE_ID, {object_id}}}));
  BasicBlock* block = context()->get_instr_block(insert_before);
  if (block) context()->set_instr_block(bitcast, block);
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse)) {
    context()->get_def_use_mgr()->AnalyzeInstDefUse(bitcast);
  }
  return result_id;
}

}  // namespace opt
}  // namespace spvtools
