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

Instruction* MoveTypeAfter(Instruction* type, Instruction* insert_after) {
  if (!type || !insert_after || type == insert_after) return nullptr;
  type->RemoveFromList();
  std::unique_ptr<Instruction> owned_type(type);
  return insert_after->NextNode()->InsertBefore(std::move(owned_type));
}

}  // namespace

bool HwLowerToStandardPass::CollectHwTypes() {
  std::vector<Instruction*> hw_types;
  get_module()->ForEachInst([this, &hw_types](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpTypeCooperativeMatrixHW ||
        inst->opcode() == spv::Op::OpTypeCooperativeVectorHW) {
      hw_types.push_back(inst);
    }
  });

  for (Instruction* inst : hw_types) {
    if (inst->opcode() == spv::Op::OpTypeCooperativeMatrixHW) {
      if (inst->NumInOperands() < 3) {
        ReportError(inst, "OpTypeCooperativeMatrixHW is missing operands");
        return false;
      }

      MatrixTypeInfo info;
      info.type_id = inst->result_id();
      info.component_type_id = inst->GetSingleWordInOperand(0);
      if (inst->NumInOperands() >= 4) {
        info.has_matrix_use = true;
        info.matrix_use = static_cast<spv::CooperativeMatrixUseHW>(
            inst->GetSingleWordInOperand(3));
      }
      Instruction* rows_def =
          get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(1));
      Instruction* cols_def =
          get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(2));
      if ((rows_def && IsSpecConstantInst(rows_def->opcode())) ||
          (cols_def && IsSpecConstantInst(cols_def->opcode()))) {
        ReportError(inst,
                    "HW cooperative matrix specialization-constant shape is "
                    "not supported");
        return false;
      }
      if (!GetConstantU32(inst->GetSingleWordInOperand(1), &info.rows) ||
          !GetConstantU32(inst->GetSingleWordInOperand(2), &info.cols)) {
        ReportError(inst,
                    "HW cooperative matrix rows/columns must be constants");
        return false;
      }
      const uint64_t element_count =
          static_cast<uint64_t>(info.rows) * static_cast<uint64_t>(info.cols);
      if (element_count == 0 || element_count > max_elements_) {
        ReportError(inst, "HW cooperative matrix shape is unsupported");
        return false;
      }

      Instruction* component_type =
          get_def_use_mgr()->GetDef(info.component_type_id);
      if (!IsSupportedHwComponentType(component_type)) {
        ReportError(inst,
                    "HW cooperative matrix component type must be 16/32-bit "
                    "floating-point or 8/16/32-bit integer");
        return false;
      }

      if (IsFloat16Type(info.component_type_id) &&
          ShouldUsePackedVec2(info.cols)) {
        info.packed_f16vec2 = true;
        info.packed_cols = info.cols / kPackedVec2Width;
      } else if (IsFloat32Type(info.component_type_id) &&
                 ShouldUsePackedVec2(info.cols)) {
        info.packed_f32vec2 = true;
        info.packed_cols = info.cols / kPackedVec2Width;
      }
      matrix_types_[info.type_id] = info;
      continue;
    }

    if (inst->NumInOperands() < 2) {
      ReportError(inst, "OpTypeCooperativeVectorHW is missing operands");
      return false;
    }

    VectorTypeInfo info;
    info.type_id = inst->result_id();
    info.component_type_id = inst->GetSingleWordInOperand(0);
    Instruction* length_def =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(1));
    if (length_def && IsSpecConstantInst(length_def->opcode())) {
      ReportError(inst,
                  "HW cooperative vector specialization-constant shape is "
                  "not supported");
      return false;
    }
    if (!GetConstantU32(inst->GetSingleWordInOperand(1), &info.length)) {
      ReportError(inst, "HW cooperative vector length must be constant");
      return false;
    }
    if (info.length == 0 || info.length > max_elements_) {
      ReportError(inst, "HW cooperative vector length is unsupported");
      return false;
    }

    Instruction* component_type =
        get_def_use_mgr()->GetDef(info.component_type_id);
    if (!IsSupportedHwComponentType(component_type)) {
      ReportError(inst,
                  "HW cooperative vector component type must be 16/32-bit "
                  "floating-point or 8/16/32-bit integer");
      return false;
    }

    if (IsFloat16Type(info.component_type_id) &&
        ShouldUsePackedVec2(info.length)) {
      info.packed_f16vec2 = true;
      info.packed_length = info.length / kPackedVec2Width;
    } else if (IsFloat32Type(info.component_type_id) &&
               ShouldUsePackedVec2(info.length)) {
      info.packed_f32vec2 = true;
      info.packed_length = info.length / kPackedVec2Width;
    }
    vector_types_[info.type_id] = info;
  }

  return true;
}

bool HwLowerToStandardPass::MaterializeLoweredTypes() {
  // Materialize ordinary types only after every cooperative declaration and
  // operation has passed read-only legality and limit analysis.  A rejected
  // module must not be left partially rewritten.
  std::vector<uint32_t> type_ids;
  for (Instruction& inst : get_module()->types_values()) {
    if (matrix_types_.count(inst.result_id()) ||
        vector_types_.count(inst.result_id())) {
      type_ids.push_back(inst.result_id());
    }
  }
  for (uint32_t type_id : type_ids) {
    auto matrix_entry = matrix_types_.find(type_id);
    if (matrix_entry != matrix_types_.end()) {
      MatrixTypeInfo& info = matrix_entry->second;
      Instruction* insertion_point = get_def_use_mgr()->GetDef(info.type_id);
      if (!insertion_point) return false;
      if (IsPackedVec2(info)) {
        info.packed_vec2_type_id = GetOrCreateVectorType(
            info.component_type_id, kPackedVec2Width, &insertion_point);
        if (info.packed_vec2_type_id == 0) return false;
        info.lowered_type_id = GetOrCreatePackedArrayType(
            info.packed_vec2_type_id, info.rows * info.packed_cols,
            insertion_point);
      } else {
        info.lowered_type_id = GetOrCreateArrayType(
            info.component_type_id, info.rows * info.cols, insertion_point);
      }
      if (info.lowered_type_id == 0) return false;
      lowered_types_[info.type_id] = info.lowered_type_id;
      continue;
    }

    auto vector_entry = vector_types_.find(type_id);
    if (vector_entry != vector_types_.end()) {
      VectorTypeInfo& info = vector_entry->second;
      Instruction* insertion_point = get_def_use_mgr()->GetDef(info.type_id);
      if (!insertion_point) return false;
      if (IsPackedVec2(info)) {
        info.packed_vec2_type_id = GetOrCreateVectorType(
            info.component_type_id, kPackedVec2Width, &insertion_point);
        if (info.packed_vec2_type_id == 0) return false;
        info.lowered_type_id = GetOrCreatePackedArrayType(
            info.packed_vec2_type_id, info.packed_length, insertion_point);
      } else {
        info.lowered_type_id = GetOrCreateArrayType(
            info.component_type_id, info.length, insertion_point);
      }
      if (info.lowered_type_id == 0) return false;
      lowered_types_[info.type_id] = info.lowered_type_id;
      continue;
    }
  }

  return true;
}

void HwLowerToStandardPass::RecordOriginalHwValueTypes() {
  get_module()->ForEachInst([this](Instruction* inst) {
    if (inst->result_id() != 0 && IsHwType(inst->type_id())) {
      original_hw_value_types_[inst->result_id()] = inst->type_id();
    }
  });
}

uint32_t HwLowerToStandardPass::GetOrCreateArrayType(
    uint32_t component_type_id, uint32_t length, Instruction* insert_after) {
  Instruction* insertion_point = insert_after;
  uint32_t length_id = GetOrCreateUIntConstantAfter(length, &insertion_point);
  if (length_id == 0) return 0;

  bool passed_insertion_point = false;
  Instruction* later_array_type = nullptr;
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeArray &&
        inst.GetSingleWordInOperand(0) == component_type_id) {
      uint32_t existing_length = 0;
      if (GetConstantU32(inst.GetSingleWordInOperand(1), &existing_length) &&
          existing_length == length) {
        if (!passed_insertion_point) return inst.result_id();
        later_array_type = &inst;
        break;
      }
    }
    if (&inst == insertion_point) passed_insertion_point = true;
  }

  if (later_array_type) {
    // Preserve the identity of an existing aggregate type so function types
    // that become equivalent after lowering can still be canonicalized. The
    // replacement length constant is guaranteed to precede the insertion
    // point, unlike the existing array's length declaration in general.
    if (later_array_type->GetSingleWordInOperand(1) != length_id) {
      later_array_type->SetInOperand(1, {length_id});
      context()->UpdateDefUse(later_array_type);
    }
    Instruction* moved = MoveTypeAfter(later_array_type, insertion_point);
    return moved ? moved->result_id() : 0;
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::unique_ptr<Instruction> array_type = MakeUnique<Instruction>(
      context(), spv::Op::OpTypeArray, 0, result_id,
      std::initializer_list<Operand>{IdOperand(component_type_id),
                                     IdOperand(length_id)});
  AddTypeOrGlobalAfter(context(), insertion_point, std::move(array_type));
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateVectorType(
    uint32_t component_type_id, uint32_t component_count,
    Instruction** insert_after) {
  bool passed_insertion_point = false;
  Instruction* later_vector_type = nullptr;
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeVector &&
        inst.GetSingleWordInOperand(0) == component_type_id &&
        inst.GetSingleWordInOperand(1) == component_count) {
      if (!passed_insertion_point) return inst.result_id();
      later_vector_type = &inst;
      break;
    }
    if (&inst == *insert_after) passed_insertion_point = true;
  }

  if (later_vector_type) {
    Instruction* moved = MoveTypeAfter(later_vector_type, *insert_after);
    if (!moved) return 0;
    *insert_after = moved;
    return moved->result_id();
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::unique_ptr<Instruction> vector_type = MakeUnique<Instruction>(
      context(), spv::Op::OpTypeVector, 0, result_id,
      std::initializer_list<Operand>{
          IdOperand(component_type_id),
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {component_count}}});
  Instruction* added =
      AddTypeOrGlobalAfter(context(), *insert_after, std::move(vector_type));
  *insert_after = added;
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreatePackedArrayType(
    uint32_t vec2_type_id, uint32_t length, Instruction* insert_after) {
  return GetOrCreateArrayType(vec2_type_id, length, insert_after);
}

uint32_t HwLowerToStandardPass::GetOrCreatePointerType(
    uint32_t pointee_type_id, spv::StorageClass storage_class) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypePointer &&
        inst.GetSingleWordInOperand(0) ==
            static_cast<uint32_t>(storage_class) &&
        inst.GetSingleWordInOperand(1) == pointee_type_id) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddType(MakeUnique<Instruction>(
      context(), spv::Op::OpTypePointer, 0, result_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_STORAGE_CLASS,
                                      {static_cast<uint32_t>(storage_class)}},
                                     IdOperand(pointee_type_id)}));
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateVoidType() {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeVoid) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddType(MakeUnique<Instruction>(context(), spv::Op::OpTypeVoid, 0,
                                             result_id,
                                             std::initializer_list<Operand>{}));
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateBoolType() {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeBool) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddType(MakeUnique<Instruction>(context(), spv::Op::OpTypeBool, 0,
                                             result_id,
                                             std::initializer_list<Operand>{}));
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateUIntType() {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeInt &&
        inst.GetSingleWordInOperand(0) == 32 &&
        inst.GetSingleWordInOperand(1) == 0) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddType(MakeUnique<Instruction>(
      context(), spv::Op::OpTypeInt, 0, result_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_LITERAL_INTEGER, {32}},
                                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}}));
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateIntegerType(uint32_t width,
                                                       bool is_signed) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeInt && inst.NumInOperands() >= 2 &&
        inst.GetSingleWordInOperand(0) == width &&
        (inst.GetSingleWordInOperand(1) != 0) == is_signed) {
      return inst.result_id();
    }
  }
  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddType(MakeUnique<Instruction>(
      context(), spv::Op::OpTypeInt, 0, result_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {width}},
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {is_signed ? 1u : 0u}}}));
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateUIntConstant(uint32_t value) {
  const uint32_t uint_type_id = GetOrCreateUIntType();
  if (uint_type_id == 0) return 0;
  return GetOrCreateConstant(uint_type_id, value);
}

uint32_t HwLowerToStandardPass::GetOrCreateUIntConstantAfter(
    uint32_t value, Instruction** insert_after) {
  const uint32_t uint_type_id = GetOrCreateUIntTypeAfter(insert_after);
  if (uint_type_id == 0) return 0;

  bool can_reuse = true;
  for (Instruction& inst : get_module()->types_values()) {
    if (can_reuse && inst.opcode() == spv::Op::OpConstant &&
        inst.type_id() == uint_type_id && inst.NumInOperands() == 1 &&
        inst.GetSingleWordInOperand(0) == value) {
      return inst.result_id();
    }
    if (&inst == *insert_after) can_reuse = false;
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::unique_ptr<Instruction> constant = MakeUnique<Instruction>(
      context(), spv::Op::OpConstant, uint_type_id, result_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {value}}});
  Instruction* added =
      AddTypeOrGlobalAfter(context(), *insert_after, std::move(constant));
  *insert_after = added;
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateUIntTypeAfter(
    Instruction** insert_after) {
  bool can_reuse = true;
  for (Instruction& inst : get_module()->types_values()) {
    if (can_reuse && inst.opcode() == spv::Op::OpTypeInt &&
        inst.GetSingleWordInOperand(0) == 32 &&
        inst.GetSingleWordInOperand(1) == 0) {
      return inst.result_id();
    }
    if (&inst == *insert_after) can_reuse = false;
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::unique_ptr<Instruction> uint_type = MakeUnique<Instruction>(
      context(), spv::Op::OpTypeInt, 0, result_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_LITERAL_INTEGER, {32}},
                                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}});
  Instruction* added =
      AddTypeOrGlobalAfter(context(), *insert_after, std::move(uint_type));
  *insert_after = added;
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateConstant(uint32_t type_id,
                                                    uint32_t value) {
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  if (!type || type->opcode() != spv::Op::OpTypeInt ||
      type->NumInOperands() < 2 || type->GetSingleWordInOperand(0) == 0 ||
      type->GetSingleWordInOperand(0) > 64) {
    return 0;
  }
  const uint32_t word_count = type->GetSingleWordInOperand(0) > 32 ? 2 : 1;

  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpConstant && inst.type_id() == type_id &&
        inst.NumInOperands() == 1) {
      const Operand& literal = inst.GetInOperand(0);
      if (literal.words.size() == word_count && literal.words[0] == value &&
          (word_count == 1 || literal.words[1] == 0)) {
        return inst.result_id();
      }
    }
    if (value == 0 && inst.opcode() == spv::Op::OpConstantNull &&
        inst.type_id() == type_id) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::vector<uint32_t> words = {value};
  if (word_count == 2) words.push_back(0);
  context()->AddGlobalValue(MakeUnique<Instruction>(
      context(), spv::Op::OpConstant, type_id, result_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, std::move(words)}}));
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateZero(uint32_t type_id) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.type_id() != type_id) continue;
    if (inst.opcode() == spv::Op::OpConstantNull) {
      return inst.result_id();
    }
    if (inst.opcode() == spv::Op::OpConstant && inst.NumInOperands() == 1) {
      const Operand& literal = inst.GetInOperand(0);
      if (!literal.words.empty() &&
          std::all_of(literal.words.begin(), literal.words.end(),
                      [](uint32_t word) { return word == 0; })) {
        return inst.result_id();
      }
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddGlobalValue(
      MakeUnique<Instruction>(context(), spv::Op::OpConstantNull, type_id,
                              result_id, std::initializer_list<Operand>{}));
  return result_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateCompositeConstant(
    uint32_t type_id, const std::vector<uint32_t>& constituent_ids,
    Instruction** insert_after, Instruction* reuse_before, spv::Op opcode) {
  if (opcode != spv::Op::OpConstantComposite &&
      opcode != spv::Op::OpSpecConstantComposite) {
    return 0;
  }
  bool can_reuse = true;
  for (Instruction& inst : get_module()->types_values()) {
    if (&inst == reuse_before) can_reuse = false;
    if (can_reuse && inst.opcode() == opcode && inst.type_id() == type_id &&
        inst.NumInOperands() == constituent_ids.size()) {
      bool matches = true;
      for (uint32_t i = 0; i < constituent_ids.size(); ++i) {
        if (inst.GetSingleWordInOperand(i) != constituent_ids[i]) {
          matches = false;
          break;
        }
      }
      if (matches) return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::vector<Operand> operands;
  operands.reserve(constituent_ids.size());
  for (uint32_t id : constituent_ids) operands.push_back(IdOperand(id));
  Instruction* added = AddTypeOrGlobalAfter(
      context(), insert_after ? *insert_after : nullptr,
      MakeUnique<Instruction>(context(), opcode, type_id, result_id, operands));
  if (!added) return 0;
  if (insert_after) *insert_after = added;
  return result_id;
}

const HwLowerToStandardPass::MatrixTypeInfo*
HwLowerToStandardPass::GetMatrixType(uint32_t type_id) const {
  auto it = matrix_types_.find(type_id);
  if (it != matrix_types_.end()) return &it->second;
  return nullptr;
}

const HwLowerToStandardPass::MatrixTypeInfo*
HwLowerToStandardPass::GetMatrixTypeForValue(const Instruction* value) const {
  if (!value) return nullptr;
  auto original = original_hw_value_types_.find(value->result_id());
  if (original != original_hw_value_types_.end()) {
    return GetMatrixType(original->second);
  }
  return GetMatrixType(value->type_id());
}

const HwLowerToStandardPass::VectorTypeInfo*
HwLowerToStandardPass::GetVectorType(uint32_t type_id) const {
  auto it = vector_types_.find(type_id);
  if (it != vector_types_.end()) return &it->second;
  for (const auto& id_and_info : vector_types_) {
    if (id_and_info.second.lowered_type_id == type_id)
      return &id_and_info.second;
  }
  return nullptr;
}

uint32_t HwLowerToStandardPass::GetLoweredType(uint32_t type_id) const {
  auto it = lowered_types_.find(type_id);
  return it == lowered_types_.end() ? 0 : it->second;
}

uint32_t HwLowerToStandardPass::GetPointerTypeId(uint32_t pointer_id) const {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  return pointer ? pointer->type_id() : 0;
}

uint32_t HwLowerToStandardPass::GetPointeeType(uint32_t pointer_type_id) const {
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer_type_id);
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    return 0;
  }
  return pointer_type->GetSingleWordInOperand(1);
}

bool HwLowerToStandardPass::GetConstantU32(uint32_t id, uint32_t* value) const {
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  if (!inst) return false;
  if (inst->opcode() == spv::Op::OpConstantNull) {
    Instruction* type = get_def_use_mgr()->GetDef(inst->type_id());
    if (!type || type->opcode() != spv::Op::OpTypeInt) return false;
    *value = 0;
    return true;
  }
  if (inst->opcode() != spv::Op::OpConstant || inst->NumInOperands() != 1)
    return false;
  Instruction* type = get_def_use_mgr()->GetDef(inst->type_id());
  const Operand& literal = inst->GetInOperand(0);
  if (!type || type->opcode() != spv::Op::OpTypeInt ||
      type->GetSingleWordInOperand(0) > 64 || literal.words.empty() ||
      (type->GetSingleWordInOperand(0) > 32 &&
       (literal.words.size() < 2 || literal.words[1] != 0))) {
    return false;
  }
  *value = literal.words[0];
  return true;
}

bool HwLowerToStandardPass::IsFloat16Type(uint32_t type_id) const {
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  return type && type->opcode() == spv::Op::OpTypeFloat &&
         type->GetSingleWordInOperand(0) == 16;
}

bool HwLowerToStandardPass::IsFloat32Type(uint32_t type_id) const {
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  return type && type->opcode() == spv::Op::OpTypeFloat &&
         type->GetSingleWordInOperand(0) == 32;
}

bool HwLowerToStandardPass::IsHwType(uint32_t type_id) const {
  return matrix_types_.find(type_id) != matrix_types_.end() ||
         vector_types_.find(type_id) != vector_types_.end();
}

bool HwLowerToStandardPass::TypeContainsHw(uint32_t type_id) const {
  std::unordered_set<uint32_t> visited;
  return TypeContainsHwImpl(type_id, &visited);
}

bool HwLowerToStandardPass::TypeContainsHwImpl(
    uint32_t type_id, std::unordered_set<uint32_t>* visited) const {
  if (type_id == 0) return false;
  if (IsHwType(type_id)) return true;
  if (!visited || !visited->insert(type_id).second) return false;

  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  if (!type) return false;
  switch (type->opcode()) {
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
      return TypeContainsHwImpl(
          type->GetSingleWordInOperand(
              type->opcode() == spv::Op::OpTypePointer ? 1 : 0),
          visited);
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeStruct:
      for (uint32_t i = 0; i < type->NumInOperands(); ++i) {
        if (TypeContainsHwImpl(type->GetSingleWordInOperand(i), visited)) {
          return true;
        }
      }
      return false;
    default:
      return false;
  }
}

bool HwLowerToStandardPass::InstructionTouchesHw(
    const Instruction* inst) const {
  if (!inst) return false;
  if (TypeContainsHw(inst->type_id())) return true;
  if (inst->result_id() != 0 && IsTypeInst(inst->opcode()) &&
      TypeContainsHw(inst->result_id())) {
    return true;
  }

  bool touches_hw = false;
  inst->ForEachInId([this, &touches_hw](const uint32_t* id) {
    if (touches_hw || !id || *id == 0) return;
    if (TypeContainsHw(*id)) {
      touches_hw = true;
      return;
    }
    Instruction* value = get_def_use_mgr()->GetDef(*id);
    if (!value) return;
    uint32_t value_type_id = value->type_id();
    auto original = original_hw_value_types_.find(value->result_id());
    if (original != original_hw_value_types_.end()) {
      value_type_id = original->second;
    }
    touches_hw = TypeContainsHw(value_type_id);
  });
  return touches_hw;
}

bool HwLowerToStandardPass::HasHwTypeReference(const Instruction* inst) const {
  if (!inst) return false;
  if (TypeContainsHw(inst->type_id())) return true;
  if (inst->result_id() != 0 && IsTypeInst(inst->opcode()) &&
      TypeContainsHw(inst->result_id())) {
    return true;
  }

  bool has_hw_type_ref = false;
  inst->ForEachInId([this, &has_hw_type_ref](const uint32_t* id) {
    if (has_hw_type_ref) return;
    if (TypeContainsHw(*id)) has_hw_type_ref = true;
  });
  return has_hw_type_ref;
}

}  // namespace opt
}  // namespace spvtools
