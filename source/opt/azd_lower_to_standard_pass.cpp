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

#include "source/opt/azd_lower_to_standard_pass.h"

#include <algorithm>
#include <string>
#include <utility>

#include "source/opt/basic_block.h"
#include "source/opt/constants.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {

constexpr uint32_t kAzdMatrixLoadPointerInIdx = 0;
constexpr uint32_t kAzdMatrixLoadShapeInIdx = 1;
constexpr uint32_t kAzdMatrixLoadOffsetInIdx = 2;
constexpr uint32_t kAzdMatrixLoadLayoutInIdx = 3;

constexpr uint32_t kAzdMatrixStorePointerInIdx = 0;
constexpr uint32_t kAzdMatrixStoreObjectInIdx = 1;
constexpr uint32_t kAzdMatrixStoreShapeInIdx = 2;
constexpr uint32_t kAzdMatrixStoreOffsetInIdx = 3;
constexpr uint32_t kAzdMatrixStoreLayoutInIdx = 4;

constexpr uint32_t kAzdMatrixMulAddAInIdx = 0;
constexpr uint32_t kAzdMatrixMulAddBInIdx = 1;
constexpr uint32_t kAzdMatrixMulAddCInIdx = 2;

constexpr uint32_t kAzdVectorMatrixMulInputInIdx = 0;
constexpr uint32_t kAzdVectorMatrixMulMatrixInIdx = 1;
constexpr uint32_t kAzdVectorMatrixMulAddBiasInIdx = 2;

constexpr uint32_t kDefaultMaxLoweredElements = 4096;

Operand IdOperand(uint32_t id) { return {SPV_OPERAND_TYPE_ID, {id}}; }

Instruction* AddTypeOrGlobalAfter(IRContext* context, Instruction* insert_after,
                                  std::unique_ptr<Instruction>&& inst) {
  if (!insert_after || insert_after->NextNode() == nullptr) {
    const bool is_type = IsTypeInst(inst->opcode());
    if (is_type) {
      context->AddType(std::move(inst));
    } else {
      context->AddGlobalValue(std::move(inst));
    }
    return &*(--context->types_values_end());
  }

  Instruction* added = insert_after->NextNode()->InsertBefore(std::move(inst));
  if (context->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse)) {
    context->get_def_use_mgr()->AnalyzeInstDefUse(added);
  }
  return added;
}

}  // namespace

Pass::Status AzdLowerToStandardPass::Process() {
  matrix_types_.clear();
  vector_types_.clear();
  lowered_types_.clear();

  bool has_azd = false;
  get_module()->ForEachInst([this, &has_azd](Instruction* inst) {
    has_azd |= IsAzdOpcode(inst->opcode()) || IsAzdCapabilityOrExtension(inst);
  });
  if (!has_azd) return Status::SuccessWithoutChange;

  if (!CollectAzdTypes()) return Status::Failure;
  if (!LegalizeModule()) return Status::Failure;

  std::vector<Instruction*> to_kill;
  if (!LowerAzdInstructions(&to_kill)) return Status::Failure;
  if (!ReplaceAzdTypeUses()) return Status::Failure;
  if (!CleanupAzdDeclarations(to_kill)) return Status::Failure;
  if (!FinalAzdCheck()) return Status::Failure;

  context()->InvalidateAnalyses(IRContext::kAnalysisTypes |
                                IRContext::kAnalysisConstants);
  return Status::SuccessWithChange;
}

bool AzdLowerToStandardPass::CollectAzdTypes() {
  std::vector<Instruction*> azd_types;
  get_module()->ForEachInst([this, &azd_types](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpTypeCooperativeMatrixAZD ||
        inst->opcode() == spv::Op::OpTypeCooperativeVectorAZD) {
      azd_types.push_back(inst);
    }
  });

  for (Instruction* inst : azd_types) {
    if (inst->opcode() == spv::Op::OpTypeCooperativeMatrixAZD) {
      if (inst->NumInOperands() < 3) {
        ReportError(inst, "OpTypeCooperativeMatrixAZD is missing operands");
        return false;
      }

      MatrixTypeInfo info;
      info.type_id = inst->result_id();
      info.component_type_id = inst->GetSingleWordInOperand(0);
      if (!GetConstantU32(inst->GetSingleWordInOperand(1), &info.rows) ||
          !GetConstantU32(inst->GetSingleWordInOperand(2), &info.cols)) {
        ReportError(inst,
                    "AZD cooperative matrix rows/columns must be constants");
        return false;
      }
      if (!IsFloat32Type(info.component_type_id)) {
        ReportError(inst,
                    "AZD lowering MVP only supports f32 cooperative matrices");
        return false;
      }
      const uint64_t element_count =
          static_cast<uint64_t>(info.rows) * static_cast<uint64_t>(info.cols);
      if (element_count == 0 || element_count > kDefaultMaxLoweredElements) {
        ReportError(inst, "AZD cooperative matrix shape is unsupported");
        return false;
      }
      info.lowered_type_id =
          GetOrCreateArrayType(info.component_type_id,
                               static_cast<uint32_t>(element_count), inst);
      if (info.lowered_type_id == 0) return false;
      matrix_types_[info.type_id] = info;
      lowered_types_[info.type_id] = info.lowered_type_id;
      continue;
    }

    if (inst->NumInOperands() < 2) {
      ReportError(inst, "OpTypeCooperativeVectorAZD is missing operands");
      return false;
    }

    VectorTypeInfo info;
    info.type_id = inst->result_id();
    info.component_type_id = inst->GetSingleWordInOperand(0);
    if (!GetConstantU32(inst->GetSingleWordInOperand(1), &info.length)) {
      ReportError(inst, "AZD cooperative vector length must be constant");
      return false;
    }
    if (!IsFloat32Type(info.component_type_id)) {
      ReportError(inst,
                  "AZD lowering MVP only supports f32 cooperative vectors");
      return false;
    }
    if (info.length == 0 || info.length > kDefaultMaxLoweredElements) {
      ReportError(inst, "AZD cooperative vector length is unsupported");
      return false;
    }
    info.lowered_type_id =
        GetOrCreateArrayType(info.component_type_id, info.length, inst);
    if (info.lowered_type_id == 0) return false;
    vector_types_[info.type_id] = info;
    lowered_types_[info.type_id] = info.lowered_type_id;
  }

  return true;
}

bool AzdLowerToStandardPass::LegalizeModule() {
  bool ok = true;
  get_module()->ForEachInst([this, &ok](Instruction* inst) {
    if (!ok) return;

    if (inst->opcode() == spv::Op::OpTypePointer &&
        IsAzdType(inst->GetSingleWordInOperand(1))) {
      const auto storage_class =
          static_cast<spv::StorageClass>(inst->GetSingleWordInOperand(0));
      if (storage_class != spv::StorageClass::Function) {
        ReportError(inst,
                    "AZD cooperative values may only be stored in Function "
                    "variables before lowering");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpTypeFunction) {
      for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
        if (TypeContainsAzd(inst->GetSingleWordInOperand(i))) {
          ReportError(inst,
                      "AZD cooperative values across function boundaries are "
                      "not supported");
          ok = false;
          return;
        }
      }
    }

    if (inst->opcode() == spv::Op::OpFunction &&
        TypeContainsAzd(inst->type_id())) {
      ReportError(inst, "AZD cooperative function return is not supported");
      ok = false;
      return;
    }

    if (inst->opcode() == spv::Op::OpFunctionParameter &&
        TypeContainsAzd(inst->type_id())) {
      ReportError(inst, "AZD cooperative function parameter is not supported");
      ok = false;
      return;
    }

    if (inst->opcode() == spv::Op::OpReturnValue &&
        inst->NumInOperands() > 0) {
      Instruction* object =
          get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
      if (object && TypeContainsAzd(object->type_id())) {
        ReportError(inst, "AZD cooperative function return is not supported");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpPhi && TypeContainsAzd(inst->type_id())) {
      ReportError(inst, "AZD cooperative OpPhi is not supported");
      ok = false;
      return;
    }

    if (inst->opcode() == spv::Op::OpCooperativeMatrixReduceAZD) {
      ReportError(inst, "OpCooperativeMatrixReduceAZD is not supported");
      ok = false;
      return;
    }

    if (inst->opcode() == spv::Op::OpCooperativeMatrixMulAddAZD) {
      const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
      const MatrixTypeInfo* a = nullptr;
      const MatrixTypeInfo* b = nullptr;
      const MatrixTypeInfo* c = nullptr;
      if (inst->NumInOperands() >= 3) {
        Instruction* a_inst =
            get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
        Instruction* b_inst =
            get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(1));
        Instruction* c_inst =
            get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(2));
        a = a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
        b = b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
        c = c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
      }
      if (!result || !a || !b || !c || a->cols != b->rows ||
          result->rows != a->rows || result->cols != b->cols ||
          c->rows != result->rows || c->cols != result->cols) {
        ReportError(inst, "AZD cooperative matrix multiply shapes do not match");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAZD ||
        inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddAZD) {
      const VectorTypeInfo* result = GetVectorType(inst->type_id());
      Instruction* input_inst =
          inst->NumInOperands() > kAzdVectorMatrixMulInputInIdx
              ? get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(
                    kAzdVectorMatrixMulInputInIdx))
              : nullptr;
      Instruction* matrix_inst =
          inst->NumInOperands() > kAzdVectorMatrixMulMatrixInIdx
              ? get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(
                    kAzdVectorMatrixMulMatrixInIdx))
              : nullptr;
      const VectorTypeInfo* input =
          input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
      const MatrixTypeInfo* matrix =
          matrix_inst ? GetMatrixType(matrix_inst->type_id()) : nullptr;
      const bool has_bias =
          inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddAZD;
      const VectorTypeInfo* bias = nullptr;
      if (has_bias && inst->NumInOperands() > kAzdVectorMatrixMulAddBiasInIdx) {
        Instruction* bias_inst = get_def_use_mgr()->GetDef(
            inst->GetSingleWordInOperand(kAzdVectorMatrixMulAddBiasInIdx));
        bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
      }
      if (!result || !input || !matrix || input->length != matrix->rows ||
          result->length != matrix->cols ||
          (has_bias && (!bias || bias->length != result->length))) {
        ReportError(inst, "AZD cooperative vector matrix multiply shapes do "
                          "not match");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpBitcast &&
        TypeContainsAzd(inst->type_id())) {
      if (inst->NumInOperands() != 1) {
        ReportError(inst, "unsupported AZD OpBitcast");
        ok = false;
        return;
      }
      Instruction* object =
          get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
      if (!object || GetLoweredType(inst->type_id()) == 0 ||
          GetLoweredType(object->type_id()) == 0 ||
          GetLoweredType(inst->type_id()) != GetLoweredType(object->type_id())) {
        ReportError(inst, "unsupported AZD OpBitcast");
        ok = false;
        return;
      }
    }

    if (!IsAzdOpcode(inst->opcode()) && TypeContainsAzd(inst->type_id())) {
      switch (inst->opcode()) {
        case spv::Op::OpUndef:
        case spv::Op::OpConstantNull:
        case spv::Op::OpVariable:
        case spv::Op::OpLoad:
        case spv::Op::OpStore:
        case spv::Op::OpCopyObject:
        case spv::Op::OpCompositeConstruct:
        case spv::Op::OpCompositeExtract:
        case spv::Op::OpBitcast:
          break;
        default:
          ReportError(inst, "unsupported AZD cooperative value use");
          ok = false;
          return;
      }
    }
  });

  return ok;
}

bool AzdLowerToStandardPass::LowerAzdInstructions(
    std::vector<Instruction*>* to_kill) {
  bool ok = true;
  get_module()->ForEachInst([this, to_kill, &ok](Instruction* inst) {
    if (!ok) return;

    switch (inst->opcode()) {
      case spv::Op::OpCooperativeMatrixLoadAZD:
        ok = LowerMatrixLoad(inst);
        break;
      case spv::Op::OpCooperativeMatrixStoreAZD:
        ok = LowerMatrixStore(inst, to_kill);
        break;
      case spv::Op::OpCooperativeMatrixMulAddAZD:
        ok = LowerMatrixMulAdd(inst);
        break;
      case spv::Op::OpCooperativeMatrixLengthAZD:
        ok = LowerMatrixLength(inst, to_kill);
        break;
      case spv::Op::OpCooperativeVectorLoadAZD:
        ok = LowerVectorLoad(inst);
        break;
      case spv::Op::OpCooperativeVectorStoreAZD:
        ok = LowerVectorStore(inst, to_kill);
        break;
      case spv::Op::OpCooperativeVectorMatrixMulAZD:
        ok = LowerVectorMatrixMul(inst, false);
        break;
      case spv::Op::OpCooperativeVectorMatrixMulAddAZD:
        ok = LowerVectorMatrixMul(inst, true);
        break;
      case spv::Op::OpBitcast:
        if (TypeContainsAzd(inst->type_id())) ok = LowerAzdBitcast(inst);
        break;
      default:
        break;
    }
  });

  return ok;
}

bool AzdLowerToStandardPass::ReplaceAzdTypeUses() {
  for (const auto& type_pair : lowered_types_) {
    Instruction* old_type = get_def_use_mgr()->GetDef(type_pair.first);
    Instruction* new_type = get_def_use_mgr()->GetDef(type_pair.second);
    if (!old_type || !new_type) return false;
    context()->ReplaceAllUsesWith(type_pair.first, type_pair.second);
  }
  return true;
}

bool AzdLowerToStandardPass::CleanupAzdDeclarations(
    const std::vector<Instruction*>& to_kill) {
  bool modified = false;
  for (Instruction* inst : to_kill) {
    if (inst && !inst->IsNop()) {
      context()->KillInst(inst);
      modified = true;
    }
  }

  for (const auto& type_pair : matrix_types_) {
    Instruction* inst = get_def_use_mgr()->GetDef(type_pair.first);
    if (inst) {
      context()->KillInst(inst);
      modified = true;
    }
  }
  for (const auto& type_pair : vector_types_) {
    Instruction* inst = get_def_use_mgr()->GetDef(type_pair.first);
    if (inst) {
      context()->KillInst(inst);
      modified = true;
    }
  }

  modified |= context()->RemoveCapability(spv::Capability::CooperativeMatrixAZD);
  modified |= context()->RemoveCapability(spv::Capability::CooperativeVectorAZD);
  modified |= RemoveExtensionByName("SPV_AZD_neural_matrix");
  modified |= RemoveExtensionByName("SPV_AZD_cooperative_vector");
  (void)modified;
  return true;
}

bool AzdLowerToStandardPass::FinalAzdCheck() const {
  bool ok = true;
  get_module()->ForEachInst([this, &ok](Instruction* inst) {
    if (!ok) return;
    if (IsAzdOpcode(inst->opcode()) || IsAzdCapabilityOrExtension(inst) ||
        TypeContainsAzd(inst->type_id())) {
      ReportError(inst, "AZD lowering left AZD op/type/capability/extension");
      ok = false;
    }
  });
  return ok;
}

bool AzdLowerToStandardPass::LowerMatrixLoad(Instruction* inst) {
  const MatrixTypeInfo* info = GetMatrixType(inst->type_id());
  if (!info || inst->NumInOperands() < 4) {
    ReportError(inst, "invalid OpCooperativeMatrixLoadAZD");
    return false;
  }

  uint32_t layout = 0;
  if (!GetConstantU32(inst->GetSingleWordInOperand(kAzdMatrixLoadLayoutInIdx),
                      &layout) ||
      layout > 1) {
    ReportError(inst, "AZD matrix load layout must be RowMajor or ColumnMajor");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;
  element_ids.reserve(info->rows * info->cols);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kAzdMatrixLoadPointerInIdx);
  const uint32_t shape_id =
      inst->GetSingleWordInOperand(kAzdMatrixLoadShapeInIdx);
  const uint32_t offset_id =
      inst->GetSingleWordInOperand(kAzdMatrixLoadOffsetInIdx);

  for (uint32_t row = 0; row < info->rows; ++row) {
    for (uint32_t col = 0; col < info->cols; ++col) {
      const uint32_t index_id = BuildMatrixElementIndex(
          &builder, inst, *info, shape_id, offset_id, layout, row, col);
      const uint32_t elem_ptr_id = BuildElementAccess(
          &builder, inst, pointer_id, info->component_type_id, index_id);
      if (index_id == 0 || elem_ptr_id == 0) return false;
      Instruction* load =
          builder.AddLoad(info->component_type_id, elem_ptr_id, 0);
      if (!load) return false;
      element_ids.push_back(load->result_id());
    }
  }

  RebuildAsCompositeConstruct(inst, info->lowered_type_id, element_ids);
  return true;
}

bool AzdLowerToStandardPass::LowerMatrixStore(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 5) {
    ReportError(inst, "invalid OpCooperativeMatrixStoreAZD");
    return false;
  }

  Instruction* object =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(
          kAzdMatrixStoreObjectInIdx));
  const MatrixTypeInfo* info = object ? GetMatrixType(object->type_id())
                                      : nullptr;
  if (!info) {
    ReportError(inst, "invalid AZD matrix store object");
    return false;
  }

  uint32_t layout = 0;
  if (!GetConstantU32(inst->GetSingleWordInOperand(kAzdMatrixStoreLayoutInIdx),
                      &layout) ||
      layout > 1) {
    ReportError(inst,
                "AZD matrix store layout must be RowMajor or ColumnMajor");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kAzdMatrixStorePointerInIdx);
  const uint32_t object_id =
      inst->GetSingleWordInOperand(kAzdMatrixStoreObjectInIdx);
  const uint32_t shape_id =
      inst->GetSingleWordInOperand(kAzdMatrixStoreShapeInIdx);
  const uint32_t offset_id =
      inst->GetSingleWordInOperand(kAzdMatrixStoreOffsetInIdx);

  for (uint32_t row = 0; row < info->rows; ++row) {
    for (uint32_t col = 0; col < info->cols; ++col) {
      const uint32_t flat_index = row * info->cols + col;
      const uint32_t value_id = ExtractCompositeElement(
          &builder, info->component_type_id, object_id, flat_index);
      const uint32_t index_id = BuildMatrixElementIndex(
          &builder, inst, *info, shape_id, offset_id, layout, row, col);
      const uint32_t elem_ptr_id = BuildElementAccess(
          &builder, inst, pointer_id, info->component_type_id, index_id);
      if (value_id == 0 || index_id == 0 || elem_ptr_id == 0) return false;
      if (!builder.AddStore(elem_ptr_id, value_id)) return false;
    }
  }

  to_kill->push_back(inst);
  return true;
}

bool AzdLowerToStandardPass::LowerMatrixMulAdd(Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(kAzdMatrixMulAddAInIdx));
  Instruction* b_inst =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(kAzdMatrixMulAddBInIdx));
  Instruction* c_inst =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(kAzdMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
  const MatrixTypeInfo* b = b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
  const MatrixTypeInfo* c = c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddAZD");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;
  element_ids.reserve(result->rows * result->cols);

  const uint32_t a_id = a_inst->result_id();
  const uint32_t b_id = b_inst->result_id();
  const uint32_t c_id = c_inst->result_id();
  const uint32_t float_type_id = result->component_type_id;

  for (uint32_t row = 0; row < result->rows; ++row) {
    for (uint32_t col = 0; col < result->cols; ++col) {
      uint32_t acc_id = ExtractCompositeElement(
          &builder, float_type_id, c_id, row * c->cols + col);
      if (acc_id == 0) return false;
      for (uint32_t k = 0; k < a->cols; ++k) {
        const uint32_t a_elem = ExtractCompositeElement(
            &builder, float_type_id, a_id, row * a->cols + k);
        const uint32_t b_elem = ExtractCompositeElement(
            &builder, float_type_id, b_id, k * b->cols + col);
        if (a_elem == 0 || b_elem == 0) return false;
        Instruction* mul =
            builder.AddBinaryOp(float_type_id, spv::Op::OpFMul, a_elem,
                                b_elem);
        if (!mul) return false;
        Instruction* add =
            builder.AddBinaryOp(float_type_id, spv::Op::OpFAdd, acc_id,
                                mul->result_id());
        if (!add) return false;
        acc_id = add->result_id();
      }
      element_ids.push_back(acc_id);
    }
  }

  RebuildAsCompositeConstruct(inst, result->lowered_type_id, element_ids);
  return true;
}

bool AzdLowerToStandardPass::LowerMatrixLength(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 1) {
    ReportError(inst, "invalid OpCooperativeMatrixLengthAZD");
    return false;
  }
  const MatrixTypeInfo* info = GetMatrixType(inst->GetSingleWordInOperand(0));
  if (!info) {
    ReportError(inst, "invalid OpCooperativeMatrixLengthAZD type operand");
    return false;
  }
  const uint32_t length_id =
      GetOrCreateConstant(inst->type_id(), info->rows * info->cols);
  if (length_id == 0) {
    ReportError(inst, "OpCooperativeMatrixLengthAZD result type must be int32");
    return false;
  }
  context()->ReplaceAllUsesWith(inst->result_id(), length_id);
  to_kill->push_back(inst);
  return true;
}

bool AzdLowerToStandardPass::LowerVectorLoad(Instruction* inst) {
  const VectorTypeInfo* info = GetVectorType(inst->type_id());
  if (!info || inst->NumInOperands() < 1) {
    ReportError(inst, "invalid OpCooperativeVectorLoadAZD");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;
  element_ids.reserve(info->length);
  const uint32_t pointer_id = inst->GetSingleWordInOperand(0);

  for (uint32_t i = 0; i < info->length; ++i) {
    const uint32_t index_id = GetOrCreateUIntConstant(i);
    const uint32_t elem_ptr_id = BuildElementAccess(
        &builder, inst, pointer_id, info->component_type_id, index_id);
    if (elem_ptr_id == 0) return false;
    Instruction* load = builder.AddLoad(info->component_type_id, elem_ptr_id);
    if (!load) return false;
    element_ids.push_back(load->result_id());
  }

  RebuildAsCompositeConstruct(inst, info->lowered_type_id, element_ids);
  return true;
}

bool AzdLowerToStandardPass::LowerVectorStore(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 2) {
    ReportError(inst, "invalid OpCooperativeVectorStoreAZD");
    return false;
  }

  Instruction* object =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(1));
  const VectorTypeInfo* info = object ? GetVectorType(object->type_id())
                                      : nullptr;
  if (!info) {
    ReportError(inst, "invalid AZD vector store object");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t pointer_id = inst->GetSingleWordInOperand(0);
  const uint32_t object_id = inst->GetSingleWordInOperand(1);
  for (uint32_t i = 0; i < info->length; ++i) {
    const uint32_t value_id =
        ExtractCompositeElement(&builder, info->component_type_id, object_id, i);
    const uint32_t index_id = GetOrCreateUIntConstant(i);
    const uint32_t elem_ptr_id = BuildElementAccess(
        &builder, inst, pointer_id, info->component_type_id, index_id);
    if (value_id == 0 || elem_ptr_id == 0) return false;
    if (!builder.AddStore(elem_ptr_id, value_id)) return false;
  }

  to_kill->push_back(inst);
  return true;
}

bool AzdLowerToStandardPass::LowerVectorMatrixMul(Instruction* inst,
                                                  bool has_bias) {
  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix =
      matrix_inst ? GetMatrixType(matrix_inst->type_id()) : nullptr;
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kAzdVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  if (!result || !input || !matrix || (has_bias && !bias)) {
    ReportError(inst, "invalid AZD vector matrix multiply");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;
  element_ids.reserve(result->length);
  const uint32_t float_type_id = result->component_type_id;
  const uint32_t zero_id = GetOrCreateZero(float_type_id);

  for (uint32_t col = 0; col < result->length; ++col) {
    uint32_t acc_id =
        has_bias ? ExtractCompositeElement(&builder, float_type_id,
                                           bias_inst->result_id(), col)
                 : zero_id;
    if (acc_id == 0) return false;
    for (uint32_t k = 0; k < input->length; ++k) {
      const uint32_t x_id = ExtractCompositeElement(
          &builder, float_type_id, input_inst->result_id(), k);
      const uint32_t w_id = ExtractCompositeElement(
          &builder, float_type_id, matrix_inst->result_id(),
          k * matrix->cols + col);
      if (x_id == 0 || w_id == 0) return false;
      Instruction* mul =
          builder.AddBinaryOp(float_type_id, spv::Op::OpFMul, x_id, w_id);
      if (!mul) return false;
      Instruction* add =
          builder.AddBinaryOp(float_type_id, spv::Op::OpFAdd, acc_id,
                              mul->result_id());
      if (!add) return false;
      acc_id = add->result_id();
    }
    element_ids.push_back(acc_id);
  }

  RebuildAsCompositeConstruct(inst, result->lowered_type_id, element_ids);
  return true;
}

bool AzdLowerToStandardPass::LowerAzdBitcast(Instruction* inst) {
  inst->SetOpcode(spv::Op::OpCopyObject);
  inst->SetResultType(GetLoweredType(inst->type_id()));
  context()->UpdateDefUse(inst);
  return true;
}

uint32_t AzdLowerToStandardPass::GetOrCreateArrayType(
    uint32_t component_type_id, uint32_t length, Instruction* insert_after) {
  Instruction* insertion_point = insert_after;
  uint32_t length_id =
      GetOrCreateUIntConstantAfter(length, &insertion_point);
  if (length_id == 0) return 0;

  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() != spv::Op::OpTypeArray ||
        inst.GetSingleWordInOperand(0) != component_type_id) {
      continue;
    }
    uint32_t existing_length = 0;
    if (GetConstantU32(inst.GetSingleWordInOperand(1), &existing_length) &&
        existing_length == length) {
      return inst.result_id();
    }
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

uint32_t AzdLowerToStandardPass::GetOrCreatePointerType(
    uint32_t pointee_type_id, spv::StorageClass storage_class) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypePointer &&
        inst.GetSingleWordInOperand(0) == static_cast<uint32_t>(storage_class) &&
        inst.GetSingleWordInOperand(1) == pointee_type_id) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddType(MakeUnique<Instruction>(
      context(), spv::Op::OpTypePointer, 0, result_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS,
           {static_cast<uint32_t>(storage_class)}},
          IdOperand(pointee_type_id)}));
  return result_id;
}

uint32_t AzdLowerToStandardPass::GetOrCreateUIntType() {
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
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {32}},
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}}));
  return result_id;
}

uint32_t AzdLowerToStandardPass::GetOrCreateUIntConstant(uint32_t value) {
  const uint32_t uint_type_id = GetOrCreateUIntType();
  if (uint_type_id == 0) return 0;
  return GetOrCreateConstant(uint_type_id, value);
}

uint32_t AzdLowerToStandardPass::GetOrCreateUIntConstantAfter(
    uint32_t value, Instruction** insert_after) {
  const uint32_t uint_type_id = GetOrCreateUIntTypeAfter(insert_after);
  if (uint_type_id == 0) return 0;

  for (Instruction& inst : get_module()->types_values()) {
    if ((inst.opcode() == spv::Op::OpConstant ||
         inst.opcode() == spv::Op::OpSpecConstant) &&
        inst.type_id() == uint_type_id && inst.GetSingleWordInOperand(0) == value) {
      return inst.result_id();
    }
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

uint32_t AzdLowerToStandardPass::GetOrCreateUIntTypeAfter(
    Instruction** insert_after) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeInt &&
        inst.GetSingleWordInOperand(0) == 32 &&
        inst.GetSingleWordInOperand(1) == 0) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::unique_ptr<Instruction> uint_type = MakeUnique<Instruction>(
      context(), spv::Op::OpTypeInt, 0, result_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {32}},
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}});
  Instruction* added =
      AddTypeOrGlobalAfter(context(), *insert_after, std::move(uint_type));
  *insert_after = added;
  return result_id;
}

uint32_t AzdLowerToStandardPass::GetOrCreateConstant(uint32_t type_id,
                                                     uint32_t value) {
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  if (!type || type->opcode() != spv::Op::OpTypeInt ||
      type->GetSingleWordInOperand(0) != 32) {
    return 0;
  }

  for (Instruction& inst : get_module()->types_values()) {
    if ((inst.opcode() == spv::Op::OpConstant ||
         inst.opcode() == spv::Op::OpSpecConstant) &&
        inst.type_id() == type_id && inst.GetSingleWordInOperand(0) == value) {
      return inst.result_id();
    }
    if (value == 0 && inst.opcode() == spv::Op::OpConstantNull &&
        inst.type_id() == type_id) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddGlobalValue(MakeUnique<Instruction>(
      context(), spv::Op::OpConstant, type_id, result_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {value}}}));
  return result_id;
}

uint32_t AzdLowerToStandardPass::GetOrCreateZero(uint32_t type_id) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.type_id() == type_id &&
        (inst.opcode() == spv::Op::OpConstantNull ||
         (inst.opcode() == spv::Op::OpConstant && inst.NumInOperands() > 0 &&
          inst.GetSingleWordInOperand(0) == 0))) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  context()->AddGlobalValue(MakeUnique<Instruction>(
      context(), spv::Op::OpConstantNull, type_id, result_id,
      std::initializer_list<Operand>{}));
  return result_id;
}

uint32_t AzdLowerToStandardPass::BuildPairComponentAsUInt(
    InstructionBuilder* builder, Instruction* user, uint32_t pair_id,
    uint32_t component_index) {
  Instruction* pair = get_def_use_mgr()->GetDef(pair_id);
  if (!pair || pair->type_id() == 0) {
    ReportError(user, "AZD matrix shape/offset must be a two-component value");
    return 0;
  }

  Instruction* pair_type = get_def_use_mgr()->GetDef(pair->type_id());
  if (!pair_type || pair_type->opcode() != spv::Op::OpTypeVector ||
      pair_type->GetSingleWordInOperand(1) < 2) {
    ReportError(user, "AZD matrix shape/offset must be a two-component vector");
    return 0;
  }

  const uint32_t component_type_id = pair_type->GetSingleWordInOperand(0);
  Instruction* extract =
      builder->AddCompositeExtract(component_type_id, pair_id, {component_index});
  if (!extract) return 0;

  const uint32_t uint_type_id = GetOrCreateUIntType();
  if (component_type_id == uint_type_id) return extract->result_id();

  Instruction* component_type = get_def_use_mgr()->GetDef(component_type_id);
  if (!component_type) return 0;
  if (component_type->opcode() == spv::Op::OpTypeFloat &&
      component_type->GetSingleWordInOperand(0) == 32) {
    Instruction* converted = builder->AddUnaryOp(
        uint_type_id, spv::Op::OpConvertFToU, extract->result_id());
    return converted ? converted->result_id() : 0;
  }
  if (component_type->opcode() == spv::Op::OpTypeInt &&
      component_type->GetSingleWordInOperand(0) == 32) {
    Instruction* converted =
        builder->AddUnaryOp(uint_type_id, spv::Op::OpBitcast,
                            extract->result_id());
    return converted ? converted->result_id() : 0;
  }

  ReportError(user, "AZD matrix shape/offset component type is unsupported");
  return 0;
}

uint32_t AzdLowerToStandardPass::BuildMatrixElementIndex(
    InstructionBuilder* builder, Instruction* user, const MatrixTypeInfo& info,
    uint32_t shape_id, uint32_t offset_id, uint32_t layout, uint32_t row,
    uint32_t col) {
  const uint32_t shape_rows =
      BuildPairComponentAsUInt(builder, user, shape_id, 0);
  const uint32_t shape_cols =
      BuildPairComponentAsUInt(builder, user, shape_id, 1);
  const uint32_t offset_row =
      BuildPairComponentAsUInt(builder, user, offset_id, 0);
  const uint32_t offset_col =
      BuildPairComponentAsUInt(builder, user, offset_id, 1);
  if (shape_rows == 0 || shape_cols == 0 || offset_row == 0 ||
      offset_col == 0) {
    return 0;
  }

  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t row_const_id = GetOrCreateUIntConstant(row);
  const uint32_t col_const_id = GetOrCreateUIntConstant(col);
  Instruction* global_row =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd, offset_row,
                           row_const_id);
  Instruction* global_col =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd, offset_col,
                           col_const_id);
  if (!global_row || !global_col) return 0;

  Instruction* major_mul = nullptr;
  if (layout == static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    major_mul = builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                     global_row->result_id(), shape_cols);
    if (!major_mul) return 0;
    Instruction* index = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, major_mul->result_id(),
        global_col->result_id());
    return index ? index->result_id() : 0;
  }

  (void)info;
  major_mul = builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                   global_col->result_id(), shape_rows);
  if (!major_mul) return 0;
  Instruction* index = builder->AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, major_mul->result_id(),
      global_row->result_id());
  return index ? index->result_id() : 0;
}

uint32_t AzdLowerToStandardPass::BuildElementAccess(
    InstructionBuilder* builder, Instruction* user, uint32_t pointer_id,
    uint32_t component_type_id, uint32_t element_index_id) {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer || pointer->type_id() == 0) {
    ReportError(user, "AZD load/store pointer is invalid");
    return 0;
  }
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer->type_id());
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    ReportError(user, "AZD load/store pointer must be a pointer");
    return 0;
  }

  const uint32_t pointee_type_id = pointer_type->GetSingleWordInOperand(1);
  Instruction* pointee_type = get_def_use_mgr()->GetDef(pointee_type_id);
  if (!pointee_type) return 0;

  if (pointee_type_id == component_type_id) {
    uint32_t index_value = 0;
    if (GetConstantU32(element_index_id, &index_value) && index_value == 0) {
      return pointer_id;
    }
    ReportError(user,
                "AZD scalar pointer load/store only supports element zero");
    return 0;
  }

  if ((pointee_type->opcode() == spv::Op::OpTypeRuntimeArray ||
       pointee_type->opcode() == spv::Op::OpTypeArray) &&
      pointee_type->GetSingleWordInOperand(0) == component_type_id) {
    const uint32_t storage_class = pointer_type->GetSingleWordInOperand(0);
    const uint32_t component_pointer_type_id = GetOrCreatePointerType(
        component_type_id, static_cast<spv::StorageClass>(storage_class));
    Instruction* access = builder->AddAccessChain(component_pointer_type_id,
                                                  pointer_id,
                                                  {element_index_id});
    return access ? access->result_id() : 0;
  }

  ReportError(user,
              "AZD load/store pointer must point to the component or a "
              "component array");
  return 0;
}

uint32_t AzdLowerToStandardPass::ExtractCompositeElement(
    InstructionBuilder* builder, uint32_t component_type_id,
    uint32_t composite_id, uint32_t index) {
  Instruction* extract =
      builder->AddCompositeExtract(component_type_id, composite_id, {index});
  return extract ? extract->result_id() : 0;
}

const AzdLowerToStandardPass::MatrixTypeInfo*
AzdLowerToStandardPass::GetMatrixType(uint32_t type_id) const {
  auto it = matrix_types_.find(type_id);
  if (it != matrix_types_.end()) return &it->second;
  for (const auto& id_and_info : matrix_types_) {
    if (id_and_info.second.lowered_type_id == type_id) return &id_and_info.second;
  }
  return nullptr;
}

const AzdLowerToStandardPass::VectorTypeInfo*
AzdLowerToStandardPass::GetVectorType(uint32_t type_id) const {
  auto it = vector_types_.find(type_id);
  if (it != vector_types_.end()) return &it->second;
  for (const auto& id_and_info : vector_types_) {
    if (id_and_info.second.lowered_type_id == type_id) return &id_and_info.second;
  }
  return nullptr;
}

uint32_t AzdLowerToStandardPass::GetLoweredType(uint32_t type_id) const {
  auto it = lowered_types_.find(type_id);
  return it == lowered_types_.end() ? 0 : it->second;
}

uint32_t AzdLowerToStandardPass::GetPointeeType(
    uint32_t pointer_type_id) const {
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer_type_id);
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    return 0;
  }
  return pointer_type->GetSingleWordInOperand(1);
}

bool AzdLowerToStandardPass::GetConstantU32(uint32_t id,
                                            uint32_t* value) const {
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  if (!inst) return false;
  if (inst->opcode() == spv::Op::OpConstantNull) {
    Instruction* type = get_def_use_mgr()->GetDef(inst->type_id());
    if (!type || type->opcode() != spv::Op::OpTypeInt) return false;
    *value = 0;
    return true;
  }
  if (inst->opcode() != spv::Op::OpConstant &&
      inst->opcode() != spv::Op::OpSpecConstant) {
    return false;
  }
  Instruction* type = get_def_use_mgr()->GetDef(inst->type_id());
  if (!type || type->opcode() != spv::Op::OpTypeInt ||
      type->GetSingleWordInOperand(0) > 32 || inst->NumInOperands() == 0) {
    return false;
  }
  *value = inst->GetSingleWordInOperand(0);
  return true;
}

bool AzdLowerToStandardPass::IsFloat32Type(uint32_t type_id) const {
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  return type && type->opcode() == spv::Op::OpTypeFloat &&
         type->GetSingleWordInOperand(0) == 32;
}

bool AzdLowerToStandardPass::IsAzdType(uint32_t type_id) const {
  return matrix_types_.find(type_id) != matrix_types_.end() ||
         vector_types_.find(type_id) != vector_types_.end();
}

bool AzdLowerToStandardPass::TypeContainsAzd(uint32_t type_id) const {
  if (type_id == 0) return false;
  if (IsAzdType(type_id)) return true;

  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  if (!type) return false;
  switch (type->opcode()) {
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
      return TypeContainsAzd(type->GetSingleWordInOperand(
          type->opcode() == spv::Op::OpTypePointer ? 1 : 0));
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeStruct:
      for (uint32_t i = 0; i < type->NumInOperands(); ++i) {
        if (TypeContainsAzd(type->GetSingleWordInOperand(i))) return true;
      }
      return false;
    default:
      return false;
  }
}

bool AzdLowerToStandardPass::IsAzdOpcode(spv::Op opcode) const {
  switch (opcode) {
    case spv::Op::OpTypeCooperativeMatrixAZD:
    case spv::Op::OpCooperativeMatrixLoadAZD:
    case spv::Op::OpCooperativeMatrixStoreAZD:
    case spv::Op::OpCooperativeMatrixMulAddAZD:
    case spv::Op::OpCooperativeMatrixLengthAZD:
    case spv::Op::OpCooperativeMatrixReduceAZD:
    case spv::Op::OpTypeCooperativeVectorAZD:
    case spv::Op::OpCooperativeVectorLoadAZD:
    case spv::Op::OpCooperativeVectorStoreAZD:
    case spv::Op::OpCooperativeVectorMatrixMulAZD:
    case spv::Op::OpCooperativeVectorMatrixMulAddAZD:
      return true;
    default:
      return false;
  }
}

bool AzdLowerToStandardPass::IsAzdCapabilityOrExtension(
    const Instruction* inst) const {
  if (inst->opcode() == spv::Op::OpCapability) {
    const auto capability =
        static_cast<spv::Capability>(inst->GetSingleWordInOperand(0));
    return capability == spv::Capability::CooperativeMatrixAZD ||
           capability == spv::Capability::CooperativeVectorAZD;
  }
  if (inst->opcode() == spv::Op::OpExtension) {
    const std::string extension = inst->GetInOperand(0).AsString();
    return extension == "SPV_AZD_neural_matrix" ||
           extension == "SPV_AZD_cooperative_vector";
  }
  return false;
}

bool AzdLowerToStandardPass::RemoveExtensionByName(
    const char* extension_name) {
  return context()->KillInstructionIf(
      get_module()->extension_begin(), get_module()->extension_end(),
      [extension_name](Instruction* inst) {
        return inst->opcode() == spv::Op::OpExtension &&
               inst->GetInOperand(0).AsString() == extension_name;
      });
}

void AzdLowerToStandardPass::RebuildAsCompositeConstruct(
    Instruction* inst, uint32_t type_id, const std::vector<uint32_t>& element_ids) {
  std::vector<Operand> operands;
  operands.reserve(element_ids.size());
  for (uint32_t id : element_ids) operands.push_back(IdOperand(id));
  inst->SetOpcode(spv::Op::OpCompositeConstruct);
  inst->SetResultType(type_id);
  inst->SetInOperands(std::move(operands));
  context()->UpdateDefUse(inst);
}

void AzdLowerToStandardPass::ReportError(
    const Instruction*, const std::string& message) const {
  if (!consumer()) return;
  consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
}

}  // namespace opt
}  // namespace spvtools
