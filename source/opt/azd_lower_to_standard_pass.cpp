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
#include "source/opt/def_use_manager.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"
#include "spirv/unified1/GLSL.std.450.h"

namespace spvtools {
namespace opt {
namespace {

constexpr uint32_t kAzdMatrixLoadPointerInIdx = 0;
constexpr uint32_t kAzdMatrixLoadShapeInIdx = 1;
constexpr uint32_t kAzdMatrixLoadOffsetInIdx = 2;
constexpr uint32_t kAzdMatrixLoadLayoutInIdx = 3;
constexpr uint32_t kAzdMatrixLoadMemoryOperandsInIdx = 4;

constexpr uint32_t kAzdMatrixStorePointerInIdx = 0;
constexpr uint32_t kAzdMatrixStoreObjectInIdx = 1;
constexpr uint32_t kAzdMatrixStoreShapeInIdx = 2;
constexpr uint32_t kAzdMatrixStoreOffsetInIdx = 3;
constexpr uint32_t kAzdMatrixStoreLayoutInIdx = 4;
constexpr uint32_t kAzdMatrixStoreMemoryOperandsInIdx = 5;

constexpr uint32_t kAzdMatrixMulAddAInIdx = 0;
constexpr uint32_t kAzdMatrixMulAddBInIdx = 1;
constexpr uint32_t kAzdMatrixMulAddCInIdx = 2;

constexpr uint32_t kAzdVectorMatrixMulInputInIdx = 0;
constexpr uint32_t kAzdVectorMatrixMulMatrixInIdx = 1;
constexpr uint32_t kAzdVectorMatrixMulAddBiasInIdx = 2;

constexpr uint32_t kAzdVectorLoadPointerInIdx = 0;
constexpr uint32_t kAzdVectorLoadMemoryOperandsInIdx = 1;
constexpr uint32_t kAzdVectorStorePointerInIdx = 0;
constexpr uint32_t kAzdVectorStoreObjectInIdx = 1;
constexpr uint32_t kAzdVectorStoreMemoryOperandsInIdx = 2;

constexpr uint32_t kDefaultMaxLoweredElements = 131072;
constexpr uint32_t kDefaultMaxLoweredMatmulMacs = 65536;
constexpr uint32_t kDefaultMatrixTileM = 2;
constexpr uint32_t kDefaultMatrixTileN = 4;
constexpr uint32_t kDefaultVectorMatmulTileN = 4;
constexpr uint32_t kPackedVec4Width = 4;

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
  packed_load_chunk_functions_.clear();
  packed_store_chunk_functions_.clear();
  tile_weight_functions_.clear();
  vector_matmul_pattern_functions_.clear();
  matmul_pattern_functions_.clear();
  matmul_pattern_function_ids_.clear();
  generated_function_ids_.clear();

  bool has_azd = false;
  get_module()->ForEachInst([this, &has_azd](Instruction* inst) {
    has_azd |= IsAzdOpcode(inst->opcode()) || IsAzdCapabilityOrExtension(inst);
  });
  if (!has_azd) return Status::SuccessWithoutChange;

  if (!CollectAzdTypes()) return Status::Failure;
  if (!LegalizeModule()) return Status::Failure;
  if (!PrepareMatmulPatternFunctions()) return Status::Failure;

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
      const uint64_t element_count =
          static_cast<uint64_t>(info.rows) * static_cast<uint64_t>(info.cols);
      if (element_count == 0 || element_count > kDefaultMaxLoweredElements) {
        ReportError(inst, "AZD cooperative matrix shape is unsupported");
        return false;
      }

      if (IsFloat16Type(info.component_type_id) &&
          ShouldUsePackedVec4(info.cols)) {
        Instruction* insertion_point = inst;
        info.packed_f16vec4 = true;
        info.packed_cols = info.cols / kPackedVec4Width;
        info.packed_vec4_type_id = GetOrCreateVectorType(
            info.component_type_id, kPackedVec4Width, &insertion_point);
        if (info.packed_vec4_type_id == 0) return false;
        info.lowered_type_id = GetOrCreatePackedArrayType(
            info.packed_vec4_type_id, info.rows * info.packed_cols,
            insertion_point);
      } else if (IsFloat32Type(info.component_type_id) &&
                 ShouldUsePackedVec4(info.cols)) {
        Instruction* insertion_point = inst;
        info.packed_f32vec4 = true;
        info.packed_cols = info.cols / kPackedVec4Width;
        info.packed_vec4_type_id = GetOrCreateVectorType(
            info.component_type_id, kPackedVec4Width, &insertion_point);
        if (info.packed_vec4_type_id == 0) return false;
        info.lowered_type_id = GetOrCreatePackedArrayType(
            info.packed_vec4_type_id, info.rows * info.packed_cols,
            insertion_point);
      } else if (IsFloat16Type(info.component_type_id) ||
                 IsFloat32Type(info.component_type_id)) {
        info.lowered_type_id = GetOrCreateArrayType(
            info.component_type_id, static_cast<uint32_t>(element_count), inst);
      } else {
        ReportError(inst, "unsupported AZD cooperative matrix component type");
        return false;
      }
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
    if (info.length == 0 || info.length > kDefaultMaxLoweredElements) {
      ReportError(inst, "AZD cooperative vector length is unsupported");
      return false;
    }
    if (IsFloat16Type(info.component_type_id) &&
        ShouldUsePackedVec4(info.length)) {
      Instruction* insertion_point = inst;
      info.packed_f16vec4 = true;
      info.packed_length = info.length / kPackedVec4Width;
      info.packed_vec4_type_id = GetOrCreateVectorType(
          info.component_type_id, kPackedVec4Width, &insertion_point);
      if (info.packed_vec4_type_id == 0) return false;
      info.lowered_type_id = GetOrCreatePackedArrayType(
          info.packed_vec4_type_id, info.packed_length, insertion_point);
    } else if (IsFloat32Type(info.component_type_id) &&
               ShouldUsePackedVec4(info.length)) {
      Instruction* insertion_point = inst;
      info.packed_f32vec4 = true;
      info.packed_length = info.length / kPackedVec4Width;
      info.packed_vec4_type_id = GetOrCreateVectorType(
          info.component_type_id, kPackedVec4Width, &insertion_point);
      if (info.packed_vec4_type_id == 0) return false;
      info.lowered_type_id = GetOrCreatePackedArrayType(
          info.packed_vec4_type_id, info.packed_length, insertion_point);
    } else if (IsFloat16Type(info.component_type_id) ||
               IsFloat32Type(info.component_type_id)) {
      info.lowered_type_id =
          GetOrCreateArrayType(info.component_type_id, info.length, inst);
    } else {
      ReportError(inst, "unsupported AZD cooperative vector component type");
      return false;
    }
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
        TypeContainsAzd(inst->GetSingleWordInOperand(1))) {
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

    if (inst->opcode() == spv::Op::OpReturnValue && inst->NumInOperands() > 0) {
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
        ReportError(inst,
                    "AZD cooperative matrix multiply shapes do not match");
        ok = false;
        return;
      }
      const uint64_t mac_count = static_cast<uint64_t>(result->rows) *
                                 static_cast<uint64_t>(result->cols) *
                                 static_cast<uint64_t>(a->cols);
      if (mac_count > kDefaultMaxLoweredMatmulMacs) {
        ReportError(inst,
                    "AZD cooperative matrix multiply expansion is too large");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAZD ||
        inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddAZD) {
      const VectorTypeInfo* result = GetVectorType(inst->type_id());
      Instruction* input_inst =
          inst->NumInOperands() > kAzdVectorMatrixMulInputInIdx
              ? get_def_use_mgr()->GetDef(
                    inst->GetSingleWordInOperand(kAzdVectorMatrixMulInputInIdx))
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
      if (!result || !input || !matrix || input->length != matrix->cols ||
          result->length != matrix->rows ||
          (has_bias && (!bias || bias->length != result->length))) {
        ReportError(inst,
                    "AZD cooperative vector matrix multiply shapes do "
                    "not match");
        ok = false;
        return;
      }
      const uint64_t mac_count = static_cast<uint64_t>(result->length) *
                                 static_cast<uint64_t>(input->length);
      if (mac_count > kDefaultMaxLoweredMatmulMacs) {
        ReportError(inst,
                    "AZD cooperative vector matrix multiply expansion is too "
                    "large");
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
          GetLoweredType(inst->type_id()) !=
              GetLoweredType(object->type_id())) {
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

bool AzdLowerToStandardPass::PrepareMatmulPatternFunctions() {
  std::vector<Instruction*> matmul_insts;
  get_module()->ForEachInst([&matmul_insts](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpCooperativeMatrixMulAddAZD) {
      matmul_insts.push_back(inst);
    }
  });

  for (Instruction* inst : matmul_insts) {
    const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
    Instruction* a_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kAzdMatrixMulAddAInIdx));
    Instruction* b_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kAzdMatrixMulAddBInIdx));
    Instruction* c_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kAzdMatrixMulAddCInIdx));
    const MatrixTypeInfo* a =
        a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
    const MatrixTypeInfo* b =
        b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
    const MatrixTypeInfo* c =
        c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
    if (!result || !a || !b || !c) {
      ReportError(inst, "invalid OpCooperativeMatrixMulAddAZD");
      return false;
    }
    if (CanUsePackedVec4MatrixMulAdd(*result, *a, *b, *c) &&
        GetOrCreateMatmulPatternFunctionPackedVec4(*result, *a, *b, *c) == 0) {
      return false;
    }
  }

  return true;
}

bool AzdLowerToStandardPass::LowerAzdInstructions(
    std::vector<Instruction*>* to_kill) {
  std::vector<Instruction*> worklist;
  get_module()->ForEachInst([this, &worklist](Instruction* inst) {
    BasicBlock* block = context()->get_instr_block(inst);
    Function* function = block ? block->GetParent() : nullptr;
    if (function && generated_function_ids_.find(function->result_id()) !=
                        generated_function_ids_.end()) {
      return;
    }

    switch (inst->opcode()) {
      case spv::Op::OpCooperativeMatrixLoadAZD:
      case spv::Op::OpCooperativeMatrixStoreAZD:
      case spv::Op::OpCooperativeMatrixMulAddAZD:
      case spv::Op::OpCooperativeMatrixLengthAZD:
      case spv::Op::OpCooperativeVectorLoadAZD:
      case spv::Op::OpCooperativeVectorStoreAZD:
      case spv::Op::OpCooperativeVectorMatrixMulAZD:
      case spv::Op::OpCooperativeVectorMatrixMulAddAZD:
        worklist.push_back(inst);
        break;
      case spv::Op::OpCompositeConstruct:
        if (IsAzdType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpCompositeExtract:
        worklist.push_back(inst);
        break;
      case spv::Op::OpConstantNull:
      case spv::Op::OpUndef:
        if (IsAzdType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpBitcast:
        if (TypeContainsAzd(inst->type_id())) worklist.push_back(inst);
        break;
      default:
        break;
    }
  });

  for (Instruction* inst : worklist) {
    bool ok = true;
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
      case spv::Op::OpCompositeConstruct:
        ok = LowerCompositeConstruct(inst);
        break;
      case spv::Op::OpCompositeExtract:
        ok = LowerCompositeExtract(inst);
        break;
      case spv::Op::OpConstantNull:
      case spv::Op::OpUndef:
        ok = LowerNullOrUndef(inst);
        break;
      case spv::Op::OpBitcast:
        ok = LowerAzdBitcast(inst);
        break;
      default:
        break;
    }
    if (!ok) return false;
  }

  return true;
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

  modified |=
      context()->RemoveCapability(spv::Capability::CooperativeMatrixAZD);
  modified |=
      context()->RemoveCapability(spv::Capability::CooperativeVectorAZD);
  modified |= RemoveExtensionByName("SPV_AZD_neural_matrix");
  modified |= RemoveExtensionByName("SPV_AZD_cooperative_vector");
  modified |= RemoveSourceExtensionByName("GL_AZD_neural_matrix");
  modified |= RemoveSourceExtensionByName("GL_AZD_cooperative_vector");
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
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kAzdMatrixLoadMemoryOperandsInIdx);

  if (IsPackedVec4(*info)) {
    if (layout ==
            static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR) &&
        CanCapturePointer(pointer_id)) {
      const uint32_t pointer_type_id = GetPointerTypeId(pointer_id);
      if (pointer_type_id == 0) return false;

      uint32_t result_id = 0;
      if (!BuildPackedMatrixLoadOuterLoop(
              inst, *info, pointer_id, pointer_type_id, shape_id, offset_id,
              layout, memory_operands, &result_id)) {
        return false;
      }
      inst->SetOpcode(spv::Op::OpCopyObject);
      inst->SetResultType(info->lowered_type_id);
      inst->SetInOperands({IdOperand(result_id)});
      context()->UpdateDefUse(inst);
      return true;
    }

    element_ids.reserve(info->rows * info->packed_cols);
    for (uint32_t row = 0; row < info->rows; ++row) {
      for (uint32_t col_pack = 0; col_pack < info->packed_cols; ++col_pack) {
        std::vector<uint32_t> lane_ids;
        lane_ids.reserve(kPackedVec4Width);
        for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
          const uint32_t col = col_pack * kPackedVec4Width + lane;
          const uint32_t index_id = BuildMatrixElementIndex(
              &builder, inst, *info, shape_id, offset_id, layout, row, col);
          const uint32_t elem_ptr_id = BuildElementAccess(
              &builder, inst, pointer_id, info->component_type_id, index_id);
          if (index_id == 0 || elem_ptr_id == 0) return false;
          const uint32_t load_id = AddLoad(&builder, info->component_type_id,
                                           elem_ptr_id, memory_operands);
          if (load_id == 0) return false;
          lane_ids.push_back(load_id);
        }
        Instruction* vec =
            builder.AddCompositeConstruct(info->packed_vec4_type_id, lane_ids);
        if (!vec) return false;
        element_ids.push_back(vec->result_id());
      }
    }

    RebuildAsCompositeConstruct(inst, info->lowered_type_id, element_ids);
    return true;
  }

  element_ids.reserve(info->rows * info->cols);
  for (uint32_t row = 0; row < info->rows; ++row) {
    for (uint32_t col = 0; col < info->cols; ++col) {
      const uint32_t index_id = BuildMatrixElementIndex(
          &builder, inst, *info, shape_id, offset_id, layout, row, col);
      const uint32_t elem_ptr_id = BuildElementAccess(
          &builder, inst, pointer_id, info->component_type_id, index_id);
      if (index_id == 0 || elem_ptr_id == 0) return false;
      const uint32_t load_id = AddLoad(&builder, info->component_type_id,
                                       elem_ptr_id, memory_operands);
      if (load_id == 0) return false;
      element_ids.push_back(load_id);
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

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixStoreObjectInIdx));
  const MatrixTypeInfo* info =
      object ? GetMatrixType(object->type_id()) : nullptr;
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
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kAzdMatrixStoreMemoryOperandsInIdx);

  if (IsPackedVec4(*info)) {
    if (layout ==
            static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR) &&
        CanCapturePointer(pointer_id)) {
      const uint32_t pointer_type_id = GetPointerTypeId(pointer_id);
      if (pointer_type_id == 0) return false;

      if (!BuildPackedMatrixStoreOuterLoop(
              inst, *info, pointer_id, pointer_type_id, object_id, shape_id,
              offset_id, layout, memory_operands)) {
        return false;
      }
      to_kill->push_back(inst);
      return true;
    }

    for (uint32_t row = 0; row < info->rows; ++row) {
      for (uint32_t col_pack = 0; col_pack < info->packed_cols; ++col_pack) {
        const uint32_t vec_id = ExtractCompositeElement(
            &builder, info->packed_vec4_type_id, object_id,
            MatrixPackedIndex(*info, row, col_pack));
        if (vec_id == 0) return false;
        for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
          const uint32_t col = col_pack * kPackedVec4Width + lane;
          const uint32_t value_id = ExtractCompositeElement(
              &builder, info->component_type_id, vec_id, lane);
          const uint32_t index_id = BuildMatrixElementIndex(
              &builder, inst, *info, shape_id, offset_id, layout, row, col);
          const uint32_t elem_ptr_id = BuildElementAccess(
              &builder, inst, pointer_id, info->component_type_id, index_id);
          if (value_id == 0 || index_id == 0 || elem_ptr_id == 0) {
            return false;
          }
          if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands)) {
            return false;
          }
        }
      }
    }

    to_kill->push_back(inst);
    return true;
  }

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
      if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands)) {
        return false;
      }
    }
  }

  to_kill->push_back(inst);
  return true;
}

bool AzdLowerToStandardPass::LowerMatrixMulAdd(Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
  const MatrixTypeInfo* b = b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
  const MatrixTypeInfo* c = c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddAZD");
    return false;
  }

  if (CanUsePackedVec4MatrixMulAdd(*result, *a, *b, *c)) {
    return LowerMatrixMulAddPackedVec4(inst);
  }
  return LowerMatrixMulAddScalarFallback(inst);
}

bool AzdLowerToStandardPass::LowerMatrixMulAddPackedVec4(Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
  const MatrixTypeInfo* b = b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
  const MatrixTypeInfo* c = c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddAZD");
    return false;
  }

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

bool AzdLowerToStandardPass::LowerMatrixMulAddScalarFallback(
    Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdMatrixMulAddCInIdx));
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
  std::vector<uint32_t> scalar_ids(result->rows * result->cols, 0);

  const uint32_t a_id = a_inst->result_id();
  const uint32_t b_id = b_inst->result_id();
  const uint32_t c_id = c_inst->result_id();
  const uint32_t float_type_id = result->component_type_id;

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
            Instruction* mul = builder.AddBinaryOp(
                float_type_id, spv::Op::OpFMul, a_elems[i], b_elems[j]);
            if (!mul) return false;
            Instruction* add =
                builder.AddBinaryOp(float_type_id, spv::Op::OpFAdd,
                                    acc[i * tile_n + j], mul->result_id());
            if (!add) return false;
            acc[i * tile_n + j] = add->result_id();
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
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kAzdVectorLoadPointerInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kAzdVectorLoadMemoryOperandsInIdx);

  if (IsPackedVec4(*info)) {
    const uint32_t pointer_type_id = GetPointerTypeId(pointer_id);
    if (pointer_type_id == 0) return false;

    if (CanCapturePointer(pointer_id)) {
      uint32_t result_id = 0;
      if (!BuildPackedVectorLoadOuterLoop(inst, *info, pointer_id,
                                          pointer_type_id, memory_operands,
                                          &result_id)) {
        return false;
      }
      inst->SetOpcode(spv::Op::OpCopyObject);
      inst->SetResultType(info->lowered_type_id);
      inst->SetInOperands({IdOperand(result_id)});
      context()->UpdateDefUse(inst);
      return true;
    }

    element_ids.reserve(info->packed_length);
    for (uint32_t pack = 0; pack < info->packed_length; ++pack) {
      std::vector<uint32_t> lane_ids;
      lane_ids.reserve(kPackedVec4Width);
      for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
        const uint32_t index_id =
            GetOrCreateUIntConstant(pack * kPackedVec4Width + lane);
        const uint32_t elem_ptr_id = BuildElementAccess(
            &builder, inst, pointer_id, info->component_type_id, index_id);
        if (index_id == 0 || elem_ptr_id == 0) return false;
        const uint32_t load_id = AddLoad(&builder, info->component_type_id,
                                         elem_ptr_id, memory_operands);
        if (load_id == 0) return false;
        lane_ids.push_back(load_id);
      }
      Instruction* vec =
          builder.AddCompositeConstruct(info->packed_vec4_type_id, lane_ids);
      if (!vec) return false;
      element_ids.push_back(vec->result_id());
    }

    RebuildAsCompositeConstruct(inst, info->lowered_type_id, element_ids);
    return true;
  }

  for (uint32_t i = 0; i < info->length; ++i) {
    const uint32_t index_id = GetOrCreateUIntConstant(i);
    const uint32_t elem_ptr_id = BuildElementAccess(
        &builder, inst, pointer_id, info->component_type_id, index_id);
    if (elem_ptr_id == 0) return false;
    const uint32_t load_id = AddLoad(&builder, info->component_type_id,
                                     elem_ptr_id, memory_operands);
    if (load_id == 0) return false;
    element_ids.push_back(load_id);
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

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kAzdVectorStoreObjectInIdx));
  const VectorTypeInfo* info =
      object ? GetVectorType(object->type_id()) : nullptr;
  if (!info) {
    ReportError(inst, "invalid AZD vector store object");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kAzdVectorStorePointerInIdx);
  const uint32_t object_id =
      inst->GetSingleWordInOperand(kAzdVectorStoreObjectInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kAzdVectorStoreMemoryOperandsInIdx);

  if (IsPackedVec4(*info)) {
    const uint32_t pointer_type_id = GetPointerTypeId(pointer_id);
    if (pointer_type_id == 0) return false;

    if (CanCapturePointer(pointer_id)) {
      if (!BuildPackedVectorStoreOuterLoop(inst, *info, pointer_id,
                                           pointer_type_id, object_id,
                                           memory_operands)) {
        return false;
      }
      to_kill->push_back(inst);
      return true;
    }

    for (uint32_t pack = 0; pack < info->packed_length; ++pack) {
      const uint32_t vec_id = ExtractCompositeElement(
          &builder, info->packed_vec4_type_id, object_id, pack);
      if (vec_id == 0) return false;
      for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
        const uint32_t value_id = ExtractCompositeElement(
            &builder, info->component_type_id, vec_id, lane);
        const uint32_t index_id =
            GetOrCreateUIntConstant(pack * kPackedVec4Width + lane);
        const uint32_t elem_ptr_id = BuildElementAccess(
            &builder, inst, pointer_id, info->component_type_id, index_id);
        if (value_id == 0 || index_id == 0 || elem_ptr_id == 0) return false;
        if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands)) {
          return false;
        }
      }
    }

    to_kill->push_back(inst);
    return true;
  }

  for (uint32_t i = 0; i < info->length; ++i) {
    const uint32_t value_id = ExtractCompositeElement(
        &builder, info->component_type_id, object_id, i);
    const uint32_t index_id = GetOrCreateUIntConstant(i);
    const uint32_t elem_ptr_id = BuildElementAccess(
        &builder, inst, pointer_id, info->component_type_id, index_id);
    if (value_id == 0 || elem_ptr_id == 0) return false;
    if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands)) {
      return false;
    }
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

  if (CanUsePackedVec4VectorMatrixMul(*result, *input, *matrix, bias)) {
    return LowerVectorMatrixMulPackedVec4(inst, has_bias);
  }
  return LowerVectorMatrixMulScalarFallback(inst, has_bias);
}

bool AzdLowerToStandardPass::LowerVectorMatrixMulPackedVec4(Instruction* inst,
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

  const uint32_t function_id = GetOrCreateVectorMatmulPatternFunctionPackedVec4(
      *result, *input, *matrix, bias, has_bias);
  if (function_id == 0) return false;
  std::vector<uint32_t> argument_ids = {input_inst->result_id(),
                                        matrix_inst->result_id()};
  if (has_bias) argument_ids.push_back(bias_inst->result_id());
  RebuildAsFunctionCall(inst, result->lowered_type_id, function_id,
                        argument_ids);
  return true;
}

bool AzdLowerToStandardPass::LowerVectorMatrixMulScalarFallback(
    Instruction* inst, bool has_bias) {
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
  std::vector<uint32_t> scalar_ids(result->length, 0);
  const uint32_t float_type_id = result->component_type_id;
  const uint32_t zero_id = has_bias ? 0 : GetOrCreateZero(float_type_id);
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
            &builder, *matrix, matrix_inst->result_id(), col, k);
        if (w_id == 0) return false;
        Instruction* mul =
            builder.AddBinaryOp(float_type_id, spv::Op::OpFMul, x_id, w_id);
        if (!mul) return false;
        Instruction* add = builder.AddBinaryOp(float_type_id, spv::Op::OpFAdd,
                                               acc[j], mul->result_id());
        if (!add) return false;
        acc[j] = add->result_id();
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

bool AzdLowerToStandardPass::LowerCompositeConstruct(Instruction* inst) {
  const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
  const VectorTypeInfo* vector = GetVectorType(inst->type_id());
  if (!matrix && !vector) {
    ReportError(inst, "invalid AZD OpCompositeConstruct result type");
    return false;
  }

  if (matrix && !IsPackedVec4(*matrix)) {
    inst->SetResultType(matrix->lowered_type_id);
    context()->UpdateDefUse(inst);
    return true;
  }
  if (vector && !IsPackedVec4(*vector)) {
    inst->SetResultType(vector->lowered_type_id);
    context()->UpdateDefUse(inst);
    return true;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;

  if (matrix) {
    const uint32_t expected_operands = matrix->rows * matrix->cols;
    if (inst->NumInOperands() != expected_operands) {
      ReportError(inst,
                  "AZD matrix OpCompositeConstruct operand count is invalid");
      return false;
    }

    element_ids.resize(matrix->rows * matrix->packed_cols, 0);
    for (uint32_t row = 0; row < matrix->rows; ++row) {
      for (uint32_t col_pack = 0; col_pack < matrix->packed_cols; ++col_pack) {
        std::vector<uint32_t> lane_ids;
        lane_ids.reserve(kPackedVec4Width);
        for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
          const uint32_t col = col_pack * kPackedVec4Width + lane;
          lane_ids.push_back(
              inst->GetSingleWordInOperand(MatrixFlatIndex(*matrix, row, col)));
        }
        Instruction* vec = builder.AddCompositeConstruct(
            matrix->packed_vec4_type_id, lane_ids);
        if (!vec) return false;
        element_ids[MatrixPackedIndex(*matrix, row, col_pack)] =
            vec->result_id();
      }
    }
    RebuildAsCompositeConstruct(inst, matrix->lowered_type_id, element_ids);
    return true;
  }

  const uint32_t expected_operands = vector->length;
  if (inst->NumInOperands() != expected_operands) {
    ReportError(inst,
                "AZD vector OpCompositeConstruct operand count is invalid");
    return false;
  }

  element_ids.resize(vector->packed_length, 0);
  for (uint32_t pack = 0; pack < vector->packed_length; ++pack) {
    std::vector<uint32_t> lane_ids;
    lane_ids.reserve(kPackedVec4Width);
    for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
      lane_ids.push_back(
          inst->GetSingleWordInOperand(pack * kPackedVec4Width + lane));
    }
    Instruction* vec =
        builder.AddCompositeConstruct(vector->packed_vec4_type_id, lane_ids);
    if (!vec) return false;
    element_ids[pack] = vec->result_id();
  }
  RebuildAsCompositeConstruct(inst, vector->lowered_type_id, element_ids);
  return true;
}

bool AzdLowerToStandardPass::LowerCompositeExtract(Instruction* inst) {
  if (inst->NumInOperands() < 1) return true;
  Instruction* object =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  if (!object) return true;

  const MatrixTypeInfo* matrix = GetMatrixType(object->type_id());
  const VectorTypeInfo* vector = GetVectorType(object->type_id());
  if (!matrix && !vector) return true;

  if (matrix) {
    if (inst->NumInOperands() != 3 ||
        inst->type_id() != matrix->component_type_id) {
      ReportError(inst, "unsupported AZD matrix OpCompositeExtract");
      return false;
    }
    const uint32_t row = inst->GetSingleWordInOperand(1);
    const uint32_t col = inst->GetSingleWordInOperand(2);
    if (row >= matrix->rows || col >= matrix->cols) {
      ReportError(inst, "AZD matrix OpCompositeExtract index is out of range");
      return false;
    }

    std::vector<Operand> operands;
    if (!IsPackedVec4(*matrix)) {
      operands.push_back(IdOperand(object->result_id()));
      operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {MatrixFlatIndex(*matrix, row, col)}});
      inst->SetInOperands(std::move(operands));
      context()->UpdateDefUse(inst);
      return true;
    }

    InstructionBuilder builder(
        context(), inst,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    const uint32_t vec_id = ExtractCompositeElement(
        &builder, matrix->packed_vec4_type_id, object->result_id(),
        MatrixPackedIndex(*matrix, row, VectorPackedIndex(col)));
    if (vec_id == 0) return false;
    operands.push_back(IdOperand(vec_id));
    operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {PackedLane(col)}});
    inst->SetInOperands(std::move(operands));
    context()->UpdateDefUse(inst);
    return true;
  }

  if (inst->NumInOperands() != 2 ||
      inst->type_id() != vector->component_type_id) {
    ReportError(inst, "unsupported AZD vector OpCompositeExtract");
    return false;
  }
  const uint32_t index = inst->GetSingleWordInOperand(1);
  if (index >= vector->length) {
    ReportError(inst, "AZD vector OpCompositeExtract index is out of range");
    return false;
  }

  std::vector<Operand> operands;
  if (!IsPackedVec4(*vector)) {
    operands.push_back(IdOperand(object->result_id()));
    operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {index}});
    inst->SetInOperands(std::move(operands));
    context()->UpdateDefUse(inst);
    return true;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t vec_id =
      ExtractCompositeElement(&builder, vector->packed_vec4_type_id,
                              object->result_id(), VectorPackedIndex(index));
  if (vec_id == 0) return false;
  operands.push_back(IdOperand(vec_id));
  operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {PackedLane(index)}});
  inst->SetInOperands(std::move(operands));
  context()->UpdateDefUse(inst);
  return true;
}

bool AzdLowerToStandardPass::LowerNullOrUndef(Instruction* inst) {
  const uint32_t lowered_type_id = GetLoweredType(inst->type_id());
  if (lowered_type_id == 0) {
    ReportError(inst, "invalid AZD null/undef result type");
    return false;
  }
  inst->SetResultType(lowered_type_id);
  context()->UpdateDefUse(inst);
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
  uint32_t length_id = GetOrCreateUIntConstantAfter(length, &insertion_point);
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

uint32_t AzdLowerToStandardPass::GetOrCreateVectorType(
    uint32_t component_type_id, uint32_t component_count,
    Instruction** insert_after) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeVector &&
        inst.GetSingleWordInOperand(0) == component_type_id &&
        inst.GetSingleWordInOperand(1) == component_count) {
      return inst.result_id();
    }
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

uint32_t AzdLowerToStandardPass::GetOrCreatePackedArrayType(
    uint32_t vec4_type_id, uint32_t length, Instruction* insert_after) {
  return GetOrCreateArrayType(vec4_type_id, length, insert_after);
}

uint32_t AzdLowerToStandardPass::GetOrCreatePointerType(
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

uint32_t AzdLowerToStandardPass::GetOrCreateVoidType() {
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

uint32_t AzdLowerToStandardPass::GetOrCreateBoolType() {
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
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_LITERAL_INTEGER, {32}},
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

  bool can_reuse = true;
  for (Instruction& inst : get_module()->types_values()) {
    if (can_reuse &&
        (inst.opcode() == spv::Op::OpConstant ||
         inst.opcode() == spv::Op::OpSpecConstant) &&
        inst.type_id() == uint_type_id &&
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

uint32_t AzdLowerToStandardPass::GetOrCreateUIntTypeAfter(
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
  context()->AddGlobalValue(
      MakeUnique<Instruction>(context(), spv::Op::OpConstantNull, type_id,
                              result_id, std::initializer_list<Operand>{}));
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
  Instruction* extract = builder->AddCompositeExtract(
      component_type_id, pair_id, {component_index});
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
    Instruction* converted = builder->AddUnaryOp(
        uint_type_id, spv::Op::OpBitcast, extract->result_id());
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
  Instruction* global_row = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_row, row_const_id);
  Instruction* global_col = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_col, col_const_id);
  if (!global_row || !global_col) return 0;

  Instruction* major_mul = nullptr;
  if (layout ==
      static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    major_mul = builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                     global_row->result_id(), shape_cols);
    if (!major_mul) return 0;
    Instruction* index =
        builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                             major_mul->result_id(), global_col->result_id());
    return index ? index->result_id() : 0;
  }

  (void)info;
  major_mul = builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                   global_col->result_id(), shape_rows);
  if (!major_mul) return 0;
  Instruction* index =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                           major_mul->result_id(), global_row->result_id());
  return index ? index->result_id() : 0;
}

uint32_t AzdLowerToStandardPass::BuildElementAccess(InstructionBuilder* builder,
                                                    Instruction* user,
                                                    uint32_t pointer_id,
                                                    uint32_t component_type_id,
                                                    uint32_t element_index_id) {
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
    Instruction* access = builder->AddAccessChain(
        component_pointer_type_id, pointer_id, {element_index_id});
    return access ? access->result_id() : 0;
  }

  ReportError(user,
              "AZD load/store pointer must point to the component or a "
              "component array");
  return 0;
}

uint32_t AzdLowerToStandardPass::BuildElementAccessFromPointerType(
    InstructionBuilder* builder, uint32_t pointer_type_id, uint32_t pointer_id,
    uint32_t component_type_id, uint32_t element_index_id) {
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer_type_id);
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    ReportError(nullptr, "AZD load/store pointer must be a pointer");
    return 0;
  }

  const uint32_t pointee_type_id = pointer_type->GetSingleWordInOperand(1);
  Instruction* pointee_type = get_def_use_mgr()->GetDef(pointee_type_id);
  if (!pointee_type) return 0;

  if (pointee_type_id == component_type_id) {
    ReportError(nullptr,
                "AZD scalar pointer load/store cannot be chunk-lowered");
    return 0;
  }

  if ((pointee_type->opcode() == spv::Op::OpTypeRuntimeArray ||
       pointee_type->opcode() == spv::Op::OpTypeArray) &&
      pointee_type->GetSingleWordInOperand(0) == component_type_id) {
    const uint32_t storage_class = pointer_type->GetSingleWordInOperand(0);
    const uint32_t component_pointer_type_id = GetOrCreatePointerType(
        component_type_id, static_cast<spv::StorageClass>(storage_class));
    Instruction* access = builder->AddAccessChain(
        component_pointer_type_id, pointer_id, {element_index_id});
    return access ? access->result_id() : 0;
  }

  ReportError(nullptr,
              "AZD load/store pointer must point to the component or a "
              "component array");
  return 0;
}

Instruction* AzdLowerToStandardPass::AddFunctionVariable(
    Function* function, uint32_t pointer_type_id, uint32_t initializer_id) {
  if (!function || function->begin() == function->end()) return nullptr;

  BasicBlock* entry_block = &*function->begin();
  auto insert_iter = entry_block->begin();
  while (insert_iter != entry_block->end() &&
         insert_iter->opcode() == spv::Op::OpVariable) {
    ++insert_iter;
  }

  const uint32_t var_id = TakeNextId();
  if (var_id == 0) return nullptr;
  std::unique_ptr<Instruction> variable = MakeUnique<Instruction>(
      context(), spv::Op::OpVariable, pointer_type_id, var_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS,
           {uint32_t(spv::StorageClass::Function)}}});
  Instruction* variable_ptr = variable.get();
  insert_iter.InsertBefore(std::move(variable));
  context()->AnalyzeDefUse(variable_ptr);
  context()->set_instr_block(variable_ptr, entry_block);

  if (initializer_id != 0) {
    InstructionBuilder builder(
        context(), entry_block,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    if (!builder.AddStore(var_id, initializer_id)) return nullptr;
  }
  return variable_ptr;
}

BasicBlock* AzdLowerToStandardPass::MakeBasicBlock(uint32_t label_id) {
  if (label_id == 0) return nullptr;
  BasicBlock* block = new BasicBlock(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));
  context()->AnalyzeDefUse(block->GetLabelInst());
  context()->set_instr_block(block->GetLabelInst(), block);
  return block;
}

uint32_t AzdLowerToStandardPass::BuildRowMajorMatrixMemoryIndex(
    InstructionBuilder* builder, Instruction* user, uint32_t shape_id,
    uint32_t offset_id, uint32_t cols, uint32_t base_id) {
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t cols_id = GetOrCreateUIntConstant(cols);
  if (uint_type_id == 0 || cols_id == 0) return 0;

  const uint32_t shape_cols =
      BuildPairComponentAsUInt(builder, user, shape_id, 1);
  const uint32_t offset_row =
      BuildPairComponentAsUInt(builder, user, offset_id, 0);
  const uint32_t offset_col =
      BuildPairComponentAsUInt(builder, user, offset_id, 1);
  if (shape_cols == 0 || offset_row == 0 || offset_col == 0) return 0;

  Instruction* row =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpUDiv, base_id, cols_id);
  Instruction* col =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpUMod, base_id, cols_id);
  if (!row || !col) return 0;

  Instruction* global_row = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_row, row->result_id());
  Instruction* global_col = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_col, col->result_id());
  if (!global_row || !global_col) return 0;

  Instruction* major_mul = builder->AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, global_row->result_id(), shape_cols);
  if (!major_mul) return 0;
  Instruction* index =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                           major_mul->result_id(), global_col->result_id());
  return index ? index->result_id() : 0;
}

bool AzdLowerToStandardPass::BuildPackedMatrixLoadOuterLoop(
    Instruction* insert_before, const MatrixTypeInfo& info, uint32_t pointer_id,
    uint32_t pointer_type_id, uint32_t shape_id, uint32_t offset_id,
    uint32_t layout, const std::vector<Operand>& memory_operands,
    uint32_t* result_id) {
  if (!insert_before || !result_id || !IsPackedVec4(info)) return false;
  if (layout !=
      static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    return false;
  }
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t load_function_id = GetOrCreatePackedLoadChunkFunction(
      pointer_id, pointer_type_id, info.component_type_id,
      info.packed_vec4_type_id, memory_operands);
  const uint32_t lowered_function_ptr_type_id =
      GetOrCreatePointerType(info.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec4_function_ptr_type_id = GetOrCreatePointerType(
      info.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t element_count_id =
      GetOrCreateUIntConstant(info.rows * info.cols);
  if (load_function_id == 0 || lowered_function_ptr_type_id == 0 ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || four_uint_id == 0 || element_count_id == 0) {
    return false;
  }

  Instruction* result_var =
      AddFunctionVariable(function, lowered_function_ptr_type_id);
  Instruction* base_var =
      AddFunctionVariable(function, uint_function_ptr_type_id);
  if (!result_var || !base_var) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }

  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(base_var->result_id(), zero_uint_id)) {
    return false;
  }
  if (!preheader_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* base_load =
      header_builder.AddLoad(uint_type_id, base_var->result_id());
  if (!base_load) return false;
  Instruction* cond =
      header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                 base_load->result_id(), element_count_id);
  if (!cond) return false;
  if (!header_builder.AddLoopMerge(merge_label_id, continue_label_id)) {
    return false;
  }
  if (!header_builder.AddConditionalBranch(cond->result_id(), body_label_id,
                                           merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t memory_base_id = BuildRowMajorMatrixMemoryIndex(
      &body_builder, insert_before, shape_id, offset_id, info.cols,
      base_load->result_id());
  if (memory_base_id == 0) return false;
  Instruction* vec = body_builder.AddFunctionCall(
      info.packed_vec4_type_id, load_function_id, {memory_base_id});
  if (!vec) return false;
  Instruction* packed_index = body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpUDiv, base_load->result_id(), four_uint_id);
  if (!packed_index) return false;
  Instruction* result_elem_ptr = body_builder.AddAccessChain(
      vec4_function_ptr_type_id, result_var->result_id(),
      {packed_index->result_id()});
  if (!result_elem_ptr) return false;
  if (!body_builder.AddStore(result_elem_ptr->result_id(), vec->result_id())) {
    return false;
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_i = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, base_load->result_id(), four_uint_id);
  if (!next_i) return false;
  if (!continue_builder.AddStore(base_var->result_id(), next_i->result_id())) {
    return false;
  }
  if (!continue_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder merge_builder(
      context(), insert_before,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* result =
      merge_builder.AddLoad(info.lowered_type_id, result_var->result_id());
  if (!result) return false;
  *result_id = result->result_id();
  return true;
}

bool AzdLowerToStandardPass::BuildPackedMatrixStoreOuterLoop(
    Instruction* insert_before, const MatrixTypeInfo& info, uint32_t pointer_id,
    uint32_t pointer_type_id, uint32_t object_id, uint32_t shape_id,
    uint32_t offset_id, uint32_t layout,
    const std::vector<Operand>& memory_operands) {
  if (!insert_before || object_id == 0 || !IsPackedVec4(info)) return false;
  if (layout !=
      static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    return false;
  }
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t store_function_id = GetOrCreatePackedStoreChunkFunction(
      pointer_id, pointer_type_id, info.component_type_id,
      info.packed_vec4_type_id, memory_operands);
  const uint32_t lowered_function_ptr_type_id =
      GetOrCreatePointerType(info.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec4_function_ptr_type_id = GetOrCreatePointerType(
      info.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t element_count_id =
      GetOrCreateUIntConstant(info.rows * info.cols);
  if (store_function_id == 0 || lowered_function_ptr_type_id == 0 ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || void_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || four_uint_id == 0 ||
      element_count_id == 0) {
    return false;
  }

  Instruction* object_var =
      AddFunctionVariable(function, lowered_function_ptr_type_id);
  Instruction* base_var =
      AddFunctionVariable(function, uint_function_ptr_type_id);
  if (!object_var || !base_var) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }

  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(object_var->result_id(), object_id)) {
    return false;
  }
  if (!preheader_builder.AddStore(base_var->result_id(), zero_uint_id)) {
    return false;
  }
  if (!preheader_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* base_load =
      header_builder.AddLoad(uint_type_id, base_var->result_id());
  if (!base_load) return false;
  Instruction* cond =
      header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                 base_load->result_id(), element_count_id);
  if (!cond) return false;
  if (!header_builder.AddLoopMerge(merge_label_id, continue_label_id)) {
    return false;
  }
  if (!header_builder.AddConditionalBranch(cond->result_id(), body_label_id,
                                           merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* packed_index = body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpUDiv, base_load->result_id(), four_uint_id);
  if (!packed_index) return false;
  Instruction* value_elem_ptr = body_builder.AddAccessChain(
      vec4_function_ptr_type_id, object_var->result_id(),
      {packed_index->result_id()});
  if (!value_elem_ptr) return false;
  Instruction* vec = body_builder.AddLoad(info.packed_vec4_type_id,
                                          value_elem_ptr->result_id());
  if (!vec) return false;
  const uint32_t memory_base_id = BuildRowMajorMatrixMemoryIndex(
      &body_builder, insert_before, shape_id, offset_id, info.cols,
      base_load->result_id());
  if (memory_base_id == 0) return false;
  if (!body_builder.AddFunctionCall(void_type_id, store_function_id,
                                    {memory_base_id, vec->result_id()})) {
    return false;
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_i = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, base_load->result_id(), four_uint_id);
  if (!next_i) return false;
  if (!continue_builder.AddStore(base_var->result_id(), next_i->result_id())) {
    return false;
  }
  if (!continue_builder.AddBranch(header_label_id)) return false;
  return true;
}

bool AzdLowerToStandardPass::BuildPackedVectorLoadOuterLoop(
    Instruction* insert_before, const VectorTypeInfo& info, uint32_t pointer_id,
    uint32_t pointer_type_id, const std::vector<Operand>& memory_operands,
    uint32_t* result_id) {
  if (!insert_before || !result_id || !IsPackedVec4(info)) return false;
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t load_function_id = GetOrCreatePackedLoadChunkFunction(
      pointer_id, pointer_type_id, info.component_type_id,
      info.packed_vec4_type_id, memory_operands);
  const uint32_t lowered_function_ptr_type_id =
      GetOrCreatePointerType(info.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec4_function_ptr_type_id = GetOrCreatePointerType(
      info.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t element_count_id = GetOrCreateUIntConstant(info.length);
  if (load_function_id == 0 || lowered_function_ptr_type_id == 0 ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || four_uint_id == 0 || element_count_id == 0) {
    return false;
  }

  Instruction* result_var =
      AddFunctionVariable(function, lowered_function_ptr_type_id);
  Instruction* base_var =
      AddFunctionVariable(function, uint_function_ptr_type_id);
  if (!result_var || !base_var) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }

  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(base_var->result_id(), zero_uint_id)) {
    return false;
  }
  if (!preheader_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* base_load =
      header_builder.AddLoad(uint_type_id, base_var->result_id());
  if (!base_load) return false;
  Instruction* cond =
      header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                 base_load->result_id(), element_count_id);
  if (!cond) return false;
  if (!header_builder.AddLoopMerge(merge_label_id, continue_label_id)) {
    return false;
  }
  if (!header_builder.AddConditionalBranch(cond->result_id(), body_label_id,
                                           merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* vec = body_builder.AddFunctionCall(
      info.packed_vec4_type_id, load_function_id, {base_load->result_id()});
  if (!vec) return false;
  Instruction* packed_index = body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpUDiv, base_load->result_id(), four_uint_id);
  if (!packed_index) return false;
  Instruction* result_elem_ptr = body_builder.AddAccessChain(
      vec4_function_ptr_type_id, result_var->result_id(),
      {packed_index->result_id()});
  if (!result_elem_ptr) return false;
  if (!body_builder.AddStore(result_elem_ptr->result_id(), vec->result_id())) {
    return false;
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_i = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, base_load->result_id(), four_uint_id);
  if (!next_i) return false;
  if (!continue_builder.AddStore(base_var->result_id(), next_i->result_id())) {
    return false;
  }
  if (!continue_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder merge_builder(
      context(), insert_before,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* result =
      merge_builder.AddLoad(info.lowered_type_id, result_var->result_id());
  if (!result) return false;
  *result_id = result->result_id();
  return true;
}

bool AzdLowerToStandardPass::BuildPackedVectorStoreOuterLoop(
    Instruction* insert_before, const VectorTypeInfo& info, uint32_t pointer_id,
    uint32_t pointer_type_id, uint32_t object_id,
    const std::vector<Operand>& memory_operands) {
  if (!insert_before || object_id == 0 || !IsPackedVec4(info)) return false;
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t store_function_id = GetOrCreatePackedStoreChunkFunction(
      pointer_id, pointer_type_id, info.component_type_id,
      info.packed_vec4_type_id, memory_operands);
  const uint32_t lowered_function_ptr_type_id =
      GetOrCreatePointerType(info.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec4_function_ptr_type_id = GetOrCreatePointerType(
      info.packed_vec4_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t element_count_id = GetOrCreateUIntConstant(info.length);
  if (store_function_id == 0 || lowered_function_ptr_type_id == 0 ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || void_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || four_uint_id == 0 ||
      element_count_id == 0) {
    return false;
  }

  Instruction* object_var =
      AddFunctionVariable(function, lowered_function_ptr_type_id);
  Instruction* base_var =
      AddFunctionVariable(function, uint_function_ptr_type_id);
  if (!object_var || !base_var) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }

  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(object_var->result_id(), object_id)) {
    return false;
  }
  if (!preheader_builder.AddStore(base_var->result_id(), zero_uint_id)) {
    return false;
  }
  if (!preheader_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* base_load =
      header_builder.AddLoad(uint_type_id, base_var->result_id());
  if (!base_load) return false;
  Instruction* cond =
      header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                 base_load->result_id(), element_count_id);
  if (!cond) return false;
  if (!header_builder.AddLoopMerge(merge_label_id, continue_label_id)) {
    return false;
  }
  if (!header_builder.AddConditionalBranch(cond->result_id(), body_label_id,
                                           merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* packed_index = body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpUDiv, base_load->result_id(), four_uint_id);
  if (!packed_index) return false;
  Instruction* value_elem_ptr = body_builder.AddAccessChain(
      vec4_function_ptr_type_id, object_var->result_id(),
      {packed_index->result_id()});
  if (!value_elem_ptr) return false;
  Instruction* vec = body_builder.AddLoad(info.packed_vec4_type_id,
                                          value_elem_ptr->result_id());
  if (!vec) return false;
  if (!body_builder.AddFunctionCall(
          void_type_id, store_function_id,
          {base_load->result_id(), vec->result_id()})) {
    return false;
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_i = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, base_load->result_id(), four_uint_id);
  if (!next_i) return false;
  if (!continue_builder.AddStore(base_var->result_id(), next_i->result_id())) {
    return false;
  }
  if (!continue_builder.AddBranch(header_label_id)) return false;
  return true;
}

uint32_t AzdLowerToStandardPass::BuildCapturedPointer(
    InstructionBuilder* builder, uint32_t pointer_id) {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer) return 0;

  if (pointer->opcode() == spv::Op::OpVariable &&
      context()->get_instr_block(pointer) == nullptr) {
    return pointer_id;
  }

  if (pointer->opcode() != spv::Op::OpAccessChain &&
      pointer->opcode() != spv::Op::OpInBoundsAccessChain) {
    return 0;
  }

  const uint32_t base_pointer_id = pointer->GetSingleWordInOperand(0);
  const uint32_t captured_base_id =
      BuildCapturedPointer(builder, base_pointer_id);
  if (captured_base_id == 0) return 0;

  std::vector<uint32_t> index_ids;
  index_ids.reserve(pointer->NumInOperands() - 1);
  for (uint32_t i = 1; i < pointer->NumInOperands(); ++i) {
    const uint32_t index_id = pointer->GetSingleWordInOperand(i);
    if (!IsModuleVisibleValue(index_id)) return 0;
    index_ids.push_back(index_id);
  }

  Instruction* access =
      builder->AddAccessChain(pointer->type_id(), captured_base_id, index_ids);
  return access ? access->result_id() : 0;
}

bool AzdLowerToStandardPass::CanCapturePointer(uint32_t pointer_id) const {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer) return false;

  if (pointer->opcode() == spv::Op::OpVariable &&
      context()->get_instr_block(pointer) == nullptr) {
    return true;
  }

  if (pointer->opcode() != spv::Op::OpAccessChain &&
      pointer->opcode() != spv::Op::OpInBoundsAccessChain) {
    return false;
  }

  if (!CanCapturePointer(pointer->GetSingleWordInOperand(0))) {
    return false;
  }

  for (uint32_t i = 1; i < pointer->NumInOperands(); ++i) {
    if (!IsModuleVisibleValue(pointer->GetSingleWordInOperand(i))) {
      return false;
    }
  }
  return true;
}

bool AzdLowerToStandardPass::IsModuleVisibleValue(uint32_t id) const {
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  if (!inst) return false;
  return context()->get_instr_block(inst) == nullptr;
}

uint32_t AzdLowerToStandardPass::ExtractCompositeElement(
    InstructionBuilder* builder, uint32_t component_type_id,
    uint32_t composite_id, uint32_t index) {
  Instruction* extract =
      builder->AddCompositeExtract(component_type_id, composite_id, {index});
  return extract ? extract->result_id() : 0;
}

uint32_t AzdLowerToStandardPass::AddLoad(
    InstructionBuilder* builder, uint32_t type_id, uint32_t pointer_id,
    const std::vector<Operand>& memory_operands) {
  std::vector<Operand> operands;
  operands.reserve(1 + memory_operands.size());
  operands.push_back(IdOperand(pointer_id));
  operands.insert(operands.end(), memory_operands.begin(),
                  memory_operands.end());
  std::unique_ptr<Instruction> load = MakeUnique<Instruction>(
      context(), spv::Op::OpLoad, type_id, TakeNextId(), operands);
  Instruction* added = builder->AddInstruction(std::move(load));
  return added ? added->result_id() : 0;
}

bool AzdLowerToStandardPass::AddStore(
    InstructionBuilder* builder, uint32_t pointer_id, uint32_t object_id,
    const std::vector<Operand>& memory_operands) {
  std::vector<Operand> operands;
  operands.reserve(2 + memory_operands.size());
  operands.push_back(IdOperand(pointer_id));
  operands.push_back(IdOperand(object_id));
  operands.insert(operands.end(), memory_operands.begin(),
                  memory_operands.end());
  std::unique_ptr<Instruction> store =
      MakeUnique<Instruction>(context(), spv::Op::OpStore, 0, 0, operands);
  return builder->AddInstruction(std::move(store)) != nullptr;
}

std::vector<Operand> AzdLowerToStandardPass::CopyMemoryOperands(
    const Instruction* inst, uint32_t first_in_operand) const {
  std::vector<Operand> operands;
  if (inst->NumInOperands() <= first_in_operand) return operands;
  operands.reserve(inst->NumInOperands() - first_in_operand);
  for (uint32_t i = first_in_operand; i < inst->NumInOperands(); ++i) {
    operands.push_back(inst->GetInOperand(i));
  }
  return operands;
}

uint32_t AzdLowerToStandardPass::ExtractVectorScalar(
    InstructionBuilder* builder, const VectorTypeInfo& info, uint32_t vector_id,
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

uint32_t AzdLowerToStandardPass::ExtractMatrixScalar(
    InstructionBuilder* builder, const MatrixTypeInfo& info, uint32_t matrix_id,
    uint32_t row, uint32_t col) {
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

uint32_t AzdLowerToStandardPass::BuildMatrixRowVector(
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

uint32_t AzdLowerToStandardPass::BuildMatrixColumnVector(
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

uint32_t AzdLowerToStandardPass::BuildVectorTimesScalar(
    InstructionBuilder* builder, uint32_t vec4_type_id, uint32_t vector_id,
    uint32_t scalar_id) {
  const uint32_t scalar_vec_id =
      BuildScalarSplat(builder, vec4_type_id, scalar_id);
  if (scalar_vec_id == 0) return 0;
  Instruction* mul = builder->AddBinaryOp(vec4_type_id, spv::Op::OpFMul,
                                          vector_id, scalar_vec_id);
  return mul ? mul->result_id() : 0;
}

uint32_t AzdLowerToStandardPass::BuildScalarSplat(InstructionBuilder* builder,
                                                  uint32_t vec4_type_id,
                                                  uint32_t scalar_id) {
  std::vector<uint32_t> lane_ids(kPackedVec4Width, scalar_id);
  Instruction* scalar_vec =
      builder->AddCompositeConstruct(vec4_type_id, lane_ids);
  return scalar_vec ? scalar_vec->result_id() : 0;
}

uint32_t AzdLowerToStandardPass::BuildFma(InstructionBuilder* builder,
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
  return added ? added->result_id() : 0;
}

uint32_t AzdLowerToStandardPass::BuildHorizontalReduce(
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
    sum = add->result_id();
  }
  return sum;
}

bool AzdLowerToStandardPass::BuildVectorMatrixMulPatternPackedVec4(
    InstructionBuilder* builder, const VectorTypeInfo& result,
    const VectorTypeInfo& input, const MatrixTypeInfo& matrix,
    const VectorTypeInfo* bias, uint32_t input_id, uint32_t matrix_id,
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

    const uint32_t out_row = out_pack * kPackedVec4Width;
    for (uint32_t k = 0; k < input.length; ++k) {
      const uint32_t scalar = ExtractVectorScalar(builder, input, input_id, k);
      const uint32_t weight4 = BuildMatrixColumnVector(
          builder, matrix, matrix_id, out_row, k, vec4_type_id);
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

bool AzdLowerToStandardPass::BuildMatmulPatternPackedVec4(
    InstructionBuilder* builder, const MatrixTypeInfo& result,
    const MatrixTypeInfo& a, const MatrixTypeInfo& b, const MatrixTypeInfo& c,
    uint32_t a_id, uint32_t b_id, uint32_t c_id,
    std::vector<uint32_t>* element_ids) {
  element_ids->assign(result.rows * result.packed_cols, 0);

  const uint32_t vec4_type_id = result.packed_vec4_type_id;
  const uint32_t zero4_id = GetOrCreateZero(vec4_type_id);
  if (zero4_id == 0) return false;
  const uint32_t tile_weight_function_id =
      GetOrCreateMatmulTileWeightFunctionPackedVec4(b);
  Instruction* insertion_point = get_def_use_mgr()->GetDef(b.lowered_type_id);
  if (tile_weight_function_id == 0 || !insertion_point) return false;
  const uint32_t weight_array_type_id = GetOrCreatePackedArrayType(
      b.packed_vec4_type_id, kPackedVec4Width, insertion_point);
  if (weight_array_type_id == 0) return false;

  for (uint32_t row0 = 0; row0 < result.rows; row0 += kPackedVec4Width) {
    const uint32_t tile_m = std::min(kPackedVec4Width, result.rows - row0);
    for (uint32_t col_pack = 0; col_pack < result.packed_cols; ++col_pack) {
      const uint32_t col0 = col_pack * kPackedVec4Width;
      std::vector<uint32_t> acc(tile_m * kPackedVec4Width, zero4_id);
      const uint32_t col_pack_id = GetOrCreateUIntConstant(col_pack);
      if (col_pack_id == 0) return false;
      const bool full_col_tile = col0 + kPackedVec4Width <= b.cols;

      for (uint32_t k0 = 0; k0 < a.cols; k0 += kPackedVec4Width) {
        Instruction* weights = nullptr;
        const bool full_k_tile = k0 + kPackedVec4Width <= a.cols;
        if (full_k_tile && full_col_tile) {
          const uint32_t k0_id = GetOrCreateUIntConstant(k0);
          if (k0_id == 0) return false;
          weights = builder->AddFunctionCall(weight_array_type_id,
                                             tile_weight_function_id,
                                             {b_id, k0_id, col_pack_id});
          if (!weights) return false;
        }

        std::vector<uint32_t> weight(kPackedVec4Width, 0);
        for (uint32_t col_lane = 0; col_lane < kPackedVec4Width; ++col_lane) {
          if (weights) {
            weight[col_lane] = ExtractCompositeElement(
                builder, vec4_type_id, weights->result_id(), col_lane);
          } else {
            weight[col_lane] = BuildMatrixColumnVector(
                builder, b, b_id, k0, col0 + col_lane, vec4_type_id);
          }
          if (weight[col_lane] == 0) return false;
        }

        for (uint32_t row_lane = 0; row_lane < tile_m; ++row_lane) {
          const uint32_t v = BuildMatrixRowVector(
              builder, a, a_id, row0 + row_lane, k0, vec4_type_id);
          if (v == 0) return false;

          for (uint32_t col_lane = 0; col_lane < kPackedVec4Width; ++col_lane) {
            const uint32_t index = row_lane * kPackedVec4Width + col_lane;
            acc[index] = BuildFma(builder, vec4_type_id, v, weight[col_lane],
                                  acc[index]);
            if (acc[index] == 0) return false;
          }
        }
      }

      for (uint32_t row_lane = 0; row_lane < tile_m; ++row_lane) {
        const uint32_t row = row0 + row_lane;
        std::vector<uint32_t> lane_ids;
        lane_ids.reserve(kPackedVec4Width);
        for (uint32_t col_lane = 0; col_lane < kPackedVec4Width; ++col_lane) {
          const uint32_t col = col0 + col_lane;
          const uint32_t reduced = BuildHorizontalReduce(
              builder, result.component_type_id,
              acc[row_lane * kPackedVec4Width + col_lane]);
          const uint32_t bias = ExtractMatrixScalar(builder, c, c_id, row, col);
          if (reduced == 0 || bias == 0) return false;
          Instruction* add = builder->AddBinaryOp(
              result.component_type_id, spv::Op::OpFAdd, bias, reduced);
          if (!add) return false;
          lane_ids.push_back(add->result_id());
        }

        Instruction* vec =
            builder->AddCompositeConstruct(vec4_type_id, lane_ids);
        if (!vec) return false;
        (*element_ids)[MatrixPackedIndex(result, row, col_pack)] =
            vec->result_id();
      }
    }
  }

  for (uint32_t id : *element_ids) {
    if (id == 0) return false;
  }
  return true;
}

void AzdLowerToStandardPass::AddGeneratedFunction(
    std::unique_ptr<Function> function, uint32_t function_id) {
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
  generated_function_ids_.insert(function_id);
}

std::string AzdLowerToStandardPass::MemoryOperandsKey(
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

uint32_t AzdLowerToStandardPass::GetOrCreateFunctionType(
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

uint32_t AzdLowerToStandardPass::GetOrCreatePackedLoadChunkFunction(
    uint32_t pointer_id, uint32_t pointer_type_id, uint32_t component_type_id,
    uint32_t vec4_type_id, const std::vector<Operand>& memory_operands) {
  const std::string key =
      std::to_string(pointer_id) + "|" + std::to_string(pointer_type_id) + "|" +
      std::to_string(component_type_id) + "|" + std::to_string(vec4_type_id) +
      "|" + MemoryOperandsKey(memory_operands);
  auto cached = packed_load_chunk_functions_.find(key);
  if (cached != packed_load_chunk_functions_.end()) return cached->second;
  if (!CanCapturePointer(pointer_id)) return 0;

  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t function_type_id =
      GetOrCreateFunctionType(vec4_type_id, {uint_type_id});
  if (uint_type_id == 0 || function_type_id == 0) return 0;

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start =
      MakeUnique<Instruction>(context(), spv::Op::OpFunction, vec4_type_id,
                              function_id, std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t base_param_id = TakeNextId();
  if (base_param_id == 0) return 0;
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, base_param_id,
      std::initializer_list<Operand>{}));

  const uint32_t entry_label_id = TakeNextId();
  if (entry_label_id == 0) return 0;
  std::unique_ptr<BasicBlock> entry_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_label_id));
  if (!entry_block) return 0;

  InstructionBuilder builder(context(), entry_block.get());
  const uint32_t captured_pointer_id =
      BuildCapturedPointer(&builder, pointer_id);
  if (captured_pointer_id == 0) return 0;

  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    const uint32_t lane_offset_id = GetOrCreateUIntConstant(lane);
    if (lane_offset_id == 0) return 0;
    uint32_t element_index_id = base_param_id;
    if (lane != 0) {
      Instruction* element_index = builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, base_param_id, lane_offset_id);
      if (!element_index) return 0;
      element_index_id = element_index->result_id();
    }
    const uint32_t elem_ptr_id = BuildElementAccessFromPointerType(
        &builder, pointer_type_id, captured_pointer_id, component_type_id,
        element_index_id);
    if (elem_ptr_id == 0) return 0;
    const uint32_t load_id =
        AddLoad(&builder, component_type_id, elem_ptr_id, memory_operands);
    if (load_id == 0) return 0;
    lane_ids.push_back(load_id);
  }

  Instruction* vec = builder.AddCompositeConstruct(vec4_type_id, lane_ids);
  if (!vec) return 0;
  if (!builder.AddUnaryOp(0, spv::Op::OpReturnValue, vec->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(entry_block));
  AddGeneratedFunction(std::move(function), function_id);

  packed_load_chunk_functions_[key] = function_id;
  return function_id;
}

uint32_t AzdLowerToStandardPass::GetOrCreatePackedStoreChunkFunction(
    uint32_t pointer_id, uint32_t pointer_type_id, uint32_t component_type_id,
    uint32_t vec4_type_id, const std::vector<Operand>& memory_operands) {
  const std::string key =
      std::to_string(pointer_id) + "|" + std::to_string(pointer_type_id) + "|" +
      std::to_string(component_type_id) + "|" + std::to_string(vec4_type_id) +
      "|" + MemoryOperandsKey(memory_operands);
  auto cached = packed_store_chunk_functions_.find(key);
  if (cached != packed_store_chunk_functions_.end()) return cached->second;
  if (!CanCapturePointer(pointer_id)) return 0;

  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t function_type_id =
      GetOrCreateFunctionType(void_type_id, {uint_type_id, vec4_type_id});
  if (void_type_id == 0 || uint_type_id == 0 || function_type_id == 0) {
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

  const uint32_t base_param_id = TakeNextId();
  const uint32_t value_param_id = TakeNextId();
  if (base_param_id == 0 || value_param_id == 0) return 0;
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, base_param_id,
      std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, vec4_type_id, value_param_id,
      std::initializer_list<Operand>{}));

  const uint32_t entry_label_id = TakeNextId();
  if (entry_label_id == 0) return 0;
  std::unique_ptr<BasicBlock> entry_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_label_id));
  if (!entry_block) return 0;

  InstructionBuilder builder(context(), entry_block.get());
  const uint32_t captured_pointer_id =
      BuildCapturedPointer(&builder, pointer_id);
  if (captured_pointer_id == 0) return 0;

  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    const uint32_t lane_offset_id = GetOrCreateUIntConstant(lane);
    if (lane_offset_id == 0) return 0;
    uint32_t element_index_id = base_param_id;
    if (lane != 0) {
      Instruction* element_index = builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, base_param_id, lane_offset_id);
      if (!element_index) return 0;
      element_index_id = element_index->result_id();
    }
    const uint32_t elem_ptr_id = BuildElementAccessFromPointerType(
        &builder, pointer_type_id, captured_pointer_id, component_type_id,
        element_index_id);
    const uint32_t value_id = ExtractCompositeElement(
        &builder, component_type_id, value_param_id, lane);
    if (elem_ptr_id == 0 || value_id == 0) return 0;
    if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands)) return 0;
  }

  if (!builder.AddNullaryOp(0, spv::Op::OpReturn)) return 0;

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(entry_block));
  AddGeneratedFunction(std::move(function), function_id);

  packed_store_chunk_functions_[key] = function_id;
  return function_id;
}

uint32_t AzdLowerToStandardPass::GetOrCreateTileWeightFunctionPackedVec4(
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
  AddGeneratedFunction(std::move(function), function_id);

  tile_weight_functions_[key] = function_id;
  return function_id;
}

uint32_t AzdLowerToStandardPass::GetOrCreateMatmulTileWeightFunctionPackedVec4(
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
  AddGeneratedFunction(std::move(function), function_id);

  matmul_tile_weight_functions_[key] = function_id;
  return function_id;
}

uint32_t
AzdLowerToStandardPass::GetOrCreateVectorMatmulPatternFunctionPackedVec4(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias) {
  const std::string key =
      VectorMatmulPatternFunctionKey(result, input, matrix, bias, has_bias);
  auto cached = vector_matmul_pattern_functions_.find(key);
  if (cached != vector_matmul_pattern_functions_.end()) return cached->second;

  const bool use_tile_weight = IsPackedVec4(input) && IsPackedVec4(matrix);
  uint32_t tile_weight_function_id = 0;
  uint32_t weight_array_type_id = 0;
  if (use_tile_weight) {
    tile_weight_function_id = GetOrCreateTileWeightFunctionPackedVec4(matrix);
    if (tile_weight_function_id == 0) return 0;

    Instruction* insertion_point =
        get_def_use_mgr()->GetDef(matrix.lowered_type_id);
    if (!insertion_point) return 0;
    weight_array_type_id = GetOrCreatePackedArrayType(
        matrix.packed_vec4_type_id, kPackedVec4Width, insertion_point);
    if (weight_array_type_id == 0) return 0;
  }

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

  if (!use_tile_weight) {
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
    AddGeneratedFunction(std::move(function), function_id);

    vector_matmul_pattern_functions_[key] = function_id;
    return function_id;
  }

  std::vector<uint32_t> element_ids(result.packed_length, 0);
  for (uint32_t out_pack = 0; out_pack < result.packed_length; ++out_pack) {
    std::vector<uint32_t> acc(kPackedVec4Width, zero4_id);
    const uint32_t row_id =
        GetOrCreateUIntConstant(out_pack * kPackedVec4Width);
    if (row_id == 0) return 0;

    for (uint32_t k_pack = 0; k_pack < input.packed_length; ++k_pack) {
      const uint32_t v = ExtractCompositeElement(
          &builder, input.packed_vec4_type_id, input_param_id, k_pack);
      const uint32_t col_id = GetOrCreateUIntConstant(k_pack);
      if (v == 0 || col_id == 0) return 0;
      Instruction* weights =
          builder.AddFunctionCall(weight_array_type_id, tile_weight_function_id,
                                  {matrix_param_id, row_id, col_id});
      if (!weights) return 0;

      for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
        const uint32_t weight = ExtractCompositeElement(
            &builder, vec4_type_id, weights->result_id(), lane);
        if (weight == 0) return 0;
        acc[lane] = BuildFma(&builder, vec4_type_id, v, weight, acc[lane]);
        if (acc[lane] == 0) return 0;
      }
    }

    std::vector<uint32_t> lane_ids;
    lane_ids.reserve(kPackedVec4Width);
    for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
      uint32_t reduced =
          BuildHorizontalReduce(&builder, result.component_type_id, acc[lane]);
      if (reduced == 0) return 0;
      if (has_bias) {
        const uint32_t bias_value = ExtractVectorScalar(
            &builder, *bias, bias_param_id, out_pack * kPackedVec4Width + lane);
        if (bias_value == 0) return 0;
        Instruction* add = builder.AddBinaryOp(
            result.component_type_id, spv::Op::OpFAdd, reduced, bias_value);
        if (!add) return 0;
        reduced = add->result_id();
      }
      lane_ids.push_back(reduced);
    }

    Instruction* vec = builder.AddCompositeConstruct(vec4_type_id, lane_ids);
    if (!vec) return 0;
    element_ids[out_pack] = vec->result_id();
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
  AddGeneratedFunction(std::move(function), function_id);

  vector_matmul_pattern_functions_[key] = function_id;
  return function_id;
}

uint32_t AzdLowerToStandardPass::GetOrCreateMatmulPatternFunctionPackedVec4(
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
  std::unique_ptr<BasicBlock> block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));

  InstructionBuilder builder(context(), block.get());
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
  function->AddBasicBlock(std::move(block));
  AddGeneratedFunction(std::move(function), function_id);
  matmul_pattern_functions_[key] = function_id;
  matmul_pattern_function_ids_.insert(function_id);
  return function_id;
}

std::string AzdLowerToStandardPass::TileWeightFunctionKey(
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

std::string AzdLowerToStandardPass::MatmulTileWeightFunctionKey(
    const MatrixTypeInfo& matrix) const {
  return std::string("matmul|") + TileWeightFunctionKey(matrix);
}

std::string AzdLowerToStandardPass::VectorMatmulPatternFunctionKey(
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
  return key;
}

std::string AzdLowerToStandardPass::MatmulPatternFunctionKey(
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
  return key;
}

bool AzdLowerToStandardPass::IsPackedVec4(const MatrixTypeInfo& info) const {
  return info.packed_f16vec4 || info.packed_f32vec4;
}

bool AzdLowerToStandardPass::IsPackedVec4(const VectorTypeInfo& info) const {
  return info.packed_f16vec4 || info.packed_f32vec4;
}

bool AzdLowerToStandardPass::IsSamePackedVec4Kind(
    const MatrixTypeInfo& a, const MatrixTypeInfo& b) const {
  return IsPackedVec4(a) && IsPackedVec4(b) &&
         a.packed_f16vec4 == b.packed_f16vec4 &&
         a.packed_f32vec4 == b.packed_f32vec4 &&
         a.component_type_id == b.component_type_id &&
         a.packed_vec4_type_id == b.packed_vec4_type_id;
}

bool AzdLowerToStandardPass::IsSamePackedVec4Kind(
    const VectorTypeInfo& a, const VectorTypeInfo& b) const {
  return IsPackedVec4(a) && IsPackedVec4(b) &&
         a.packed_f16vec4 == b.packed_f16vec4 &&
         a.packed_f32vec4 == b.packed_f32vec4 &&
         a.component_type_id == b.component_type_id &&
         a.packed_vec4_type_id == b.packed_vec4_type_id;
}

bool AzdLowerToStandardPass::CanUsePackedVec4MatrixMulAdd(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c) const {
  return IsPackedVec4(result) &&
         result.component_type_id == a.component_type_id &&
         IsSamePackedVec4Kind(result, b) && IsSamePackedVec4Kind(result, c);
}

bool AzdLowerToStandardPass::CanUsePackedVec4VectorMatrixMul(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias) const {
  if (!IsPackedVec4(result) ||
      result.component_type_id != input.component_type_id ||
      result.component_type_id != matrix.component_type_id) {
    return false;
  }
  if (result.length != matrix.rows || input.length != matrix.cols) {
    return false;
  }
  return !bias || IsSamePackedVec4Kind(result, *bias);
}

bool AzdLowerToStandardPass::ShouldUsePackedVec4(uint32_t extent) const {
  return lowering_mode_ == LoweringMode::kPreferPackedVec4 &&
         extent % kPackedVec4Width == 0;
}

uint32_t AzdLowerToStandardPass::MatrixFlatIndex(const MatrixTypeInfo& info,
                                                 uint32_t row,
                                                 uint32_t col) const {
  return row * info.cols + col;
}

uint32_t AzdLowerToStandardPass::MatrixPackedIndex(const MatrixTypeInfo& info,
                                                   uint32_t row,
                                                   uint32_t col_pack) const {
  return row * info.packed_cols + col_pack;
}

uint32_t AzdLowerToStandardPass::VectorPackedIndex(
    uint32_t scalar_index) const {
  return scalar_index / kPackedVec4Width;
}

uint32_t AzdLowerToStandardPass::PackedLane(uint32_t scalar_index) const {
  return scalar_index % kPackedVec4Width;
}

uint32_t AzdLowerToStandardPass::GetOrCreateGLSLStd450Import() {
  uint32_t import_id =
      context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  if (import_id != 0) return import_id;

  import_id = get_module()->GetExtInstImportId("GLSL.std.450");
  if (import_id != 0) return import_id;

  context()->AddExtInstImport("GLSL.std.450");
  return context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
}

const AzdLowerToStandardPass::MatrixTypeInfo*
AzdLowerToStandardPass::GetMatrixType(uint32_t type_id) const {
  auto it = matrix_types_.find(type_id);
  if (it != matrix_types_.end()) return &it->second;
  for (const auto& id_and_info : matrix_types_) {
    if (id_and_info.second.lowered_type_id == type_id)
      return &id_and_info.second;
  }
  return nullptr;
}

const AzdLowerToStandardPass::VectorTypeInfo*
AzdLowerToStandardPass::GetVectorType(uint32_t type_id) const {
  auto it = vector_types_.find(type_id);
  if (it != vector_types_.end()) return &it->second;
  for (const auto& id_and_info : vector_types_) {
    if (id_and_info.second.lowered_type_id == type_id)
      return &id_and_info.second;
  }
  return nullptr;
}

uint32_t AzdLowerToStandardPass::GetLoweredType(uint32_t type_id) const {
  auto it = lowered_types_.find(type_id);
  return it == lowered_types_.end() ? 0 : it->second;
}

uint32_t AzdLowerToStandardPass::GetPointerTypeId(uint32_t pointer_id) const {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  return pointer ? pointer->type_id() : 0;
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

bool AzdLowerToStandardPass::IsFloat16Type(uint32_t type_id) const {
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  return type && type->opcode() == spv::Op::OpTypeFloat &&
         type->GetSingleWordInOperand(0) == 16;
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
  if (inst->opcode() == spv::Op::OpSourceExtension) {
    const std::string extension = inst->GetInOperand(0).AsString();
    return extension == "GL_AZD_neural_matrix" ||
           extension == "GL_AZD_cooperative_vector";
  }
  return false;
}

bool AzdLowerToStandardPass::RemoveExtensionByName(const char* extension_name) {
  return context()->KillInstructionIf(
      get_module()->extension_begin(), get_module()->extension_end(),
      [extension_name](Instruction* inst) {
        return inst->opcode() == spv::Op::OpExtension &&
               inst->GetInOperand(0).AsString() == extension_name;
      });
}

bool AzdLowerToStandardPass::RemoveSourceExtensionByName(
    const char* extension_name) {
  return context()->KillInstructionIf(
      get_module()->debug1_begin(), get_module()->debug1_end(),
      [extension_name](Instruction* inst) {
        return inst->opcode() == spv::Op::OpSourceExtension &&
               inst->GetInOperand(0).AsString() == extension_name;
      });
}

void AzdLowerToStandardPass::RebuildAsCompositeConstruct(
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

void AzdLowerToStandardPass::RebuildAsFunctionCall(
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

void AzdLowerToStandardPass::ReportError(const Instruction*,
                                         const std::string& message) const {
  if (!consumer()) return;
  consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
}

}  // namespace opt
}  // namespace spvtools
