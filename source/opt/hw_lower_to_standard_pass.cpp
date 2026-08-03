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

Pass::Status HwLowerToStandardPass::Process() {
  if (max_elements_ == 0 || max_matmul_macs_ == 0 ||
      max_unrolled_elements_ == 0 || max_unrolled_matmul_macs_ == 0 ||
      max_unrolled_elements_ > max_elements_ ||
      max_unrolled_matmul_macs_ > max_matmul_macs_) {
    ReportError(nullptr, "invalid HW lowering size configuration");
    return Status::Failure;
  }
  // An instruction word count is 16-bit.  OpCompositeConstruct has three
  // fixed words in addition to its constituents, so user-provided unroll
  // thresholds must never force an unserializable aggregate instruction.
  max_unrolled_elements_ =
      std::min(max_unrolled_elements_, kMaxCompositeConstituents);
  max_unrolled_matmul_macs_ =
      std::min(max_unrolled_matmul_macs_, uint64_t(kMaxCompositeConstituents));
  matrix_types_.clear();
  vector_types_.clear();
  lowered_types_.clear();
  original_hw_value_types_.clear();
  packed_load_chunk_functions_.clear();
  packed_store_chunk_functions_.clear();
  tile_weight_functions_.clear();
  vector_matmul_pattern_functions_.clear();
  matmul_pattern_functions_.clear();
  matmul_pattern_function_ids_.clear();
  generated_function_ids_.clear();
  read_only_generated_function_ids_.clear();
  pending_fp_fast_math_modes_.clear();
  active_fp_fast_math_mode_ = 0;

  bool has_hw = false;
  get_module()->ForEachInst([this, &has_hw](Instruction* inst) {
    has_hw |=
        completeness_mode_ == CompletenessMode::kExtensionFree
            ? (IsAnyHwOpcode(inst->opcode()) ||
               IsAnyHwCapabilityOrExtension(inst) || HasHwOperand(inst))
            : (IsHwOpcode(inst->opcode()) || IsHwCapabilityOrExtension(inst));
  });
  if (!has_hw) return Status::SuccessWithoutChange;

  if (completeness_mode_ == CompletenessMode::kExtensionFree &&
      !PreflightExtensionFreeMode()) {
    return Status::Failure;
  }

  if (!PreflightNoContractionVectorMatmul()) return Status::Failure;

  if (lowering_mode_ == LoweringMode::kPreferPackedVec4) {
    HwFuseTwoLayerVectorMatmulPass fusion(max_unrolled_matmul_macs_);
    fusion.SetMessageConsumer(consumer());
    if (fusion.Run(context()) == Status::Failure) return Status::Failure;
  }

  if (!CollectHwTypes()) return Status::Failure;
  if (!LegalizeModule()) return Status::Failure;
  if (!MaterializeLoweredTypes()) return Status::Failure;
  RecordOriginalHwValueTypes();
  if (!EliminateHwFunctionVariables()) return Status::Failure;
  if (!PrepareMatmulPatternFunctions()) return Status::Failure;

  std::vector<Instruction*> to_kill;
  if (!LowerHwInstructions(&to_kill)) return Status::Failure;
  if (!ReplaceHwTypeUses()) return Status::Failure;
  if (!CleanupHwDeclarations(to_kill)) return Status::Failure;
  if (!FinalHwCheck()) return Status::Failure;

  context()->InvalidateAnalyses(IRContext::kAnalysisTypes |
                                IRContext::kAnalysisConstants);
  return Status::SuccessWithChange;
}

bool HwLowerToStandardPass::LowerHwInstructions(
    std::vector<Instruction*>* to_kill) {
  std::vector<Instruction*> vector_stores;
  get_module()->ForEachInst([&vector_stores](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpCooperativeVectorStoreHW) {
      vector_stores.push_back(inst);
    }
  });
  for (Instruction* inst : vector_stores) {
    if (!inst || inst->IsNop()) continue;
    Instruction* object = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorStoreObjectInIdx));
    std::vector<Instruction*> chain;
    Instruction* matmul = TraceFunctionValueSource(object, inst, &chain);
    const uint32_t saved_fast_math_mode = active_fp_fast_math_mode_;
    active_fp_fast_math_mode_ =
        matmul ? GetFPFastMathMode(matmul->result_id()) : 0;
    bool handled = false;
    const bool ok = TryLowerFusedVectorMatmulStore(inst, &handled);
    active_fp_fast_math_mode_ = saved_fast_math_mode;
    if (!ok) return false;
  }

  std::vector<Instruction*> matrix_stores;
  get_module()->ForEachInst([&matrix_stores](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpCooperativeMatrixStoreHW) {
      matrix_stores.push_back(inst);
    }
  });
  for (Instruction* inst : matrix_stores) {
    if (!inst || inst->IsNop()) continue;
    Instruction* object = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwMatrixStoreObjectInIdx));
    std::vector<Instruction*> chain;
    Instruction* matmul = TraceFunctionValueSource(object, inst, &chain);
    const uint32_t saved_fast_math_mode = active_fp_fast_math_mode_;
    active_fp_fast_math_mode_ =
        matmul ? GetFPFastMathMode(matmul->result_id()) : 0;
    bool handled = false;
    const bool ok = TryLowerFusedMatrixMatmulStore(inst, &handled);
    active_fp_fast_math_mode_ = saved_fast_math_mode;
    if (!ok) return false;
  }

  std::vector<Instruction*> direct_matmuls;
  get_module()->ForEachInst([&direct_matmuls](Instruction* inst) {
    switch (inst->opcode()) {
      case spv::Op::OpCooperativeMatrixMulAddHW:
      case spv::Op::OpCooperativeVectorMatrixMulHW:
      case spv::Op::OpCooperativeVectorMatrixMulAddHW:
        direct_matmuls.push_back(inst);
        break;
      default:
        break;
    }
  });
  for (Instruction* inst : direct_matmuls) {
    if (!inst || inst->IsNop()) continue;
    const uint32_t saved_fast_math_mode = active_fp_fast_math_mode_;
    active_fp_fast_math_mode_ = GetFPFastMathMode(inst->result_id());
    bool handled = false;
    bool ok = true;
    switch (inst->opcode()) {
      case spv::Op::OpCooperativeMatrixMulAddHW:
        ok = TryLowerDirectMatrixMulAddPackedVec4(inst, &handled);
        break;
      case spv::Op::OpCooperativeVectorMatrixMulHW:
        ok = TryLowerDirectVectorMatrixMulPackedVec4(inst, false, &handled);
        break;
      case spv::Op::OpCooperativeVectorMatrixMulAddHW:
        ok = TryLowerDirectVectorMatrixMulPackedVec4(inst, true, &handled);
        break;
      default:
        break;
    }
    if (handled) RemoveFPFastMathMode(inst->result_id());
    active_fp_fast_math_mode_ = saved_fast_math_mode;
    if (!ok) return false;
  }

  std::vector<Instruction*> worklist;
  get_module()->ForEachInst([this, &worklist](Instruction* inst) {
    BasicBlock* block = context()->get_instr_block(inst);
    Function* function = block ? block->GetParent() : nullptr;
    if (function && generated_function_ids_.find(function->result_id()) !=
                        generated_function_ids_.end()) {
      return;
    }

    switch (inst->opcode()) {
      case spv::Op::OpCooperativeMatrixLoadHW:
      case spv::Op::OpCooperativeMatrixStoreHW:
      case spv::Op::OpCooperativeMatrixMulAddHW:
      case spv::Op::OpCooperativeMatrixLengthHW:
      case spv::Op::OpCooperativeMatrixReduceHW:
      case spv::Op::OpCooperativeVectorLoadHW:
      case spv::Op::OpCooperativeVectorStoreHW:
      case spv::Op::OpCooperativeVectorMatrixMulHW:
      case spv::Op::OpCooperativeVectorMatrixMulAddHW:
        worklist.push_back(inst);
        break;
      case spv::Op::OpCompositeConstruct:
      case spv::Op::OpCompositeConstructReplicateEXT:
        if (IsHwType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpConstantComposite:
      case spv::Op::OpSpecConstantComposite:
      case spv::Op::OpConstantCompositeReplicateEXT:
      case spv::Op::OpSpecConstantCompositeReplicateEXT:
        if (IsHwType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpCompositeExtract:
        worklist.push_back(inst);
        break;
      case spv::Op::OpCompositeInsert:
        if (TypeContainsHw(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpSelect:
        if (IsHwType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpVectorExtractDynamic: {
        Instruction* object =
            inst->NumInOperands() >= 1
                ? get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0))
                : nullptr;
        if (object && GetVectorType(object->type_id()))
          worklist.push_back(inst);
        break;
      }
      case spv::Op::OpVectorInsertDynamic:
        if (GetVectorType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
      case spv::Op::OpPtrAccessChain:
      case spv::Op::OpInBoundsPtrAccessChain: {
        Instruction* base =
            inst->NumInOperands() >= 1
                ? get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0))
                : nullptr;
        if (base && TypeContainsHw(base->type_id())) worklist.push_back(inst);
        break;
      }
      case spv::Op::OpConstantNull:
      case spv::Op::OpUndef:
        if (IsHwType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpBitcast:
        if (TypeContainsHw(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpExtInst:
        if (GetVectorType(inst->type_id()) != nullptr) worklist.push_back(inst);
        break;
      default:
        if (TypeContainsHw(inst->type_id()) &&
            (IsHwConversionOpcode(inst->opcode()) ||
             IsHwArithmeticOpcode(inst->opcode()) ||
             IsHwScaleOpcode(inst->opcode()))) {
          worklist.push_back(inst);
        }
        break;
    }
  });

  for (Instruction* inst : worklist) {
    const uint32_t saved_fast_math_mode = active_fp_fast_math_mode_;
    const uint32_t source_fast_math_mode =
        inst ? GetFPFastMathMode(inst->result_id()) : 0;
    active_fp_fast_math_mode_ = source_fast_math_mode;
    bool ok = true;
    switch (inst->opcode()) {
      case spv::Op::OpCooperativeMatrixLoadHW:
        ok = LowerMatrixLoad(inst);
        break;
      case spv::Op::OpCooperativeMatrixStoreHW:
        ok = LowerMatrixStore(inst, to_kill);
        break;
      case spv::Op::OpCooperativeMatrixMulAddHW:
        ok = LowerMatrixMulAdd(inst);
        break;
      case spv::Op::OpCooperativeMatrixLengthHW:
        ok = LowerMatrixLength(inst, to_kill);
        break;
      case spv::Op::OpCooperativeMatrixReduceHW:
        ok = LowerMatrixReduce(inst);
        break;
      case spv::Op::OpCooperativeVectorLoadHW:
        ok = LowerVectorLoad(inst);
        break;
      case spv::Op::OpCooperativeVectorStoreHW:
        ok = LowerVectorStore(inst, to_kill);
        break;
      case spv::Op::OpCooperativeVectorMatrixMulHW:
        ok = LowerVectorMatrixMul(inst, false);
        break;
      case spv::Op::OpCooperativeVectorMatrixMulAddHW:
        ok = LowerVectorMatrixMul(inst, true);
        break;
      case spv::Op::OpCompositeConstruct:
      case spv::Op::OpCompositeConstructReplicateEXT:
        ok = LowerCompositeConstruct(inst);
        break;
      case spv::Op::OpConstantComposite:
      case spv::Op::OpSpecConstantComposite:
      case spv::Op::OpConstantCompositeReplicateEXT:
      case spv::Op::OpSpecConstantCompositeReplicateEXT:
        ok = LowerConstantComposite(inst);
        break;
      case spv::Op::OpCompositeExtract:
        ok = LowerCompositeExtract(inst);
        break;
      case spv::Op::OpCompositeInsert:
        ok = LowerCompositeInsert(inst);
        break;
      case spv::Op::OpSelect:
        ok = LowerSelect(inst);
        break;
      case spv::Op::OpVectorExtractDynamic:
        ok = LowerVectorExtractDynamic(inst);
        break;
      case spv::Op::OpVectorInsertDynamic:
        ok = LowerVectorInsertDynamic(inst);
        break;
      case spv::Op::OpAccessChain:
      case spv::Op::OpInBoundsAccessChain:
      case spv::Op::OpPtrAccessChain:
      case spv::Op::OpInBoundsPtrAccessChain:
        ok = LowerAccessChain(inst);
        break;
      case spv::Op::OpConstantNull:
      case spv::Op::OpUndef:
        ok = LowerNullOrUndef(inst);
        break;
      case spv::Op::OpBitcast:
        ok = LowerHwBitcast(inst);
        break;
      case spv::Op::OpExtInst:
        ok = LowerExtInstOnCooperativeVector(inst);
        break;
      default:
        if (IsHwConversionOpcode(inst->opcode())) {
          ok = LowerHwConversion(inst);
        } else if (IsHwArithmeticOpcode(inst->opcode())) {
          ok = LowerHwArithmetic(inst);
        } else if (IsHwScaleOpcode(inst->opcode())) {
          ok = LowerHwScale(inst);
        }
        break;
    }
    if (ok && source_fast_math_mode != 0) {
      RemoveFPFastMathMode(inst->result_id());
    }
    active_fp_fast_math_mode_ = saved_fast_math_mode;
    if (!ok) return false;
  }

  return true;
}

bool HwLowerToStandardPass::ReplaceHwTypeUses() {
  for (const auto& type_pair : lowered_types_) {
    Instruction* old_type = get_def_use_mgr()->GetDef(type_pair.first);
    Instruction* new_type = get_def_use_mgr()->GetDef(type_pair.second);
    if (!old_type || !new_type) return false;
    context()->ReplaceAllUsesWith(type_pair.first, type_pair.second);
  }

  // Unlike aggregate type declarations, OpTypeFunction declarations must be
  // unique.  Distinct signatures that use cooperative types can become
  // identical after the leaf types above are replaced, so canonicalize them
  // before validation observes the rewritten module.
  std::map<std::vector<uint32_t>, uint32_t> canonical_function_types;
  std::vector<Instruction*> duplicate_function_types;
  for (Instruction& type : get_module()->types_values()) {
    if (type.opcode() != spv::Op::OpTypeFunction) continue;

    std::vector<uint32_t> signature;
    signature.reserve(type.NumInOperands());
    for (uint32_t i = 0; i < type.NumInOperands(); ++i) {
      signature.push_back(type.GetSingleWordInOperand(i));
    }
    const auto inserted = canonical_function_types.emplace(std::move(signature),
                                                           type.result_id());
    if (inserted.second) continue;

    context()->ReplaceAllUsesWith(type.result_id(), inserted.first->second);
    duplicate_function_types.push_back(&type);
  }
  for (Instruction* duplicate : duplicate_function_types) {
    context()->KillInst(duplicate);
  }

  // Distinct, logically matching cooperative types can canonicalize to the
  // same lowered aggregate type.  OpCopyLogical requires its two types to be
  // distinct, while OpCopyObject expresses the resulting same-type copy.
  get_module()->ForEachInst([this](Instruction* inst) {
    if (inst->opcode() != spv::Op::OpCopyLogical ||
        inst->NumInOperands() != 1) {
      return;
    }
    Instruction* source =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
    if (source && inst->type_id() == source->type_id()) {
      inst->SetOpcode(spv::Op::OpCopyObject);
      context()->UpdateDefUse(inst);
    }
  });
  return true;
}

bool HwLowerToStandardPass::CleanupHwDeclarations(
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

  modified |= context()->RemoveCapability(spv::Capability::CooperativeMatrixHW);
  modified |= context()->RemoveCapability(spv::Capability::CooperativeVectorHW);
  modified |= RemoveExtensionByName("SPV_AZD_neural_matrix");
  modified |= RemoveExtensionByName("SPV_AZD_cooperative_vector");
  modified |= RemoveSourceExtensionByName("GL_AZD_neural_matrix");
  modified |= RemoveSourceExtensionByName("GL_AZD_cooperative_vector");
  if (!ModuleRequiresHwNeuralShaderExtension()) {
    modified |= RemoveExtensionByName("SPV_HW_neural_shader");
    modified |= RemoveSourceExtensionByName("GL_HW_neural_shader");
  }
  (void)modified;
  return true;
}

bool HwLowerToStandardPass::FinalHwCheck() const {
  bool ok = true;
  get_module()->ForEachInst([this, &ok](Instruction* inst) {
    if (!ok) return;
    const bool has_forbidden_hw =
        completeness_mode_ == CompletenessMode::kExtensionFree
            ? (IsAnyHwOpcode(inst->opcode()) ||
               IsAnyHwCapabilityOrExtension(inst) || HasHwOperand(inst))
            : (IsHwOpcode(inst->opcode()) || IsHwCapabilityOrExtension(inst));
    if (has_forbidden_hw || HasHwTypeReference(inst)) {
      ReportError(inst, "HW lowering left HW op/type/capability/extension");
      ok = false;
    }
  });
  return ok;
}

bool HwLowerToStandardPass::RemoveExtensionByName(const char* extension_name) {
  return context()->KillInstructionIf(
      get_module()->extension_begin(), get_module()->extension_end(),
      [extension_name](Instruction* inst) {
        return inst->opcode() == spv::Op::OpExtension &&
               inst->GetInOperand(0).AsString() == extension_name;
      });
}

bool HwLowerToStandardPass::RemoveSourceExtensionByName(
    const char* extension_name) {
  return context()->KillInstructionIf(
      get_module()->debug1_begin(), get_module()->debug1_end(),
      [extension_name](Instruction* inst) {
        return inst->opcode() == spv::Op::OpSourceExtension &&
               inst->GetInOperand(0).AsString() == extension_name;
      });
}

void HwLowerToStandardPass::ReportError(const Instruction*,
                                        const std::string& message) const {
  if (!consumer()) return;
  consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
}

}  // namespace opt
}  // namespace spvtools
