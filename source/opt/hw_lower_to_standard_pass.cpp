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
#include "source/opt/build_module.h"
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
  if (!transactional_) return ProcessImpl();

  if (completeness_mode_ == CompletenessMode::kExtensionFree &&
      (!PreflightExtensionFreeMode() || !PreflightReplicatedComposites()))
    return Status::Failure;
  // Preserve precise diagnostics for malformed cooperative instructions used
  // by defensive callers, before serializing their in-memory representation.
  if (!CollectHwTypes()) return Status::Failure;
  bool valid_extinst = true;
  get_module()->ForEachInst([this, &valid_extinst](Instruction* inst) {
    if (valid_extinst && inst->opcode() == spv::Op::OpExtInst &&
        InstructionTouchesHw(inst))
      valid_extinst = LegalizeCooperativeCoreInstruction(inst);
  });
  if (!valid_extinst) return Status::Failure;

  std::vector<uint32_t> original;
  get_module()->ToBinary(&original, false);
  SpirvTools validator(context()->GetTargetEnv());
  // Some internal callers deliberately pass pre-legalization input. Only use
  // the target's ordinary validation rules when they accept the input itself.
  const bool validate_output = validator.Validate(original);
  struct Diagnostic {
    spv_message_level_t level;
    std::string source;
    spv_position_t position;
    std::string message;
  };
  std::vector<Diagnostic> diagnostics;
  const auto capture = [&diagnostics](
                           spv_message_level_t level, const char* source,
                           const spv_position_t& pos, const char* text) {
    diagnostics.push_back({level, source ? source : "", pos, text});
  };
  validator.SetMessageConsumer(capture);
  auto flush = [&]() {
    if (!consumer()) return;
    for (const auto& d : diagnostics)
      consumer()(d.level, d.source.c_str(), d.position, d.message.c_str());
  };

  LoweringMode mode = lowering_mode_;
  for (;;) {
    // Never reuse a partially rewritten attempt, including its cached analyses.
    auto attempt = BuildModule(context()->GetTargetEnv(), capture,
                               original.data(), original.size(), false);
    if (!attempt) {
      flush();
      return Status::Failure;
    }
    attempt->set_max_id_bound(context()->max_id_bound());
    attempt->set_preserve_bindings(context()->preserve_bindings());
    attempt->set_preserve_spec_constants(context()->preserve_spec_constants());
    bool recoverable = false;
    Status status = RunLoweringAttempt(attempt.get(), mode, &recoverable);
    std::vector<uint32_t> lowered;
    if (status == Status::SuccessWithChange) {
      attempt->module()->ToBinary(&lowered, false);
      if (validate_output && !validator.Validate(lowered)) {
        status = Status::Failure;
        recoverable = true;
      }
    }
    if (status != Status::Failure) {
      if (status == Status::SuccessWithChange) {
        if (!context()->ReplaceModule(lowered)) return Status::Failure;
      }
      flush();
      return status;
    }
    if (!recoverable ||
        (mode == LoweringMode::kForceScalar && !enable_direct_) ||
        attempt->module()->IdBound() >= attempt->max_id_bound()) {
      flush();
      return Status::Failure;
    }
    if (consumer()) {
      std::string message =
          "HW optimized lowering failed; retrying scalar from the original "
          "module";
      if (!diagnostics.empty()) message += ": " + diagnostics.back().message;
      consumer()(SPV_MSG_WARNING, "", {0, 0, 0}, message.c_str());
    }
    diagnostics.clear();
    enable_direct_ = false;
    enable_fusion_ = false;
    mode = LoweringMode::kForceScalar;
  }
}

Pass::Status HwLowerToStandardPass::RunLoweringAttempt(IRContext* attempt,
                                                       LoweringMode mode,
                                                       bool* recoverable) {
  HwLowerToStandardPass worker(
      mode, completeness_mode_, max_elements_, max_matmul_macs_,
      max_unrolled_elements_, max_unrolled_matmul_macs_,
      enable_direct_ && mode == lowering_mode_, enable_fusion_);
  worker.transactional_ = false;
  worker.SetMessageConsumer(attempt->consumer());
  const Status status = worker.Run(attempt);
  *recoverable = worker.recoverable_failure_;
  return status;
}

Pass::Status HwLowerToStandardPass::ProcessImpl() {
  if (max_unrolled_elements_ == 0 || max_unrolled_matmul_macs_ == 0 ||
      (max_elements_ && max_unrolled_elements_ > max_elements_) ||
      (max_matmul_macs_ && max_unrolled_matmul_macs_ > max_matmul_macs_)) {
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
  loop_carried_packed_memory_roots_.clear();
  contraction_off_function_ids_.clear();
  pending_fp_fast_math_modes_.clear();
  active_arithmetic_contract_ = {};

  bool has_hw = false;
  get_module()->ForEachInst([this, &has_hw](Instruction* inst) {
    has_hw |=
        completeness_mode_ == CompletenessMode::kExtensionFree
            ? (IsAnyHwOpcode(inst->opcode()) ||
               IsAnyHwCapabilityOrExtension(inst) || HasHwOperand(inst) ||
               IsReplicatedCompositeOpcode(inst->opcode()) ||
               IsReplicatedCompositeDeclaration(inst))
            : (IsHwOpcode(inst->opcode()) || IsHwCapabilityOrExtension(inst));
  });
  if (!has_hw) return Status::SuccessWithoutChange;

  if (completeness_mode_ == CompletenessMode::kExtensionFree &&
      !PreflightExtensionFreeMode()) {
    return Status::Failure;
  }
  if (completeness_mode_ == CompletenessMode::kExtensionFree &&
      !PreflightReplicatedComposites()) {
    return Status::Failure;
  }

  // Reject unsupported input and explicit resource policies before any attempt
  // is classified as a recoverable code-generation failure.
  // Constant-expression extracts must be resolved before checking the core
  // use closure. All normalization still happens in the isolated attempt.
  if (!AnalyzeLargeConstants() || !FoldLargeConstantExtracts() ||
      !CollectHwTypes() || !LegalizeModule())
    return Status::Failure;
  recoverable_failure_ = true;

  RecordContractionOffFunctions();

  if (lowering_mode_ == LoweringMode::kPreferPackedVec2) {
    RecordLoopCarriedPackedMemoryRoots();
  }

  if (enable_fusion_ && lowering_mode_ == LoweringMode::kPreferPackedVec2) {
    HwFuseTwoLayerVectorMatmulPass fusion(max_unrolled_matmul_macs_);
    fusion.SetMessageConsumer(consumer());
    if (fusion.Run(context()) == Status::Failure) return Status::Failure;
  }

  if (!CollectHwTypes()) return Status::Failure;
  if (!LegalizeModule()) return Status::Failure;
  if (!MaterializeLoweredTypes()) return Status::Failure;
  RecordOriginalHwValueTypes();
  if (!MaterializeLargeConstants()) return Status::Failure;
  if (!EliminateHwFunctionVariables()) return Status::Failure;
  if (!PrepareMatmulPatternFunctions()) return Status::Failure;

  std::vector<Instruction*> to_kill;
  if (!LowerHwInstructions(&to_kill)) return Status::Failure;
  for (const auto& materialized : materialized_constant_variables_) {
    Instruction* load = get_def_use_mgr()->GetDef(materialized.first);
    if (load && get_def_use_mgr()->NumUsers(load) == 0) context()->KillInst(load);
  }
  if (!ReplaceHwTypeUses()) return Status::Failure;
  if (completeness_mode_ == CompletenessMode::kExtensionFree &&
      !LowerRemainingReplicatedComposites()) {
    return Status::Failure;
  }
  if (!CleanupHwDeclarations(to_kill)) return Status::Failure;
  if (!FinalHwCheck()) return Status::Failure;

  context()->InvalidateAnalyses(IRContext::kAnalysisTypes |
                                IRContext::kAnalysisConstants);
  return Status::SuccessWithChange;
}

void HwLowerToStandardPass::RecordContractionOffFunctions() {
  for (const Instruction& execution_mode : get_module()->execution_modes()) {
    if (execution_mode.opcode() != spv::Op::OpExecutionMode ||
        execution_mode.NumInOperands() < 2 ||
        execution_mode.GetSingleWordInOperand(1) !=
            uint32_t(spv::ExecutionMode::ContractionOff)) {
      continue;
    }
    context()->CollectCallTreeFromRoots(
        execution_mode.GetSingleWordInOperand(0),
        &contraction_off_function_ids_);
  }
}

void HwLowerToStandardPass::RecordLoopCarriedPackedMemoryRoots() {
  // Packed memory helpers introduce their own loops and may sit behind user
  // function calls.  Conservatively strengthen accesses when an existing
  // structured loop can carry a RAW dependency through the same module
  // pointer.  Volatile preserves that dependency without abandoning packed
  // compute or expanding the whole module through scalar fallback.
  bool has_loop = false;
  std::unordered_set<uint32_t> load_roots;
  std::unordered_set<uint32_t> store_roots;

  get_module()->ForEachInst(
      [this, &has_loop, &load_roots, &store_roots](Instruction* inst) {
        if (inst->opcode() == spv::Op::OpLoopMerge) {
          has_loop = true;
          return;
        }

        bool is_load = false;
        switch (inst->opcode()) {
          case spv::Op::OpCooperativeMatrixLoadHW:
          case spv::Op::OpCooperativeVectorLoadHW:
            is_load = true;
            break;
          case spv::Op::OpCooperativeMatrixStoreHW:
          case spv::Op::OpCooperativeVectorStoreHW:
            break;
          default:
            return;
        }

        const uint32_t root =
            GetRootModulePointerId(GetMemoryPointerOperandId(inst));
        if (root == 0) return;
        (is_load ? load_roots : store_roots).insert(root);
      });

  if (!has_loop) return;
  for (uint32_t root : load_roots) {
    if (store_roots.count(root) != 0) {
      loop_carried_packed_memory_roots_.insert(root);
    }
  }
}

bool HwLowerToStandardPass::LowerHwInstructions(
    std::vector<Instruction*>* to_kill) {
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
    if (!enable_direct_) break;
    if (!inst || inst->IsNop()) continue;
    const ArithmeticContract saved_contract = active_arithmetic_contract_;
    active_arithmetic_contract_ = GetArithmeticContract(inst);
    bool handled = false;
    bool ok = true;
    switch (inst->opcode()) {
      case spv::Op::OpCooperativeMatrixMulAddHW:
        ok = TryLowerDirectMatrixMulAdd(inst, &handled);
        break;
      case spv::Op::OpCooperativeVectorMatrixMulHW:
        ok = TryLowerDirectVectorMatrixMulPackedVec2(inst, false, &handled);
        break;
      case spv::Op::OpCooperativeVectorMatrixMulAddHW:
        ok = TryLowerDirectVectorMatrixMulPackedVec2(inst, true, &handled);
        break;
      default:
        break;
    }
    if (handled && active_arithmetic_contract_.has_explicit_fp_fast_math_mode) {
      RemoveFPFastMathMode(inst->result_id());
    }
    active_arithmetic_contract_ = saved_contract;
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
    const ArithmeticContract saved_contract = active_arithmetic_contract_;
    const ArithmeticContract source_contract = GetArithmeticContract(inst);
    active_arithmetic_contract_ = source_contract;
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
    if (ok && source_contract.has_explicit_fp_fast_math_mode) {
      RemoveFPFastMathMode(inst->result_id());
    }
    if (ok && source_contract.propagate_no_contraction) {
      RemoveNoContraction(inst->result_id());
    }
    active_arithmetic_contract_ = saved_contract;
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
  if (completeness_mode_ == CompletenessMode::kExtensionFree) {
    modified |=
        context()->RemoveCapability(spv::Capability::ReplicatedCompositesEXT);
    modified |= RemoveExtensionByName("SPV_EXT_replicated_composites");
    modified |= RemoveSourceExtensionByName("GL_EXT_replicated_composites");
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
    const bool has_forbidden_replicated_composite =
        completeness_mode_ == CompletenessMode::kExtensionFree &&
        (IsReplicatedCompositeOpcode(inst->opcode()) ||
         IsReplicatedCompositeDeclaration(inst));
    if (has_forbidden_hw || has_forbidden_replicated_composite ||
        HasHwTypeReference(inst)) {
      ReportError(inst, has_forbidden_replicated_composite
                            ? "extension-free HW lowering left replicated "
                              "composite residue"
                            : "HW lowering left HW op/type/declaration");
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
