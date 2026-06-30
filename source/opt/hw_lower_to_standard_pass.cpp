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

constexpr uint32_t kHwMatrixLoadPointerInIdx = 0;
constexpr uint32_t kHwMatrixLoadShapeInIdx = 1;
constexpr uint32_t kHwMatrixLoadOffsetInIdx = 2;
constexpr uint32_t kHwMatrixLoadLayoutInIdx = 3;
constexpr uint32_t kHwMatrixLoadMemoryOperandsInIdx = 4;

constexpr uint32_t kHwMatrixStorePointerInIdx = 0;
constexpr uint32_t kHwMatrixStoreObjectInIdx = 1;
constexpr uint32_t kHwMatrixStoreShapeInIdx = 2;
constexpr uint32_t kHwMatrixStoreOffsetInIdx = 3;
constexpr uint32_t kHwMatrixStoreLayoutInIdx = 4;
constexpr uint32_t kHwMatrixStoreMemoryOperandsInIdx = 5;

constexpr uint32_t kHwMatrixMulAddAInIdx = 0;
constexpr uint32_t kHwMatrixMulAddBInIdx = 1;
constexpr uint32_t kHwMatrixMulAddCInIdx = 2;

constexpr uint32_t kHwVectorMatrixMulInputInIdx = 0;
constexpr uint32_t kHwVectorMatrixMulMatrixInIdx = 1;
constexpr uint32_t kHwVectorMatrixMulAddBiasInIdx = 2;

constexpr uint32_t kHwVectorLoadPointerInIdx = 0;
constexpr uint32_t kHwVectorLoadOffsetInIdx = 1;
constexpr uint32_t kHwVectorLoadMemoryOperandsInIdx = 2;
constexpr uint32_t kHwVectorStorePointerInIdx = 0;
constexpr uint32_t kHwVectorStoreOffsetInIdx = 1;
constexpr uint32_t kHwVectorStoreObjectInIdx = 2;
constexpr uint32_t kHwVectorStoreMemoryOperandsInIdx = 3;

constexpr uint32_t kDefaultMaxLoweredElements = 131072;
constexpr uint32_t kDefaultMaxLoweredMatmulMacs = 65536;
constexpr uint32_t kDefaultMatrixTileM = 2;
constexpr uint32_t kDefaultMatrixTileN = 4;
constexpr uint32_t kDefaultVectorMatmulTileN = 4;
constexpr uint32_t kPackedVec4Width = 4;

struct DirectVectorLoadCandidate {
  Instruction* source_load = nullptr;
  std::vector<Instruction*> chain;
  uint32_t pointer_id = 0;
  uint32_t pointer_type_id = 0;
  std::vector<Operand> memory_operands;
};

struct DirectMatrixLoadCandidate {
  Instruction* source_load = nullptr;
  std::vector<Instruction*> chain;
  uint32_t pointer_id = 0;
  uint32_t pointer_type_id = 0;
  uint32_t shape_id = 0;
  uint32_t offset_id = 0;
  uint32_t layout = 0;
  std::vector<Operand> memory_operands;
};

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

Pass::Status HwLowerToStandardPass::Process() {
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

  bool has_hw = false;
  get_module()->ForEachInst([this, &has_hw](Instruction* inst) {
    has_hw |= IsHwOpcode(inst->opcode()) || IsHwCapabilityOrExtension(inst);
  });
  if (!has_hw) return Status::SuccessWithoutChange;

  if (!CollectHwTypes()) return Status::Failure;
  if (!EliminateHwFunctionVariables()) return Status::Failure;
  if (!LegalizeModule()) return Status::Failure;
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
      if (!GetConstantU32(inst->GetSingleWordInOperand(1), &info.rows) ||
          !GetConstantU32(inst->GetSingleWordInOperand(2), &info.cols)) {
        ReportError(inst,
                    "HW cooperative matrix rows/columns must be constants");
        return false;
      }
      const uint64_t element_count =
          static_cast<uint64_t>(info.rows) * static_cast<uint64_t>(info.cols);
      if (element_count == 0 || element_count > kDefaultMaxLoweredElements) {
        ReportError(inst, "HW cooperative matrix shape is unsupported");
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
        ReportError(inst, "unsupported HW cooperative matrix component type");
        return false;
      }
      if (info.lowered_type_id == 0) return false;
      matrix_types_[info.type_id] = info;
      lowered_types_[info.type_id] = info.lowered_type_id;
      continue;
    }

    if (inst->NumInOperands() < 2) {
      ReportError(inst, "OpTypeCooperativeVectorHW is missing operands");
      return false;
    }

    VectorTypeInfo info;
    info.type_id = inst->result_id();
    info.component_type_id = inst->GetSingleWordInOperand(0);
    if (!GetConstantU32(inst->GetSingleWordInOperand(1), &info.length)) {
      ReportError(inst, "HW cooperative vector length must be constant");
      return false;
    }
    if (info.length == 0 || info.length > kDefaultMaxLoweredElements) {
      ReportError(inst, "HW cooperative vector length is unsupported");
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
      ReportError(inst, "unsupported HW cooperative vector component type");
      return false;
    }
    if (info.lowered_type_id == 0) return false;
    vector_types_[info.type_id] = info;
    lowered_types_[info.type_id] = info.lowered_type_id;
  }

  return true;
}

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
    return user && (user->opcode() == spv::Op::OpName ||
                    user->opcode() == spv::Op::OpMemberName ||
                    user->IsDecoration() || user->IsNonSemanticInstruction() ||
                    user->IsDebugLineInst());
  };

  for (Instruction* var : hw_vars) {
    const uint32_t var_id = var->result_id();

    // Walk the block in program order, tracking the last stored value.
    // Every load from this variable is replaced with the most recent store.
    std::vector<Instruction*> var_stores;
    std::vector<Instruction*> var_loads;
    BasicBlock* block = nullptr;

    get_def_use_mgr()->ForEachUser(var, [&](Instruction* user) {
      BasicBlock* user_block = context()->get_instr_block(user);
      if (!user_block) return;
      if (!block) block = user_block;
      if (user->opcode() == spv::Op::OpStore &&
          user->GetSingleWordInOperand(0) == var_id) {
        var_stores.push_back(user);
      } else if (user->opcode() == spv::Op::OpLoad &&
                 user->GetSingleWordInOperand(0) == var_id) {
        var_loads.push_back(user);
      }
    });

    if (var_stores.empty() || var_loads.empty() || !block) continue;

    // Walk block in order: track the last stored value and record
    // replacements for each load.
    uint32_t last_stored_value = 0;
    for (Instruction& inst : *block) {
      if (inst.opcode() == spv::Op::OpStore &&
          inst.NumInOperands() >= 2 &&
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
          user->GetSingleWordInOperand(0) == var_id &&
          elim_loads.count(user)) {
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

bool HwLowerToStandardPass::LegalizeModule() {
  bool ok = true;
  get_module()->ForEachInst([this, &ok](Instruction* inst) {
    if (!ok) return;

    if (inst->opcode() == spv::Op::OpTypePointer &&
        TypeContainsHw(inst->GetSingleWordInOperand(1))) {
      const auto storage_class =
          static_cast<spv::StorageClass>(inst->GetSingleWordInOperand(0));
      if (storage_class != spv::StorageClass::Function) {
        ReportError(inst,
                    "HW cooperative values may only be stored in Function "
                    "variables before lowering");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpTypeFunction) {
      for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
        if (TypeContainsHw(inst->GetSingleWordInOperand(i))) {
          ReportError(inst,
                      "HW cooperative values across function boundaries are "
                      "not supported");
          ok = false;
          return;
        }
      }
    }

    if (inst->opcode() == spv::Op::OpFunction &&
        TypeContainsHw(inst->type_id())) {
      ReportError(inst, "HW cooperative function return is not supported");
      ok = false;
      return;
    }

    if (inst->opcode() == spv::Op::OpFunctionParameter &&
        TypeContainsHw(inst->type_id())) {
      ReportError(inst, "HW cooperative function parameter is not supported");
      ok = false;
      return;
    }

    if (inst->opcode() == spv::Op::OpReturnValue && inst->NumInOperands() > 0) {
      Instruction* object =
          get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
      if (object && TypeContainsHw(object->type_id())) {
        ReportError(inst, "HW cooperative function return is not supported");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpPhi && TypeContainsHw(inst->type_id())) {
      ReportError(inst, "HW cooperative OpPhi is not supported");
      ok = false;
      return;
    }

    if (inst->opcode() == spv::Op::OpCooperativeMatrixReduceHW) {
      ReportError(inst, "OpCooperativeMatrixReduceHW is not supported");
      ok = false;
      return;
    }

    if (inst->opcode() == spv::Op::OpCooperativeMatrixMulAddHW) {
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
                    "HW cooperative matrix multiply shapes do not match");
        ok = false;
        return;
      }
      const uint64_t mac_count = static_cast<uint64_t>(result->rows) *
                                 static_cast<uint64_t>(result->cols) *
                                 static_cast<uint64_t>(a->cols);
      if (mac_count > kDefaultMaxLoweredMatmulMacs) {
        ReportError(inst,
                    "HW cooperative matrix multiply expansion is too large");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulHW ||
        inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddHW) {
      const VectorTypeInfo* result = GetVectorType(inst->type_id());
      Instruction* input_inst =
          inst->NumInOperands() > kHwVectorMatrixMulInputInIdx
              ? get_def_use_mgr()->GetDef(
                    inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx))
              : nullptr;
      Instruction* matrix_inst =
          inst->NumInOperands() > kHwVectorMatrixMulMatrixInIdx
              ? get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(
                    kHwVectorMatrixMulMatrixInIdx))
              : nullptr;
      const VectorTypeInfo* input =
          input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
      const MatrixTypeInfo* matrix =
          matrix_inst ? GetMatrixType(matrix_inst->type_id()) : nullptr;
      const bool has_bias =
          inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddHW;
      const VectorTypeInfo* bias = nullptr;
      if (has_bias && inst->NumInOperands() > kHwVectorMatrixMulAddBiasInIdx) {
        Instruction* bias_inst = get_def_use_mgr()->GetDef(
            inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
        bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
      }
      if (!result || !input || !matrix || input->length != matrix->rows ||
          result->length != matrix->cols ||
          (has_bias && (!bias || bias->length != result->length))) {
        ReportError(inst,
                    "HW cooperative vector matrix multiply shapes do "
                    "not match");
        ok = false;
        return;
      }
      const uint64_t mac_count = static_cast<uint64_t>(result->length) *
                                 static_cast<uint64_t>(input->length);
      if (mac_count > kDefaultMaxLoweredMatmulMacs) {
        ReportError(inst,
                    "HW cooperative vector matrix multiply expansion is too "
                    "large");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpBitcast &&
        TypeContainsHw(inst->type_id())) {
      if (inst->NumInOperands() != 1) {
        ReportError(inst, "unsupported HW OpBitcast");
        ok = false;
        return;
      }
      Instruction* object =
          get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
      if (!object || GetLoweredType(inst->type_id()) == 0 ||
          GetLoweredType(object->type_id()) == 0 ||
          GetLoweredType(inst->type_id()) !=
              GetLoweredType(object->type_id())) {
        ReportError(inst, "unsupported HW OpBitcast");
        ok = false;
        return;
      }
    }

    if (!IsHwOpcode(inst->opcode()) && TypeContainsHw(inst->type_id())) {
      switch (inst->opcode()) {
        case spv::Op::OpUndef:
        case spv::Op::OpConstantNull:
        case spv::Op::OpConstantComposite:
        case spv::Op::OpConstantCompositeReplicateEXT:
        case spv::Op::OpVariable:
        case spv::Op::OpLoad:
        case spv::Op::OpStore:
        case spv::Op::OpCopyObject:
        case spv::Op::OpCompositeConstruct:
        case spv::Op::OpCompositeExtract:
        case spv::Op::OpBitcast:
        case spv::Op::OpExtInst:
          break;
        default:
          ReportError(inst, "unsupported HW cooperative value use");
          ok = false;
          return;
      }
    }
  });

  return ok;
}

bool HwLowerToStandardPass::PrepareMatmulPatternFunctions() {
  // Validation only — pattern functions are created lazily via
  // GetOrCreateMatmulPatternFunctionPackedVec4 when needed (by the direct
  // matmul path or the non-fused store path).  This avoids generating dead
  // pattern functions for matmuls that will be handled by the fused
  // matrix-matmul-store path in LowerHwInstructions.
  std::vector<Instruction*> matmul_insts;
  get_module()->ForEachInst([&matmul_insts](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpCooperativeMatrixMulAddHW) {
      matmul_insts.push_back(inst);
    }
  });

  for (Instruction* inst : matmul_insts) {
    const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
    Instruction* a_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
    Instruction* b_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
    Instruction* c_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
    const MatrixTypeInfo* a =
        a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
    const MatrixTypeInfo* b =
        b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
    const MatrixTypeInfo* c =
        c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
    if (!result || !a || !b || !c) {
      ReportError(inst, "invalid OpCooperativeMatrixMulAddHW");
      return false;
    }
  }

  return true;
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
    bool handled = false;
    if (!TryLowerFusedVectorMatmulStore(inst, &handled)) return false;
  }

  std::vector<Instruction*> matrix_stores;
  get_module()->ForEachInst([&matrix_stores](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpCooperativeMatrixStoreHW) {
      matrix_stores.push_back(inst);
    }
  });
  for (Instruction* inst : matrix_stores) {
    if (!inst || inst->IsNop()) continue;
    bool handled = false;
    if (!TryLowerFusedMatrixMatmulStore(inst, &handled)) return false;
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
      case spv::Op::OpCooperativeVectorLoadHW:
      case spv::Op::OpCooperativeVectorStoreHW:
      case spv::Op::OpCooperativeVectorMatrixMulHW:
      case spv::Op::OpCooperativeVectorMatrixMulAddHW:
        worklist.push_back(inst);
        break;
      case spv::Op::OpCompositeConstruct:
        if (IsHwType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpConstantComposite:
      case spv::Op::OpConstantCompositeReplicateEXT:
        if (IsHwType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpCompositeExtract:
        worklist.push_back(inst);
        break;
      case spv::Op::OpConstantNull:
      case spv::Op::OpUndef:
        if (IsHwType(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpBitcast:
        if (TypeContainsHw(inst->type_id())) worklist.push_back(inst);
        break;
      case spv::Op::OpExtInst:
        if (GetVectorType(inst->type_id()) != nullptr)
          worklist.push_back(inst);
        break;
      default:
        break;
    }
  });

  for (Instruction* inst : worklist) {
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
        ok = LowerCompositeConstruct(inst);
        break;
      case spv::Op::OpConstantComposite:
      case spv::Op::OpConstantCompositeReplicateEXT:
        ok = LowerConstantComposite(inst);
        break;
      case spv::Op::OpCompositeExtract:
        ok = LowerCompositeExtract(inst);
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
        break;
    }
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

  modified |=
      context()->RemoveCapability(spv::Capability::CooperativeMatrixHW);
  modified |=
      context()->RemoveCapability(spv::Capability::CooperativeVectorHW);
  modified |= RemoveExtensionByName("SPV_AZD_neural_matrix");
  modified |= RemoveExtensionByName("SPV_AZD_cooperative_vector");
  modified |= RemoveExtensionByName("SPV_HW_neural_shader");
  modified |= RemoveSourceExtensionByName("GL_AZD_neural_matrix");
  modified |= RemoveSourceExtensionByName("GL_AZD_cooperative_vector");
  modified |= RemoveSourceExtensionByName("GL_HW_neural_shader");
  (void)modified;
  return true;
}

bool HwLowerToStandardPass::FinalHwCheck() const {
  bool ok = true;
  get_module()->ForEachInst([this, &ok](Instruction* inst) {
    if (!ok) return;
    if (IsHwOpcode(inst->opcode()) || IsHwCapabilityOrExtension(inst) ||
        HasHwTypeReference(inst)) {
      ReportError(inst, "HW lowering left HW op/type/capability/extension");
      ok = false;
    }
  });
  return ok;
}

bool HwLowerToStandardPass::LowerMatrixLoad(Instruction* inst) {
  const MatrixTypeInfo* info = GetMatrixType(inst->type_id());
  if (!info || inst->NumInOperands() < 4) {
    ReportError(inst, "invalid OpCooperativeMatrixLoadHW");
    return false;
  }

  uint32_t layout = 0;
  if (!GetConstantU32(inst->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
                      &layout) ||
      layout > 1) {
    ReportError(inst, "HW matrix load layout must be RowMajor or ColumnMajor");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;
  element_ids.reserve(info->rows * info->cols);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
  const uint32_t shape_id =
      inst->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
  const uint32_t offset_id =
      inst->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kHwMatrixLoadMemoryOperandsInIdx);

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

bool HwLowerToStandardPass::LowerMatrixStore(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 5) {
    ReportError(inst, "invalid OpCooperativeMatrixStoreHW");
    return false;
  }

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixStoreObjectInIdx));
  const MatrixTypeInfo* info =
      object ? GetMatrixType(object->type_id()) : nullptr;
  if (!info) {
    ReportError(inst, "invalid HW matrix store object");
    return false;
  }

  uint32_t layout = 0;
  if (!GetConstantU32(inst->GetSingleWordInOperand(kHwMatrixStoreLayoutInIdx),
                      &layout) ||
      layout > 1) {
    ReportError(inst,
                "HW matrix store layout must be RowMajor or ColumnMajor");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kHwMatrixStorePointerInIdx);
  const uint32_t object_id =
      inst->GetSingleWordInOperand(kHwMatrixStoreObjectInIdx);
  const uint32_t shape_id =
      inst->GetSingleWordInOperand(kHwMatrixStoreShapeInIdx);
  const uint32_t offset_id =
      inst->GetSingleWordInOperand(kHwMatrixStoreOffsetInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kHwMatrixStoreMemoryOperandsInIdx);

  if (IsPackedVec4(*info)) {
    bool fused = false;
    if (!TryLowerFusedMatrixMatmulStore(inst, &fused)) return false;
    if (fused) return true;

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

bool HwLowerToStandardPass::LowerMatrixMulAdd(Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
  const MatrixTypeInfo* b = b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
  const MatrixTypeInfo* c = c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddHW");
    return false;
  }

  if (CanUsePackedVec4MatrixMulAdd(*result, *a, *b, *c)) {
    return LowerMatrixMulAddPackedVec4(inst);
  }
  return LowerMatrixMulAddScalarFallback(inst);
}

bool HwLowerToStandardPass::LowerMatrixMulAddPackedVec4(Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
  const MatrixTypeInfo* b = b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
  const MatrixTypeInfo* c = c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddHW");
    return false;
  }

  bool handled = false;
  if (!TryLowerDirectMatrixMulAddPackedVec4(inst, &handled)) return false;
  if (handled) return true;

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

bool HwLowerToStandardPass::LowerMatrixMulAddScalarFallback(
    Instruction* inst) {
  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
  const MatrixTypeInfo* b = b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
  const MatrixTypeInfo* c = c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
  if (!result || !a || !b || !c) {
    ReportError(inst, "invalid OpCooperativeMatrixMulAddHW");
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
            const uint32_t fma =
                BuildFma(&builder, float_type_id, a_elems[i], b_elems[j],
                         acc[i * tile_n + j]);
            if (fma == 0) return false;
            acc[i * tile_n + j] = fma;
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

bool HwLowerToStandardPass::LowerMatrixLength(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 1) {
    ReportError(inst, "invalid OpCooperativeMatrixLengthHW");
    return false;
  }
  const MatrixTypeInfo* info = GetMatrixType(inst->GetSingleWordInOperand(0));
  if (!info) {
    ReportError(inst, "invalid OpCooperativeMatrixLengthHW type operand");
    return false;
  }
  const uint32_t length_id =
      GetOrCreateConstant(inst->type_id(), info->rows * info->cols);
  if (length_id == 0) {
    ReportError(inst, "OpCooperativeMatrixLengthHW result type must be int32");
    return false;
  }
  context()->ReplaceAllUsesWith(inst->result_id(), length_id);
  to_kill->push_back(inst);
  return true;
}

bool HwLowerToStandardPass::LowerVectorLoad(Instruction* inst) {
  const VectorTypeInfo* info = GetVectorType(inst->type_id());
  if (!info || inst->NumInOperands() < 1) {
    ReportError(inst, "invalid OpCooperativeVectorLoadHW");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;
  element_ids.reserve(info->length);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kHwVectorLoadPointerInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kHwVectorLoadMemoryOperandsInIdx);

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

bool HwLowerToStandardPass::LowerVectorStore(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 2) {
    ReportError(inst, "invalid OpCooperativeVectorStoreHW");
    return false;
  }

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorStoreObjectInIdx));
  const VectorTypeInfo* info =
      object ? GetVectorType(object->type_id()) : nullptr;
  if (!info) {
    ReportError(inst, "invalid HW vector store object");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kHwVectorStorePointerInIdx);
  const uint32_t object_id =
      inst->GetSingleWordInOperand(kHwVectorStoreObjectInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kHwVectorStoreMemoryOperandsInIdx);

  if (IsPackedVec4(*info)) {
    bool fused = false;
    if (!TryLowerFusedVectorMatmulStore(inst, &fused)) return false;
    if (fused) return true;

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

bool HwLowerToStandardPass::TryLowerFusedVectorMatmulStore(Instruction* inst,
                                                            bool* handled) {
  if (handled) *handled = false;
  if (!handled || !inst || inst->NumInOperands() < 2) return false;

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorStoreObjectInIdx));
  std::vector<Instruction*> object_chain;
  Instruction* matmul = TraceFunctionValueSource(object, inst, &object_chain);
  if (!matmul || matmul->opcode() != spv::Op::OpCooperativeVectorMatrixMulHW)
    return true;

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

  const VectorTypeInfo* result = GetVectorType(matmul->type_id());
  const VectorTypeInfo* input = GetVectorType(input_load->type_id());
  const MatrixTypeInfo* matrix = GetMatrixType(matrix_load->type_id());
  if (!result || !input || !matrix ||
      !CanUsePackedVec4VectorMatrixMul(*result, *input, *matrix, nullptr)) {
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
    return user && (user->opcode() == spv::Op::OpName ||
                    user->opcode() == spv::Op::OpMemberName ||
                    user->IsDecoration() || user->IsNonSemanticInstruction() ||
                    user->IsDebugLineInst());
  };

  std::vector<Instruction*> kill_list;
  std::unordered_set<Instruction*> kill_set;
  auto add_kill = [&kill_list, &kill_set](Instruction* kill) {
    if (kill && kill_set.insert(kill).second) kill_list.push_back(kill);
  };
  auto add_dead_function_store_users =
      [this, inst, &is_ignorable_user, &add_kill, &kill_set](
          Instruction* value) {
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
  add_kill(matmul);
  add_kill(input_load);
  add_kill(matrix_load);
  if (!add_dead_function_store_users(matmul) ||
      !add_dead_function_store_users(input_load) ||
      !add_dead_function_store_users(matrix_load)) {
    return true;
  }

  for (Instruction* kill : kill_list) {
    if (!kill) continue;
    if (kill->result_id() != 0) {
      bool only_killed_users = true;
      get_def_use_mgr()->ForEachUser(
          kill, [&kill_set, &only_killed_users, &is_ignorable_user](
                    Instruction* user) {
            if (!is_ignorable_user(user) &&
                kill_set.find(user) == kill_set.end()) {
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
      get_def_use_mgr()->ForEachUser(
          pointer, [&kill_set, &only_killed_users, &is_ignorable_user](
                      Instruction* user) {
            if (!is_ignorable_user(user) &&
                kill_set.find(user) == kill_set.end()) {
              only_killed_users = false;
            }
          });
      if (!only_killed_users) return true;
    }
  }

  const uint32_t function_id = BuildFusedVectorMatmulStoreFunctionPackedVec4(
      *result, *input, *matrix, input_pointer_id, input_pointer_type_id,
      CopyMemoryOperands(input_load, kHwVectorLoadMemoryOperandsInIdx),
      matrix_pointer_id, matrix_pointer_type_id, matrix_shape_id,
      matrix_offset_id,
      CopyMemoryOperands(matrix_load, kHwMatrixLoadMemoryOperandsInIdx),
      output_pointer_id, output_pointer_type_id,
      CopyMemoryOperands(inst, kHwVectorStoreMemoryOperandsInIdx));
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

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixStoreObjectInIdx));
  std::vector<Instruction*> object_chain;
  Instruction* matmul = TraceFunctionValueSource(object, inst, &object_chain);
  if (!matmul ||
      matmul->opcode() != spv::Op::OpCooperativeMatrixMulAddHW) {
    return true;
  }

  Instruction* a_value = get_def_use_mgr()->GetDef(
      matmul->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_value = get_def_use_mgr()->GetDef(
      matmul->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_value = get_def_use_mgr()->GetDef(
      matmul->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  std::vector<Instruction*> a_chain;
  std::vector<Instruction*> b_chain;
  std::vector<Instruction*> c_chain;
  Instruction* a_load =
      TraceFunctionValueSource(a_value, matmul, &a_chain);
  Instruction* b_load =
      TraceFunctionValueSource(b_value, matmul, &b_chain);
  Instruction* c_load =
      TraceFunctionValueSource(c_value, matmul, &c_chain);
  if (!a_load || !b_load || !c_load ||
      a_load->opcode() != spv::Op::OpCooperativeMatrixLoadHW ||
      b_load->opcode() != spv::Op::OpCooperativeMatrixLoadHW ||
      c_load->opcode() != spv::Op::OpCooperativeMatrixLoadHW) {
    return true;
  }

  const MatrixTypeInfo* result = GetMatrixType(matmul->type_id());
  const MatrixTypeInfo* a = GetMatrixType(a_load->type_id());
  const MatrixTypeInfo* b = GetMatrixType(b_load->type_id());
  const MatrixTypeInfo* c = GetMatrixType(c_load->type_id());
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
      !GetConstantU32(
          inst->GetSingleWordInOperand(kHwMatrixStoreLayoutInIdx),
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
      !CanCapturePointer(a_pointer_id) ||
      !CanCapturePointer(b_pointer_id) ||
      !CanCapturePointer(c_pointer_id) ||
      !CanCapturePointer(output_pointer_id) ||
      !IsModuleVisibleValue(a_shape_id) ||
      !IsModuleVisibleValue(a_offset_id) ||
      !IsModuleVisibleValue(b_shape_id) ||
      !IsModuleVisibleValue(b_offset_id) ||
      !IsModuleVisibleValue(c_shape_id) ||
      !IsModuleVisibleValue(c_offset_id) ||
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
    return user && (user->opcode() == spv::Op::OpName ||
                    user->opcode() == spv::Op::OpMemberName ||
                    user->IsDecoration() || user->IsNonSemanticInstruction() ||
                    user->IsDebugLineInst());
  };

  std::vector<Instruction*> kill_list;
  std::unordered_set<Instruction*> kill_set;
  auto add_kill = [&kill_list, &kill_set](Instruction* kill) {
    if (kill && kill_set.insert(kill).second) kill_list.push_back(kill);
  };
  auto add_dead_function_store_users =
      [this, inst, &is_ignorable_user, &add_kill, &kill_set](
          Instruction* value) {
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
      get_def_use_mgr()->ForEachUser(
          kill, [&kill_set, &only_killed_users, &is_ignorable_user](
                    Instruction* user) {
            if (!is_ignorable_user(user) &&
                kill_set.find(user) == kill_set.end()) {
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
      get_def_use_mgr()->ForEachUser(
          pointer, [&kill_set, &only_killed_users, &is_ignorable_user](
                      Instruction* user) {
            if (!is_ignorable_user(user) &&
                kill_set.find(user) == kill_set.end()) {
              only_killed_users = false;
            }
          });
      if (!only_killed_users) return true;
    }
  }

  const uint32_t function_id = BuildFusedMatrixMatmulStoreFunctionPackedVec4(
      *result, *a, *b, *c, a_pointer_id, a_pointer_type_id, a_shape_id,
      a_offset_id,
      CopyMemoryOperands(a_load, kHwMatrixLoadMemoryOperandsInIdx),
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

bool HwLowerToStandardPass::TryLowerDirectMatrixMulAddPackedVec4(
    Instruction* inst, bool* handled) {
  if (handled) *handled = false;
  if (!handled || !inst ||
      inst->opcode() != spv::Op::OpCooperativeMatrixMulAddHW) {
    return false;
  }

  const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
  Instruction* a_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddAInIdx));
  Instruction* b_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddBInIdx));
  Instruction* c_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixMulAddCInIdx));
  const MatrixTypeInfo* a = a_inst ? GetMatrixType(a_inst->type_id()) : nullptr;
  const MatrixTypeInfo* b = b_inst ? GetMatrixType(b_inst->type_id()) : nullptr;
  const MatrixTypeInfo* c = c_inst ? GetMatrixType(c_inst->type_id()) : nullptr;
  if (!result || !a || !b || !c || !CanUsePackedVec4MatrixMulAdd(*result, *a, *b, *c)) {
    return true;
  }

  auto resolve_matrix_load = [this, inst](Instruction* value_inst,
                                          DirectMatrixLoadCandidate* candidate) {
    if (!candidate) return false;
    *candidate = {};
    Instruction* source =
        TraceFunctionValueSource(value_inst, inst, &candidate->chain);
    if (!source || source->opcode() != spv::Op::OpCooperativeMatrixLoadHW) {
      return false;
    }
    if (!GetConstantU32(
            source->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
            &candidate->layout) ||
        candidate->layout != static_cast<uint32_t>(
                                 spv::CooperativeMatrixLayout::RowMajorKHR)) {
      return false;
    }

    candidate->source_load = source;
    candidate->pointer_id =
        source->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
    candidate->pointer_type_id = GetPointerTypeId(candidate->pointer_id);
    candidate->shape_id =
        source->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
    candidate->offset_id =
        source->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
    candidate->memory_operands =
        CopyMemoryOperands(source, kHwMatrixLoadMemoryOperandsInIdx);
    return candidate->pointer_type_id != 0 &&
           CanCapturePointer(candidate->pointer_id) &&
           IsModuleVisibleValue(candidate->shape_id) &&
           IsModuleVisibleValue(candidate->offset_id) &&
           CanMoveLoadToUse(source, inst, /*function_memory=*/false,
                            kHwMatrixLoadMemoryOperandsInIdx);
  };

  auto users_are_closed = [this, inst](
                              const std::vector<Instruction*>& kills) {
    std::unordered_set<Instruction*> allowed(kills.begin(), kills.end());
    allowed.insert(inst);
    for (Instruction* kill : kills) {
      if (!kill) continue;
      if (kill->result_id() != 0) {
        bool only_allowed_users = true;
        get_def_use_mgr()->ForEachUser(
            kill, [this, kill, &allowed, &only_allowed_users](
                      Instruction* user) {
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
      if (kill->opcode() == spv::Op::OpStore && kill->NumInOperands() >= 1) {
        const uint32_t pointer_id = kill->GetSingleWordInOperand(0);
        if (!IsFunctionPointer(pointer_id)) continue;
        Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
        if (!pointer) return false;
        bool only_allowed_users = true;
        get_def_use_mgr()->ForEachUser(
            pointer, [this, pointer, &allowed, &only_allowed_users](
                        Instruction* user) {
              if (user->opcode() == spv::Op::OpStore &&
                  user->NumInOperands() >= 1 &&
                  user->GetSingleWordInOperand(0) == pointer->result_id()) {
                return;
              }
              if (!IsIgnorableDirectUser(user) &&
                  allowed.find(user) == allowed.end()) {
                only_allowed_users = false;
              }
            });
        if (!only_allowed_users) return false;
      }
    }
    return true;
  };

  DirectMatrixLoadCandidate direct_a;
  DirectMatrixLoadCandidate direct_b;
  DirectMatrixLoadCandidate direct_c;
  std::vector<std::pair<uint32_t, uint32_t>> value_arguments;

  // Resolve A: buffer load or constant
  bool a_is_value = false;
  uint32_t a_constant_id = 0;
  uint32_t a_value_id = a_inst ? a_inst->result_id() : 0;
  if (resolve_matrix_load(a_inst, &direct_a)) {
    a_is_value = false;
  } else {
    std::vector<Instruction*> a_chain;
    Instruction* a_source = TraceFunctionValueSource(a_inst, inst, &a_chain);
    if (a_source &&
        a_source->opcode() == spv::Op::OpConstantComposite) {
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
        // Keep a_inst as the original for type info, but use const_id for access
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
  if (resolve_matrix_load(b_inst, &direct_b)) {
    b_is_value = false;
  } else {
    std::vector<Instruction*> b_chain;
    Instruction* b_source = TraceFunctionValueSource(b_inst, inst, &b_chain);
    if (b_source &&
        b_source->opcode() == spv::Op::OpConstantComposite) {
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
  if (resolve_matrix_load(c_inst, &direct_c)) {
    c_is_value = false;
  } else {
    std::vector<Instruction*> c_chain;
    Instruction* c_source = TraceFunctionValueSource(c_inst, inst, &c_chain);
    if (c_source &&
        c_source->opcode() == spv::Op::OpConstantComposite) {
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
  auto keep_direct_chain_alive = [this, &remove_kill](
                                     const std::vector<Instruction*>& chain) {
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
  auto has_live_users_outside_kill = [this, inst, &kill_set](
                                          Instruction* source_load) {
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
  if (!a_is_value &&
      HasLiveSafeSharedDirectSourceUsers(inst, direct_a.source_load,
                                         kill_set)) {
    remove_kill(direct_a.source_load);
    KeepSharedDirectSourceAlive(inst, direct_a.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_a.chain);
  }
  if (!b_is_value &&
      HasLiveSafeSharedDirectSourceUsers(inst, direct_b.source_load,
                                         kill_set)) {
    remove_kill(direct_b.source_load);
    KeepSharedDirectSourceAlive(inst, direct_b.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_b.chain);
  }
  if (!c_is_value &&
      HasLiveSafeSharedDirectSourceUsers(inst, direct_c.source_load,
                                         kill_set)) {
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
  if (!users_are_closed(kill_list)) return true;

  const uint32_t function_id = BuildDirectMatmulFunctionPackedVec4(
      *result, *a, *b, *c,
      a_is_value ? 0 : direct_a.pointer_id,
      a_is_value ? 0 : direct_a.pointer_type_id,
      a_is_value ? 0 : direct_a.shape_id,
      a_is_value ? 0 : direct_a.offset_id,
      a_is_value ? std::vector<Operand>{} : direct_a.memory_operands,
      a_constant_id, a_is_value,
      b_is_value ? 0 : direct_b.pointer_id,
      b_is_value ? 0 : direct_b.pointer_type_id,
      b_is_value ? 0 : direct_b.shape_id,
      b_is_value ? 0 : direct_b.offset_id,
      b_is_value ? std::vector<Operand>{} : direct_b.memory_operands,
      b_constant_id, b_is_value,
      c_is_value ? 0 : direct_c.pointer_id,
      c_is_value ? 0 : direct_c.pointer_type_id,
      c_is_value ? 0 : direct_c.shape_id,
      c_is_value ? 0 : direct_c.offset_id,
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

  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix =
      matrix_inst ? GetMatrixType(matrix_inst->type_id()) : nullptr;
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  if (!result || !input || !matrix ||
      (has_bias && !bias) ||
      !CanUsePackedVec4VectorMatrixMul(*result, *input, *matrix, bias)) {
    return true;
  }

  auto resolve_vector_load = [this, inst](Instruction* value_inst,
                                          DirectVectorLoadCandidate* candidate) {
    if (!candidate) return false;
    *candidate = {};
    Instruction* source =
        TraceFunctionValueSource(value_inst, inst, &candidate->chain);
    if (!source || source->opcode() != spv::Op::OpCooperativeVectorLoadHW) {
      return false;
    }
    candidate->source_load = source;
    candidate->pointer_id =
        source->GetSingleWordInOperand(kHwVectorLoadPointerInIdx);
    candidate->pointer_type_id = GetPointerTypeId(candidate->pointer_id);
    candidate->memory_operands =
        CopyMemoryOperands(source, kHwVectorLoadMemoryOperandsInIdx);
    return candidate->pointer_type_id != 0 &&
           CanCapturePointer(candidate->pointer_id) &&
           CanMoveLoadToUse(source, inst, /*function_memory=*/false,
                            kHwVectorLoadMemoryOperandsInIdx);
  };

  auto resolve_matrix_load = [this, inst](Instruction* value_inst,
                                          DirectMatrixLoadCandidate* candidate) {
    if (!candidate) return false;
    *candidate = {};
    Instruction* source =
        TraceFunctionValueSource(value_inst, inst, &candidate->chain);
    if (!source || source->opcode() != spv::Op::OpCooperativeMatrixLoadHW) {
      return false;
    }
    if (!GetConstantU32(
            source->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
            &candidate->layout) ||
        candidate->layout != static_cast<uint32_t>(
                                 spv::CooperativeMatrixLayout::RowMajorKHR)) {
      return false;
    }
    candidate->source_load = source;
    candidate->pointer_id =
        source->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
    candidate->pointer_type_id = GetPointerTypeId(candidate->pointer_id);
    candidate->shape_id =
        source->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
    candidate->offset_id =
        source->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
    candidate->memory_operands =
        CopyMemoryOperands(source, kHwMatrixLoadMemoryOperandsInIdx);
    return candidate->pointer_type_id != 0 &&
           CanCapturePointer(candidate->pointer_id) &&
           IsModuleVisibleValue(candidate->shape_id) &&
           IsModuleVisibleValue(candidate->offset_id) &&
           CanMoveLoadToUse(source, inst, /*function_memory=*/false,
                            kHwMatrixLoadMemoryOperandsInIdx);
  };

  auto users_are_closed = [this, inst](
                              const std::vector<Instruction*>& kills) {
    std::unordered_set<Instruction*> allowed(kills.begin(), kills.end());
    allowed.insert(inst);
    for (Instruction* kill : kills) {
      if (!kill) continue;
      if (kill->result_id() != 0) {
        bool only_allowed_users = true;
        get_def_use_mgr()->ForEachUser(
            kill, [this, kill, &allowed, &only_allowed_users](
                      Instruction* user) {
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
      if (kill->opcode() == spv::Op::OpStore && kill->NumInOperands() >= 1) {
        const uint32_t pointer_id = kill->GetSingleWordInOperand(0);
        if (!IsFunctionPointer(pointer_id)) continue;
        Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
        if (!pointer) return false;
        bool only_allowed_users = true;
        get_def_use_mgr()->ForEachUser(
            pointer, [this, pointer, &allowed, &only_allowed_users](
                        Instruction* user) {
              if (user->opcode() == spv::Op::OpStore &&
                  user->NumInOperands() >= 1 &&
                  user->GetSingleWordInOperand(0) == pointer->result_id()) {
                return;
              }
              if (!IsIgnorableDirectUser(user) &&
                  allowed.find(user) == allowed.end()) {
                only_allowed_users = false;
              }
            });
        if (!only_allowed_users) return false;
      }
    }
    return true;
  };

  DirectVectorLoadCandidate direct_input;
  DirectMatrixLoadCandidate direct_matrix;
  DirectVectorLoadCandidate direct_bias;

  // Track operands passed as function parameters (hybrid direct path).
  // Each pair: (operand_id, lowered_type_id).
  std::vector<std::pair<uint32_t, uint32_t>> value_arguments;

  // Resolve input: buffer load, constant, or parameter
  bool input_is_value = false;
  uint32_t input_constant_id = 0;
  uint32_t input_value_id = input_inst->result_id();
  if (resolve_vector_load(input_inst, &direct_input)) {
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
  if (resolve_matrix_load(matrix_inst, &direct_matrix)) {
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
    if (resolve_vector_load(bias_inst, &direct_bias)) {
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
  auto keep_direct_chain_alive = [this, &remove_kill](
                                     const std::vector<Instruction*>& chain) {
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
  auto has_live_users_outside_kill = [this, inst, &kill_set](
                                          Instruction* source_load) {
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
  if (!input_is_value &&
      HasLiveSafeSharedDirectSourceUsers(inst, direct_input.source_load,
                                         kill_set)) {
    remove_kill(direct_input.source_load);
    KeepSharedDirectSourceAlive(inst, direct_input.source_load, &kill_list,
                                &kill_set);
    keep_direct_chain_alive(direct_input.chain);
  }
  if (!matrix_is_value &&
      HasLiveSafeSharedDirectSourceUsers(inst, direct_matrix.source_load,
                                         kill_set)) {
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
  if (!input_is_value && has_live_users_outside_kill(direct_input.source_load)) {
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
  if (!users_are_closed(kill_list)) return true;

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
      bias_constant_id, bias_is_value,
      value_arguments);
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

bool HwLowerToStandardPass::LowerVectorMatrixMul(Instruction* inst,
                                                  bool has_bias) {
  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix =
      matrix_inst ? GetMatrixType(matrix_inst->type_id()) : nullptr;
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  if (!result || !input || !matrix || (has_bias && !bias)) {
    ReportError(inst, "invalid HW vector matrix multiply");
    return false;
  }

  if (CanUsePackedVec4VectorMatrixMul(*result, *input, *matrix, bias)) {
    return LowerVectorMatrixMulPackedVec4(inst, has_bias);
  }
  return LowerVectorMatrixMulScalarFallback(inst, has_bias);
}

bool HwLowerToStandardPass::LowerVectorMatrixMulPackedVec4(Instruction* inst,
                                                            bool has_bias) {
  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix =
      matrix_inst ? GetMatrixType(matrix_inst->type_id()) : nullptr;
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  if (!result || !input || !matrix || (has_bias && !bias)) {
    ReportError(inst, "invalid HW vector matrix multiply");
    return false;
  }

  bool handled = false;
  if (!TryLowerDirectVectorMatrixMulPackedVec4(inst, has_bias, &handled)) {
    return false;
  }
  if (handled) return true;

  const uint32_t function_id = GetOrCreateVectorMatmulPatternFunctionPackedVec4(
      *result, *input, *matrix, bias, has_bias);
  std::vector<uint32_t> argument_ids = {input_inst->result_id(),
                                        matrix_inst->result_id()};
  if (has_bias) argument_ids.push_back(bias_inst->result_id());
  if (function_id == 0) return false;
  RebuildAsFunctionCall(inst, result->lowered_type_id, function_id,
                        argument_ids);
  return true;
}

bool HwLowerToStandardPass::LowerVectorMatrixMulScalarFallback(
    Instruction* inst, bool has_bias) {
  const VectorTypeInfo* result = GetVectorType(inst->type_id());
  Instruction* input_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulInputInIdx));
  Instruction* matrix_inst = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx));
  const VectorTypeInfo* input =
      input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
  const MatrixTypeInfo* matrix =
      matrix_inst ? GetMatrixType(matrix_inst->type_id()) : nullptr;
  Instruction* bias_inst = nullptr;
  const VectorTypeInfo* bias = nullptr;
  if (has_bias) {
    bias_inst = get_def_use_mgr()->GetDef(
        inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
    bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
  }
  if (!result || !input || !matrix || (has_bias && !bias)) {
    ReportError(inst, "invalid HW vector matrix multiply");
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
            &builder, *matrix, matrix_inst->result_id(), k, col);
        if (w_id == 0) return false;
        const uint32_t fma =
            BuildFma(&builder, float_type_id, x_id, w_id, acc[j]);
        if (fma == 0) return false;
        acc[j] = fma;
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

bool HwLowerToStandardPass::LowerCompositeConstruct(Instruction* inst) {
  const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
  const VectorTypeInfo* vector = GetVectorType(inst->type_id());
  if (!matrix && !vector) {
    ReportError(inst, "invalid HW OpCompositeConstruct result type");
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
                  "HW matrix OpCompositeConstruct operand count is invalid");
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
                "HW vector OpCompositeConstruct operand count is invalid");
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

bool HwLowerToStandardPass::LowerConstantComposite(Instruction* inst) {
  const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
  const VectorTypeInfo* vector = GetVectorType(inst->type_id());
  if (!matrix && !vector) {
    ReportError(inst, "invalid HW OpConstantComposite result type");
    return false;
  }
  const bool is_replicate =
      inst->opcode() == spv::Op::OpConstantCompositeReplicateEXT;

  const uint32_t component_type_id =
      matrix ? matrix->component_type_id : vector->component_type_id;
  const uint32_t expected_operands =
      matrix ? matrix->rows * matrix->cols : vector->length;

  std::vector<uint32_t> scalar_ids;
  auto append_scalar_constants =
      [this, component_type_id](Instruction* operand,
                                std::vector<uint32_t>* out) {
        if (!operand || !out) return false;
        if (operand->type_id() == component_type_id) {
          out->push_back(operand->result_id());
          return true;
        }
        if (operand->opcode() != spv::Op::OpConstantComposite) {
          return false;
        }
        for (uint32_t i = 0; i < operand->NumInOperands(); ++i) {
          Instruction* nested =
              get_def_use_mgr()->GetDef(operand->GetSingleWordInOperand(i));
          if (!nested || nested->type_id() != component_type_id) {
            return false;
          }
          out->push_back(nested->result_id());
        }
        return true;
      };

  for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
    Instruction* operand =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(i));
    if (!append_scalar_constants(operand, &scalar_ids)) {
      ReportError(inst, "unsupported HW OpConstantComposite operand");
      return false;
    }
  }

  if (scalar_ids.size() == 1 && expected_operands > 1) {
    scalar_ids.resize(expected_operands, scalar_ids[0]);
  }
  if (scalar_ids.size() != expected_operands) {
    ReportError(inst, "HW OpConstantComposite operand count is invalid");
    return false;
  }

  if ((matrix && !IsPackedVec4(*matrix)) || (vector && !IsPackedVec4(*vector))) {
    if (is_replicate) inst->SetOpcode(spv::Op::OpConstantComposite);
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
            matrix->packed_vec4_type_id, lane_ids, &insert_after);
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
          vector->packed_vec4_type_id, lane_ids, &insert_after);
      if (element_ids[pack] == 0) return false;
    }
  }

  std::vector<Operand> operands;
  operands.reserve(element_ids.size());
  for (uint32_t id : element_ids) operands.push_back(IdOperand(id));
  if (is_replicate) inst->SetOpcode(spv::Op::OpConstantComposite);
  inst->SetResultType(matrix ? matrix->lowered_type_id : vector->lowered_type_id);
  inst->SetInOperands(std::move(operands));
  context()->UpdateDefUse(inst);
  return true;
}

bool HwLowerToStandardPass::LowerCompositeExtract(Instruction* inst) {
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
      ReportError(inst, "unsupported HW matrix OpCompositeExtract");
      return false;
    }
    const uint32_t row = inst->GetSingleWordInOperand(1);
    const uint32_t col = inst->GetSingleWordInOperand(2);
    if (row >= matrix->rows || col >= matrix->cols) {
      ReportError(inst, "HW matrix OpCompositeExtract index is out of range");
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
    ReportError(inst, "unsupported HW vector OpCompositeExtract");
    return false;
  }
  const uint32_t index = inst->GetSingleWordInOperand(1);
  if (index >= vector->length) {
    ReportError(inst, "HW vector OpCompositeExtract index is out of range");
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

bool HwLowerToStandardPass::LowerHwBitcast(Instruction* inst) {
  inst->SetOpcode(spv::Op::OpCopyObject);
  inst->SetResultType(GetLoweredType(inst->type_id()));
  context()->UpdateDefUse(inst);
  return true;
}

bool HwLowerToStandardPass::LowerExtInstOnCooperativeVector(
    Instruction* inst) {
  const VectorTypeInfo* info = GetVectorType(inst->type_id());
  if (!info) return true;  // Not a cooperative vector result; nothing to do.

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  const uint32_t glsl_std450_id = inst->GetSingleWordInOperand(0);
  const uint32_t ext_opcode = inst->GetSingleWordInOperand(1);
  const uint32_t num_ext_operands = inst->NumInOperands() - 2;

  // Determine the result type for each element-wise sub-operation.
  const uint32_t elem_type_id = IsPackedVec4(*info)
                                     ? info->packed_vec4_type_id
                                     : info->component_type_id;
  const uint32_t count =
      IsPackedVec4(*info) ? info->packed_length : info->length;

  std::vector<uint32_t> result_ids;
  result_ids.reserve(count);

  for (uint32_t i = 0; i < count; ++i) {
    std::vector<Operand> operands;
    operands.push_back(IdOperand(glsl_std450_id));
    operands.push_back(
        Operand(SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER, {ext_opcode}));

    for (uint32_t op = 0; op < num_ext_operands; ++op) {
      const uint32_t operand_id = inst->GetSingleWordInOperand(2 + op);
      const VectorTypeInfo* op_info =
          GetVectorType(get_def_use_mgr()->GetDef(operand_id)->type_id());
      if (op_info) {
        // Operand is a cooperative vector — extract the element.
        const uint32_t elem_id =
            ExtractCompositeElement(&builder, elem_type_id, operand_id, i);
        if (elem_id == 0) return false;
        operands.push_back(IdOperand(elem_id));
      } else {
        // Operand is a scalar or regular vector — pass through unchanged.
        operands.push_back(IdOperand(operand_id));
      }
    }

    const uint32_t result_id = TakeNextId();
    if (result_id == 0) return false;
    std::unique_ptr<Instruction> ext_inst = MakeUnique<Instruction>(
        context(), spv::Op::OpExtInst, elem_type_id, result_id,
        std::move(operands));
    Instruction* added = builder.AddInstruction(std::move(ext_inst));
    if (!added) return false;
    result_ids.push_back(added->result_id());
  }

  RebuildAsCompositeConstruct(inst, info->lowered_type_id, result_ids);
  return true;
}

uint32_t HwLowerToStandardPass::GetOrCreateArrayType(
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

uint32_t HwLowerToStandardPass::GetOrCreateVectorType(
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

uint32_t HwLowerToStandardPass::GetOrCreatePackedArrayType(
    uint32_t vec4_type_id, uint32_t length, Instruction* insert_after) {
  return GetOrCreateArrayType(vec4_type_id, length, insert_after);
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

uint32_t HwLowerToStandardPass::GetOrCreateZero(uint32_t type_id) {
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

uint32_t HwLowerToStandardPass::GetOrCreateCompositeConstant(
    uint32_t type_id, const std::vector<uint32_t>& constituent_ids,
    Instruction** insert_after) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() != spv::Op::OpConstantComposite ||
        inst.type_id() != type_id ||
        inst.NumInOperands() != constituent_ids.size()) {
      continue;
    }
    bool matches = true;
    for (uint32_t i = 0; i < constituent_ids.size(); ++i) {
      if (inst.GetSingleWordInOperand(i) != constituent_ids[i]) {
        matches = false;
        break;
      }
    }
    if (matches) return inst.result_id();
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  std::vector<Operand> operands;
  operands.reserve(constituent_ids.size());
  for (uint32_t id : constituent_ids) operands.push_back(IdOperand(id));
  Instruction* added = AddTypeOrGlobalAfter(
      context(), insert_after ? *insert_after : nullptr,
      MakeUnique<Instruction>(context(), spv::Op::OpConstantComposite, type_id,
                              result_id, operands));
  if (!added) return 0;
  if (insert_after) *insert_after = added;
  return result_id;
}

uint32_t HwLowerToStandardPass::BuildPairComponentAsUInt(
    InstructionBuilder* builder, Instruction* user, uint32_t pair_id,
    uint32_t component_index) {
  Instruction* pair = get_def_use_mgr()->GetDef(pair_id);
  if (!pair || pair->type_id() == 0) {
    ReportError(user, "HW matrix shape/offset must be a two-component value");
    return 0;
  }

  Instruction* pair_type = get_def_use_mgr()->GetDef(pair->type_id());
  if (!pair_type || pair_type->opcode() != spv::Op::OpTypeVector ||
      pair_type->GetSingleWordInOperand(1) < 2) {
    ReportError(user, "HW matrix shape/offset must be a two-component vector");
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

  ReportError(user, "HW matrix shape/offset component type is unsupported");
  return 0;
}

uint32_t HwLowerToStandardPass::BuildMatrixElementIndex(
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

uint32_t HwLowerToStandardPass::BuildElementAccess(InstructionBuilder* builder,
                                                    Instruction* user,
                                                    uint32_t pointer_id,
                                                    uint32_t component_type_id,
                                                    uint32_t element_index_id) {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer || pointer->type_id() == 0) {
    ReportError(user, "HW load/store pointer is invalid");
    return 0;
  }
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer->type_id());
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    ReportError(user, "HW load/store pointer must be a pointer");
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
                "HW scalar pointer load/store only supports element zero");
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
              "HW load/store pointer must point to the component or a "
              "component array");
  return 0;
}

uint32_t HwLowerToStandardPass::BuildElementAccessFromPointerType(
    InstructionBuilder* builder, uint32_t pointer_type_id, uint32_t pointer_id,
    uint32_t component_type_id, uint32_t element_index_id) {
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer_type_id);
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    ReportError(nullptr, "HW load/store pointer must be a pointer");
    return 0;
  }

  const uint32_t pointee_type_id = pointer_type->GetSingleWordInOperand(1);
  Instruction* pointee_type = get_def_use_mgr()->GetDef(pointee_type_id);
  if (!pointee_type) return 0;

  if (pointee_type_id == component_type_id) {
    ReportError(nullptr,
                "HW scalar pointer load/store cannot be chunk-lowered");
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
              "HW load/store pointer must point to the component or a "
              "component array");
  return 0;
}

Instruction* HwLowerToStandardPass::AddFunctionVariable(
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

BasicBlock* HwLowerToStandardPass::MakeBasicBlock(uint32_t label_id) {
  if (label_id == 0) return nullptr;
  BasicBlock* block = new BasicBlock(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));
  context()->AnalyzeDefUse(block->GetLabelInst());
  context()->set_instr_block(block->GetLabelInst(), block);
  return block;
}

uint32_t HwLowerToStandardPass::BuildRowMajorMatrixMemoryIndex(
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

bool HwLowerToStandardPass::BuildPackedMatrixLoadOuterLoop(
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

bool HwLowerToStandardPass::BuildPackedMatrixStoreOuterLoop(
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

bool HwLowerToStandardPass::BuildPackedVectorLoadOuterLoop(
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

bool HwLowerToStandardPass::BuildPackedVectorStoreOuterLoop(
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
  const uint32_t one_uint_id = GetOrCreateUIntConstant(1);
  const uint32_t four_uint_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t result_packed_length_id =
      GetOrCreateUIntConstant(result.packed_length);
  const uint32_t input_packed_length_id =
      GetOrCreateUIntConstant(input.packed_length);
  const uint32_t matrix_cols_id = GetOrCreateUIntConstant(matrix.cols);
  const uint32_t zero4_id = GetOrCreateZero(result.packed_vec4_type_id);
  if (input_load_function_id == 0 || matrix_load_function_id == 0 ||
      output_store_function_id == 0 || void_type_id == 0 ||
      function_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || vec4_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || one_uint_id == 0 ||
      four_uint_id == 0 || result_packed_length_id == 0 ||
      input_packed_length_id == 0 || matrix_cols_id == 0 || zero4_id == 0) {
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
  Instruction* k_cond =
      k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                   k_base_load->result_id(),
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
  Instruction* input_vec = k_body_builder.AddFunctionCall(
      input.packed_vec4_type_id, input_load_function_id,
      {k_element_base->result_id()});
  if (!input_vec) return 0;
  Instruction* output_col_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, out_pack_load->result_id(), four_uint_id);
  if (!output_col_base) return 0;
  std::array<uint32_t, kPackedVec4Width> weight_row_ids = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
    const uint32_t lane_id = GetOrCreateUIntConstant(row_lane);
    if (lane_id == 0) return 0;
    Instruction* matrix_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_element_base->result_id(), lane_id);
    if (!matrix_row) return 0;
    Instruction* matrix_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, matrix_row->result_id(), matrix_cols_id);
    if (!matrix_row_offset) return 0;
    Instruction* matrix_local_base = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, matrix_row_offset->result_id(),
        output_col_base->result_id());
    if (!matrix_local_base) return 0;
    const uint32_t matrix_memory_base_id = BuildRowMajorMatrixMemoryIndex(
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
      const uint32_t weight_scalar = ExtractCompositeElement(
          &k_body_builder, result.component_type_id, weight_row_ids[row_lane],
          lane);
      if (weight_scalar == 0) return 0;
      weight_lane_ids.push_back(weight_scalar);
    }
    Instruction* weight = k_body_builder.AddCompositeConstruct(
        result.packed_vec4_type_id, weight_lane_ids);
    Instruction* acc =
        k_body_builder.AddLoad(result.packed_vec4_type_id,
                               acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, result.packed_vec4_type_id,
                 input_vec->result_id(), weight->result_id(),
                 acc->result_id());
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
    Instruction* acc =
        k_merge_builder.AddLoad(result.packed_vec4_type_id,
                                acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(&k_merge_builder,
                                             result.component_type_id,
                                             acc->result_id());
    if (reduced == 0) return 0;
    lane_ids.push_back(reduced);
  }
  Instruction* result_vec = k_merge_builder.AddCompositeConstruct(
      result.packed_vec4_type_id, lane_ids);
  if (!result_vec) return 0;
  Instruction* output_base = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, out_pack_load->result_id(), four_uint_id);
  if (!output_base) return 0;
  if (!k_merge_builder.AddFunctionCall(
          void_type_id, output_store_function_id,
          {output_base->result_id(), result_vec->result_id()}) ||
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
  AddGeneratedFunction(std::move(function), function_id);
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
  std::unique_ptr<Instruction> function_start = MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, void_type_id, function_id,
      std::initializer_list<Operand>{});
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
      !k_header_block || !k_body_block || !k_continue_block ||
      !k_merge_block) {
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
    Instruction* b_col_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
        four_uint_id);
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
      uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
      four_uint_id);
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
    const uint32_t bias_value =
        ExtractCompositeElement(&k_merge_builder, result.component_type_id,
                                c_vec->result_id(), lane);
    if (bias_value == 0) return 0;
    Instruction* add = k_merge_builder.AddBinaryOp(
        result.component_type_id, spv::Op::OpFAdd, bias_value, reduced);
    if (!add) return 0;
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
      &k_merge_builder, nullptr, output_shape_id, output_offset_id,
      result.cols, out_local_base->result_id());
  if (out_memory_base_id == 0) return 0;
  if (!k_merge_builder.AddFunctionCall(void_type_id, output_store_function_id,
                                       {out_memory_base_id,
                                        result_vec->result_id()}) ||
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
  AddGeneratedFunction(std::move(function), function_id);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildDirectVectorMatmulFunctionPackedVec4(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
    uint32_t input_pointer_id, uint32_t input_pointer_type_id,
    const std::vector<Operand>& input_memory_operands,
    uint32_t input_constant_id, bool input_is_value,
    uint32_t matrix_pointer_id, uint32_t matrix_pointer_type_id,
    uint32_t matrix_shape_id, uint32_t matrix_offset_id,
    const std::vector<Operand>& matrix_memory_operands,
    uint32_t matrix_constant_id, bool matrix_is_value,
    uint32_t bias_pointer_id, uint32_t bias_pointer_type_id,
    const std::vector<Operand>& bias_memory_operands,
    uint32_t bias_constant_id, bool bias_is_value,
    const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments) {
  if (!IsPackedVec4(result) || !IsPackedVec4(input) || !IsPackedVec4(matrix) ||
      input.length != matrix.rows || result.length != matrix.cols ||
      (has_bias && (!bias || !IsPackedVec4(*bias)))) {
    return 0;
  }
  // For value operands: constant_id may be 0 if passed as function parameter.
  if ((!input_is_value && (input_pointer_id == 0 || input_pointer_type_id == 0)) ||
      (!matrix_is_value && (matrix_pointer_id == 0 || matrix_pointer_type_id == 0 ||
                            matrix_shape_id == 0 || matrix_offset_id == 0)) ||
      (has_bias && !bias_is_value &&
       (bias_pointer_id == 0 || bias_pointer_type_id == 0))) {
    return 0;
  }

  const uint32_t input_load_function_id =
      input_is_value
          ? 0
          : GetOrCreatePackedLoadChunkFunction(
                input_pointer_id, input_pointer_type_id, input.component_type_id,
                input.packed_vec4_type_id, input_memory_operands);
  const uint32_t matrix_load_function_id =
      matrix_is_value
          ? 0
          : GetOrCreatePackedLoadChunkFunction(
                matrix_pointer_id, matrix_pointer_type_id, matrix.component_type_id,
                matrix.packed_vec4_type_id, matrix_memory_operands);
  const uint32_t bias_load_function_id =
      (has_bias && !bias_is_value)
          ? GetOrCreatePackedLoadChunkFunction(
                bias_pointer_id, bias_pointer_type_id,
                bias->component_type_id, bias->packed_vec4_type_id,
                bias_memory_operands)
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
  Instruction* k_cond =
      k_header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                   k_base_load->result_id(),
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
    Instruction* input_vec = k_body_builder.AddLoad(
        input.packed_vec4_type_id, input_vec_ptr->result_id());
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
  const uint32_t matrix_packed_cols_id = GetOrCreateUIntConstant(matrix.packed_cols);
  if (matrix_packed_cols_id == 0) return 0;
  const uint32_t matrix_vec4_ptr_type_id =
      matrix_is_value
          ? GetOrCreatePointerType(matrix.packed_vec4_type_id,
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
      Instruction* matrix_row_offset = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, matrix_row->result_id(), matrix_cols_id);
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
      const uint32_t weight_scalar = ExtractCompositeElement(
          &k_body_builder, result.component_type_id, weight_row_ids[row_lane],
          lane);
      if (weight_scalar == 0) return 0;
      weight_lane_ids.push_back(weight_scalar);
    }
    Instruction* weight = k_body_builder.AddCompositeConstruct(
        result.packed_vec4_type_id, weight_lane_ids);
    Instruction* acc =
        k_body_builder.AddLoad(result.packed_vec4_type_id,
                               acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, result.packed_vec4_type_id,
                 input_vec_id, weight->result_id(),
                 acc->result_id());
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
    Instruction* acc =
        k_merge_builder.AddLoad(result.packed_vec4_type_id,
                                acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(&k_merge_builder,
                                             result.component_type_id,
                                             acc->result_id());
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
      Instruction* bias_base = k_merge_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, out_pack_load->result_id(),
          four_uint_id);
      if (!bias_base) return 0;
      Instruction* bias_vec = k_merge_builder.AddFunctionCall(
          bias->packed_vec4_type_id, bias_load_function_id,
          {bias_base->result_id()});
      if (!bias_vec) return 0;
      bias_vec_id = bias_vec->result_id();
    }
    result_vec = k_merge_builder.AddBinaryOp(result.packed_vec4_type_id,
                                             spv::Op::OpFAdd,
                                             result_vec->result_id(),
                                             bias_vec_id);
    if (!result_vec) return 0;
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
  Instruction* result_value =
      out_merge_builder.AddLoad(result.lowered_type_id, result_var->result_id());
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
  AddGeneratedFunction(std::move(function), function_id);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildDirectMatmulFunctionPackedVec4(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c, uint32_t a_pointer_id,
    uint32_t a_pointer_type_id, uint32_t a_shape_id, uint32_t a_offset_id,
    const std::vector<Operand>& a_memory_operands,
    uint32_t a_constant_id, bool a_is_value,
    uint32_t b_pointer_id,
    uint32_t b_pointer_type_id, uint32_t b_shape_id, uint32_t b_offset_id,
    const std::vector<Operand>& b_memory_operands,
    uint32_t b_constant_id, bool b_is_value,
    uint32_t c_pointer_id,
    uint32_t c_pointer_type_id, uint32_t c_shape_id, uint32_t c_offset_id,
    const std::vector<Operand>& c_memory_operands,
    uint32_t c_constant_id, bool c_is_value,
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
      a_is_value
          ? 0
          : GetOrCreatePackedLoadChunkFunction(
                a_pointer_id, a_pointer_type_id, a.component_type_id,
                a.packed_vec4_type_id, a_memory_operands);
  const uint32_t b_load_function_id =
      b_is_value
          ? 0
          : GetOrCreatePackedLoadChunkFunction(
                b_pointer_id, b_pointer_type_id, b.component_type_id,
                b.packed_vec4_type_id, b_memory_operands);
  const uint32_t c_load_function_id =
      c_is_value
          ? 0
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
      !k_header_block || !k_body_block || !k_continue_block ||
      !k_merge_block) {
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
    a_function_ptr_type_id = GetOrCreatePointerType(
        a.lowered_type_id, spv::StorageClass::Function);
    if (a_function_ptr_type_id == 0) return 0;
  }
  uint32_t b_function_ptr_type_id = 0;
  if (b_is_value) {
    b_function_ptr_type_id = GetOrCreatePointerType(
        b.lowered_type_id, spv::StorageClass::Function);
    if (b_function_ptr_type_id == 0) return 0;
  }
  uint32_t c_function_ptr_type_id = 0;
  if (c_is_value) {
    c_function_ptr_type_id = GetOrCreatePointerType(
        c.lowered_type_id, spv::StorageClass::Function);
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
    Instruction* a_vec = k_body_builder.AddLoad(a.packed_vec4_type_id,
                                                 a_vec_ptr->result_id());
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
      Instruction* b_vec = k_body_builder.AddLoad(b.packed_vec4_type_id,
                                                   b_vec_ptr->result_id());
      if (!b_vec) return 0;
      b_vecs[row_lane] = b_vec->result_id();
    } else {
      // Load from buffer via helper function
      Instruction* b_row_offset = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, b_row->result_id(), b_cols_id);
      if (!b_row_offset) return 0;
      Instruction* b_col_offset = k_body_builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
          four_uint_id);
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
        BuildFma(&k_body_builder, result.packed_vec4_type_id,
                 a_vec_id, weight_vec->result_id(), acc->result_id());
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
    Instruction* c_vec = k_merge_builder.AddLoad(c.packed_vec4_type_id,
                                                  c_vec_ptr->result_id());
    if (!c_vec) return 0;
    c_vec_id = c_vec->result_id();
  } else {
    // Load from buffer via helper function
    Instruction* c_row_offset = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, row_load->result_id(), c_cols_id);
    if (!c_row_offset) return 0;
    Instruction* c_col_offset = k_merge_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, col_pack_load->result_id(),
        four_uint_id);
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
    const uint32_t bias_value =
        ExtractCompositeElement(&k_merge_builder, result.component_type_id,
                                c_vec_id, lane);
    if (bias_value == 0) return 0;
    Instruction* add = k_merge_builder.AddBinaryOp(
        result.component_type_id, spv::Op::OpFAdd, bias_value, reduced);
    if (!add) return 0;
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
  Instruction* result_value =
      row_merge_builder.AddLoad(result.lowered_type_id, result_var->result_id());
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
  AddGeneratedFunction(std::move(function), function_id);
  return function_id;
}

uint32_t HwLowerToStandardPass::BuildCapturedPointer(
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

bool HwLowerToStandardPass::CanCapturePointer(uint32_t pointer_id) const {
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

bool HwLowerToStandardPass::IsModuleVisibleValue(uint32_t id) const {
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  if (!inst) return false;
  return context()->get_instr_block(inst) == nullptr;
}

uint32_t HwLowerToStandardPass::ExtractCompositeElement(
    InstructionBuilder* builder, uint32_t component_type_id,
    uint32_t composite_id, uint32_t index) {
  Instruction* extract =
      builder->AddCompositeExtract(component_type_id, composite_id, {index});
  return extract ? extract->result_id() : 0;
}

uint32_t HwLowerToStandardPass::AddLoad(
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

bool HwLowerToStandardPass::AddStore(
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

std::vector<Operand> HwLowerToStandardPass::CopyMemoryOperands(
    const Instruction* inst, uint32_t first_in_operand) const {
  std::vector<Operand> operands;
  if (inst->NumInOperands() <= first_in_operand) return operands;
  operands.reserve(inst->NumInOperands() - first_in_operand);
  for (uint32_t i = first_in_operand; i < inst->NumInOperands(); ++i) {
    operands.push_back(inst->GetInOperand(i));
  }
  return operands;
}

uint32_t HwLowerToStandardPass::ExtractVectorScalar(
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

uint32_t HwLowerToStandardPass::ExtractMatrixScalar(
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

uint32_t HwLowerToStandardPass::BuildVectorTimesScalar(
    InstructionBuilder* builder, uint32_t vec4_type_id, uint32_t vector_id,
    uint32_t scalar_id) {
  const uint32_t scalar_vec_id =
      BuildScalarSplat(builder, vec4_type_id, scalar_id);
  if (scalar_vec_id == 0) return 0;
  Instruction* mul = builder->AddBinaryOp(vec4_type_id, spv::Op::OpFMul,
                                          vector_id, scalar_vec_id);
  return mul ? mul->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildScalarSplat(InstructionBuilder* builder,
                                                  uint32_t vec4_type_id,
                                                  uint32_t scalar_id) {
  std::vector<uint32_t> lane_ids(kPackedVec4Width, scalar_id);
  Instruction* scalar_vec =
      builder->AddCompositeConstruct(vec4_type_id, lane_ids);
  return scalar_vec ? scalar_vec->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildFma(InstructionBuilder* builder,
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

uint32_t HwLowerToStandardPass::BuildHorizontalReduce(
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

bool HwLowerToStandardPass::BuildVectorMatrixMulPatternPackedVec4(
    InstructionBuilder* builder, const VectorTypeInfo& result,
    const VectorTypeInfo& input, const MatrixTypeInfo& matrix,
    const VectorTypeInfo* /*bias*/, uint32_t input_id, uint32_t matrix_id,
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

    const uint32_t out_col = out_pack * kPackedVec4Width;
    for (uint32_t k = 0; k < input.length; ++k) {
      const uint32_t scalar = ExtractVectorScalar(builder, input, input_id, k);
      const uint32_t weight4 = BuildMatrixRowVector(
          builder, matrix, matrix_id, k, out_col, vec4_type_id);
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

bool HwLowerToStandardPass::BuildMatmulPatternPackedVec4(
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

void HwLowerToStandardPass::AddGeneratedFunction(
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

std::string HwLowerToStandardPass::MemoryOperandsKey(
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

uint32_t HwLowerToStandardPass::GetOrCreateFunctionType(
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

uint32_t HwLowerToStandardPass::GetOrCreatePackedLoadChunkFunction(
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

uint32_t HwLowerToStandardPass::GetOrCreatePackedStoreChunkFunction(
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

uint32_t HwLowerToStandardPass::GetOrCreateTileWeightFunctionPackedVec4(
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

uint32_t HwLowerToStandardPass::GetOrCreateMatmulTileWeightFunctionPackedVec4(
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
HwLowerToStandardPass::GetOrCreateVectorMatmulPatternFunctionPackedVec4(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias) {
  const std::string key =
      VectorMatmulPatternFunctionKey(result, input, matrix, bias, has_bias);
  auto cached = vector_matmul_pattern_functions_.find(key);
  if (cached != vector_matmul_pattern_functions_.end()) return cached->second;

  const bool use_loop_path = true;

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

  if (!use_loop_path) {
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

  const uint32_t result_function_ptr_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t input_function_ptr_type_id = GetOrCreatePointerType(
      input.lowered_type_id, spv::StorageClass::Function);
  const uint32_t matrix_function_ptr_type_id = GetOrCreatePointerType(
      matrix.lowered_type_id, spv::StorageClass::Function);
  const uint32_t bias_function_ptr_type_id =
      has_bias ? GetOrCreatePointerType(bias->lowered_type_id,
                                        spv::StorageClass::Function)
               : 0;
  const uint32_t vec4_function_ptr_type_id =
      GetOrCreatePointerType(vec4_type_id, spv::StorageClass::Function);
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
  const uint32_t matrix_packed_cols_id =
      GetOrCreateUIntConstant(matrix.packed_cols);
  if (result_function_ptr_type_id == 0 || input_function_ptr_type_id == 0 ||
      matrix_function_ptr_type_id == 0 ||
      (has_bias && bias_function_ptr_type_id == 0) ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || four_uint_id == 0 ||
      result_packed_length_id == 0 || input_packed_length_id == 0 ||
      matrix_packed_cols_id == 0) {
    return 0;
  }

  const uint32_t out_header_label_id = TakeNextId();
  const uint32_t out_body_label_id = TakeNextId();
  const uint32_t out_continue_label_id = TakeNextId();
  const uint32_t out_merge_label_id = TakeNextId();
  const uint32_t k_header_label_id = TakeNextId();
  const uint32_t k_body_label_id = TakeNextId();
  const uint32_t k_continue_label_id = TakeNextId();
  const uint32_t k_merge_label_id = TakeNextId();
  if (out_header_label_id == 0 || out_body_label_id == 0 ||
      out_continue_label_id == 0 || out_merge_label_id == 0 ||
      k_header_label_id == 0 || k_body_label_id == 0 ||
      k_continue_label_id == 0 || k_merge_label_id == 0) {
    return 0;
  }
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
  if (!out_header_block || !out_body_block || !out_continue_block ||
      !out_merge_block || !k_header_block || !k_body_block ||
      !k_continue_block || !k_merge_block) {
    return 0;
  }

  Instruction* result_var =
      builder.AddVariable(result_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* input_var =
      builder.AddVariable(input_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* matrix_var =
      builder.AddVariable(matrix_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* bias_var =
      has_bias ? builder.AddVariable(
                     bias_function_ptr_type_id,
                     static_cast<uint32_t>(spv::StorageClass::Function))
               : nullptr;
  Instruction* out_pack_var =
      builder.AddVariable(uint_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_pack_var =
      builder.AddVariable(uint_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] =
        builder.AddVariable(vec4_function_ptr_type_id,
                            static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!result_var || !input_var || !matrix_var || (has_bias && !bias_var) ||
      !out_pack_var || !k_pack_var) {
    return 0;
  }
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }
  if (!builder.AddStore(input_var->result_id(), input_param_id) ||
      !builder.AddStore(matrix_var->result_id(), matrix_param_id) ||
      (has_bias && !builder.AddStore(bias_var->result_id(), bias_param_id)) ||
      !builder.AddStore(out_pack_var->result_id(), zero_uint_id) ||
      !builder.AddBranch(out_header_label_id)) {
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
                                       out_continue_label_id)) {
    return 0;
  }
  if (!out_header_builder.AddConditionalBranch(
          out_cond->result_id(), out_body_label_id, out_merge_label_id)) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!out_body_builder.AddStore(acc_var->result_id(), zero4_id)) return 0;
  }
  if (!out_body_builder.AddStore(k_pack_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k_pack_load =
      k_header_builder.AddLoad(uint_type_id, k_pack_var->result_id());
  if (!k_pack_load) return 0;
  Instruction* k_cond = k_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, k_pack_load->result_id(),
      input_packed_length_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id)) {
    return 0;
  }
  if (!k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* input_vec_ptr = k_body_builder.AddAccessChain(
      vec4_function_ptr_type_id, input_var->result_id(),
      {k_pack_load->result_id()});
  if (!input_vec_ptr) return 0;
  Instruction* input_vec = k_body_builder.AddLoad(input.packed_vec4_type_id,
                                                  input_vec_ptr->result_id());
  if (!input_vec) return 0;
  Instruction* k_element_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_pack_load->result_id(), four_uint_id);
  if (!k_element_base) return 0;
  std::array<uint32_t, kPackedVec4Width> weight_row_ids = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
    const uint32_t lane_id = GetOrCreateUIntConstant(row_lane);
    if (lane_id == 0) return 0;
    Instruction* matrix_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_element_base->result_id(), lane_id);
    if (!matrix_row) return 0;
    Instruction* matrix_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, matrix_row->result_id(),
        matrix_packed_cols_id);
    if (!matrix_row_offset) return 0;
    Instruction* matrix_flat_index = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, matrix_row_offset->result_id(),
        out_pack_load->result_id());
    if (!matrix_flat_index) return 0;
    Instruction* weight_ptr = k_body_builder.AddAccessChain(
        vec4_function_ptr_type_id, matrix_var->result_id(),
        {matrix_flat_index->result_id()});
    if (!weight_ptr) return 0;
    Instruction* weight_row = k_body_builder.AddLoad(
        matrix.packed_vec4_type_id, weight_ptr->result_id());
    if (!weight_row) return 0;
    weight_row_ids[row_lane] = weight_row->result_id();
  }
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    std::vector<uint32_t> weight_lane_ids;
    weight_lane_ids.reserve(kPackedVec4Width);
    for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
      const uint32_t weight_scalar = ExtractCompositeElement(
          &k_body_builder, result.component_type_id, weight_row_ids[row_lane],
          lane);
      if (weight_scalar == 0) return 0;
      weight_lane_ids.push_back(weight_scalar);
    }
    Instruction* weight = k_body_builder.AddCompositeConstruct(
        vec4_type_id, weight_lane_ids);
    Instruction* acc =
        k_body_builder.AddLoad(vec4_type_id, acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, vec4_type_id, input_vec->result_id(),
                 weight->result_id(), acc->result_id());
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
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc =
        k_merge_builder.AddLoad(vec4_type_id, acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(&k_merge_builder,
                                             result.component_type_id,
                                             acc->result_id());
    if (reduced == 0) return 0;
    lane_ids.push_back(reduced);
  }
  Instruction* result_vec =
      k_merge_builder.AddCompositeConstruct(vec4_type_id, lane_ids);
  if (!result_vec) return 0;
  if (has_bias) {
    Instruction* bias_vec_ptr = k_merge_builder.AddAccessChain(
        vec4_function_ptr_type_id, bias_var->result_id(),
        {out_pack_load->result_id()});
    if (!bias_vec_ptr) return 0;
    Instruction* bias_vec = k_merge_builder.AddLoad(bias->packed_vec4_type_id,
                                                    bias_vec_ptr->result_id());
    if (!bias_vec) return 0;
    result_vec = k_merge_builder.AddBinaryOp(
        vec4_type_id, spv::Op::OpFAdd, result_vec->result_id(),
        bias_vec->result_id());
    if (!result_vec) return 0;
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
  function->AddBasicBlock(std::move(block));
  function->AddBasicBlock(std::move(out_header_block));
  function->AddBasicBlock(std::move(out_body_block));
  function->AddBasicBlock(std::move(k_header_block));
  function->AddBasicBlock(std::move(k_body_block));
  function->AddBasicBlock(std::move(k_continue_block));
  function->AddBasicBlock(std::move(k_merge_block));
  function->AddBasicBlock(std::move(out_continue_block));
  function->AddBasicBlock(std::move(out_merge_block));
  AddGeneratedFunction(std::move(function), function_id);

  vector_matmul_pattern_functions_[key] = function_id;
  return function_id;
}

uint32_t
HwLowerToStandardPass::GetOrCreateVectorMatmulPatternPointerFunctionPackedVec4(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
    uint32_t input_pointer_type_id, uint32_t matrix_pointer_type_id,
    uint32_t bias_pointer_type_id) {
  if (!IsPackedVec4(result) || !IsPackedVec4(input) || !IsPackedVec4(matrix) ||
      (has_bias && (!bias || !IsPackedVec4(*bias)))) {
    return 0;
  }

  if (input_pointer_type_id == 0 || matrix_pointer_type_id == 0 ||
      (has_bias && bias_pointer_type_id == 0)) {
    return 0;
  }

  std::string key =
      std::string("ptr|") +
      VectorMatmulPatternFunctionKey(result, input, matrix, bias, has_bias);
  key += "|ip:";
  key += std::to_string(input_pointer_type_id);
  key += "|mp:";
  key += std::to_string(matrix_pointer_type_id);
  key += "|bp:";
  key += std::to_string(bias_pointer_type_id);
  auto cached = vector_matmul_pattern_functions_.find(key);
  if (cached != vector_matmul_pattern_functions_.end()) return cached->second;

  std::vector<uint32_t> param_type_ids = {input_pointer_type_id,
                                          matrix_pointer_type_id};
  if (has_bias) param_type_ids.push_back(bias_pointer_type_id);
  const uint32_t function_type_id =
      GetOrCreateFunctionType(result.lowered_type_id, param_type_ids);
  if (function_type_id == 0) {
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

  const uint32_t input_param_id = TakeNextId();
  const uint32_t matrix_param_id = TakeNextId();
  uint32_t bias_param_id = 0;
  if (input_param_id == 0 || matrix_param_id == 0) return 0;
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, input_pointer_type_id,
      input_param_id, std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, matrix_pointer_type_id,
      matrix_param_id, std::initializer_list<Operand>{}));
  if (has_bias) {
    bias_param_id = TakeNextId();
    if (bias_param_id == 0) return 0;
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, bias_pointer_type_id,
        bias_param_id, std::initializer_list<Operand>{}));
  }

  const uint32_t label_id = TakeNextId();
  if (label_id == 0) return 0;
  std::unique_ptr<BasicBlock> entry_block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));
  InstructionBuilder builder(context(), entry_block.get());

  const uint32_t vec4_type_id = result.packed_vec4_type_id;
  const uint32_t vec4_function_ptr_type_id =
      GetOrCreatePointerType(vec4_type_id, spv::StorageClass::Function);
  const uint32_t result_function_ptr_type_id = GetOrCreatePointerType(
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
  const uint32_t matrix_packed_cols_id =
      GetOrCreateUIntConstant(matrix.packed_cols);
  const uint32_t zero4_id = GetOrCreateZero(vec4_type_id);
  if (vec4_function_ptr_type_id == 0 || result_function_ptr_type_id == 0 ||
      uint_type_id == 0 || uint_function_ptr_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || one_uint_id == 0 ||
      four_uint_id == 0 || result_packed_length_id == 0 ||
      input_packed_length_id == 0 || matrix_packed_cols_id == 0 ||
      zero4_id == 0) {
    return 0;
  }

  const uint32_t out_header_label_id = TakeNextId();
  const uint32_t out_body_label_id = TakeNextId();
  const uint32_t out_continue_label_id = TakeNextId();
  const uint32_t out_merge_label_id = TakeNextId();
  const uint32_t k_header_label_id = TakeNextId();
  const uint32_t k_body_label_id = TakeNextId();
  const uint32_t k_continue_label_id = TakeNextId();
  const uint32_t k_merge_label_id = TakeNextId();
  if (out_header_label_id == 0 || out_body_label_id == 0 ||
      out_continue_label_id == 0 || out_merge_label_id == 0 ||
      k_header_label_id == 0 || k_body_label_id == 0 ||
      k_continue_label_id == 0 || k_merge_label_id == 0) {
    return 0;
  }

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
  if (!out_header_block || !out_body_block || !out_continue_block ||
      !out_merge_block || !k_header_block || !k_body_block ||
      !k_continue_block || !k_merge_block) {
    return 0;
  }

  Instruction* result_var =
      builder.AddVariable(result_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* out_pack_var =
      builder.AddVariable(uint_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* k_pack_var =
      builder.AddVariable(uint_function_ptr_type_id,
                          static_cast<uint32_t>(spv::StorageClass::Function));
  std::array<Instruction*, kPackedVec4Width> acc_vars = {};
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    acc_vars[lane] =
        builder.AddVariable(vec4_function_ptr_type_id,
                            static_cast<uint32_t>(spv::StorageClass::Function));
  }
  if (!result_var || !out_pack_var || !k_pack_var) return 0;
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }
  if (!builder.AddStore(out_pack_var->result_id(), zero_uint_id) ||
      !builder.AddBranch(out_header_label_id)) {
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
                                       out_continue_label_id)) {
    return 0;
  }
  if (!out_header_builder.AddConditionalBranch(
          out_cond->result_id(), out_body_label_id, out_merge_label_id)) {
    return 0;
  }

  InstructionBuilder out_body_builder(context(), out_body_block.get());
  for (Instruction* acc_var : acc_vars) {
    if (!out_body_builder.AddStore(acc_var->result_id(), zero4_id)) return 0;
  }
  if (!out_body_builder.AddStore(k_pack_var->result_id(), zero_uint_id) ||
      !out_body_builder.AddBranch(k_header_label_id)) {
    return 0;
  }

  InstructionBuilder k_header_builder(context(), k_header_block.get());
  Instruction* k_pack_load =
      k_header_builder.AddLoad(uint_type_id, k_pack_var->result_id());
  if (!k_pack_load) return 0;
  Instruction* k_cond = k_header_builder.AddBinaryOp(
      bool_type_id, spv::Op::OpULessThan, k_pack_load->result_id(),
      input_packed_length_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id)) {
    return 0;
  }
  if (!k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* input_vec_ptr = k_body_builder.AddAccessChain(
      vec4_function_ptr_type_id, input_param_id, {k_pack_load->result_id()});
  if (!input_vec_ptr) return 0;
  Instruction* input_vec = k_body_builder.AddLoad(input.packed_vec4_type_id,
                                                  input_vec_ptr->result_id());
  if (!input_vec) return 0;
  Instruction* out_row_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, out_pack_load->result_id(), four_uint_id);
  if (!out_row_base) return 0;
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    const uint32_t lane_id = GetOrCreateUIntConstant(lane);
    if (lane_id == 0) return 0;
    Instruction* matrix_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, out_row_base->result_id(), lane_id);
    if (!matrix_row) return 0;
    Instruction* matrix_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, matrix_row->result_id(),
        matrix_packed_cols_id);
    if (!matrix_row_offset) return 0;
    Instruction* matrix_flat_index = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, matrix_row_offset->result_id(),
        k_pack_load->result_id());
    if (!matrix_flat_index) return 0;
    Instruction* weight_ptr = k_body_builder.AddAccessChain(
        vec4_function_ptr_type_id, matrix_param_id,
        {matrix_flat_index->result_id()});
    if (!weight_ptr) return 0;
    Instruction* weight = k_body_builder.AddLoad(matrix.packed_vec4_type_id,
                                                 weight_ptr->result_id());
    Instruction* acc =
        k_body_builder.AddLoad(vec4_type_id, acc_vars[lane]->result_id());
    if (!weight || !acc) return 0;
    const uint32_t fma =
        BuildFma(&k_body_builder, vec4_type_id, input_vec->result_id(),
                 weight->result_id(), acc->result_id());
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
  Instruction* bias_vec = nullptr;
  if (has_bias) {
    Instruction* bias_vec_ptr = k_merge_builder.AddAccessChain(
        vec4_function_ptr_type_id, bias_param_id, {out_pack_load->result_id()});
    if (!bias_vec_ptr) return 0;
    bias_vec = k_merge_builder.AddLoad(bias->packed_vec4_type_id,
                                       bias_vec_ptr->result_id());
    if (!bias_vec) return 0;
  }
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc =
        k_merge_builder.AddLoad(vec4_type_id, acc_vars[lane]->result_id());
    if (!acc) return 0;
    uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    if (reduced == 0) return 0;
    if (has_bias) {
      const uint32_t bias_value =
          ExtractCompositeElement(&k_merge_builder, result.component_type_id,
                                  bias_vec->result_id(), lane);
      if (bias_value == 0) return 0;
      Instruction* add = k_merge_builder.AddBinaryOp(
          result.component_type_id, spv::Op::OpFAdd, reduced, bias_value);
      if (!add) return 0;
      reduced = add->result_id();
    }
    lane_ids.push_back(reduced);
  }
  Instruction* result_vec =
      k_merge_builder.AddCompositeConstruct(vec4_type_id, lane_ids);
  if (!result_vec) return 0;
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
  AddGeneratedFunction(std::move(function), function_id);

  vector_matmul_pattern_functions_[key] = function_id;
  return function_id;
}

uint32_t HwLowerToStandardPass::GetOrCreateMatmulPatternFunctionPackedVec4(
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
  std::unique_ptr<BasicBlock> entry_block = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));

  if (!IsPackedVec4(a)) {
    InstructionBuilder builder(context(), entry_block.get());
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
    function->AddBasicBlock(std::move(entry_block));
    AddGeneratedFunction(std::move(function), function_id);
    matmul_pattern_functions_[key] = function_id;
    matmul_pattern_function_ids_.insert(function_id);
    return function_id;
  }

  const uint32_t lowered_function_ptr_type_id = GetOrCreatePointerType(
      result.lowered_type_id, spv::StorageClass::Function);
  const uint32_t a_function_ptr_type_id =
      GetOrCreatePointerType(a.lowered_type_id, spv::StorageClass::Function);
  const uint32_t b_function_ptr_type_id =
      GetOrCreatePointerType(b.lowered_type_id, spv::StorageClass::Function);
  const uint32_t c_function_ptr_type_id =
      GetOrCreatePointerType(c.lowered_type_id, spv::StorageClass::Function);
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
  const uint32_t a_packed_cols_id = GetOrCreateUIntConstant(a.packed_cols);
  const uint32_t b_packed_cols_id = GetOrCreateUIntConstant(b.packed_cols);
  const uint32_t k_pack_count_id = GetOrCreateUIntConstant(a.packed_cols);
  const uint32_t zero4_id = GetOrCreateZero(result.packed_vec4_type_id);
  if (lowered_function_ptr_type_id == 0 || a_function_ptr_type_id == 0 ||
      b_function_ptr_type_id == 0 || c_function_ptr_type_id == 0 ||
      vec4_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || one_uint_id == 0 || four_uint_id == 0 ||
      row_count_id == 0 || result_packed_cols_id == 0 ||
      a_packed_cols_id == 0 || b_packed_cols_id == 0 || k_pack_count_id == 0 ||
      zero4_id == 0) {
    return 0;
  }

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
  if (row_header_label_id == 0 || row_body_label_id == 0 ||
      row_continue_label_id == 0 || row_merge_label_id == 0 ||
      col_header_label_id == 0 || col_body_label_id == 0 ||
      col_continue_label_id == 0 || col_merge_label_id == 0 ||
      k_header_label_id == 0 || k_body_label_id == 0 ||
      k_continue_label_id == 0 || k_merge_label_id == 0) {
    return 0;
  }
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
  if (!row_header_block || !row_body_block || !row_continue_block ||
      !row_merge_block || !col_header_block || !col_body_block ||
      !col_continue_block || !col_merge_block || !k_header_block ||
      !k_body_block || !k_continue_block || !k_merge_block) {
    return 0;
  }

  InstructionBuilder entry_builder(context(), entry_block.get());
  Instruction* result_var = entry_builder.AddVariable(
      lowered_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* a_var = entry_builder.AddVariable(
      a_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* b_var = entry_builder.AddVariable(
      b_function_ptr_type_id,
      static_cast<uint32_t>(spv::StorageClass::Function));
  Instruction* c_var = entry_builder.AddVariable(
      c_function_ptr_type_id,
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
  if (!result_var || !a_var || !b_var || !c_var || !row_var || !col_pack_var ||
      !k_pack_var) {
    return 0;
  }
  for (Instruction* acc_var : acc_vars) {
    if (!acc_var) return 0;
  }
  if (!entry_builder.AddStore(a_var->result_id(), a_param_id) ||
      !entry_builder.AddStore(b_var->result_id(), b_param_id) ||
      !entry_builder.AddStore(c_var->result_id(), c_param_id) ||
      !entry_builder.AddStore(row_var->result_id(), zero_uint_id) ||
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
                                       row_continue_label_id)) {
    return 0;
  }
  if (!row_header_builder.AddConditionalBranch(
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
                                       col_continue_label_id)) {
    return 0;
  }
  if (!col_header_builder.AddConditionalBranch(
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
                                   k_pack_load->result_id(), k_pack_count_id);
  if (!k_cond) return 0;
  if (!k_header_builder.AddLoopMerge(k_merge_label_id, k_continue_label_id)) {
    return 0;
  }
  if (!k_header_builder.AddConditionalBranch(
          k_cond->result_id(), k_body_label_id, k_merge_label_id)) {
    return 0;
  }

  InstructionBuilder k_body_builder(context(), k_body_block.get());
  Instruction* a_row_offset = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, row_load->result_id(), a_packed_cols_id);
  if (!a_row_offset) return 0;
  Instruction* a_flat_index = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, a_row_offset->result_id(),
      k_pack_load->result_id());
  if (!a_flat_index) return 0;
  Instruction* a_vec_ptr = k_body_builder.AddAccessChain(
      vec4_function_ptr_type_id, a_var->result_id(),
      {a_flat_index->result_id()});
  if (!a_vec_ptr) return 0;
  Instruction* a_vec =
      k_body_builder.AddLoad(a.packed_vec4_type_id, a_vec_ptr->result_id());
  if (!a_vec) return 0;
  Instruction* k_base = k_body_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, k_pack_load->result_id(), four_uint_id);
  if (!k_base) return 0;
  std::array<uint32_t, kPackedVec4Width> b_vecs = {};
  for (uint32_t row_lane = 0; row_lane < kPackedVec4Width; ++row_lane) {
    const uint32_t row_lane_id = GetOrCreateUIntConstant(row_lane);
    if (row_lane_id == 0) return 0;
    Instruction* b_row = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, k_base->result_id(), row_lane_id);
    if (!b_row) return 0;
    Instruction* b_row_offset = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIMul, b_row->result_id(), b_packed_cols_id);
    if (!b_row_offset) return 0;
    Instruction* b_flat_index = k_body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, b_row_offset->result_id(),
        col_pack_load->result_id());
    if (!b_flat_index) return 0;
    Instruction* b_vec_ptr = k_body_builder.AddAccessChain(
        vec4_function_ptr_type_id, b_var->result_id(),
        {b_flat_index->result_id()});
    if (!b_vec_ptr) return 0;
    Instruction* b_vec =
        k_body_builder.AddLoad(b.packed_vec4_type_id, b_vec_ptr->result_id());
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

  InstructionBuilder k_merge_builder(context(), k_merge_block.get());
  Instruction* result_row_offset =
      k_merge_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                  row_load->result_id(), result_packed_cols_id);
  if (!result_row_offset) return 0;
  Instruction* result_flat_index = k_merge_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, result_row_offset->result_id(),
      col_pack_load->result_id());
  if (!result_flat_index) return 0;
  Instruction* c_vec_ptr = k_merge_builder.AddAccessChain(
      vec4_function_ptr_type_id, c_var->result_id(),
      {result_flat_index->result_id()});
  if (!c_vec_ptr) return 0;
  Instruction* c_vec =
      k_merge_builder.AddLoad(c.packed_vec4_type_id, c_vec_ptr->result_id());
  if (!c_vec) return 0;
  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec4Width);
  for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
    Instruction* acc = k_merge_builder.AddLoad(result.packed_vec4_type_id,
                                               acc_vars[lane]->result_id());
    if (!acc) return 0;
    const uint32_t reduced = BuildHorizontalReduce(
        &k_merge_builder, result.component_type_id, acc->result_id());
    const uint32_t bias = ExtractCompositeElement(
        &k_merge_builder, result.component_type_id, c_vec->result_id(), lane);
    if (reduced == 0 || bias == 0) return 0;
    Instruction* add = k_merge_builder.AddBinaryOp(
        result.component_type_id, spv::Op::OpFAdd, bias, reduced);
    if (!add) return 0;
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
  AddGeneratedFunction(std::move(function), function_id);
  matmul_pattern_functions_[key] = function_id;
  matmul_pattern_function_ids_.insert(function_id);
  return function_id;
}

std::string HwLowerToStandardPass::TileWeightFunctionKey(
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

std::string HwLowerToStandardPass::MatmulTileWeightFunctionKey(
    const MatrixTypeInfo& matrix) const {
  return std::string("matmul|") + TileWeightFunctionKey(matrix);
}

std::string HwLowerToStandardPass::VectorMatmulPatternFunctionKey(
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

std::string HwLowerToStandardPass::MatmulPatternFunctionKey(
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

bool HwLowerToStandardPass::IsPackedVec4(const MatrixTypeInfo& info) const {
  return info.packed_f16vec4 || info.packed_f32vec4;
}

bool HwLowerToStandardPass::IsPackedVec4(const VectorTypeInfo& info) const {
  return info.packed_f16vec4 || info.packed_f32vec4;
}

bool HwLowerToStandardPass::IsSamePackedVec4Kind(
    const MatrixTypeInfo& a, const MatrixTypeInfo& b) const {
  return IsPackedVec4(a) && IsPackedVec4(b) &&
         a.packed_f16vec4 == b.packed_f16vec4 &&
         a.packed_f32vec4 == b.packed_f32vec4 &&
         a.component_type_id == b.component_type_id &&
         a.packed_vec4_type_id == b.packed_vec4_type_id;
}

bool HwLowerToStandardPass::IsSamePackedVec4Kind(
    const VectorTypeInfo& a, const VectorTypeInfo& b) const {
  return IsPackedVec4(a) && IsPackedVec4(b) &&
         a.packed_f16vec4 == b.packed_f16vec4 &&
         a.packed_f32vec4 == b.packed_f32vec4 &&
         a.component_type_id == b.component_type_id &&
         a.packed_vec4_type_id == b.packed_vec4_type_id;
}

bool HwLowerToStandardPass::CanUsePackedVec4MatrixMulAdd(
    const MatrixTypeInfo& result, const MatrixTypeInfo& a,
    const MatrixTypeInfo& b, const MatrixTypeInfo& c) const {
  return IsPackedVec4(result) &&
         IsSamePackedVec4Kind(result, a) &&
         IsSamePackedVec4Kind(result, b) && IsSamePackedVec4Kind(result, c);
}

bool HwLowerToStandardPass::CanUsePackedVec4VectorMatrixMul(
    const VectorTypeInfo& result, const VectorTypeInfo& input,
    const MatrixTypeInfo& matrix, const VectorTypeInfo* bias) const {
  if (!IsPackedVec4(result) || !IsPackedVec4(input) ||
      !IsPackedVec4(matrix) ||
      result.component_type_id != input.component_type_id ||
      result.component_type_id != matrix.component_type_id) {
    return false;
  }
  if (result.length != matrix.cols || input.length != matrix.rows) {
    return false;
  }
  return !bias || IsSamePackedVec4Kind(result, *bias);
}

bool HwLowerToStandardPass::ShouldUsePackedVec4(uint32_t extent) const {
  return lowering_mode_ == LoweringMode::kPreferPackedVec4 &&
         extent % kPackedVec4Width == 0;
}

uint32_t HwLowerToStandardPass::MatrixFlatIndex(const MatrixTypeInfo& info,
                                                 uint32_t row,
                                                 uint32_t col) const {
  return row * info.cols + col;
}

uint32_t HwLowerToStandardPass::MatrixPackedIndex(const MatrixTypeInfo& info,
                                                   uint32_t row,
                                                   uint32_t col_pack) const {
  return row * info.packed_cols + col_pack;
}

uint32_t HwLowerToStandardPass::VectorPackedIndex(
    uint32_t scalar_index) const {
  return scalar_index / kPackedVec4Width;
}

uint32_t HwLowerToStandardPass::PackedLane(uint32_t scalar_index) const {
  return scalar_index % kPackedVec4Width;
}

uint32_t HwLowerToStandardPass::GetOrCreateGLSLStd450Import() {
  uint32_t import_id =
      context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  if (import_id != 0) return import_id;

  import_id = get_module()->GetExtInstImportId("GLSL.std.450");
  if (import_id != 0) return import_id;

  context()->AddExtInstImport("GLSL.std.450");
  return context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
}

const HwLowerToStandardPass::MatrixTypeInfo*
HwLowerToStandardPass::GetMatrixType(uint32_t type_id) const {
  auto it = matrix_types_.find(type_id);
  if (it != matrix_types_.end()) return &it->second;
  for (const auto& id_and_info : matrix_types_) {
    if (id_and_info.second.lowered_type_id == type_id)
      return &id_and_info.second;
  }
  return nullptr;
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
  return user && (user->opcode() == spv::Op::OpName ||
                  user->opcode() == spv::Op::OpMemberName ||
                  user->IsDecoration() || user->IsNonSemanticInstruction() ||
                  user->IsDebugLineInst());
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
      user_state = AnalyzeSharedDirectValueUses(current_inst, user, kill_set,
                                                keep_alive, visited_values,
                                                visited_pointers);
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
              current_inst, pointer, user, kill_set, keep_alive,
              visited_values, visited_pointers);
          if (user_state == SharedDirectUserState::kSafeLive &&
              keep_alive) {
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
      // Function calls to lowering-generated functions (direct-path matmul/
      // vecmatmul) only read from SSBO and are safe to move loads past.
      if (inst->NumInOperands() >= 1 &&
          generated_function_ids_.count(
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

uint32_t HwLowerToStandardPass::GetRootModulePointerId(uint32_t pointer_id) const {
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
  return load_root_id != 0 && write_root_id != 0 && load_root_id != write_root_id;
}

bool HwLowerToStandardPass::MemoryAccessOperandsAreMovable(
    const Instruction* inst, uint32_t first_in_operand) const {
  if (!inst || inst->NumInOperands() <= first_in_operand) return true;
  const Operand& access = inst->GetInOperand(first_in_operand);
  if (access.type != SPV_OPERAND_TYPE_MEMORY_ACCESS || access.words.empty()) {
    return false;
  }
  const uint32_t mask = access.words[0];
  const uint32_t allowed = uint32_t(spv::MemoryAccessMask::Aligned);
  return (mask & ~allowed) == 0;
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

uint32_t HwLowerToStandardPass::GetLoweredType(uint32_t type_id) const {
  auto it = lowered_types_.find(type_id);
  return it == lowered_types_.end() ? 0 : it->second;
}

uint32_t HwLowerToStandardPass::GetPointerTypeId(uint32_t pointer_id) const {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  return pointer ? pointer->type_id() : 0;
}

uint32_t HwLowerToStandardPass::GetPointeeType(
    uint32_t pointer_type_id) const {
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer_type_id);
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    return 0;
  }
  return pointer_type->GetSingleWordInOperand(1);
}

bool HwLowerToStandardPass::GetConstantU32(uint32_t id,
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
  if (type_id == 0) return false;
  if (IsHwType(type_id)) return true;

  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  if (!type) return false;
  switch (type->opcode()) {
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
      return TypeContainsHw(type->GetSingleWordInOperand(
          type->opcode() == spv::Op::OpTypePointer ? 1 : 0));
    case spv::Op::OpTypeFunction:
    case spv::Op::OpTypeStruct:
      for (uint32_t i = 0; i < type->NumInOperands(); ++i) {
        if (TypeContainsHw(type->GetSingleWordInOperand(i))) return true;
      }
      return false;
    default:
      return false;
  }
}

bool HwLowerToStandardPass::HasHwTypeReference(
    const Instruction* inst) const {
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

bool HwLowerToStandardPass::IsHwOpcode(spv::Op opcode) const {
  switch (opcode) {
    case spv::Op::OpTypeCooperativeMatrixHW:
    case spv::Op::OpCooperativeMatrixLoadHW:
    case spv::Op::OpCooperativeMatrixStoreHW:
    case spv::Op::OpCooperativeMatrixMulAddHW:
    case spv::Op::OpCooperativeMatrixLengthHW:
    case spv::Op::OpCooperativeMatrixReduceHW:
    case spv::Op::OpTypeCooperativeVectorHW:
    case spv::Op::OpCooperativeVectorLoadHW:
    case spv::Op::OpCooperativeVectorStoreHW:
    case spv::Op::OpCooperativeVectorMatrixMulHW:
    case spv::Op::OpCooperativeVectorMatrixMulAddHW:
      return true;
    default:
      return false;
  }
}

bool HwLowerToStandardPass::IsHwCapabilityOrExtension(
    const Instruction* inst) const {
  if (inst->opcode() == spv::Op::OpCapability) {
    const auto capability =
        static_cast<spv::Capability>(inst->GetSingleWordInOperand(0));
    return capability == spv::Capability::CooperativeMatrixHW ||
           capability == spv::Capability::CooperativeVectorHW;
  }
  if (inst->opcode() == spv::Op::OpExtension) {
    const std::string extension = inst->GetInOperand(0).AsString();
    return extension == "SPV_AZD_neural_matrix" ||
           extension == "SPV_AZD_cooperative_vector" ||
           extension == "SPV_HW_neural_shader";
  }
  if (inst->opcode() == spv::Op::OpSourceExtension) {
    const std::string extension = inst->GetInOperand(0).AsString();
    return extension == "GL_AZD_neural_matrix" ||
           extension == "GL_AZD_cooperative_vector" ||
           extension == "GL_HW_neural_shader";
  }
  return false;
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
        op != spv::Op::OpConstantComposite && op != spv::Op::OpSpecConstant &&
        op != spv::Op::OpSpecConstantTrue &&
        op != spv::Op::OpSpecConstantFalse &&
        op != spv::Op::OpConstantTrue && op != spv::Op::OpConstantFalse) {
      return 0;  // Not all operands are constants
    }
    const_operands.push_back(IdOperand(operand_id));
  }

  // Create a module-level OpConstantComposite
  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;

  std::unique_ptr<Instruction> new_const = MakeUnique<Instruction>(
      context(), spv::Op::OpConstantComposite,
      composite_construct->type_id(), result_id, std::move(const_operands));

  // Add as global value
  context()->AddGlobalValue(std::move(new_const));
  Instruction* inserted = &*(--context()->types_values_end());
  if (!inserted) return 0;

  get_def_use_mgr()->AnalyzeInstDefUse(inserted);
  return result_id;
}

void HwLowerToStandardPass::ReportError(const Instruction*,
                                         const std::string& message) const {
  if (!consumer()) return;
  consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message.c_str());
}

}  // namespace opt
}  // namespace spvtools
