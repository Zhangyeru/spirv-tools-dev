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

#include "source/opcode.h"
#include "source/opt/hw_lower_to_standard_pass.h"
#include "source/opt/hw_lower_to_standard_pass_internal.h"
#include "source/opt/ir_builder.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
using namespace hw_lower_internal;

namespace {
// Resolve literal extraction through the compact input constant DAG. In
// particular, a scalar used as an array length must remain a module constant.
uint32_t ExtractConstant(IRContext* context, uint32_t id,
                         const std::vector<uint32_t>& indices) {
  size_t position = 0;
  while (position < indices.size()) {
    auto* value = context->get_def_use_mgr()->GetDef(id);
    if (!value) return 0;
    auto* type = context->get_def_use_mgr()->GetDef(value->type_id());
    const uint32_t index = indices[position++];
    const bool replicate =
        value->opcode() == spv::Op::OpConstantCompositeReplicateEXT ||
        value->opcode() == spv::Op::OpSpecConstantCompositeReplicateEXT;
    const bool composite = value->opcode() == spv::Op::OpConstantComposite ||
                           value->opcode() == spv::Op::OpSpecConstantComposite;
    if (type && type->opcode() == spv::Op::OpTypeCooperativeMatrixHW &&
        (composite || replicate) && value->NumInOperands() == 1) {
      if (position == indices.size()) return 0;
      ++position;  // HW matrix constants broadcast a scalar across rows/cols.
      id = value->GetSingleWordInOperand(0);
    } else if (replicate) {
      id = value->GetSingleWordInOperand(0);
    } else if (composite && index < value->NumInOperands()) {
      id = value->GetSingleWordInOperand(index);
    } else {
      return 0;
    }
  }
  return id;
}
}  // namespace

bool HwLowerToStandardPass::AnalyzeLargeConstants() {
  large_constants_.clear();
  constant_functions_.clear();
  repeat_functions_.clear();
  repeat_pointer_functions_.clear();
  materialized_constant_variables_.clear();
  for (Instruction& inst : get_module()->types_values()) {
    if (!spvOpcodeIsConstant(inst.opcode())) continue;
    Instruction* type = get_def_use_mgr()->GetDef(inst.type_id());
    if (!type) continue;
    uint32_t rows = 1, cols = 0;
    bool hw = false;
    if (type->opcode() == spv::Op::OpTypeCooperativeMatrixHW &&
        type->NumInOperands() >= 3) {
      hw = true;
      GetConstantU32(type->GetSingleWordInOperand(1), &rows);
      GetConstantU32(type->GetSingleWordInOperand(2), &cols);
    } else if (type->opcode() == spv::Op::OpTypeCooperativeVectorHW &&
               type->NumInOperands() >= 2) {
      hw = true;
      GetConstantU32(type->GetSingleWordInOperand(1), &cols);
    } else if (type->opcode() == spv::Op::OpTypeArray &&
               type->NumInOperands() >= 2) {
      GetConstantU32(type->GetSingleWordInOperand(1), &cols);
    }
    uint64_t count = uint64_t(rows) * cols;
    Instruction* component = get_def_use_mgr()->GetDef(
        type->NumInOperands() ? type->GetSingleWordInOperand(0) : 0);
    if (hw && component && component->opcode() == spv::Op::OpTypeFloat &&
        lowering_mode_ == LoweringMode::kPreferPackedVec2 && cols % 2 == 0) {
      count /= 2;
    }
    const bool composite = inst.opcode() == spv::Op::OpConstantComposite ||
                           inst.opcode() == spv::Op::OpSpecConstantComposite ||
                           IsReplicatedCompositeOpcode(inst.opcode());
    bool large = composite && count > kMaxCompositeConstituents;
    inst.ForEachInId(
        [&](const uint32_t* id) { large |= large_constants_.count(*id) != 0; });
    if (large) large_constants_.insert(inst.result_id());
  }
  bool ok = true;
  for (uint32_t id : large_constants_) {
    get_def_use_mgr()->ForEachUser(id, [&](Instruction* user) {
      if (user->opcode() == spv::Op::OpVariable) {
        const auto storage = spv::StorageClass(user->GetSingleWordInOperand(0));
        if (storage != spv::StorageClass::Function &&
            storage != spv::StorageClass::Private &&
            storage != spv::StorageClass::Output) {
          ReportError(user,
                      "large constant initialization requires Function, "
                      "Private or Output storage");
          ok = false;
        }
      }
    });
  }
  return ok;
}

uint32_t HwLowerToStandardPass::BuildRepeatedValueFunction(
    uint32_t type_id, uint32_t element_type_id, uint32_t count,
    bool into_pointer) {
  auto& cache = into_pointer ? repeat_pointer_functions_ : repeat_functions_;
  auto cached = cache.find(type_id);
  if (cached != cache.end()) return cached->second;
  const uint32_t ptr_type =
      GetOrCreatePointerType(type_id, spv::StorageClass::Function);
  const uint32_t result_type = into_pointer ? GetOrCreateVoidType() : type_id;
  std::vector<uint32_t> params;
  if (into_pointer) params.push_back(ptr_type);
  params.push_back(element_type_id);
  const uint32_t fn_type = GetOrCreateFunctionType(result_type, params);
  const uint32_t elem_ptr =
      GetOrCreatePointerType(element_type_id, spv::StorageClass::Function);
  const uint32_t uint_type = GetOrCreateUIntType();
  const uint32_t bool_type = GetOrCreateBoolType();
  const uint32_t zero = GetOrCreateUIntConstant(0);
  const uint32_t one = GetOrCreateUIntConstant(1);
  const uint32_t bound = GetOrCreateUIntConstant(count);
  const uint32_t fn_id = TakeNextId(), param = TakeNextId();
  const uint32_t output_param = into_pointer ? TakeNextId() : 0;
  const uint32_t entry_id = TakeNextId(), header_id = TakeNextId();
  const uint32_t body_id = TakeNextId(), continue_id = TakeNextId();
  const uint32_t merge_id = TakeNextId(), next_id = TakeNextId();
  if (!fn_type || !ptr_type || !elem_ptr || !uint_type || !bool_type || !zero ||
      !one || !bound || !fn_id || !param || !entry_id || !header_id ||
      !body_id || !continue_id || !merge_id || !next_id || !result_type ||
      (into_pointer && !output_param))
    return 0;
  auto function = MakeUnique<Function>(MakeUnique<Instruction>(
      context(), spv::Op::OpFunction, result_type, fn_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}},
                                     IdOperand(fn_type)}));
  if (into_pointer)
    function->AddParameter(MakeUnique<Instruction>(
        context(), spv::Op::OpFunctionParameter, ptr_type, output_param,
        std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, element_type_id, param,
      std::initializer_list<Operand>{}));
  auto entry = std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_id));
  auto header = std::unique_ptr<BasicBlock>(MakeBasicBlock(header_id));
  auto body = std::unique_ptr<BasicBlock>(MakeBasicBlock(body_id));
  auto cont = std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_id));
  auto merge = std::unique_ptr<BasicBlock>(MakeBasicBlock(merge_id));
  InstructionBuilder eb(context(), entry.get());
  uint32_t output = output_param;
  if (!into_pointer) {
    auto* var = eb.AddVariable(ptr_type, uint32_t(spv::StorageClass::Function));
    if (!var) return 0;
    output = var->result_id();
  }
  if (!eb.AddBranch(header_id)) return 0;
  InstructionBuilder hb(context(), header.get());
  auto* index = hb.AddPhi(uint_type, {zero, entry_id, next_id, continue_id});
  if (!index) return 0;
  auto* cond = hb.AddBinaryOp(bool_type, spv::Op::OpULessThan,
                              index->result_id(), bound);
  if (!cond || !hb.AddLoopMerge(merge_id, continue_id, 0) ||
      !hb.AddConditionalBranch(cond->result_id(), body_id, merge_id))
    return 0;
  InstructionBuilder bb(context(), body.get());
  auto* ptr = bb.AddAccessChain(elem_ptr, output, {index->result_id()});
  if (!ptr || !bb.AddStore(ptr->result_id(), param) ||
      !bb.AddBranch(continue_id))
    return 0;
  InstructionBuilder cb(context(), cont.get());
  cb.AddInstruction(MakeUnique<Instruction>(
      context(), spv::Op::OpIAdd, uint_type, next_id,
      std::initializer_list<Operand>{IdOperand(index->result_id()),
                                     IdOperand(one)}));
  if (!cb.AddBranch(header_id)) return 0;
  InstructionBuilder mb(context(), merge.get());
  if (into_pointer) {
    if (!mb.AddNullaryOp(0, spv::Op::OpReturn)) return 0;
  } else {
    auto* value = mb.AddLoad(type_id, output);
    if (!value || !mb.AddUnaryOp(0, spv::Op::OpReturnValue, value->result_id()))
      return 0;
  }
  function->AddBasicBlock(std::move(entry));
  function->AddBasicBlock(std::move(header));
  function->AddBasicBlock(std::move(body));
  function->AddBasicBlock(std::move(cont));
  function->AddBasicBlock(std::move(merge));
  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  AddGeneratedFunction(std::move(function), fn_id, into_pointer);
  cache[type_id] = fn_id;
  return fn_id;
}

uint32_t HwLowerToStandardPass::MaterializeConstantValue(
    InstructionBuilder* builder, uint32_t id, Function* caller) {
  if (!large_constants_.count(id)) return id;
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  if (!inst) return 0;
  uint32_t type_id = GetLoweredType(inst->type_id());
  if (!type_id) type_id = inst->type_id();
  // Keep large broadcast storage addressable. Passing a giant array by value
  // through helpers and copying it for every dynamic extract can make driver
  // compilation quadratic even though the fill itself is a small loop.
  const auto* matrix_value = GetMatrixType(inst->type_id());
  const auto* vector_value = GetVectorType(inst->type_id());
  if (caller && (matrix_value || vector_value) && inst->NumInOperands() == 1 &&
      (IsReplicatedCompositeOpcode(inst->opcode()) ||
       inst->opcode() == spv::Op::OpConstantComposite ||
       inst->opcode() == spv::Op::OpSpecConstantComposite)) {
    uint32_t element = inst->GetSingleWordInOperand(0);
    auto* type = get_def_use_mgr()->GetDef(type_id);
    uint32_t count = 0;
    if (!type || !GetConstantU32(type->GetSingleWordInOperand(1), &count))
      return 0;
    const uint32_t element_type = type->GetSingleWordInOperand(0);
    const uint32_t ptr_type =
        GetOrCreatePointerType(type_id, spv::StorageClass::Function);
    auto* var = AddFunctionVariable(caller, ptr_type);
    if (!var) return 0;
    if ((matrix_value && IsPackedVec2(*matrix_value)) ||
        (vector_value && IsPackedVec2(*vector_value))) {
      auto* pair =
          builder->AddCompositeConstruct(element_type, {element, element});
      if (!pair) return 0;
      element = pair->result_id();
    }
    uint32_t fill =
        BuildRepeatedValueFunction(type_id, element_type, count, true);
    if (!fill || !builder->AddFunctionCall(GetOrCreateVoidType(), fill,
                                           {var->result_id(), element}))
      return 0;
    auto* load = builder->AddLoad(type_id, var->result_id());
    if (!load) return 0;
    original_hw_value_types_[load->result_id()] = inst->type_id();
    materialized_constant_variables_[load->result_id()] = var->result_id();
    return load->result_id();
  }
  uint32_t function_id = 0;
  auto cached = constant_functions_.find(id);
  if (cached != constant_functions_.end()) {
    function_id = cached->second;
  } else {
    function_id = TakeNextId();
    const uint32_t fn_type = GetOrCreateFunctionType(type_id, {});
    const uint32_t entry_id = TakeNextId();
    if (!function_id || !fn_type || !entry_id) return 0;
    auto function = MakeUnique<Function>(MakeUnique<Instruction>(
        context(), spv::Op::OpFunction, type_id, function_id,
        std::initializer_list<Operand>{{SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}},
                                       IdOperand(fn_type)}));
    auto block = std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_id));
    InstructionBuilder body(context(), block.get());
    uint32_t value_id = 0;
    Instruction* type = get_def_use_mgr()->GetDef(type_id);
    const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
    const VectorTypeInfo* vector = GetVectorType(inst->type_id());
    if (IsReplicatedCompositeOpcode(inst->opcode()) || matrix) {
      uint32_t element =
          MaterializeConstantValue(&body, inst->GetSingleWordInOperand(0));
      if (!element) return 0;
      if ((matrix && IsPackedVec2(*matrix)) ||
          (vector && IsPackedVec2(*vector))) {
        const uint32_t vec2 =
            matrix ? matrix->packed_vec2_type_id : vector->packed_vec2_type_id;
        auto* pair = body.AddCompositeConstruct(vec2, {element, element});
        if (!pair) return 0;
        element = pair->result_id();
      }
      if (type && type->opcode() == spv::Op::OpTypeStruct) {
        // A replicated parent inherits largeness from its child, even when
        // it has only one member. Materialize the child once and repeat that
        // value across the struct; array indexing is not valid for structs.
        const uint32_t count = type->NumInOperands();
        if (count <= kMaxCompositeConstituents) {
          auto* value = body.AddCompositeConstruct(
              type_id, std::vector<uint32_t>(count, element));
          if (!value) return 0;
          value_id = value->result_id();
        } else {
          // OpTypeStruct has one fewer fixed word than OpCompositeConstruct.
          // Its largest legal member count therefore needs addressable storage.
          const uint32_t pointer_type =
              GetOrCreatePointerType(type_id, spv::StorageClass::Function);
          auto* variable = body.AddVariable(
              pointer_type, uint32_t(spv::StorageClass::Function));
          if (!variable) return 0;
          // Variables must precede the child materialization instructions.
          variable->RemoveFromList();
          block->begin().InsertBefore(std::unique_ptr<Instruction>(variable));
          for (uint32_t member = 0; member < count; ++member) {
            const uint32_t member_pointer_type =
                GetOrCreatePointerType(type->GetSingleWordInOperand(member),
                                       spv::StorageClass::Function);
            auto* pointer =
                body.AddAccessChain(member_pointer_type, variable->result_id(),
                                    {GetOrCreateUIntConstant(member)});
            if (!pointer || !body.AddStore(pointer->result_id(), element))
              return 0;
          }
          auto* value = body.AddLoad(type_id, variable->result_id());
          if (!value) return 0;
          value_id = value->result_id();
        }
      } else {
        uint32_t count = 0;
        if (!type || type->opcode() != spv::Op::OpTypeArray ||
            !GetConstantU32(type->GetSingleWordInOperand(1), &count)) {
          ReportError(inst,
                      "large replicated constant requires a fixed array or "
                      "homogeneous struct");
          return 0;
        }
        uint32_t element_type = type->GetSingleWordInOperand(0);
        const uint32_t repeat =
            BuildRepeatedValueFunction(type_id, element_type, count);
        if (!repeat) return 0;
        auto* call = body.AddFunctionCall(type_id, repeat, {element});
        if (!call) return 0;
        value_id = call->result_id();
      }
    } else if (inst->opcode() == spv::Op::OpSpecConstantOp) {
      const auto opcode = spv::Op(inst->GetSingleWordInOperand(0));
      std::vector<Operand> operands;
      for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
        Operand operand = inst->GetInOperand(i);
        if (operand.type == SPV_OPERAND_TYPE_ID) {
          operand.words[0] = MaterializeConstantValue(&body, operand.words[0]);
          if (!operand.words[0]) return 0;
        }
        operands.push_back(std::move(operand));
      }
      value_id = TakeNextId();
      if (!value_id || !body.AddInstruction(MakeUnique<Instruction>(
                           context(), opcode, type_id, value_id, operands)))
        return 0;
    } else {
      std::vector<uint32_t> values;
      for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
        uint32_t child =
            MaterializeConstantValue(&body, inst->GetSingleWordInOperand(i));
        if (!child) return 0;
        values.push_back(child);
      }
      if (values.size() > kMaxCompositeConstituents) return 0;
      auto* value = body.AddCompositeConstruct(type_id, values);
      if (!value) return 0;
      value_id = value->result_id();
    }
    if (!body.AddUnaryOp(0, spv::Op::OpReturnValue, value_id)) return 0;
    function->AddBasicBlock(std::move(block));
    function->SetFunctionEnd(
        MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                                std::initializer_list<Operand>{}));
    AddGeneratedFunction(std::move(function), function_id, false);
    constant_functions_[id] = function_id;
  }
  auto* call = builder->AddFunctionCall(type_id, function_id, {});
  if (!call) return 0;
  original_hw_value_types_[call->result_id()] = inst->type_id();
  return call->result_id();
}

bool HwLowerToStandardPass::FoldLargeConstantExtracts() {
  if (large_constants_.empty()) return true;
  // Fold extracts symbolically, before replacing constant-only uses (array
  // lengths, nested constants, and decorations) with runtime values.
  std::vector<Instruction*> extracts;
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpSpecConstantOp &&
        inst.GetSingleWordInOperand(0) ==
            uint32_t(spv::Op::OpCompositeExtract)) {
      extracts.push_back(&inst);
    }
  }
  for (Instruction* inst : extracts) {
    std::vector<uint32_t> indices;
    for (uint32_t i = 2; i < inst->NumInOperands(); ++i)
      indices.push_back(inst->GetSingleWordInOperand(i));
    uint32_t value =
        ExtractConstant(context(), inst->GetSingleWordInOperand(1), indices);
    if (!value) continue;
    context()->ReplaceAllUsesWith(inst->result_id(), value);
    large_constants_.erase(inst->result_id());
    context()->KillInst(inst);
  }
  // Extract folding can remove large dependencies from other spec ops and
  // parent constants. Keep those users static.
  return AnalyzeLargeConstants();
}

bool HwLowerToStandardPass::MaterializeLargeConstants() {
  if (large_constants_.empty()) return true;
  std::vector<std::pair<Instruction*, uint32_t>> initializers;
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpVariable && inst.NumInOperands() == 2 &&
        large_constants_.count(inst.GetSingleWordInOperand(1))) {
      initializers.emplace_back(&inst, inst.GetSingleWordInOperand(1));
    }
  }
  std::vector<Function*> functions;
  for (Function& function : *get_module()) functions.push_back(&function);
  for (Function* function : functions) {
    if (function->begin() == function->end()) continue;
    auto pos = function->begin()->begin();
    while (pos != function->begin()->end() &&
           pos->opcode() == spv::Op::OpVariable)
      ++pos;
    if (pos == function->begin()->end()) return false;
    InstructionBuilder entry(
        context(), &*pos,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    std::unordered_map<uint32_t, uint32_t> values;
    auto materialize = [&](uint32_t id) -> uint32_t {
      auto found = values.find(id);
      if (found != values.end()) return found->second;
      uint32_t value = MaterializeConstantValue(&entry, id, function);
      values[id] = value;
      return value;
    };
    std::vector<Instruction*> instructions;
    function->ForEachInst(
        [&](Instruction* inst) { instructions.push_back(inst); });
    for (Instruction* inst : instructions) {
      if (inst->opcode() == spv::Op::OpVariable && inst->NumInOperands() == 2 &&
          large_constants_.count(inst->GetSingleWordInOperand(1))) {
        uint32_t value = materialize(inst->GetSingleWordInOperand(1));
        if (!value || !entry.AddStore(inst->result_id(), value)) return false;
        inst->RemoveInOperand(1);
        context()->UpdateDefUse(inst);
        continue;
      }
      bool ok = true, changed = false;
      inst->ForEachInId([&](uint32_t* id) {
        if (!large_constants_.count(*id)) return;
        *id = materialize(*id);
        ok &= *id != 0;
        changed = true;
      });
      if (!ok) return false;
      if (changed) context()->UpdateDefUse(inst);
    }
    // Only initialize globals belonging to this entry point's interface or
    // reachable call tree. Shared functions must not reinitialize Private data.
    for (Instruction& ep : get_module()->entry_points()) {
      if (ep.GetSingleWordInOperand(1) != function->result_id()) continue;
      std::unordered_set<uint32_t> reachable;
      context()->CollectCallTreeFromRoots(function->result_id(), &reachable);
      for (auto initializer : initializers) {
        const uint32_t global = initializer.first->result_id();
        bool used = false, listed = false;
        for (uint32_t i = 3; i < ep.NumInOperands(); ++i)
          listed |= ep.GetSingleWordInOperand(i) == global;
        // A global may only be reached through another module variable's
        // pointer initializer. Follow those aliases as well as direct uses;
        // dropping the initializer based only on direct call-tree users would
        // leave a valid Private pointer-to-pointer read uninitialized.
        std::vector<uint32_t> pending{global};
        std::unordered_set<uint32_t> visited;
        while (!pending.empty()) {
          const uint32_t current = pending.back();
          pending.pop_back();
          if (!visited.insert(current).second) continue;
          for (uint32_t i = 3; i < ep.NumInOperands(); ++i)
            used |= ep.GetSingleWordInOperand(i) == current;
          get_def_use_mgr()->ForEachUser(current, [&](Instruction* user) {
            BasicBlock* block = context()->get_instr_block(user);
            if (block && reachable.count(block->GetParent()->result_id()))
              used = true;
            if (!block && user->opcode() == spv::Op::OpVariable)
              pending.push_back(user->result_id());
          });
        }
        if (!used && !listed) continue;
        uint32_t value = materialize(initializer.second);
        if (!value || !entry.AddStore(global, value)) return false;
        if (!listed && (get_module()->version() >= 0x10400 ||
                        initializer.first->GetSingleWordInOperand(0) ==
                            uint32_t(spv::StorageClass::Output))) {
          ep.AddOperand(IdOperand(global));
          context()->UpdateDefUse(&ep);
        }
      }
    }
  }
  for (auto initializer : initializers) {
    initializer.first->RemoveInOperand(1);
    context()->UpdateDefUse(initializer.first);
  }
  // Kill parents before children. All executable uses have been replaced;
  // removing metadata along with constants also prevents dangling debug IDs.
  std::vector<Instruction*> dead;
  for (Instruction& inst : get_module()->types_values())
    if (large_constants_.count(inst.result_id())) dead.push_back(&inst);
  for (Instruction* inst : dead) {
    bool valid_uses = true;
    get_def_use_mgr()->ForEachUser(inst, [&](Instruction* user) {
      if (!large_constants_.count(user->result_id()) && !user->IsDecoration() &&
          user->opcode() != spv::Op::OpName)
        valid_uses = false;
    });
    if (!valid_uses) {
      ReportError(inst, "large constant has an unresolved constant-only use");
      return false;
    }
  }
  for (auto it = dead.rbegin(); it != dead.rend(); ++it)
    context()->KillInst(*it);
  return true;
}

}  // namespace opt
}  // namespace spvtools
