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

bool HwLowerToStandardPass::LowerHwBitcast(Instruction* inst) {
  if (inst && inst->NumInOperands() == 1) {
    Instruction* object =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
    const uint32_t result_type = GetLoweredType(inst->type_id());
    uint32_t input_type = 0;
    if (object) {
      input_type = GetLoweredType(object->type_id());
      auto original = original_hw_value_types_.find(object->result_id());
      if (original != original_hw_value_types_.end()) {
        input_type = GetLoweredType(original->second);
      }
      if (input_type == 0) input_type = object->type_id();
    }
    if (result_type != 0 && result_type == input_type) {
      inst->SetOpcode(spv::Op::OpCopyObject);
      inst->SetResultType(result_type);
      context()->UpdateDefUse(inst);
      return true;
    }
  }
  // Cooperative values with the same logical shape may still choose different
  // lowered layouts (for example, packed f32 versus scalar u32).  Preserve the
  // component-wise bitcast semantics instead of assuming those aggregate
  // representations are identical.
  return LowerElementwiseWithLoop(inst, ElementwiseLoopKind::kConversion);
}

bool HwLowerToStandardPass::LowerHwConversion(Instruction* inst) {
  if (inst->NumInOperands() != 1) {
    ReportError(inst, "unsupported HW conversion");
    return false;
  }
  return LowerElementwiseWithLoop(inst, ElementwiseLoopKind::kConversion);
}

bool HwLowerToStandardPass::LowerHwArithmetic(Instruction* inst) {
  const uint32_t expected_operands =
      IsHwUnaryArithmeticOpcode(inst->opcode()) ? 1 : 2;
  if (inst->NumInOperands() != expected_operands) {
    ReportError(inst, "unsupported HW arithmetic");
    return false;
  }
  return LowerElementwiseWithLoop(inst, ElementwiseLoopKind::kArithmetic);
}

bool HwLowerToStandardPass::LowerHwScale(Instruction* inst) {
  if (inst->NumInOperands() != 2) {
    ReportError(inst, "unsupported HW scale operation");
    return false;
  }
  return LowerElementwiseWithLoop(inst, ElementwiseLoopKind::kScale);
}

bool HwLowerToStandardPass::LowerExtInstOnCooperativeVector(Instruction* inst) {
  const VectorTypeInfo* info = GetVectorType(inst->type_id());
  if (!info) return true;  // Not a cooperative vector result; nothing to do.
  return LowerElementwiseWithLoop(inst, ElementwiseLoopKind::kExtInst);
}

bool HwLowerToStandardPass::LowerElementwiseWithLoop(Instruction* inst,
                                                     ElementwiseLoopKind kind) {
  if (!inst) return false;

  const VectorTypeInfo* result_vector = GetVectorType(inst->type_id());
  const MatrixTypeInfo* result_matrix = GetMatrixType(inst->type_id());
  if (!result_vector && !result_matrix) {
    ReportError(inst, "invalid HW element-wise result type");
    return false;
  }

  const bool result_packed = result_vector ? IsPackedVec4(*result_vector)
                                           : IsPackedVec4(*result_matrix);
  const uint32_t result_component_type_id =
      result_vector ? result_vector->component_type_id
                    : result_matrix->component_type_id;
  const uint32_t result_piece_type_id =
      result_packed ? (result_vector ? result_vector->packed_vec4_type_id
                                     : result_matrix->packed_vec4_type_id)
                    : result_component_type_id;
  const uint32_t result_piece_count =
      result_vector
          ? (result_packed ? result_vector->packed_length
                           : result_vector->length)
          : (result_packed ? result_matrix->rows * result_matrix->packed_cols
                           : result_matrix->rows * result_matrix->cols);
  const uint32_t logical_element_count =
      result_vector ? result_vector->length
                    : result_matrix->rows * result_matrix->cols;
  const uint32_t lowered_result_type_id = result_vector
                                              ? result_vector->lowered_type_id
                                              : result_matrix->lowered_type_id;
  if (result_piece_count == 0 || logical_element_count == 0 ||
      lowered_result_type_id == 0 || result_piece_type_id == 0) {
    ReportError(inst, "invalid HW element-wise result layout");
    return false;
  }

  const spv::Op original_opcode = inst->opcode();
  std::vector<uint32_t> original_operand_ids;
  original_operand_ids.reserve(inst->NumInOperands());
  for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
    original_operand_ids.push_back(inst->GetSingleWordInOperand(i));
  }

  struct LoopOperand {
    uint32_t value_id = 0;
    uint32_t value_type_id = 0;
    ValueLayout layout;
    uint32_t variable_id = 0;
    uint32_t scalar_pointer_type_id = 0;
    uint32_t piece_pointer_type_id = 0;
    uint32_t loop_piece_type_id = 0;
  };

  std::vector<LoopOperand> loop_operands;
  std::vector<int32_t> operand_to_loop_operand(inst->NumInOperands(), -1);
  auto add_loop_operand = [&](uint32_t operand_index,
                              const ValueLayout& layout) -> bool {
    if (operand_index >= original_operand_ids.size()) return false;
    Instruction* value =
        get_def_use_mgr()->GetDef(original_operand_ids[operand_index]);
    if (!value) return false;

    uint32_t value_type_id = value->type_id();
    auto original_type = original_hw_value_types_.find(value->result_id());
    if (original_type != original_hw_value_types_.end()) {
      const uint32_t lowered_type_id = GetLoweredType(original_type->second);
      if (lowered_type_id != 0) value_type_id = lowered_type_id;
    } else if (IsHwType(value_type_id)) {
      const uint32_t lowered_type_id = GetLoweredType(value_type_id);
      if (lowered_type_id != 0) value_type_id = lowered_type_id;
    }
    if (value_type_id == 0 || layout.component_type_id == 0 ||
        layout.piece_type_id == 0 || layout.piece_count == 0) {
      return false;
    }

    LoopOperand operand;
    operand.value_id = value->result_id();
    operand.value_type_id = value_type_id;
    operand.layout = layout;
    operand.loop_piece_type_id = layout.piece_type_id;
    loop_operands.push_back(operand);
    operand_to_loop_operand[operand_index] =
        static_cast<int32_t>(loop_operands.size() - 1);
    return true;
  };

  if (kind == ElementwiseLoopKind::kExtInst) {
    if (!result_vector || inst->NumInOperands() < 2) {
      ReportError(inst, "invalid HW cooperative vector OpExtInst");
      return false;
    }
    Instruction* import = get_def_use_mgr()->GetDef(original_operand_ids[0]);
    if (!import || import->opcode() != spv::Op::OpExtInstImport ||
        import->GetInOperand(0).AsString() != "GLSL.std.450") {
      ReportError(inst,
                  "HW cooperative vector OpExtInst must use GLSL.std.450");
      return false;
    }

    const HwGlslStd450Info ext_info =
        DescribeSupportedHwGlslStd450(original_operand_ids[1]);
    const uint32_t expected_ext_operand_count = ext_info.operand_count;
    if (expected_ext_operand_count == 0) {
      ReportError(inst,
                  "unsupported HW cooperative vector GLSL.std.450 opcode");
      return false;
    }
    if (inst->NumInOperands() != 2 + expected_ext_operand_count) {
      ReportError(inst,
                  "invalid HW cooperative vector GLSL.std.450 operand count");
      return false;
    }

    for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
      Instruction* value = get_def_use_mgr()->GetDef(original_operand_ids[i]);
      if (!value) {
        ReportError(inst, "invalid HW cooperative vector OpExtInst operand");
        return false;
      }
      uint32_t original_type_id = value->type_id();
      auto original_type = original_hw_value_types_.find(value->result_id());
      if (original_type != original_hw_value_types_.end()) {
        original_type_id = original_type->second;
      }
      const VectorTypeInfo* operand_info = GetVectorType(original_type_id);
      if (!operand_info) {
        ReportError(inst,
                    "HW cooperative vector OpExtInst operand must be a "
                    "cooperative vector");
        return false;
      }

      ValueLayout layout;
      if (operand_info->length != result_vector->length ||
          !DescribeVectorValue(value->result_id(), result_vector->length,
                               &layout) ||
          !add_loop_operand(i, layout)) {
        ReportError(inst, "invalid HW cooperative vector OpExtInst operand");
        return false;
      }
    }
  } else if (kind == ElementwiseLoopKind::kSelect) {
    if (inst->NumInOperands() != 3) {
      ReportError(inst, "invalid HW OpSelect");
      return false;
    }
    for (uint32_t i = 1; i <= 2; ++i) {
      ValueLayout layout;
      const bool described =
          result_vector ? DescribeVectorValue(original_operand_ids[i],
                                              result_vector->length, &layout)
                        : DescribeMatrixValue(original_operand_ids[i],
                                              result_matrix->rows,
                                              result_matrix->cols, &layout);
      if (!described || !add_loop_operand(i, layout)) {
        ReportError(inst, "invalid HW OpSelect object");
        return false;
      }
    }
  } else if (kind == ElementwiseLoopKind::kBroadcast) {
    Instruction* scalar =
        inst->NumInOperands() == 1
            ? get_def_use_mgr()->GetDef(original_operand_ids[0])
            : nullptr;
    if (!scalar || scalar->type_id() != result_component_type_id) {
      ReportError(inst, "invalid HW broadcast constituent");
      return false;
    }
  } else {
    const uint32_t loop_operand_count =
        kind == ElementwiseLoopKind::kArithmetic ? inst->NumInOperands() : 1;
    for (uint32_t i = 0; i < loop_operand_count; ++i) {
      ValueLayout layout;
      const bool described =
          result_vector ? DescribeVectorValue(original_operand_ids[i],
                                              result_vector->length, &layout)
                        : DescribeMatrixValue(original_operand_ids[i],
                                              result_matrix->rows,
                                              result_matrix->cols, &layout);
      if (!described || !add_loop_operand(i, layout)) {
        ReportError(inst, "invalid HW element-wise operand");
        return false;
      }
    }
  }

  if (kind == ElementwiseLoopKind::kScale ||
      kind == ElementwiseLoopKind::kSelect) {
    for (const LoopOperand& operand : loop_operands) {
      if (operand.layout.component_type_id != result_component_type_id) {
        ReportError(inst,
                    "HW element-wise arithmetic component types do not match");
        return false;
      }
    }
  }
  if (kind == ElementwiseLoopKind::kScale) {
    Instruction* scalar = get_def_use_mgr()->GetDef(original_operand_ids[1]);
    if (!scalar || scalar->type_id() != result_component_type_id) {
      ReportError(inst, "HW scale scalar type does not match result component");
      return false;
    }
  }

  Instruction* type_insertion_point = &*(--context()->types_values_end());
  for (LoopOperand& operand : loop_operands) {
    if (result_packed && !operand.layout.packed_vec4) {
      operand.loop_piece_type_id =
          GetOrCreateVectorType(operand.layout.component_type_id,
                                kPackedVec4Width, &type_insertion_point);
      if (operand.loop_piece_type_id == 0) return false;
    }
    operand.scalar_pointer_type_id = GetOrCreatePointerType(
        operand.layout.component_type_id, spv::StorageClass::Function);
    operand.piece_pointer_type_id = GetOrCreatePointerType(
        operand.layout.piece_type_id, spv::StorageClass::Function);
    if (operand.scalar_pointer_type_id == 0 ||
        operand.piece_pointer_type_id == 0) {
      return false;
    }
  }

  const uint32_t result_pointer_type_id = GetOrCreatePointerType(
      lowered_result_type_id, spv::StorageClass::Function);
  const uint32_t result_piece_pointer_type_id =
      GetOrCreatePointerType(result_piece_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_pointer_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  uint32_t packed_select_condition_type_id = 0;
  if (kind == ElementwiseLoopKind::kSelect && result_packed &&
      bool_type_id != 0) {
    Instruction* bool_type = get_def_use_mgr()->GetDef(bool_type_id);
    if (bool_type) {
      packed_select_condition_type_id =
          GetOrCreateVectorType(bool_type_id, kPackedVec4Width, &bool_type);
    }
  }
  const uint32_t zero_id = GetOrCreateUIntConstant(0);
  const uint32_t one_id = GetOrCreateUIntConstant(1);
  const uint32_t four_id = GetOrCreateUIntConstant(kPackedVec4Width);
  const uint32_t piece_count_id = GetOrCreateUIntConstant(result_piece_count);
  if (result_pointer_type_id == 0 || result_piece_pointer_type_id == 0 ||
      uint_type_id == 0 || uint_pointer_type_id == 0 || bool_type_id == 0 ||
      (kind == ElementwiseLoopKind::kSelect && result_packed &&
       packed_select_condition_type_id == 0) ||
      zero_id == 0 || one_id == 0 || four_id == 0 || piece_count_id == 0) {
    return false;
  }

  BasicBlock* preheader_block = context()->get_instr_block(inst);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) {
    ReportError(inst, "HW element-wise operation must be inside a function");
    return false;
  }

  Instruction* result_variable =
      AddFunctionVariable(function, result_pointer_type_id);
  Instruction* index_variable =
      AddFunctionVariable(function, uint_pointer_type_id);
  if (!result_variable || !index_variable) return false;
  for (LoopOperand& operand : loop_operands) {
    const uint32_t pointer_type_id = GetOrCreatePointerType(
        operand.value_type_id, spv::StorageClass::Function);
    Instruction* variable = AddFunctionVariable(function, pointer_type_id);
    if (!variable) return false;
    operand.variable_id = variable->result_id();
  }

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() && &*split_iter != inst) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  std::unique_ptr<Instruction> enclosing_loop_merge;
  if (Instruction* loop_merge = preheader_block->GetLoopMergeInst()) {
    enclosing_loop_merge.reset(loop_merge->Clone(context()));
  }

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

  if (enclosing_loop_merge) {
    Instruction* moved_loop_merge = merge_block->GetLoopMergeInst();
    if (!moved_loop_merge) return false;
    context()->KillInst(moved_loop_merge);
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  uint32_t select_condition_id =
      kind == ElementwiseLoopKind::kSelect ? original_operand_ids[0] : 0;
  if (kind == ElementwiseLoopKind::kSelect && result_packed) {
    select_condition_id =
        BuildScalarSplat(&preheader_builder, packed_select_condition_type_id,
                         original_operand_ids[0]);
    if (select_condition_id == 0) return false;
  }
  for (const LoopOperand& operand : loop_operands) {
    if (!preheader_builder.AddStore(operand.variable_id, operand.value_id)) {
      return false;
    }
  }
  if (!preheader_builder.AddStore(index_variable->result_id(), zero_id) ||
      (enclosing_loop_merge &&
       !preheader_builder.AddInstruction(std::move(enclosing_loop_merge))) ||
      !preheader_builder.AddBranch(header_label_id)) {
    return false;
  }

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* index =
      header_builder.AddLoad(uint_type_id, index_variable->result_id());
  Instruction* condition =
      index ? header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                         index->result_id(), piece_count_id)
            : nullptr;
  if (!index || !condition ||
      !header_builder.AddLoopMerge(merge_label_id, continue_label_id) ||
      !header_builder.AddConditionalBranch(condition->result_id(),
                                           body_label_id, merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  auto load_scalar = [&](const LoopOperand& operand,
                         uint32_t logical_index_id) -> uint32_t {
    if (!operand.layout.packed_vec4) {
      Instruction* pointer =
          body_builder.AddAccessChain(operand.scalar_pointer_type_id,
                                      operand.variable_id, {logical_index_id});
      Instruction* value =
          pointer ? body_builder.AddLoad(operand.layout.component_type_id,
                                         pointer->result_id())
                  : nullptr;
      return value ? value->result_id() : 0;
    }

    Instruction* packed_index = body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpUDiv, logical_index_id, four_id);
    Instruction* lane = body_builder.AddBinaryOp(uint_type_id, spv::Op::OpUMod,
                                                 logical_index_id, four_id);
    if (!packed_index || !lane) return 0;
    Instruction* pointer = body_builder.AddAccessChain(
        operand.piece_pointer_type_id, operand.variable_id,
        {packed_index->result_id()});
    Instruction* piece =
        pointer ? body_builder.AddLoad(operand.layout.piece_type_id,
                                       pointer->result_id())
                : nullptr;
    Instruction* value =
        piece ? body_builder.AddBinaryOp(operand.layout.component_type_id,
                                         spv::Op::OpVectorExtractDynamic,
                                         piece->result_id(), lane->result_id())
              : nullptr;
    return value ? value->result_id() : 0;
  };

  auto load_for_result_piece = [&](const LoopOperand& operand) -> uint32_t {
    if (!result_packed) return load_scalar(operand, index->result_id());
    if (operand.layout.packed_vec4) {
      Instruction* pointer = body_builder.AddAccessChain(
          operand.piece_pointer_type_id, operand.variable_id,
          {index->result_id()});
      Instruction* value =
          pointer ? body_builder.AddLoad(operand.layout.piece_type_id,
                                         pointer->result_id())
                  : nullptr;
      return value ? value->result_id() : 0;
    }

    Instruction* base = body_builder.AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                                 index->result_id(), four_id);
    if (!base) return 0;
    std::vector<uint32_t> constituents;
    constituents.reserve(kPackedVec4Width);
    for (uint32_t lane = 0; lane < kPackedVec4Width; ++lane) {
      uint32_t logical_index_id = base->result_id();
      if (lane != 0) {
        const uint32_t lane_id = GetOrCreateUIntConstant(lane);
        Instruction* lane_index = body_builder.AddBinaryOp(
            uint_type_id, spv::Op::OpIAdd, base->result_id(), lane_id);
        if (!lane_index) return 0;
        logical_index_id = lane_index->result_id();
      }
      const uint32_t scalar_id = load_scalar(operand, logical_index_id);
      if (scalar_id == 0) return 0;
      constituents.push_back(scalar_id);
    }
    Instruction* piece =
        body_builder.AddNaryOp(operand.loop_piece_type_id,
                               spv::Op::OpCompositeConstruct, constituents);
    return piece ? piece->result_id() : 0;
  };

  auto convert_float_arithmetic_operand = [&](const LoopOperand& operand,
                                              uint32_t operand_id) -> uint32_t {
    if (operand_id == 0 || !IsFloatArithmeticOpcode(original_opcode) ||
        operand.layout.component_type_id == result_component_type_id) {
      return operand_id;
    }
    Instruction* converted = body_builder.AddUnaryOp(
        result_piece_type_id, spv::Op::OpFConvert, operand_id);
    ApplyActiveFPFastMathMode(converted);
    return converted ? converted->result_id() : 0;
  };

  uint32_t lowered_piece_id = 0;
  if (kind == ElementwiseLoopKind::kConversion) {
    const uint32_t operand_id = load_for_result_piece(loop_operands[0]);
    Instruction* converted =
        operand_id ? body_builder.AddUnaryOp(result_piece_type_id,
                                             original_opcode, operand_id)
                   : nullptr;
    ApplyActiveFPFastMathMode(converted);
    lowered_piece_id = converted ? converted->result_id() : 0;
  } else if (kind == ElementwiseLoopKind::kArithmetic) {
    const uint32_t lhs_id = convert_float_arithmetic_operand(
        loop_operands[0], load_for_result_piece(loop_operands[0]));
    Instruction* lowered = nullptr;
    if (lhs_id != 0 && loop_operands.size() == 1) {
      lowered = body_builder.AddUnaryOp(result_piece_type_id, original_opcode,
                                        lhs_id);
    } else if (lhs_id != 0 && loop_operands.size() == 2) {
      const uint32_t rhs_id = convert_float_arithmetic_operand(
          loop_operands[1], load_for_result_piece(loop_operands[1]));
      if (rhs_id != 0) {
        lowered = body_builder.AddBinaryOp(result_piece_type_id,
                                           original_opcode, lhs_id, rhs_id);
      }
    }
    ApplyActiveFPFastMathMode(lowered);
    lowered_piece_id = lowered ? lowered->result_id() : 0;
  } else if (kind == ElementwiseLoopKind::kScale) {
    const uint32_t input_id = load_for_result_piece(loop_operands[0]);
    const spv::Op scale_opcode =
        GetScaleOpcode(get_def_use_mgr()->GetDef(result_component_type_id));
    if (input_id != 0 && result_packed) {
      lowered_piece_id = BuildVectorTimesScalar(&body_builder, scale_opcode,
                                                result_piece_type_id, input_id,
                                                original_operand_ids[1]);
    } else if (input_id != 0) {
      Instruction* scaled =
          body_builder.AddBinaryOp(result_piece_type_id, scale_opcode, input_id,
                                   original_operand_ids[1]);
      ApplyActiveFPFastMathMode(scaled);
      lowered_piece_id = scaled ? scaled->result_id() : 0;
    }
    if (lowered_piece_id != 0 && result_packed) {
      ApplyActiveFPFastMathMode(get_def_use_mgr()->GetDef(lowered_piece_id));
    }
  } else if (kind == ElementwiseLoopKind::kSelect) {
    const uint32_t lhs_id = load_for_result_piece(loop_operands[0]);
    const uint32_t rhs_id = load_for_result_piece(loop_operands[1]);
    Instruction* selected =
        lhs_id && rhs_id
            ? body_builder.AddTernaryOp(result_piece_type_id, spv::Op::OpSelect,
                                        select_condition_id, lhs_id, rhs_id)
            : nullptr;
    lowered_piece_id = selected ? selected->result_id() : 0;
  } else if (kind == ElementwiseLoopKind::kBroadcast) {
    lowered_piece_id =
        result_packed ? BuildScalarSplat(&body_builder, result_piece_type_id,
                                         original_operand_ids[0])
                      : original_operand_ids[0];
  } else {
    std::vector<Operand> operands;
    operands.push_back(IdOperand(original_operand_ids[0]));
    operands.push_back(Operand(SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
                               {original_operand_ids[1]}));
    for (uint32_t i = 2; i < original_operand_ids.size(); ++i) {
      const int32_t loop_operand_index = operand_to_loop_operand[i];
      if (loop_operand_index >= 0) {
        const uint32_t operand_id = load_for_result_piece(
            loop_operands[static_cast<size_t>(loop_operand_index)]);
        if (operand_id == 0) return false;
        operands.push_back(IdOperand(operand_id));
      } else {
        operands.push_back(IdOperand(original_operand_ids[i]));
      }
    }
    const uint32_t result_id = TakeNextId();
    if (result_id == 0) return false;
    std::unique_ptr<Instruction> ext_inst = MakeUnique<Instruction>(
        context(), spv::Op::OpExtInst, result_piece_type_id, result_id,
        std::move(operands));
    Instruction* added = body_builder.AddInstruction(std::move(ext_inst));
    ApplyActiveFPFastMathMode(added);
    lowered_piece_id = added ? added->result_id() : 0;
  }
  if (lowered_piece_id == 0) return false;

  Instruction* result_piece_pointer = body_builder.AddAccessChain(
      result_piece_pointer_type_id, result_variable->result_id(),
      {index->result_id()});
  if (!result_piece_pointer ||
      !body_builder.AddStore(result_piece_pointer->result_id(),
                             lowered_piece_id) ||
      !body_builder.AddBranch(continue_label_id)) {
    return false;
  }

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_index = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, index->result_id(), one_id);
  if (!next_index ||
      !continue_builder.AddStore(index_variable->result_id(),
                                 next_index->result_id()) ||
      !continue_builder.AddBranch(header_label_id)) {
    return false;
  }

  inst->SetOpcode(spv::Op::OpLoad);
  inst->SetResultType(lowered_result_type_id);
  inst->SetInOperands({IdOperand(result_variable->result_id())});
  context()->UpdateDefUse(inst);
  return true;
}

}  // namespace opt
}  // namespace spvtools
