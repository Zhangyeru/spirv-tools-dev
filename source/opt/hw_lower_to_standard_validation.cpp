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

bool HwLowerToStandardPass::PreflightExtensionFreeMode() const {
  bool ok = true;
  get_module()->ForEachInst([this, &ok](Instruction* inst) {
    if (!ok) return;

    // Cooperative matrix/vector instructions are handled by this pass.  The
    // remaining SPV_HW_neural_shader instructions intentionally have no
    // guessed fallback: strict mode must reject them before mutating the IR.
    if (IsAnyHwOpcode(inst->opcode()) && !IsHwOpcode(inst->opcode())) {
      ReportError(inst,
                  "extension-free HW lowering has no equivalent lowering for "
                  "this HW opcode");
      ok = false;
      return;
    }
    if (HasHwOperand(inst)) {
      ReportError(inst,
                  "extension-free HW lowering has no equivalent lowering for "
                  "this HW operand");
      ok = false;
    }
  });
  return ok;
}

bool HwLowerToStandardPass::PreflightNoContractionVectorMatmul() const {
  bool ok = true;
  get_module()->ForEachInst([this, &ok](Instruction* inst) {
    if (!ok || (inst->opcode() != spv::Op::OpCooperativeVectorMatrixMulHW &&
                inst->opcode() != spv::Op::OpCooperativeVectorMatrixMulAddHW)) {
      return;
    }

    if (context()->get_decoration_mgr()->HasDecoration(
            inst->result_id(), spv::Decoration::NoContraction)) {
      ReportError(inst,
                  "NoContraction HW cooperative vector matrix multiply "
                  "cannot be lowered");
      ok = false;
    }
  });
  return ok;
}

bool HwLowerToStandardPass::ValidateCompositeIndices(
    const Instruction* inst, uint32_t composite_in_operand,
    uint32_t first_index_in_operand) const {
  if (!inst || inst->NumInOperands() <= composite_in_operand ||
      inst->NumInOperands() <= first_index_in_operand) {
    ReportError(inst, "invalid HW composite indexing instruction");
    return false;
  }

  Instruction* composite = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(composite_in_operand));
  if (!composite) {
    ReportError(inst, "invalid HW composite indexing object");
    return false;
  }
  uint32_t current_type_id = composite->type_id();

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
      current_type_id = matrix->component_type_id;
      continue;
    }
    if (const VectorTypeInfo* vector = GetVectorType(current_type_id)) {
      if (index >= vector->length) {
        ReportError(inst, "HW vector composite index is out of range");
        return false;
      }
      current_type_id = vector->component_type_id;
      continue;
    }

    Instruction* type = get_def_use_mgr()->GetDef(current_type_id);
    if (!type) {
      ReportError(inst, "invalid HW composite index type");
      return false;
    }
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

  if (inst->opcode() == spv::Op::OpCompositeExtract) {
    if (inst->type_id() != current_type_id) {
      ReportError(inst, "HW OpCompositeExtract result type is invalid");
      return false;
    }
    return true;
  }

  Instruction* object =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  if (!object || object->type_id() != current_type_id ||
      inst->type_id() != composite->type_id()) {
    ReportError(inst, "HW OpCompositeInsert types do not match");
    return false;
  }
  return true;
}

bool HwLowerToStandardPass::ValidateAccessChain(const Instruction* inst) const {
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
    ReportError(inst, "invalid HW access-chain base pointer");
    return false;
  }

  uint32_t current_type_id = pointer_type->GetSingleWordInOperand(1);
  const bool is_ptr_chain = inst->opcode() == spv::Op::OpPtrAccessChain ||
                            inst->opcode() == spv::Op::OpInBoundsPtrAccessChain;
  const uint32_t first_index = is_ptr_chain ? 2 : 1;
  for (uint32_t i = first_index; i < inst->NumInOperands(); ++i) {
    const uint32_t index_id = inst->GetSingleWordInOperand(i);
    const MatrixTypeInfo* matrix = GetMatrixType(current_type_id);
    const VectorTypeInfo* vector = GetVectorType(current_type_id);
    if (matrix || vector) {
      Instruction* index = get_def_use_mgr()->GetDef(index_id);
      Instruction* index_type =
          index ? get_def_use_mgr()->GetDef(index->type_id()) : nullptr;
      if (!index_type || index_type->opcode() != spv::Op::OpTypeInt ||
          index_type->NumInOperands() < 2 ||
          index_type->GetSingleWordInOperand(0) == 0 ||
          index_type->GetSingleWordInOperand(0) > 64) {
        ReportError(inst,
                    "HW access chain requires an integer index of at most 64 "
                    "bits");
        return false;
      }
      current_type_id =
          matrix ? matrix->component_type_id : vector->component_type_id;
      continue;
    }

    Instruction* type = get_def_use_mgr()->GetDef(current_type_id);
    if (!type) {
      ReportError(inst, "invalid HW access-chain type");
      return false;
    }
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

  Instruction* result_pointer_type = get_def_use_mgr()->GetDef(inst->type_id());
  if (!result_pointer_type ||
      result_pointer_type->opcode() != spv::Op::OpTypePointer ||
      result_pointer_type->NumInOperands() < 2 ||
      result_pointer_type->GetSingleWordInOperand(0) !=
          pointer_type->GetSingleWordInOperand(0) ||
      result_pointer_type->GetSingleWordInOperand(1) != current_type_id) {
    ReportError(inst, "HW access-chain result pointer type is invalid");
    return false;
  }
  return true;
}

bool HwLowerToStandardPass::LegalizeCooperativeCoreInstruction(
    Instruction* inst) const {
  if (!inst) return false;

  auto get_value = [this](uint32_t id) {
    return get_def_use_mgr()->GetDef(id);
  };
  auto get_matrix_value = [this, &get_value](uint32_t id) {
    return GetMatrixTypeForValue(get_value(id));
  };
  auto get_vector_value = [this, &get_value](uint32_t id) {
    Instruction* value = get_value(id);
    return value ? GetVectorType(value->type_id()) : nullptr;
  };
  auto valid_integer_index = [this, &get_value](uint32_t id) {
    Instruction* value = get_value(id);
    Instruction* type =
        value ? get_def_use_mgr()->GetDef(value->type_id()) : nullptr;
    return type && type->opcode() == spv::Op::OpTypeInt &&
           type->NumInOperands() >= 2 && type->GetSingleWordInOperand(0) != 0 &&
           type->GetSingleWordInOperand(0) <= 64;
  };
  auto same_matrix_shape = [](const MatrixTypeInfo* lhs,
                              const MatrixTypeInfo* rhs, bool same_component) {
    return lhs && rhs && lhs->rows == rhs->rows && lhs->cols == rhs->cols &&
           (!same_component ||
            lhs->component_type_id == rhs->component_type_id);
  };
  auto same_vector_shape = [](const VectorTypeInfo* lhs,
                              const VectorTypeInfo* rhs, bool same_component) {
    return lhs && rhs && lhs->length == rhs->length &&
           (!same_component ||
            lhs->component_type_id == rhs->component_type_id);
  };

  switch (inst->opcode()) {
    case spv::Op::OpCompositeConstruct:
    case spv::Op::OpCompositeConstructReplicateEXT: {
      const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
      const VectorTypeInfo* vector = GetVectorType(inst->type_id());
      if (!matrix && !vector) return true;
      const bool is_replicate =
          inst->opcode() == spv::Op::OpCompositeConstructReplicateEXT;
      const uint32_t component_type_id =
          matrix ? matrix->component_type_id : vector->component_type_id;
      if (matrix || is_replicate) {
        Instruction* scalar = inst->NumInOperands() == 1
                                  ? get_value(inst->GetSingleWordInOperand(0))
                                  : nullptr;
        if (!scalar || scalar->type_id() != component_type_id) {
          ReportError(inst,
                      is_replicate
                          ? "HW OpCompositeConstructReplicateEXT operand is "
                            "invalid"
                          : "HW matrix OpCompositeConstruct requires one "
                            "scalar constituent");
          return false;
        }
        return true;
      }

      const uint64_t expected = vector->length;
      uint64_t constituent_count = 0;
      for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
        Instruction* constituent = get_value(inst->GetSingleWordInOperand(i));
        if (!constituent) {
          ReportError(inst,
                      "invalid HW vector OpCompositeConstruct constituent");
          return false;
        }
        if (constituent->type_id() == component_type_id) {
          ++constituent_count;
          continue;
        }
        Instruction* type = get_def_use_mgr()->GetDef(constituent->type_id());
        if (!type || type->opcode() != spv::Op::OpTypeVector ||
            type->NumInOperands() < 2 ||
            type->GetSingleWordInOperand(0) != component_type_id) {
          ReportError(inst,
                      "invalid HW vector OpCompositeConstruct constituent");
          return false;
        }
        constituent_count += type->GetSingleWordInOperand(1);
      }
      if (constituent_count != expected) {
        ReportError(inst,
                    "HW vector OpCompositeConstruct operand count is invalid");
        return false;
      }
      return true;
    }

    case spv::Op::OpConstantComposite:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpConstantCompositeReplicateEXT:
    case spv::Op::OpSpecConstantCompositeReplicateEXT: {
      const MatrixTypeInfo* matrix = GetMatrixType(inst->type_id());
      const VectorTypeInfo* vector = GetVectorType(inst->type_id());
      if (!matrix && !vector) return true;
      const bool is_replicate =
          inst->opcode() == spv::Op::OpConstantCompositeReplicateEXT ||
          inst->opcode() == spv::Op::OpSpecConstantCompositeReplicateEXT;
      const bool result_is_spec =
          inst->opcode() == spv::Op::OpSpecConstantComposite ||
          inst->opcode() == spv::Op::OpSpecConstantCompositeReplicateEXT;
      const uint32_t component_type_id =
          matrix ? matrix->component_type_id : vector->component_type_id;
      const uint64_t expected = matrix ? uint64_t(matrix->rows) * matrix->cols
                                       : uint64_t(vector->length);
      const uint64_t required_operands = matrix || is_replicate ? 1 : expected;
      if (inst->NumInOperands() != required_operands) {
        ReportError(inst, "HW OpConstantComposite operand count is invalid");
        return false;
      }
      for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
        Instruction* operand = get_value(inst->GetSingleWordInOperand(i));
        if (!operand || operand->type_id() != component_type_id ||
            (!IsConstantInst(operand->opcode()) &&
             operand->opcode() != spv::Op::OpUndef) ||
            (!result_is_spec && IsSpecConstantInst(operand->opcode()))) {
          ReportError(inst, "unsupported HW OpConstantComposite operand");
          return false;
        }
      }
      const bool packed =
          matrix ? IsPackedVec2(*matrix) : IsPackedVec2(*vector);
      const uint64_t lowered_constituent_count =
          packed ? expected / kPackedVec2Width : expected;
      if (lowered_constituent_count > kMaxCompositeConstituents) {
        ReportError(inst,
                    is_replicate
                        ? "HW replicated constant exceeds the SPIR-V composite "
                          "operand limit after lowering"
                        : "HW constant composite exceeds the SPIR-V composite "
                          "operand limit after lowering");
        return false;
      }
      return true;
    }

    case spv::Op::OpCompositeExtract:
      return ValidateCompositeIndices(inst, 0, 1);
    case spv::Op::OpCompositeInsert:
      return ValidateCompositeIndices(inst, 1, 2);
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
      return ValidateAccessChain(inst);

    case spv::Op::OpSelect: {
      const MatrixTypeInfo* result_matrix = GetMatrixType(inst->type_id());
      const VectorTypeInfo* result_vector = GetVectorType(inst->type_id());
      if (!result_matrix && !result_vector) return true;
      if (inst->NumInOperands() != 3) {
        ReportError(inst, "invalid HW OpSelect");
        return false;
      }
      Instruction* condition = get_value(inst->GetSingleWordInOperand(0));
      Instruction* condition_type =
          condition ? get_def_use_mgr()->GetDef(condition->type_id()) : nullptr;
      if (!condition_type || condition_type->opcode() != spv::Op::OpTypeBool) {
        ReportError(inst, "HW OpSelect requires a scalar bool condition");
        return false;
      }
      Instruction* true_value = get_value(inst->GetSingleWordInOperand(1));
      Instruction* false_value = get_value(inst->GetSingleWordInOperand(2));
      if (!true_value || !false_value ||
          true_value->type_id() != inst->type_id() ||
          false_value->type_id() != inst->type_id()) {
        ReportError(inst, "invalid HW OpSelect object");
        return false;
      }
      return true;
    }

    case spv::Op::OpVectorExtractDynamic: {
      if (inst->NumInOperands() != 2) {
        ReportError(inst, "invalid HW OpVectorExtractDynamic");
        return false;
      }
      const VectorTypeInfo* vector =
          get_vector_value(inst->GetSingleWordInOperand(0));
      if (!vector || inst->type_id() != vector->component_type_id ||
          !valid_integer_index(inst->GetSingleWordInOperand(1))) {
        ReportError(inst, "invalid HW OpVectorExtractDynamic object or index");
        return false;
      }
      return true;
    }

    case spv::Op::OpVectorInsertDynamic: {
      const VectorTypeInfo* result = GetVectorType(inst->type_id());
      if (!result || inst->NumInOperands() != 3) {
        ReportError(inst, "invalid HW OpVectorInsertDynamic");
        return false;
      }
      Instruction* vector = get_value(inst->GetSingleWordInOperand(0));
      Instruction* object = get_value(inst->GetSingleWordInOperand(1));
      if (!vector || vector->type_id() != inst->type_id() || !object ||
          object->type_id() != result->component_type_id ||
          !valid_integer_index(inst->GetSingleWordInOperand(2))) {
        ReportError(inst, "invalid HW OpVectorInsertDynamic object or index");
        return false;
      }
      return true;
    }

    case spv::Op::OpExtInst: {
      const VectorTypeInfo* result = GetVectorType(inst->type_id());
      if (!result) {
        ReportError(inst, "HW OpExtInst requires a cooperative vector result");
        return false;
      }
      if (inst->NumInOperands() < 2) {
        ReportError(inst, "invalid HW cooperative vector OpExtInst");
        return false;
      }
      Instruction* import = get_value(inst->GetSingleWordInOperand(0));
      if (!import || import->opcode() != spv::Op::OpExtInstImport ||
          import->GetInOperand(0).AsString() != "GLSL.std.450") {
        ReportError(inst,
                    "HW cooperative vector OpExtInst must use GLSL.std.450");
        return false;
      }
      const HwGlslStd450Info ext_info =
          DescribeSupportedHwGlslStd450(inst->GetSingleWordInOperand(1));
      if (ext_info.operand_count == 0) {
        ReportError(inst,
                    "unsupported HW cooperative vector GLSL.std.450 opcode");
        return false;
      }
      if (inst->NumInOperands() != ext_info.operand_count + 2) {
        ReportError(inst,
                    "invalid HW cooperative vector GLSL.std.450 operand count");
        return false;
      }
      Instruction* component_type =
          get_def_use_mgr()->GetDef(result->component_type_id);
      const NumericScalarInfo component_info =
          DescribeNumericScalarType(component_type);
      const bool is_float_ext_inst =
          ext_info.domain == HwGlslStd450Domain::kFloat;
      // GLSL.std.450's U/S opcode selects the interpretation of integer bit
      // patterns.  OpTypeInt signedness does not have to match that spelling;
      // the validator only requires an integer component with matching width.
      const bool domain_matches =
          component_info.valid &&
          (is_float_ext_inst == component_info.is_float);
      if (!domain_matches) {
        ReportError(inst,
                    "HW cooperative vector GLSL.std.450 component type is "
                    "incompatible with the opcode");
        return false;
      }
      for (uint32_t i = 2; i < inst->NumInOperands(); ++i) {
        const VectorTypeInfo* operand =
            get_vector_value(inst->GetSingleWordInOperand(i));
        const NumericScalarInfo operand_component_info =
            DescribeNumericScalarType(
                operand ? get_def_use_mgr()->GetDef(operand->component_type_id)
                        : nullptr);
        const bool component_matches =
            is_float_ext_inst
                ? operand &&
                      operand->component_type_id == result->component_type_id
                : operand_component_info.valid &&
                      !operand_component_info.is_float &&
                      operand_component_info.width == component_info.width;
        if (!same_vector_shape(result, operand, false) || !component_matches) {
          ReportError(inst, "invalid HW cooperative vector OpExtInst operand");
          return false;
        }
      }
      return true;
    }

    case spv::Op::OpBitcast: {
      const MatrixTypeInfo* result_matrix = GetMatrixType(inst->type_id());
      const VectorTypeInfo* result_vector = GetVectorType(inst->type_id());
      if ((!result_matrix && !result_vector) || inst->NumInOperands() != 1) {
        ReportError(inst, "unsupported HW OpBitcast");
        return false;
      }
      Instruction* object = get_value(inst->GetSingleWordInOperand(0));
      const MatrixTypeInfo* input_matrix = GetMatrixTypeForValue(object);
      const VectorTypeInfo* input_vector =
          object ? GetVectorType(object->type_id()) : nullptr;
      auto component_bit_width = [this](uint32_t type_id) {
        Instruction* type = get_def_use_mgr()->GetDef(type_id);
        return type && type->NumInOperands() >= 1
                   ? type->GetSingleWordInOperand(0)
                   : 0;
      };
      const bool matching_matrix =
          same_matrix_shape(result_matrix, input_matrix, false) &&
          component_bit_width(result_matrix->component_type_id) ==
              component_bit_width(input_matrix->component_type_id);
      const bool matching_vector =
          same_vector_shape(result_vector, input_vector, false) &&
          component_bit_width(result_vector->component_type_id) ==
              component_bit_width(input_vector->component_type_id);
      if (!object || (!matching_matrix && !matching_vector)) {
        ReportError(inst, "unsupported HW OpBitcast");
        return false;
      }
      return true;
    }

    default:
      break;
  }

  if (IsHwConversionOpcode(inst->opcode())) {
    if (inst->NumInOperands() != 1) {
      ReportError(inst, "unsupported HW conversion");
      return false;
    }
    const MatrixTypeInfo* result_matrix = GetMatrixType(inst->type_id());
    const VectorTypeInfo* result_vector = GetVectorType(inst->type_id());
    const MatrixTypeInfo* input_matrix =
        get_matrix_value(inst->GetSingleWordInOperand(0));
    const VectorTypeInfo* input_vector =
        get_vector_value(inst->GetSingleWordInOperand(0));
    if ((!same_matrix_shape(result_matrix, input_matrix, false) &&
         !same_vector_shape(result_vector, input_vector, false)) ||
        (!result_matrix && !result_vector)) {
      ReportError(inst, "unsupported HW conversion input/output types");
      return false;
    }
    const uint32_t result_component = result_matrix
                                          ? result_matrix->component_type_id
                                          : result_vector->component_type_id;
    const uint32_t input_component = input_matrix
                                         ? input_matrix->component_type_id
                                         : input_vector->component_type_id;
    const NumericScalarInfo result_info =
        DescribeNumericScalarType(get_def_use_mgr()->GetDef(result_component));
    const NumericScalarInfo input_info =
        DescribeNumericScalarType(get_def_use_mgr()->GetDef(input_component));
    bool types_match_opcode = result_info.valid && input_info.valid;
    switch (inst->opcode()) {
      case spv::Op::OpConvertFToU:
        types_match_opcode &= input_info.is_float && !result_info.is_float &&
                              !result_info.is_signed;
        break;
      case spv::Op::OpConvertFToS:
        types_match_opcode &= input_info.is_float && !result_info.is_float;
        break;
      case spv::Op::OpConvertSToF:
      case spv::Op::OpConvertUToF:
        types_match_opcode &= !input_info.is_float && result_info.is_float;
        break;
      case spv::Op::OpUConvert:
        types_match_opcode &= !input_info.is_float && !result_info.is_float &&
                              !result_info.is_signed &&
                              input_info.width != result_info.width;
        break;
      case spv::Op::OpSConvert:
        types_match_opcode &= !input_info.is_float && !result_info.is_float &&
                              input_info.width != result_info.width;
        break;
      case spv::Op::OpFConvert:
        types_match_opcode &= input_info.is_float && result_info.is_float &&
                              input_info.width != result_info.width;
        break;
      default:
        types_match_opcode = false;
        break;
    }
    if (!types_match_opcode) {
      ReportError(inst, "unsupported HW conversion component types");
      return false;
    }
    return true;
  }

  if (IsHwArithmeticOpcode(inst->opcode())) {
    const uint32_t expected_operands =
        IsHwUnaryArithmeticOpcode(inst->opcode()) ? 1 : 2;
    if (inst->NumInOperands() != expected_operands) {
      ReportError(inst, "unsupported HW arithmetic");
      return false;
    }
    const MatrixTypeInfo* result_matrix = GetMatrixType(inst->type_id());
    const VectorTypeInfo* result_vector = GetVectorType(inst->type_id());
    if (!result_matrix && !result_vector) {
      ReportError(inst, "unsupported HW arithmetic result type");
      return false;
    }
    const bool is_float_arithmetic = IsFloatArithmeticOpcode(inst->opcode());
    const bool is_shift = IsHwShiftOpcode(inst->opcode());
    const bool is_bitwise = IsHwBitwiseOpcode(inst->opcode());
    if ((is_shift || is_bitwise) && !result_vector) {
      ReportError(inst,
                  "HW shift/bitwise operations require a cooperative vector");
      return false;
    }
    const uint32_t component_type_id = result_matrix
                                           ? result_matrix->component_type_id
                                           : result_vector->component_type_id;
    Instruction* component_type = get_def_use_mgr()->GetDef(component_type_id);
    const NumericScalarInfo component_info =
        DescribeNumericScalarType(component_type);
    if (!component_info.valid ||
        (is_float_arithmetic != component_info.is_float)) {
      ReportError(inst, "unsupported HW arithmetic component type");
      return false;
    }

    if (inst->opcode() == spv::Op::OpUDiv && component_info.is_signed) {
      ReportError(inst, "OpUDiv requires unsigned HW integer components");
      return false;
    }

    for (uint32_t i = 0; i < inst->NumInOperands(); ++i) {
      const uint32_t operand_id = inst->GetSingleWordInOperand(i);
      const MatrixTypeInfo* operand_matrix = get_matrix_value(operand_id);
      const VectorTypeInfo* operand_vector = get_vector_value(operand_id);
      const bool matching_shape =
          result_matrix
              ? same_matrix_shape(result_matrix, operand_matrix, false)
              : same_vector_shape(result_vector, operand_vector, false);
      if (!matching_shape) {
        ReportError(inst, "unsupported HW arithmetic");
        return false;
      }

      const uint32_t operand_component_type_id =
          result_matrix ? operand_matrix->component_type_id
                        : operand_vector->component_type_id;
      const NumericScalarInfo operand_component_info =
          DescribeNumericScalarType(
              get_def_use_mgr()->GetDef(operand_component_type_id));
      if (!operand_component_info.valid ||
          (is_float_arithmetic != operand_component_info.is_float)) {
        ReportError(inst, "unsupported HW arithmetic component type");
        return false;
      }

      // Integer arithmetic and bitwise base operands retain their bit width.
      // Shift amounts are the sole exception: SPIR-V permits an independently
      // sized integer vector with the same number of components.
      const bool is_shift_amount = is_shift && i == 1;
      if (!is_float_arithmetic && !is_shift_amount &&
          operand_component_info.width != component_info.width) {
        if (inst->opcode() == spv::Op::OpUDiv) {
          ReportError(
              inst, "OpUDiv requires matching HW integer component bit widths");
        } else {
          ReportError(inst,
                      "HW integer arithmetic component bit widths do not "
                      "match");
        }
        return false;
      }

      if (inst->opcode() == spv::Op::OpUDiv &&
          operand_component_info.is_signed) {
        ReportError(inst, "OpUDiv requires unsigned HW integer components");
        return false;
      }
    }
    return true;
  }

  if (IsHwScaleOpcode(inst->opcode())) {
    if (inst->NumInOperands() != 2) {
      ReportError(inst, "unsupported HW scale operation");
      return false;
    }
    const MatrixTypeInfo* result_matrix = GetMatrixType(inst->type_id());
    const VectorTypeInfo* result_vector = GetVectorType(inst->type_id());
    const bool expects_matrix = inst->opcode() == spv::Op::OpMatrixTimesScalar;
    if ((expects_matrix && !result_matrix) ||
        (!expects_matrix && !result_vector)) {
      ReportError(inst, "unsupported HW scale operation");
      return false;
    }
    const bool matching =
        expects_matrix
            ? same_matrix_shape(
                  result_matrix,
                  get_matrix_value(inst->GetSingleWordInOperand(0)), true)
            : same_vector_shape(
                  result_vector,
                  get_vector_value(inst->GetSingleWordInOperand(0)), true);
    const uint32_t component_type_id = expects_matrix
                                           ? result_matrix->component_type_id
                                           : result_vector->component_type_id;
    Instruction* scalar = get_value(inst->GetSingleWordInOperand(1));
    if (!matching || !scalar || scalar->type_id() != component_type_id) {
      ReportError(inst, "unsupported HW scale operation");
      return false;
    }
    return true;
  }

  return true;
}

bool HwLowerToStandardPass::LegalizeModule() {
  bool ok = true;
  auto pair_value_legal = [this](uint32_t value_id) {
    Instruction* value = get_def_use_mgr()->GetDef(value_id);
    Instruction* type =
        value ? get_def_use_mgr()->GetDef(value->type_id()) : nullptr;
    if (!type || type->opcode() != spv::Op::OpTypeVector ||
        type->NumInOperands() < 2 || type->GetSingleWordInOperand(1) < 2) {
      return false;
    }
    Instruction* component =
        get_def_use_mgr()->GetDef(type->GetSingleWordInOperand(0));
    return component && component->NumInOperands() >= 1 &&
           ((component->opcode() == spv::Op::OpTypeInt &&
             component->GetSingleWordInOperand(0) == 32) ||
            (component->opcode() == spv::Op::OpTypeFloat &&
             component->GetSingleWordInOperand(0) == 32));
  };
  auto integer_value_legal = [this](uint32_t value_id) {
    Instruction* value = get_def_use_mgr()->GetDef(value_id);
    Instruction* type =
        value ? get_def_use_mgr()->GetDef(value->type_id()) : nullptr;
    return type && type->opcode() == spv::Op::OpTypeInt &&
           type->NumInOperands() >= 2 && type->GetSingleWordInOperand(0) != 0 &&
           type->GetSingleWordInOperand(0) <= 64;
  };
  auto pointer_legal = [this](uint32_t pointer_id, uint32_t component_type_id) {
    Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
    Instruction* pointer_type =
        pointer ? get_def_use_mgr()->GetDef(pointer->type_id()) : nullptr;
    if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer ||
        pointer_type->NumInOperands() < 2) {
      return false;
    }
    Instruction* pointee =
        get_def_use_mgr()->GetDef(pointer_type->GetSingleWordInOperand(1));
    return pointee &&
           (pointee->opcode() == spv::Op::OpTypeArray ||
            pointee->opcode() == spv::Op::OpTypeRuntimeArray) &&
           pointee->NumInOperands() >= 1 &&
           pointee->GetSingleWordInOperand(0) == component_type_id;
  };
  auto memory_access_legal =
      [this](Instruction* inst, uint32_t first_memory_operand,
             uint32_t pointer_id, uint32_t component_type_id) {
        std::vector<Operand> normalized;
        return NormalizeMemoryOperandsForAccess(
            pointer_id, component_type_id,
            CopyMemoryOperands(inst, first_memory_operand), &normalized);
      };
  auto matmul_types_legal = [this](uint32_t result_type_id, uint32_t a_type_id,
                                   uint32_t b_type_id,
                                   uint32_t accumulator_type_id) {
    if (result_type_id != accumulator_type_id) return false;
    const NumericScalarInfo result =
        DescribeNumericScalarType(get_def_use_mgr()->GetDef(result_type_id));
    const NumericScalarInfo a =
        DescribeNumericScalarType(get_def_use_mgr()->GetDef(a_type_id));
    const NumericScalarInfo b =
        DescribeNumericScalarType(get_def_use_mgr()->GetDef(b_type_id));
    if (!result.valid || !a.valid || !b.valid ||
        result.is_float != a.is_float || result.is_float != b.is_float) {
      return false;
    }
    return result.width >= a.width && result.width >= b.width;
  };
  get_module()->ForEachInst([this, &ok, &pair_value_legal, &integer_value_legal,
                             &pointer_legal, &memory_access_legal,
                             &matmul_types_legal](Instruction* inst) {
    if (!ok) return;

    if (inst->opcode() == spv::Op::OpCooperativeMatrixLoadHW ||
        inst->opcode() == spv::Op::OpCooperativeMatrixStoreHW) {
      const bool is_load = inst->opcode() == spv::Op::OpCooperativeMatrixLoadHW;
      const uint32_t minimum_operands = is_load ? 4 : 5;
      const uint32_t pointer_index =
          is_load ? kHwMatrixLoadPointerInIdx : kHwMatrixStorePointerInIdx;
      const uint32_t shape_index =
          is_load ? kHwMatrixLoadShapeInIdx : kHwMatrixStoreShapeInIdx;
      const uint32_t offset_index =
          is_load ? kHwMatrixLoadOffsetInIdx : kHwMatrixStoreOffsetInIdx;
      const uint32_t layout_index =
          is_load ? kHwMatrixLoadLayoutInIdx : kHwMatrixStoreLayoutInIdx;
      const uint32_t memory_index = is_load ? kHwMatrixLoadMemoryOperandsInIdx
                                            : kHwMatrixStoreMemoryOperandsInIdx;
      const MatrixTypeInfo* info =
          is_load ? GetMatrixType(inst->type_id()) : nullptr;
      if (!is_load && inst->NumInOperands() > kHwMatrixStoreObjectInIdx) {
        info = GetMatrixTypeForValue(get_def_use_mgr()->GetDef(
            inst->GetSingleWordInOperand(kHwMatrixStoreObjectInIdx)));
      }
      uint32_t layout = 0;
      if (inst->NumInOperands() < minimum_operands || !info ||
          !pair_value_legal(inst->GetSingleWordInOperand(shape_index)) ||
          !pair_value_legal(inst->GetSingleWordInOperand(offset_index)) ||
          !GetConstantU32(inst->GetSingleWordInOperand(layout_index),
                          &layout) ||
          layout > 1 ||
          !pointer_legal(inst->GetSingleWordInOperand(pointer_index),
                         info->component_type_id) ||
          !memory_access_legal(inst, memory_index,
                               inst->GetSingleWordInOperand(pointer_index),
                               info->component_type_id)) {
        ReportError(inst,
                    "HW cooperative matrix memory access cannot be lowered");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpCooperativeVectorLoadHW ||
        inst->opcode() == spv::Op::OpCooperativeVectorStoreHW) {
      const bool is_load = inst->opcode() == spv::Op::OpCooperativeVectorLoadHW;
      const uint32_t minimum_operands = is_load ? 2 : 3;
      const uint32_t pointer_index =
          is_load ? kHwVectorLoadPointerInIdx : kHwVectorStorePointerInIdx;
      const uint32_t offset_index =
          is_load ? kHwVectorLoadOffsetInIdx : kHwVectorStoreOffsetInIdx;
      const uint32_t memory_index = is_load ? kHwVectorLoadMemoryOperandsInIdx
                                            : kHwVectorStoreMemoryOperandsInIdx;
      const VectorTypeInfo* info =
          is_load ? GetVectorType(inst->type_id()) : nullptr;
      if (!is_load && inst->NumInOperands() > kHwVectorStoreObjectInIdx) {
        Instruction* object = get_def_use_mgr()->GetDef(
            inst->GetSingleWordInOperand(kHwVectorStoreObjectInIdx));
        info = object ? GetVectorType(object->type_id()) : nullptr;
      }
      if (inst->NumInOperands() < minimum_operands || !info ||
          !integer_value_legal(inst->GetSingleWordInOperand(offset_index)) ||
          !pointer_legal(inst->GetSingleWordInOperand(pointer_index),
                         info->component_type_id) ||
          !memory_access_legal(inst, memory_index,
                               inst->GetSingleWordInOperand(pointer_index),
                               info->component_type_id)) {
        ReportError(inst,
                    "HW cooperative vector memory access cannot be lowered");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpCooperativeMatrixLengthHW) {
      Instruction* result_type = get_def_use_mgr()->GetDef(inst->type_id());
      if (inst->NumInOperands() != 1 ||
          !GetMatrixType(inst->GetSingleWordInOperand(0)) || !result_type ||
          result_type->opcode() != spv::Op::OpTypeInt ||
          result_type->NumInOperands() < 2 ||
          result_type->GetSingleWordInOperand(0) != 32) {
        ReportError(inst, "invalid OpCooperativeMatrixLengthHW");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpTypePointer &&
        TypeContainsHw(inst->GetSingleWordInOperand(1))) {
      const auto storage_class =
          static_cast<spv::StorageClass>(inst->GetSingleWordInOperand(0));
      if (storage_class != spv::StorageClass::Function &&
          storage_class != spv::StorageClass::Private) {
        ReportError(inst,
                    "HW cooperative values may only be stored in Function or "
                    "Private variables before lowering");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpCooperativeMatrixReduceHW) {
      const MatrixTypeInfo* result = GetMatrixType(inst->type_id());
      Instruction* matrix =
          inst->NumInOperands() >= 1
              ? get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0))
              : nullptr;
      const MatrixTypeInfo* input = GetMatrixTypeForValue(matrix);
      uint32_t reduce_axis = 0;
      uint32_t combine_op = 0;
      if (!result || !input || result->rows != input->rows ||
          result->cols != input->cols ||
          result->component_type_id != input->component_type_id ||
          inst->NumInOperands() != 3 ||
          !GetConstantU32(inst->GetSingleWordInOperand(1), &reduce_axis) ||
          reduce_axis > 1 ||
          !GetConstantU32(inst->GetSingleWordInOperand(2), &combine_op) ||
          combine_op > 2) {
        ReportError(inst, "invalid OpCooperativeMatrixReduceHW");
        ok = false;
        return;
      }
    }

    if (inst->opcode() == spv::Op::OpCooperativeMatrixMulAddHW) {
      if (context()->get_decoration_mgr()->HasDecoration(
              inst->result_id(), spv::Decoration::NoContraction)) {
        ReportError(inst,
                    "NoContraction HW cooperative matrix multiply cannot be "
                    "lowered");
        ok = false;
        return;
      }
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
        a = GetMatrixTypeForValue(a_inst);
        b = GetMatrixTypeForValue(b_inst);
        c = GetMatrixTypeForValue(c_inst);
      }
      if (!result || !a || !b || !c) {
        ReportError(inst, "invalid OpCooperativeMatrixMulAddHW");
        ok = false;
        return;
      }
      if (!matmul_types_legal(result->component_type_id, a->component_type_id,
                              b->component_type_id, c->component_type_id)) {
        ReportError(inst,
                    "HW cooperative matrix multiply component types are "
                    "incompatible");
        ok = false;
        return;
      }
      if (!result || !a || !b || !c || a->cols != b->rows ||
          result->rows != a->rows || result->cols != b->cols ||
          c->rows != result->rows || c->cols != result->cols) {
        ReportError(inst, "HW cooperative matrix multiply shapes do not match");
        ok = false;
        return;
      }
      const uint64_t mac_count = static_cast<uint64_t>(result->rows) *
                                 static_cast<uint64_t>(result->cols) *
                                 static_cast<uint64_t>(a->cols);
      if (mac_count > max_matmul_macs_) {
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
              ? get_def_use_mgr()->GetDef(
                    inst->GetSingleWordInOperand(kHwVectorMatrixMulMatrixInIdx))
              : nullptr;
      const VectorTypeInfo* input =
          input_inst ? GetVectorType(input_inst->type_id()) : nullptr;
      const MatrixTypeInfo* matrix = GetMatrixTypeForValue(matrix_inst);
      const bool has_bias =
          inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddHW;
      const VectorTypeInfo* bias = nullptr;
      if (has_bias && inst->NumInOperands() > kHwVectorMatrixMulAddBiasInIdx) {
        Instruction* bias_inst = get_def_use_mgr()->GetDef(
            inst->GetSingleWordInOperand(kHwVectorMatrixMulAddBiasInIdx));
        bias = bias_inst ? GetVectorType(bias_inst->type_id()) : nullptr;
      }
      const uint32_t accumulator_type_id = has_bias && bias
                                               ? bias->component_type_id
                                           : result ? result->component_type_id
                                                    : 0;
      if (!result || !input || !matrix ||
          !matmul_types_legal(result ? result->component_type_id : 0,
                              input ? input->component_type_id : 0,
                              matrix ? matrix->component_type_id : 0,
                              accumulator_type_id)) {
        ReportError(inst,
                    "HW cooperative vector matrix multiply component types "
                    "are incompatible");
        ok = false;
        return;
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
      if (mac_count > max_matmul_macs_) {
        ReportError(inst,
                    "HW cooperative vector matrix multiply expansion is too "
                    "large");
        ok = false;
        return;
      }
    }

    const bool metadata_use =
        inst->opcode() == spv::Op::OpName ||
        inst->opcode() == spv::Op::OpMemberName || inst->IsDecoration() ||
        inst->IsNonSemanticInstruction() || inst->IsDebugLineInst();
    if (!IsHwOpcode(inst->opcode()) && InstructionTouchesHw(inst) &&
        !metadata_use) {
      if (!IsCoreOpcodeAllowedOnHwValue(inst->opcode())) {
        ReportError(inst, "unsupported HW cooperative value use");
        ok = false;
        return;
      }
      if (!LegalizeCooperativeCoreInstruction(inst)) {
        ok = false;
        return;
      }
    }
  });

  return ok;
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

bool HwLowerToStandardPass::IsAnyHwOpcode(spv::Op opcode) const {
  const char* name = spvOpcodeString(opcode);
  if (!name) return false;
  const std::string opcode_name(name);
  return opcode_name.size() >= 2 &&
         opcode_name.compare(opcode_name.size() - 2, 2, "HW") == 0;
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
           extension == "SPV_AZD_cooperative_vector";
  }
  if (inst->opcode() == spv::Op::OpSourceExtension) {
    const std::string extension = inst->GetInOperand(0).AsString();
    return extension == "GL_AZD_neural_matrix" ||
           extension == "GL_AZD_cooperative_vector";
  }
  return false;
}

bool HwLowerToStandardPass::IsAnyHwCapabilityOrExtension(
    const Instruction* inst) const {
  if (IsHwCapabilityOrExtension(inst)) return true;
  if (inst->opcode() == spv::Op::OpExtension) {
    return inst->GetInOperand(0).AsString() == "SPV_HW_neural_shader";
  }
  if (inst->opcode() == spv::Op::OpSourceExtension) {
    return inst->GetInOperand(0).AsString() == "GL_HW_neural_shader";
  }
  return false;
}

bool HwLowerToStandardPass::HasHwOperand(const Instruction* inst) const {
  return inst && inst->opcode() == spv::Op::OpSelectionMerge &&
         inst->NumInOperands() >= 2 &&
         (inst->GetSingleWordInOperand(1) &
          uint32_t(spv::SelectionControlMask::Relreg)) != 0;
}

bool HwLowerToStandardPass::RequiresHwNeuralShaderExtension(
    const Instruction* inst) const {
  if (!inst) return false;
  return (IsAnyHwOpcode(inst->opcode()) && !IsHwOpcode(inst->opcode())) ||
         HasHwOperand(inst);
}

bool HwLowerToStandardPass::ModuleRequiresHwNeuralShaderExtension() const {
  bool required = false;
  get_module()->ForEachInst([this, &required](Instruction* inst) {
    if (!required) required = RequiresHwNeuralShaderExtension(inst);
  });
  return required;
}

}  // namespace opt
}  // namespace spvtools
