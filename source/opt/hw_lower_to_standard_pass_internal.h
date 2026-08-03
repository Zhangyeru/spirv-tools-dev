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

#ifndef SOURCE_OPT_HW_LOWER_TO_STANDARD_PASS_INTERNAL_H_
#define SOURCE_OPT_HW_LOWER_TO_STANDARD_PASS_INTERNAL_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "source/opcode.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "spirv/unified1/GLSL.std.450.h"

namespace spvtools {
namespace opt {
namespace hw_lower_internal {

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

constexpr uint32_t kDefaultMatrixTileM = 2;
constexpr uint32_t kDefaultMatrixTileN = 4;
constexpr uint32_t kDefaultVectorMatmulTileN = 4;
constexpr uint32_t kPackedVec4Width = 4;
constexpr uint32_t kMaxFusedConstantBiasPacks = 4;
constexpr uint32_t kMaxCompositeConstituents = 65532;

inline Operand IdOperand(uint32_t id) { return {SPV_OPERAND_TYPE_ID, {id}}; }

inline Instruction* AddTypeOrGlobalAfter(IRContext* context,
                                         Instruction* insert_after,
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

inline bool IsHwConversionOpcode(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpConvertFToU:
    case spv::Op::OpConvertFToS:
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
    case spv::Op::OpUConvert:
    case spv::Op::OpSConvert:
    case spv::Op::OpFConvert:
      return true;
    default:
      return false;
  }
}

inline bool IsHwArithmeticOpcode(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpFAdd:
    case spv::Op::OpFSub:
    case spv::Op::OpFMul:
    case spv::Op::OpFDiv:
    case spv::Op::OpFNegate:
    case spv::Op::OpIAdd:
    case spv::Op::OpISub:
    case spv::Op::OpIMul:
    case spv::Op::OpSDiv:
    case spv::Op::OpUDiv:
    case spv::Op::OpSNegate:
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
      return true;
    default:
      return false;
  }
}

inline bool IsHwScaleOpcode(spv::Op opcode) {
  return opcode == spv::Op::OpVectorTimesScalar ||
         opcode == spv::Op::OpMatrixTimesScalar;
}

inline bool IsHwUnaryArithmeticOpcode(spv::Op opcode) {
  return opcode == spv::Op::OpFNegate || opcode == spv::Op::OpSNegate ||
         opcode == spv::Op::OpNot;
}

inline bool IsFloatArithmeticOpcode(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpFAdd:
    case spv::Op::OpFSub:
    case spv::Op::OpFMul:
    case spv::Op::OpFDiv:
    case spv::Op::OpFNegate:
      return true;
    default:
      return false;
  }
}

inline bool IsHwShiftOpcode(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
      return true;
    default:
      return false;
  }
}

inline bool IsHwBitwiseOpcode(spv::Op opcode) {
  switch (opcode) {
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
      return true;
    default:
      return false;
  }
}

enum class HwGlslStd450Domain {
  kInvalid,
  kFloat,
  kUnsignedInteger,
  kSignedInteger,
};

struct HwGlslStd450Info {
  uint32_t operand_count = 0;
  HwGlslStd450Domain domain = HwGlslStd450Domain::kInvalid;
};

inline HwGlslStd450Info DescribeSupportedHwGlslStd450(uint32_t opcode) {
  switch (static_cast<GLSLstd450>(opcode)) {
    case GLSLstd450Atan:
    case GLSLstd450Tanh:
    case GLSLstd450Exp:
    case GLSLstd450Log:
      return {1, HwGlslStd450Domain::kFloat};
    case GLSLstd450FMin:
    case GLSLstd450FMax:
    case GLSLstd450NMin:
    case GLSLstd450NMax:
    case GLSLstd450Step:
      return {2, HwGlslStd450Domain::kFloat};
    case GLSLstd450UMin:
    case GLSLstd450UMax:
      return {2, HwGlslStd450Domain::kUnsignedInteger};
    case GLSLstd450SMin:
    case GLSLstd450SMax:
      return {2, HwGlslStd450Domain::kSignedInteger};
    case GLSLstd450FClamp:
    case GLSLstd450NClamp:
    case GLSLstd450Fma:
      return {3, HwGlslStd450Domain::kFloat};
    case GLSLstd450UClamp:
      return {3, HwGlslStd450Domain::kUnsignedInteger};
    case GLSLstd450SClamp:
      return {3, HwGlslStd450Domain::kSignedInteger};
    default:
      return {};
  }
}

// Authoritative core-opcode use closure for cooperative values.  Opcodes in
// the arithmetic/conversion/scale families are classified by their semantic
// helpers above; this list contains only structural operations whose lowering
// is either dedicated or completed by recursive type replacement.
inline bool IsCoreOpcodeAllowedOnHwValue(spv::Op opcode) {
  if (IsHwConversionOpcode(opcode) || IsHwArithmeticOpcode(opcode) ||
      IsHwScaleOpcode(opcode)) {
    return true;
  }
  switch (opcode) {
    case spv::Op::OpUndef:
    case spv::Op::OpConstantNull:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpConstantCompositeReplicateEXT:
    case spv::Op::OpSpecConstantCompositeReplicateEXT:
    case spv::Op::OpVariable:
    case spv::Op::OpLoad:
    case spv::Op::OpStore:
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
    case spv::Op::OpCopyObject:
    case spv::Op::OpCopyLogical:
    case spv::Op::OpCompositeConstruct:
    case spv::Op::OpCompositeConstructReplicateEXT:
    case spv::Op::OpCompositeExtract:
    case spv::Op::OpCompositeInsert:
    case spv::Op::OpVectorExtractDynamic:
    case spv::Op::OpVectorInsertDynamic:
    case spv::Op::OpPhi:
    case spv::Op::OpSelect:
    case spv::Op::OpFunction:
    case spv::Op::OpFunctionParameter:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpReturnValue:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
    case spv::Op::OpBitcast:
    case spv::Op::OpExtInst:
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
    case spv::Op::OpTypeStruct:
    case spv::Op::OpTypePointer:
    case spv::Op::OpTypeForwardPointer:
    case spv::Op::OpTypeFunction:
    case spv::Op::OpLifetimeStart:
    case spv::Op::OpLifetimeStop:
    case spv::Op::OpArrayLength:
    case spv::Op::OpSizeOf:
    case spv::Op::OpPtrEqual:
    case spv::Op::OpPtrNotEqual:
    case spv::Op::OpPtrDiff:
      return true;
    default:
      return false;
  }
}

inline bool IsNumericScalarType(const Instruction* type) {
  return type && (type->opcode() == spv::Op::OpTypeFloat ||
                  type->opcode() == spv::Op::OpTypeInt);
}

inline bool IsSupportedHwComponentType(const Instruction* type) {
  if (!IsNumericScalarType(type) || type->NumInOperands() < 1) return false;

  const uint32_t bit_width = type->GetSingleWordInOperand(0);
  if (type->opcode() == spv::Op::OpTypeFloat) {
    return bit_width == 16 || bit_width == 32;
  }
  return bit_width == 8 || bit_width == 16 || bit_width == 32;
}

inline bool IsFloatScalarType(const Instruction* type) {
  return type && type->opcode() == spv::Op::OpTypeFloat;
}

struct NumericScalarInfo {
  bool valid = false;
  bool is_float = false;
  bool is_signed = false;
  uint32_t width = 0;
};

inline NumericScalarInfo DescribeNumericScalarType(const Instruction* type) {
  NumericScalarInfo result;
  if (!type || type->NumInOperands() < 1) return result;
  if (type->opcode() == spv::Op::OpTypeFloat) {
    result.valid = true;
    result.is_float = true;
    result.width = type->GetSingleWordInOperand(0);
  } else if (type->opcode() == spv::Op::OpTypeInt &&
             type->NumInOperands() >= 2) {
    result.valid = true;
    result.width = type->GetSingleWordInOperand(0);
    result.is_signed = type->GetSingleWordInOperand(1) != 0;
  }
  return result;
}

inline spv::Op GetScaleOpcode(const Instruction* type) {
  return IsFloatScalarType(type) ? spv::Op::OpFMul : spv::Op::OpIMul;
}

}  // namespace hw_lower_internal
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_HW_LOWER_TO_STANDARD_PASS_INTERNAL_H_
