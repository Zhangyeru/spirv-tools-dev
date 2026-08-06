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

#include "test/opt/hw_lower_to_standard_test_utils.h"

namespace spvtools {
namespace opt {
namespace {

TEST_F(HwLowerToStandardTest, NonHwModuleIsUnchanged) {
  const std::string text = R"(
; CHECK: OpCapability Shader
; CHECK-NOT: HW
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(HwLowerToStandardTest, MaterializesPackedTypesBeforeRewrittenPointer) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_16 = OpConstant %uint 16
%float = OpTypeFloat 32
%hw_vec = OpTypeCooperativeVectorHW %float %uint_16
%hw_ptr = OpTypePointer Function %hw_vec
%late_uint_4 = OpConstant %uint 4
%late_v4float = OpTypeVector %float 4
%late_array = OpTypeArray %late_v4float %late_uint_4
%value = OpUndef %hw_vec
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpTypeVector %float 4"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpTypeArray"));
}

TEST_F(HwLowerToStandardTest, PreservesSharedExtensionForBarrierOps) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%vec4 = OpTypeCooperativeVectorHW %uint %uint_4
%value = OpUndef %vec4
%main = OpFunction %void None %fn
%entry = OpLabel
OpBarrierArriveHW %int_0 %int_1
OpBarrierWaitHW %int_0 %int_1
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "CooperativeVectorHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "SPV_HW_neural_shader"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierArriveHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierWaitHW"));
}

TEST_F(HwLowerToStandardTest,
       CooperativeOnlyModeLowersValueAndPreservesNonCooperativeHwOpcode) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%a = OpUndef %vec4
%b = OpUndef %vec4
%main = OpFunction %void None %fn
%entry = OpLabel
%sum = OpFAdd %vec4 %a %b
OpBarrierArriveHW %int_0 %int_1
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "CooperativeVectorHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "SPV_HW_neural_shader"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierArriveHW"));
  EXPECT_GT(CountSubstring(disassembly, "OpFAdd %v4float"), 0u);
}

TEST_F(HwLowerToStandardTest, FailsUnsupportedCooperativeVectorExtInst) {
  const std::string text = R"(
; CHECK: unsupported HW cooperative vector GLSL.std.450 opcode
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%zero = OpConstantNull %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpExtInst %vec8 %glsl Sin %zero
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsMismatchedCooperativeVectorExtInstLength) {
  const std::string text = R"(
; CHECK: invalid HW cooperative vector OpExtInst operand
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%short = OpConstantNull %vec4
%wide = OpConstantNull %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpExtInst %vec8 %glsl FMax %wide %short
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsMatrixExtInstResult) {
  const std::string text = R"(
; CHECK: HW OpExtInst requires a cooperative vector result
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%float = OpTypeFloat 32
%mat2x2 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%zero = OpConstantNull %mat2x2
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpExtInst %mat2x2 %glsl FMax %zero %zero
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest,
       RejectsCooperativeVectorExtInstComponentTypeMismatch) {
  const std::string text = R"(
; CHECK: invalid HW cooperative vector OpExtInst operand
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%fvec8 = OpTypeCooperativeVectorHW %float %uint_8
%uvec8 = OpTypeCooperativeVectorHW %uint %uint_8
%fzero = OpConstantNull %fvec8
%uzero = OpConstantNull %uvec8
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpExtInst %fvec8 %glsl FMax %fzero %uzero
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsCooperativeVectorExtInstInvalidArity) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%zero = OpConstantNull %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpExtInst %vec8 %glsl FMax %zero %zero
OpReturn
OpFunctionEnd
)";

  ExpectFailureWithMissingExtInstOperand(this, text);
}

TEST_F(HwLowerToStandardTest,
       RejectsConversionUnlessInputAndResultAreCooperative) {
  const std::vector<std::string> instructions = {
      "%bad = OpFConvert %fvec4 %ordinary_half",
      "%bad = OpFConvert %v4float %hw_half",
      "%bad = OpFConvert %fmat2 %ordinary_half_matrix",
      "%bad = OpFConvert %m2float %hw_half_matrix",
  };
  for (const std::string& instruction : instructions) {
    SCOPED_TRACE(instruction);
    SinglePassRunAndFail<HwLowerToStandardPass>(
        MakeAsymmetricHwElementwiseModule("unsupported HW conversion",
                                          instruction));
  }
}

TEST_F(HwLowerToStandardTest,
       RejectsArithmeticUnlessInputAndResultAreCooperative) {
  const std::vector<std::string> instructions = {
      "%bad = OpFNegate %fvec4 %ordinary_float",
      "%bad = OpFNegate %v4float %hw_float",
      "%bad = OpFNegate %fmat2 %ordinary_float_matrix",
      "%bad = OpFNegate %m2float %hw_float_matrix",
  };
  for (const std::string& instruction : instructions) {
    SCOPED_TRACE(instruction);
    SinglePassRunAndFail<HwLowerToStandardPass>(
        MakeAsymmetricHwElementwiseModule("unsupported HW arithmetic",
                                          instruction));
  }
}

TEST_F(HwLowerToStandardTest, RejectsScaleUnlessInputAndResultAreCooperative) {
  const std::vector<std::string> instructions = {
      "%bad = OpVectorTimesScalar %fvec4 %ordinary_float %one",
      "%bad = OpVectorTimesScalar %v4float %hw_float %one",
      "%bad = OpMatrixTimesScalar %fmat2 %ordinary_float_matrix %one",
      "%bad = OpMatrixTimesScalar %m2float %hw_float_matrix %one",
  };
  for (const std::string& instruction : instructions) {
    SCOPED_TRACE(instruction);
    SinglePassRunAndFail<HwLowerToStandardPass>(
        MakeAsymmetricHwElementwiseModule("unsupported HW scale operation",
                                          instruction));
  }
}

TEST_F(HwLowerToStandardTest,
       RejectsPlainVectorCompositeConstructScalarBroadcast) {
  const std::string text = R"(
; CHECK: HW vector OpCompositeConstruct operand count is invalid
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%bad = OpCompositeConstruct %vec4 %one
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsLargeMatrixMulAddLowering) {
  const std::string text = R"(
; CHECK: HW cooperative matrix multiply expansion is too large
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_32 = OpConstant %uint 32
%uint_65 = OpConstant %uint 65
%float = OpTypeFloat 32
%mat32x32 = OpTypeCooperativeMatrixHW %float %uint_32 %uint_32
%mat32x65 = OpTypeCooperativeMatrixHW %float %uint_32 %uint_65
%a = OpUndef %mat32x32
%b = OpUndef %mat32x65
%c = OpConstantNull %mat32x65
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat32x65 %a %b %c
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(
      text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1048576u,
      65536ull, 4096u, 4096ull);
}

TEST_F(HwLowerToStandardTest, FailsLargeVectorMatrixMulLowering) {
  const std::string text = R"(
; CHECK: HW cooperative vector matrix multiply expansion is too large
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_64 = OpConstant %uint 64
%uint_1025 = OpConstant %uint 1025
%float = OpTypeFloat 32
%vec64 = OpTypeCooperativeVectorHW %float %uint_64
%vec1025 = OpTypeCooperativeVectorHW %float %uint_1025
%mat64x1025 = OpTypeCooperativeMatrixHW %float %uint_64 %uint_1025
%x = OpUndef %vec64
%w = OpUndef %mat64x1025
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec1025 %x %w
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(
      text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1048576u,
      65536ull, 4096u, 4096ull);
}

TEST_F(HwLowerToStandardTest, RejectsNestedVectorConstantCompositeConstituent) {
  const std::string text = R"(
; CHECK: unsupported HW OpConstantComposite operand
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%one = OpConstant %float 1
%v2float = OpTypeVector %float 2
%pair = OpConstantComposite %v2float %one %one
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%bad = OpConstantComposite %vec4 %pair %pair %pair %pair
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest,
       RejectsReplicatedConstantThatCannotBeSerializedAfterLowering) {
  const std::vector<std::string> opcodes = {
      "OpConstantCompositeReplicateEXT",
      "OpSpecConstantCompositeReplicateEXT",
  };
  for (const std::string& opcode : opcodes) {
    SCOPED_TRACE(opcode);
    const std::string text = std::string(R"(
; CHECK: HW replicated constant exceeds the SPIR-V composite operand limit
OpCapability Shader
OpCapability ReplicatedCompositesEXT
OpCapability CooperativeVectorHW
OpExtension "SPV_EXT_replicated_composites"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_65533 = OpConstant %uint 65533
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec65533 = OpTypeCooperativeVectorHW %float %uint_65533
%value = )") + opcode + R"( %vec65533 %one
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

    SinglePassRunAndFail<HwLowerToStandardPass>(
        text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
        HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 70000u,
        16777216ull, 70000u, 4096ull);
  }
}

TEST_F(HwLowerToStandardTest,
       RejectsDirectConstantCompositesAboveConstituentLimit) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"OpConstantComposite", "OpConstant %float 1"},
      {"OpSpecConstantComposite", "OpSpecConstant %float 1065353216"},
  };
  for (const auto& test_case : cases) {
    const std::string& opcode = test_case.first;
    SCOPED_TRACE(opcode);
    const std::string text = std::string(R"(
; CHECK: HW constant composite exceeds the SPIR-V composite operand limit
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_65533 = OpConstant %uint 65533
%float = OpTypeFloat 32
%one = )") + test_case.second +
                             R"(
%mat1x65533 = OpTypeCooperativeMatrixHW %float %uint_1 %uint_65533
%value = )" + opcode + R"( %mat1x65533 %one
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

    SinglePassRunAndFail<HwLowerToStandardPass>(
        text, HwLowerToStandardPass::LoweringMode::kForceScalar,
        HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 70000u,
        16777216ull, 65532u, 4096ull);
  }
}

TEST_F(HwLowerToStandardTest, FailsForInvalidMatmulShape) {
  const std::string text = R"(
; CHECK: HW cooperative matrix multiply shapes do not match
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%mat4x8 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_8
%mat4x4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%a = OpUndef %mat4x8
%b = OpUndef %mat4x4
%c = OpUndef %mat4x4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat4x4 %a %b %c
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsForInvalidVectorMatmulShape) {
  const std::string text = R"(
; CHECK: HW cooperative vector matrix multiply shapes do not match
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%mat8x4 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_4
%x = OpUndef %vec4
%w = OpUndef %mat8x4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec4 %x %w
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsForPointerEscapeOrNonFunctionHwPointer) {
  const std::string text = R"(
; CHECK: HW cooperative values may only be stored in Function or Private variables before lowering
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%_ptr_StorageBuffer_mat = OpTypePointer StorageBuffer %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsNestedHwStructInStorageBuffer) {
  const std::string text = R"(
; CHECK: HW cooperative values may only be stored in Function or Private variables before lowering
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%S = OpTypeStruct %mat
%_ptr_StorageBuffer_S = OpTypePointer StorageBuffer %S
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsNestedHwArrayInUniform) {
  const std::string text = R"(
; CHECK: HW cooperative values may only be stored in Function or Private variables before lowering
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%arr = OpTypeArray %mat %uint_2
%_ptr_Uniform_arr = OpTypePointer Uniform %arr
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, ExtensionFreeModeLowersCooperativeModule) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_1 %uint_5
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%element = OpCompositeExtract %float %a 0 4
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kExtensionFree);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, ExtensionFreeModeRejectsEveryUnloweredHwOpcode) {
  struct UnsupportedHwOpcodeCase {
    const char* name;
    const char* declarations;
    const char* body;
  };
  const UnsupportedHwOpcodeCase cases[] = {
      {"OpTypeTensorMapHW", "%tensor_map = OpTypeTensorMapHW 1\n", ""},
      {"OpCpAsyncTensorGlobalSharedHW", "",
       "OpCpAsyncTensorGlobalSharedHW 1 %uint_0 %uint_0 %uint_0\n"},
      {"OpCpAsyncCommitGroupHW", "", "OpCpAsyncCommitGroupHW\n"},
      {"OpCpAsyncWaitGroupHW", "", "OpCpAsyncWaitGroupHW %uint_0\n"},
      {"OpBarrierArriveHW", "", "OpBarrierArriveHW %uint_0 %uint_1\n"},
      {"OpBarrierWaitHW", "", "OpBarrierWaitHW %uint_0 %uint_1\n"},
      {"OpShuffleIndexHW", "",
       "%result = OpShuffleIndexHW %uint %uint_1 %uint_0\n"},
      {"OpBytePermuteHW", "",
       "%result = OpBytePermuteHW %uint %uint_1 %uint_1 %uint_0\n"},
      {"OpShuffleFillDownHW", "",
       "%result = OpShuffleFillDownHW %uint %uint_1 %uint_0 %uint_1\n"},
  };

  for (const UnsupportedHwOpcodeCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const std::string text = std::string(R"(
; CHECK: extension-free HW lowering has no equivalent lowering for this HW opcode
OpCapability Shader
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
)") + test_case.declarations +
                             R"(
%main = OpFunction %void None %fn
%entry = OpLabel
)" + test_case.body + R"(
OpReturn
OpFunctionEnd
)";

    SinglePassRunAndFail<HwLowerToStandardPass>(
        text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
        HwLowerToStandardPass::CompletenessMode::kExtensionFree);
  }
}

TEST_F(HwLowerToStandardTest, ExtensionFreeModeRejectsRelregOperand) {
  const std::string text = R"(
; CHECK: extension-free HW lowering has no equivalent lowering for this HW operand
OpCapability Shader
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%main = OpFunction %void None %fn
%entry = OpLabel
OpSelectionMerge %merge Relreg
OpBranchConditional %true %then %merge
%then = OpLabel
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(
      text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kExtensionFree);
}

TEST_F(HwLowerToStandardTest, RejectsNoContractionMatrixMatmul) {
  const std::string text = R"(
; CHECK: NoContraction HW cooperative matrix multiply cannot be lowered
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d NoContraction
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%a = OpUndef %mat4
%b = OpUndef %mat4
%c = OpUndef %mat4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat4 %a %b %c
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsNoContractionVectorMatmul) {
  const std::string text = R"(
; CHECK: NoContraction HW cooperative vector matrix multiply cannot be lowered
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y NoContraction
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%mat4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%x = OpUndef %vec4
%w = OpUndef %mat4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec4 %x %w
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsGroupedNoContractionVectorMatmul) {
  const std::string text = R"(
; CHECK: NoContraction HW cooperative vector matrix multiply cannot be lowered
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%no_contraction = OpDecorationGroup
OpDecorate %no_contraction NoContraction
OpGroupDecorate %no_contraction %y
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%mat4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%x = OpUndef %vec4
%w = OpUndef %mat4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec4 %x %w
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsNoContractionVectorMatmulAdd) {
  const std::string text = R"(
; CHECK: NoContraction HW cooperative vector matrix multiply cannot be lowered
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y NoContraction
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%mat4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%x = OpUndef %vec4
%w = OpUndef %mat4
%bias = OpUndef %vec4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulAddHW %vec4 %x %w %bias
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest,
       RejectsNoContractionBeforeTwoLayerVectorMatmulFusion) {
  const std::string text = R"(
; CHECK: NoContraction HW cooperative vector matrix multiply cannot be lowered
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %hidden NoContraction
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_7 = OpConstant %uint 7
%uint_18 = OpConstant %uint 18
%half = OpTypeFloat 16
%vec3 = OpTypeCooperativeVectorHW %half %uint_3
%vec7 = OpTypeCooperativeVectorHW %half %uint_7
%vec18 = OpTypeCooperativeVectorHW %half %uint_18
%mat3x18 = OpTypeCooperativeMatrixHW %half %uint_3 %uint_18
%mat18x7 = OpTypeCooperativeMatrixHW %half %uint_18 %uint_7
%x = OpUndef %vec3
%w0 = OpUndef %mat3x18
%w1 = OpUndef %mat18x7
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%hidden = OpCooperativeVectorMatrixMulHW %vec18 %x %w0
%out = OpCooperativeVectorMatrixMulHW %vec7 %hidden %w1
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsSpecConstantCooperativeShape) {
  const std::string text = R"(
; CHECK: HW cooperative matrix specialization-constant shape is not supported
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %rows SpecId 0
OpDecorate %cols SpecId 1
%uint = OpTypeInt 32 0
%rows = OpSpecConstant %uint 3
%cols = OpSpecConstant %uint 5
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %rows %cols
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%element = OpCompositeExtract %float %a 0 0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsFloat64CooperativeComponentType) {
  const std::string text = R"(
; CHECK: HW cooperative matrix component type must be 16/32-bit floating-point or 8/16/32-bit integer
OpCapability Shader
OpCapability Float64
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%double = OpTypeFloat 64
%mat = OpTypeCooperativeMatrixHW %double %uint_2 %uint_2
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsSignedInt64CooperativeComponentType) {
  const std::string text = R"(
; CHECK: HW cooperative vector component type must be 16/32-bit floating-point or 8/16/32-bit integer
OpCapability Shader
OpCapability Int64
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%long = OpTypeInt 64 1
%vec = OpTypeCooperativeVectorHW %long %uint_2
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsUnsignedInt64CooperativeComponentType) {
  const std::string text = R"(
; CHECK: HW cooperative matrix component type must be 16/32-bit floating-point or 8/16/32-bit integer
OpCapability Shader
OpCapability Int64
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%ulong = OpTypeInt 64 0
%mat = OpTypeCooperativeMatrixHW %ulong %uint_2 %uint_2
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsSpecConstantMatrixLayout) {
  const std::string text = R"(
; CHECK: HW cooperative matrix memory access cannot be lowered
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %layout SpecId 0
OpDecorate %data ArrayStride 4
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%layout = OpSpecConstant %int 0
%v2int = OpTypeVector %int 2
%shape = OpConstantComposite %v2int %int_0 %int_0
%offset = OpConstantComposite %v2int %int_0 %int_0
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%data = OpTypeRuntimeArray %float
%_ptr_StorageBuffer_data = OpTypePointer StorageBuffer %data
%buffer = OpVariable %_ptr_StorageBuffer_data StorageBuffer
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%value = OpCooperativeMatrixLoadHW %mat %buffer %shape %offset %layout
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsSpecConstantMatrixReduceControl) {
  const std::string text = R"(
; CHECK: invalid OpCooperativeMatrixReduceHW
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %axis SpecId 0
%uint = OpTypeInt 32 0
%axis = OpSpecConstant %uint 0
%add = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%input = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%value = OpCooperativeMatrixReduceHW %mat %input %axis %add
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsMixedWidthUnsignedDivision) {
  const std::string text = R"(
; CHECK: OpUDiv requires matching HW integer component bit widths
OpCapability Shader
OpCapability Int16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%ushort = OpTypeInt 16 0
%uint = OpTypeInt 32 0
%uint_5 = OpConstant %uint 5
%u16vec = OpTypeCooperativeVectorHW %ushort %uint_5
%u32vec = OpTypeCooperativeVectorHW %uint %uint_5
%u16 = OpUndef %u16vec
%u32 = OpUndef %u32vec
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%div = OpUDiv %u32vec %u16 %u32
OpReturn
OpFunctionEnd
)";

  // The current validator deliberately accepts this HW form even though a
  // standard OpUDiv requires matching operand/result types.  Refuse to guess
  // whether narrowing occurs before or after division.
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
