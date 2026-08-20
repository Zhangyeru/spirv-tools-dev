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

TEST_F(HwLowerToStandardTest, LowersPackedVectorFConvert) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%vec8h = OpTypeCooperativeVectorHW %half %uint_8
%vec8f = OpTypeCooperativeVectorHW %float %uint_8
%x = OpUndef %vec8h
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpFConvert %vec8f %x
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFConvert %v2float"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFConvert %float"));
}

TEST_F(HwLowerToStandardTest, LowersVectorUConvertScalarFallback) {
  const std::string text = R"(
OpCapability Shader
OpCapability Int16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%ushort = OpTypeInt 16 0
%uvec8h = OpTypeCooperativeVectorHW %ushort %uint_8
%uvec8 = OpTypeCooperativeVectorHW %uint %uint_8
%x = OpUndef %uvec8h
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpUConvert %uvec8 %x
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpUConvert %uint"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %uint 4"));
}

TEST_F(HwLowerToStandardTest, LowersPackedVectorFAdd) {
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
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%a = OpUndef %vec8
%b = OpUndef %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%c = OpFAdd %vec8 %a %b
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v2float"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFAdd %float"));
}

TEST_F(HwLowerToStandardTest, LowersPackedVectorTimesScalar) {
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
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%float_2 = OpConstant %float 2
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%x = OpUndef %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpVectorTimesScalar %vec8 %x %float_2
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpVectorTimesScalar"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFMul %v2float"));
}

TEST_F(HwLowerToStandardTest, LowersVectorIMulScalarFallback) {
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%vec8 = OpTypeCooperativeVectorHW %int %uint_8
%a = OpUndef %vec8
%b = OpUndef %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpIMul %vec8 %a %b
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpIMul %int"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFMul %int"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 2"));
}

TEST_F(HwLowerToStandardTest, LowersVectorConvertFToSWithPackedInput) {
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%fvec8 = OpTypeCooperativeVectorHW %float %uint_8
%ivec8 = OpTypeCooperativeVectorHW %int %uint_8
%x = OpUndef %fvec8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpConvertFToS %ivec8 %x
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpConvertFToS %int"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpVectorExtractDynamic %float"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 2"));
}

TEST_F(HwLowerToStandardTest, LowersVectorSNegateScalarFallback) {
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%vec8 = OpTypeCooperativeVectorHW %int %uint_8
%x = OpUndef %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpSNegate %vec8 %x
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpSNegate %int"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 2"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixIAdd) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%int = OpTypeInt 32 1
%mat2x5 = OpTypeCooperativeMatrixHW %int %uint_2 %uint_5
%a = OpUndef %mat2x5
%b = OpUndef %mat2x5
%main = OpFunction %void None %fn
%entry = OpLabel
%c = OpIAdd %mat2x5 %a %b
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpIAdd %int"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 2"));
}

TEST_F(HwLowerToStandardTest, LowersPackedMatrixFConvert) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%mat2x8h = OpTypeCooperativeMatrixHW %half %uint_2 %uint_8
%mat2x8f = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%x = OpUndef %mat2x8h
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpFConvert %mat2x8f %x
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFConvert %v2float"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFConvert %float"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixConvertFToSScalarFallback) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%mat2x5f = OpTypeCooperativeMatrixHW %float %uint_2 %uint_5
%mat2x5i = OpTypeCooperativeMatrixHW %int %uint_2 %uint_5
%x = OpUndef %mat2x5f
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpConvertFToS %mat2x5i %x
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpConvertFToS %int"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 2"));
}

TEST_F(HwLowerToStandardTest, LowersPackedMatrixFAdd) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%mat2x8 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%a = OpUndef %mat2x8
%b = OpUndef %mat2x8
%main = OpFunction %void None %fn
%entry = OpLabel
%c = OpFAdd %mat2x8 %a %b
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v2float"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFAdd %float"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixTimesScalarScalarFallback) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%float_2 = OpConstant %float 2
%mat2x5 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_5
%x = OpUndef %mat2x5
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpMatrixTimesScalar %mat2x5 %x %float_2
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpMatrixTimesScalar"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFMul %float"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %float 2"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixIMulScalarFallback) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%mat2x8 = OpTypeCooperativeMatrixHW %int %uint_2 %uint_8
%a = OpUndef %mat2x8
%b = OpUndef %mat2x8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpIMul %mat2x8 %a %b
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpIMul %int"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFMul %int"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 2"));
}

TEST_F(HwLowerToStandardTest, ForceScalarModeLowersVectorFAddToScalar) {
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
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%a = OpUndef %vec8
%b = OpUndef %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%c = OpFAdd %vec8 %a %b
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %float"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %float 2"));
}

TEST_F(HwLowerToStandardTest, LowersScalarToPackedVectorConversionWithLoop) {
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%ivec8 = OpTypeCooperativeVectorHW %int %uint_8
%fvec8 = OpTypeCooperativeVectorHW %float %uint_8
%x = OpUndef %ivec8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpConvertSToF %fvec8 %x
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpConvertSToF %v2float"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeVector %int 2"));
}

TEST_F(HwLowerToStandardTest, LowersPackedToScalarMatrixConversionWithLoop) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%fmat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%imat = OpTypeCooperativeMatrixHW %int %uint_2 %uint_8
%x = OpUndef %fmat
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpConvertFToS %imat %x
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpConvertFToS %int"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpVectorExtractDynamic %float"));
}

TEST_F(HwLowerToStandardTest, LowersCooperativeVectorExtInstWithLoop) {
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
%y = OpExtInst %vec8 %glsl FMax %zero %zero
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpExtInst %v2float"));
}

TEST_F(HwLowerToStandardTest, LowersChainedElementwiseOpsToSeparateLoops) {
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
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%a = OpUndef %vec8
%b = OpUndef %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%sum = OpFAdd %vec8 %a %b
%neg = OpFNegate %vec8 %sum
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpULessThan"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v2float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFNegate %v2float"));
}

TEST_F(HwLowerToStandardTest, PreservesEnclosingLoopHeaderWhenSplitting) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%false = OpConstantFalse %bool
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%x = OpUndef %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %outer_header
%outer_header = OpLabel
%y = OpFAdd %vec8 %x %x
OpLoopMerge %outer_merge %outer_continue None
OpBranchConditional %false %outer_body %outer_merge
%outer_body = OpLabel
OpBranch %outer_continue
%outer_continue = OpLabel
OpBranch %outer_header
%outer_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v2float"));
}

TEST_F(HwLowerToStandardTest, LowersIntegerCooperativeVectorBitwiseOps) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%vec = OpTypeCooperativeVectorHW %uint %uint_5
%a = OpUndef %vec
%b = OpUndef %vec
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%logical = OpShiftRightLogical %vec %a %b
%arithmetic = OpShiftRightArithmetic %vec %logical %b
%left = OpShiftLeftLogical %vec %arithmetic %b
%or = OpBitwiseOr %vec %left %a
%xor = OpBitwiseXor %vec %or %b
%and = OpBitwiseAnd %vec %xor %a
%not = OpNot %vec %and
%element = OpCompositeExtract %uint %not 2
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %uint %uint_5"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpShiftRightLogical %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpShiftRightArithmetic %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpShiftLeftLogical %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBitwiseOr %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBitwiseXor %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBitwiseAnd %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpNot %uint"));
}

TEST_F(HwLowerToStandardTest, LowersMixedWidthFloatVectorArithmeticPacked) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%hvec = OpTypeCooperativeVectorHW %half %uint_8
%fvec = OpTypeCooperativeVectorHW %float %uint_8
%h = OpUndef %hvec
%f = OpUndef %fvec
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%add = OpFAdd %fvec %h %f
%sub = OpFSub %hvec %f %h
%mul = OpFMul %fvec %h %h
%div = OpFDiv %hvec %f %f
%negate = OpFNegate %fvec %h
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(5u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(7u, CountSubstring(disassembly, "OpFConvert %v2"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v2float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFSub %v2half"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFMul %v2float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFDiv %v2half"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFNegate %v2float"));
}

TEST_F(HwLowerToStandardTest, LowersMixedWidthFloatMatrixArithmeticScalar) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%hmat = OpTypeCooperativeMatrixHW %half %uint_2 %uint_3
%fmat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_3
%h = OpUndef %hmat
%f = OpUndef %fmat
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%add = OpFAdd %fmat %h %f
%negate = OpFNegate %hmat %f
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFConvert %float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFConvert %half"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFNegate %half"));
}

TEST_F(HwLowerToStandardTest, LowersMixedSignednessIntegerVectorArithmetic) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%uint_5 = OpConstant %uint 5
%uvec = OpTypeCooperativeVectorHW %uint %uint_5
%svec = OpTypeCooperativeVectorHW %int %uint_5
%u = OpUndef %uvec
%s = OpUndef %svec
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%add = OpIAdd %uvec %s %u
%sub = OpISub %svec %u %s
%mul = OpIMul %uvec %s %s
%div = OpSDiv %svec %u %s
%negate = OpSNegate %uvec %s
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(5u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_GT(CountSubstring(disassembly, "OpIAdd"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpISub"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpIMul"), 0u);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpSDiv %int"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpSNegate %uint"));
}

TEST_F(HwLowerToStandardTest, LowersMixedSignednessIntegerMatrixArithmetic) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%umat = OpTypeCooperativeMatrixHW %uint %uint_2 %uint_3
%smat = OpTypeCooperativeMatrixHW %int %uint_2 %uint_3
%u = OpUndef %umat
%s = OpUndef %smat
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%add = OpIAdd %umat %s %u
%div = OpSDiv %smat %u %s
%negate = OpSNegate %umat %s
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(3u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_GT(CountSubstring(disassembly, "OpIAdd"), 0u);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpSDiv %int"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpSNegate %uint"));
}

TEST_F(HwLowerToStandardTest, LowersMixedShiftWidthAndSignedness) {
  const std::string text = R"(
OpCapability Shader
OpCapability Int8
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%char = OpTypeInt 8 1
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%uint_5 = OpConstant %uint 5
%shift_vec = OpTypeCooperativeVectorHW %char %uint_5
%uvec = OpTypeCooperativeVectorHW %uint %uint_5
%svec = OpTypeCooperativeVectorHW %int %uint_5
%shift = OpUndef %shift_vec
%u = OpUndef %uvec
%s = OpUndef %svec
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%logical = OpShiftRightLogical %uvec %s %shift
%arithmetic = OpShiftRightArithmetic %svec %u %shift
%left = OpShiftLeftLogical %uvec %s %shift
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(3u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpShiftRightLogical %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpShiftRightArithmetic %int"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpShiftLeftLogical %uint"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %char %uint_5"));
}

TEST_F(HwLowerToStandardTest, LowersMixedSignednessBitwiseOps) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%uint_5 = OpConstant %uint 5
%uvec = OpTypeCooperativeVectorHW %uint %uint_5
%svec = OpTypeCooperativeVectorHW %int %uint_5
%u = OpUndef %uvec
%s = OpUndef %svec
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%or = OpBitwiseOr %uvec %s %u
%xor = OpBitwiseXor %svec %u %s
%and = OpBitwiseAnd %uvec %s %s
%not = OpNot %svec %u
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(4u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBitwiseOr %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBitwiseXor %int"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBitwiseAnd %uint"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpNot %int"));
}

TEST_F(HwLowerToStandardTest, LowersIntegerExtInstAcrossSignedness) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%uint_5 = OpConstant %uint 5
%uvec = OpTypeCooperativeVectorHW %uint %uint_5
%svec = OpTypeCooperativeVectorHW %int %uint_5
%u = OpUndef %uvec
%s = OpUndef %svec
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%umin = OpExtInst %svec %glsl UMin %u %s
%umax = OpExtInst %uvec %glsl UMax %s %u
%uclamp = OpExtInst %svec %glsl UClamp %u %s %u
%smin = OpExtInst %uvec %glsl SMin %s %u
%smax = OpExtInst %svec %glsl SMax %u %s
%sclamp = OpExtInst %uvec %glsl SClamp %s %u %s
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(6u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(disassembly, " UMin "));
  EXPECT_EQ(1u, CountSubstring(disassembly, " UMax "));
  EXPECT_EQ(1u, CountSubstring(disassembly, " UClamp "));
  EXPECT_EQ(1u, CountSubstring(disassembly, " SMin "));
  EXPECT_EQ(1u, CountSubstring(disassembly, " SMax "));
  EXPECT_EQ(1u, CountSubstring(disassembly, " SClamp "));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
