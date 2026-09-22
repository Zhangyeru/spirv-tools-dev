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

#include <sstream>

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
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(8u, CountSubstring(disassembly, "OpFAdd %float"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %float 2"));
}

TEST_F(HwLowerToStandardTest,
       ForceScalarUnrollsSmallFloatMatrixElementwiseChains) {
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
%uint_5 = OpConstant %uint 5
%uint_7 = OpConstant %uint 7
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_5 %uint_7
%a = OpUndef %mat
%b = OpUndef %mat
%main = OpFunction %void None %fn
%entry = OpLabel
%sum = OpFAdd %mat %a %b
%product = OpFMul %mat %sum %b
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(35u, CountSubstring(disassembly, "OpFAdd %float"));
  EXPECT_EQ(35u, CountSubstring(disassembly, "OpFMul %float"));
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

// Generated from Test/spv.hw_extfree_precise_arithmetic.comp (WIDTH=4).
TEST_F(HwLowerToStandardTest,
       ElementwisePreservesNoContractionWithoutContaminatingRelaxedOps) {
  const std::string text = R"(
OpCapability Shader
OpCapability VulkanMemoryModel
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical Vulkan
OpEntryPoint GLCompute %main "main" %A %B %D
OpExecutionMode %main LocalSize 1 1 1
OpSource GLSL 460
OpSourceExtension "GL_HW_neural_shader"
OpName %main "main"
OpName %tempArg "tempArg"
OpName %InputA "InputA"
OpMemberName %InputA 0 "data"
OpName %A "A"
OpName %a "a"
OpName %tempArg_0 "tempArg"
OpName %InputB "InputB"
OpMemberName %InputB 0 "data"
OpName %B "B"
OpName %b "b"
OpName %tempArg_1 "tempArg"
OpName %c "c"
OpName %strictValue "strictValue"
OpName %strictScale "strictScale"
OpName %relaxed "relaxed"
OpName %Output "Output"
OpMemberName %Output 0 "data"
OpName %D "D"
OpDecorate %_runtimearr_float ArrayStride 4
OpDecorate %InputA Block
OpMemberDecorate %InputA 0 Offset 0
OpDecorate %A Binding 0
OpDecorate %A DescriptorSet 0
OpDecorate %_runtimearr_float_0 ArrayStride 4
OpDecorate %InputB Block
OpMemberDecorate %InputB 0 Offset 0
OpDecorate %B Binding 1
OpDecorate %B DescriptorSet 0
OpDecorate %42 NoContraction
OpDecorate %44 NoContraction
OpDecorate %51 NoContraction
OpDecorate %_runtimearr_float_1 ArrayStride 4
OpDecorate %Output Block
OpMemberDecorate %Output 0 Offset 0
OpDecorate %D Binding 2
OpDecorate %D DescriptorSet 0
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%9 = OpTypeCooperativeVectorHW %float %uint_4
%_ptr_Function_9 = OpTypePointer Function %9
%_runtimearr_float = OpTypeRuntimeArray %float
%InputA = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_InputA = OpTypePointer StorageBuffer %InputA
%A = OpVariable %_ptr_StorageBuffer_InputA StorageBuffer
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%uint_0 = OpConstant %uint 0
%_runtimearr_float_0 = OpTypeRuntimeArray %float
%InputB = OpTypeStruct %_runtimearr_float_0
%_ptr_StorageBuffer_InputB = OpTypePointer StorageBuffer %InputB
%B = OpVariable %_ptr_StorageBuffer_InputB StorageBuffer
%_ptr_StorageBuffer__runtimearr_float_0 = OpTypePointer StorageBuffer %_runtimearr_float_0
%int_4 = OpConstant %int 4
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
%_runtimearr_float_1 = OpTypeRuntimeArray %float
%Output = OpTypeStruct %_runtimearr_float_1
%_ptr_StorageBuffer_Output = OpTypePointer StorageBuffer %Output
%D = OpVariable %_ptr_StorageBuffer_Output StorageBuffer
%_ptr_StorageBuffer__runtimearr_float_1 = OpTypePointer StorageBuffer %_runtimearr_float_1
%v3uint = OpTypeVector %uint 3
%uint_1 = OpConstant %uint 1
%67 = OpConstantComposite %v3uint %uint_1 %uint_1 %uint_1
%main = OpFunction %void None %3
%5 = OpLabel
%tempArg = OpVariable %_ptr_Function_9 Function
%a = OpVariable %_ptr_Function_9 Function
%tempArg_0 = OpVariable %_ptr_Function_9 Function
%b = OpVariable %_ptr_Function_9 Function
%tempArg_1 = OpVariable %_ptr_Function_9 Function
%c = OpVariable %_ptr_Function_9 Function
%strictValue = OpVariable %_ptr_Function_9 Function
%strictScale = OpVariable %_ptr_Function_9 Function
%relaxed = OpVariable %_ptr_Function_9 Function
%19 = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %A %int_0
%21 = OpCooperativeVectorLoadHW %9 %19 %uint_0
OpStore %tempArg %21
%23 = OpLoad %9 %tempArg
OpStore %a %23
%30 = OpAccessChain %_ptr_StorageBuffer__runtimearr_float_0 %B %int_0
%31 = OpCooperativeVectorLoadHW %9 %30 %uint_0
OpStore %tempArg_0 %31
%33 = OpLoad %9 %tempArg_0
OpStore %b %33
%35 = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %A %int_0
%36 = OpCooperativeVectorLoadHW %9 %35 %uint_4
OpStore %tempArg_1 %36
%38 = OpLoad %9 %tempArg_1
OpStore %c %38
%40 = OpLoad %9 %a
%41 = OpLoad %9 %b
%42 = OpFMul %9 %40 %41
%43 = OpLoad %9 %c
%44 = OpFAdd %9 %42 %43
OpStore %strictValue %44
%46 = OpLoad %9 %strictValue
%49 = OpAccessChain %_ptr_StorageBuffer_float %B %int_0 %int_4
%50 = OpLoad %float %49
%51 = OpVectorTimesScalar %9 %46 %50
OpStore %strictScale %51
%53 = OpLoad %9 %a
%54 = OpLoad %9 %b
%55 = OpFSub %9 %53 %54
OpStore %relaxed %55
%56 = OpLoad %9 %strictScale
%62 = OpAccessChain %_ptr_StorageBuffer__runtimearr_float_1 %D %int_0
OpCooperativeVectorStoreHW %62 %uint_0 %56
%63 = OpLoad %9 %relaxed
%64 = OpAccessChain %_ptr_StorageBuffer__runtimearr_float_1 %D %int_0
OpCooperativeVectorStoreHW %64 %uint_4 %63
OpReturn
OpFunctionEnd
)";
  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SpirvTools tools(SPV_ENV_VULKAN_1_3);
  std::vector<uint32_t> binary;
  ASSERT_TRUE(tools.Assemble(text, &binary));
  ASSERT_TRUE(tools.Validate(binary));
  for (auto mode : {HwLowerToStandardPass::LoweringMode::kPreferPackedVec2,
                    HwLowerToStandardPass::LoweringMode::kForceScalar}) {
    for (uint32_t unroll : {2u, 4096u}) {
      auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
          text, true, true, mode,
          HwLowerToStandardPass::CompletenessMode::kExtensionFree, 0u, 0ull,
          unroll, 4096ull);
      ASSERT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
      const auto& out = std::get<0>(result);
      ExpectNoHwOrCoopMatrix(out);
      std::istringstream lines(out);
      std::string line;
      size_t strict_ops = 0, relaxed_ops = 0;
      while (std::getline(lines, line)) {
        const bool strict_op =
            line.find(" OpFMul ") != std::string::npos ||
            line.find(" OpFAdd ") != std::string::npos ||
            line.find(" OpVectorTimesScalar ") != std::string::npos;
        const bool relaxed_op = line.find(" OpFSub ") != std::string::npos;
        if (!strict_op && !relaxed_op) continue;
        std::istringstream tokens(line);
        std::string id;
        tokens >> id;
        const bool decorated = out.find("OpDecorate " + id +
                                        " NoContraction") != std::string::npos;
        EXPECT_EQ(strict_op, decorated) << line;
        strict_ops += strict_op;
        relaxed_ops += relaxed_op;
      }
      EXPECT_GT(strict_ops, 0u);
      EXPECT_GT(relaxed_ops, 0u);
    }
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
