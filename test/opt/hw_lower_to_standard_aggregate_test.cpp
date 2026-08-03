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

TEST_F(HwLowerToStandardTest,
       PreservesCopyLogicalBetweenDistinctNestedCooperativeTypes) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%vec = OpTypeCooperativeVectorHW %float %uint_4
%struct_a = OpTypeStruct %vec
%struct_b = OpTypeStruct %vec
%source = OpUndef %struct_a
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%copy = OpCopyLogical %struct_b %source
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_4);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  SetTargetEnv(SPV_ENV_UNIVERSAL_1_4);
  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCopyLogical"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCopyObject"));
}

TEST_F(HwLowerToStandardTest, CompatibleHwBitcastRemainsLoopFreeCopy) {
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
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat_a = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4 MatrixUseAHW
%mat_b = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4 MatrixUseBHW
%a = OpUndef %mat_a
%main = OpFunction %void None %fn
%entry = OpLabel
%b = OpBitcast %mat_b %a
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCopyObject"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpBitcast"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
}

TEST_F(HwLowerToStandardTest, LowersPackedFloatToScalarUintCooperativeBitcast) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
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
%fmat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%umat = OpTypeCooperativeMatrixHW %uint %uint_2 %uint_8
%fvec = OpTypeCooperativeVectorHW %float %uint_8
%uvec = OpTypeCooperativeVectorHW %uint %uint_8
%matrix_value = OpUndef %fmat
%vector_value = OpUndef %fvec
%main = OpFunction %void None %fn
%entry = OpLabel
%matrix_bits = OpBitcast %umat %matrix_value
%vector_bits = OpBitcast %uvec %vector_value
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpTypeArray %v4float %uint_4"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %uint %uint_16"));
  EXPECT_NE(std::string::npos,
            disassembly.find("OpTypeArray %v4float %uint_2"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %uint %uint_8"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpBitcast %uint"));
}

TEST_F(HwLowerToStandardTest, SynthesizesArrayLengthConstant) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK: OpConstant %uint 6
; CHECK: OpTypeArray %float
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_3
%undef = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
}

TEST_F(HwLowerToStandardTest,
       LowerConstantCompositePackedVectorUsesLatestScalarDefinitionOrder) {
  const std::string text = R"(
; CHECK-NOT: HW
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
%vec16 = OpTypeCooperativeVectorHW %float %uint_16
%float_n0_75 = OpConstant %float -0.75
%float_n0_5 = OpConstant %float -0.5
%float_n0_25 = OpConstant %float -0.25
%float_0 = OpConstant %float 0
%float_0_25 = OpConstant %float 0.25
%float_0_5 = OpConstant %float 0.5
%float_0_75 = OpConstant %float 0.75
%bias = OpConstantComposite %vec16
    %float_n0_75 %float_n0_5 %float_n0_25 %float_0
    %float_0_25 %float_0_5 %float_0_75 %float_n0_75
    %float_n0_5 %float_n0_25 %float_0 %float_0_25
    %float_0_5 %float_0_75 %float_n0_75 %float_n0_5
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(4u, CountSubstring(disassembly, "OpConstantComposite %v4float"));
}

TEST_F(HwLowerToStandardTest,
       LowerConstantCompositeReplicatePackedVectorForRelu) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: OpConstantCompositeReplicateEXT
; CHECK: OpConstantComposite %v4half
; CHECK: OpExtInst %v4half
OpCapability Shader
OpCapability Float16
OpCapability ReplicatedCompositesEXT
OpCapability CooperativeVectorHW
OpExtension "SPV_EXT_replicated_composites"
OpExtension "SPV_HW_neural_shader"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%half_0 = OpConstant %half 0
%vec8 = OpTypeCooperativeVectorHW %half %uint_8
%zero = OpConstantCompositeReplicateEXT %vec8 %half_0
%main = OpFunction %void None %fn
%entry = OpLabel
%relu = OpExtInst %vec8 %1 FMax %zero %zero
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpConstantCompositeReplicateEXT"));
  EXPECT_GT(CountSubstring(disassembly, "OpConstantComposite %v4half"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v4half"), 0u);
}

TEST_F(HwLowerToStandardTest, LowerCompositeConstructPackedF16Matrix) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeVector %half 4
; CHECK: OpTypeArray %v4half %uint_4
; CHECK: OpCompositeConstruct %v4half {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: OpCompositeConstruct %v4half {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: OpCompositeConstruct %v4half {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: OpCompositeConstruct %v4half {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%h0 = OpUndef %half
%h1 = OpUndef %half
%h2 = OpUndef %half
%h3 = OpUndef %half
%h4 = OpUndef %half
%h5 = OpUndef %half
%h6 = OpUndef %half
%h7 = OpUndef %half
%h8 = OpUndef %half
%h9 = OpUndef %half
%h10 = OpUndef %half
%h11 = OpUndef %half
%h12 = OpUndef %half
%h13 = OpUndef %half
%h14 = OpUndef %half
%h15 = OpUndef %half
%mat2x8 = OpTypeCooperativeMatrixHW %half %uint_2 %uint_8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%m = OpCompositeConstruct %mat2x8 %h0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeConstructPackedF32Matrix) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_4
; CHECK: OpCompositeConstruct %v4float {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: OpCompositeConstruct %v4float {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%f0 = OpUndef %float
%f1 = OpUndef %float
%f2 = OpUndef %float
%f3 = OpUndef %float
%f4 = OpUndef %float
%f5 = OpUndef %float
%f6 = OpUndef %float
%f7 = OpUndef %float
%f8 = OpUndef %float
%f9 = OpUndef %float
%f10 = OpUndef %float
%f11 = OpUndef %float
%f12 = OpUndef %float
%f13 = OpUndef %float
%f14 = OpUndef %float
%f15 = OpUndef %float
%mat2x8 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%m = OpCompositeConstruct %mat2x8 %f0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeConstructScalarF32Matrix) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %float %uint_15
; CHECK: {{%\w+}} = OpCompositeConstruct %_arr_float_uint_15 {{%\w+}} {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_5 = OpConstant %uint 5
%uint_15 = OpConstant %uint 15
%float = OpTypeFloat 32
%f0 = OpUndef %float
%f1 = OpUndef %float
%f2 = OpUndef %float
%f3 = OpUndef %float
%f4 = OpUndef %float
%f5 = OpUndef %float
%f6 = OpUndef %float
%f7 = OpUndef %float
%f8 = OpUndef %float
%f9 = OpUndef %float
%f10 = OpUndef %float
%f11 = OpUndef %float
%f12 = OpUndef %float
%f13 = OpUndef %float
%f14 = OpUndef %float
%mat3x5 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%m = OpCompositeConstruct %mat3x5 %f0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeConstructPackedF16Vector) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeVector %half 4
; CHECK: OpTypeArray %v4half %uint_2
; CHECK: OpCompositeConstruct %v4half {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: OpCompositeConstruct %v4half {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%h0 = OpUndef %half
%h1 = OpUndef %half
%h2 = OpUndef %half
%h3 = OpUndef %half
%h4 = OpUndef %half
%h5 = OpUndef %half
%h6 = OpUndef %half
%h7 = OpUndef %half
%vec8 = OpTypeCooperativeVectorHW %half %uint_8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%v = OpCompositeConstruct %vec8 %h0 %h1 %h2 %h3 %h4 %h5 %h6 %h7
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeConstructPackedF32Vector) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_2
; CHECK: OpCompositeConstruct %v4float {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: OpCompositeConstruct %v4float {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%f0 = OpUndef %float
%f1 = OpUndef %float
%f2 = OpUndef %float
%f3 = OpUndef %float
%f4 = OpUndef %float
%f5 = OpUndef %float
%f6 = OpUndef %float
%f7 = OpUndef %float
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%v = OpCompositeConstruct %vec8 %f0 %f1 %f2 %f3 %f4 %f5 %f6 %f7
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeConstructScalarF32Vector) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %float %uint_5
; CHECK: {{%\w+}} = OpCompositeConstruct %_arr_float_uint_5 {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%f0 = OpUndef %float
%f1 = OpUndef %float
%f2 = OpUndef %float
%f3 = OpUndef %float
%f4 = OpUndef %float
%vec5 = OpTypeCooperativeVectorHW %float %uint_5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%v = OpCompositeConstruct %vec5 %f0 %f1 %f2 %f3 %f4
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest,
       LowersBroadcastMatrixCompositeConstructPackedAndScalar) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_5 = OpConstant %uint 5
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%one = OpConstant %float 1
%packed_mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%scalar_mat = OpTypeCooperativeMatrixHW %float %uint_3 %uint_5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%packed = OpCompositeConstruct %packed_mat %one
%scalar = OpCompositeConstruct %scalar_mat %one
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpTypeArray %v4float %uint_4"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %float %uint_15"));
  EXPECT_EQ(4u, CountSubstring(disassembly, "OpCompositeConstruct %v4float"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpCompositeConstruct %_arr_v4float_uint_4"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpCompositeConstruct %_arr_float_uint_15"));
}

TEST_F(HwLowerToStandardTest, LowersVectorReplicateInPackedAndScalarLoopPaths) {
  const std::string text = R"(
OpCapability Shader
OpCapability ReplicatedCompositesEXT
OpCapability CooperativeVectorHW
OpExtension "SPV_EXT_replicated_composites"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%broadcast = OpCompositeConstructReplicateEXT %vec8 %one
%last = OpCompositeExtract %float %broadcast 7
OpReturn
OpFunctionEnd
)";

  auto unrolled = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u, 1024ull,
      8u, 1024ull);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(unrolled));
  ExpectNoHwOrCoopMatrix(std::get<0>(unrolled));
  EXPECT_EQ(0u, CountSubstring(std::get<0>(unrolled), "OpLoopMerge"));
  EXPECT_EQ(2u, CountSubstring(std::get<0>(unrolled),
                               "OpCompositeConstruct %v4float"));

  auto looped = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kForceScalar,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u, 1024ull,
      4u, 1024ull);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(looped));
  ExpectNoHwOrCoopMatrix(std::get<0>(looped));
  EXPECT_EQ(1u, CountSubstring(std::get<0>(looped), "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(std::get<0>(looped), "OpULessThan"));
  EXPECT_EQ(1u, CountSubstring(std::get<0>(looped), "OpBranchConditional"));
  EXPECT_GE(CountSubstring(std::get<0>(looped), "OpAccessChain"), 1u);
  EXPECT_NE(std::string::npos,
            std::get<0>(looped).find("OpTypeArray %float %uint_8"));
  EXPECT_EQ(std::string::npos,
            std::get<0>(looped).find("OpTypeVector %float 4"));
}

TEST_F(HwLowerToStandardTest,
       LowersCooperativeMatrixCompositeConstructReplicate) {
  const std::string text = R"(
OpCapability Shader
OpCapability ReplicatedCompositesEXT
OpCapability CooperativeMatrixHW
OpExtension "SPV_EXT_replicated_composites"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%one = OpConstant %float 1
%mat2x4 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%broadcast = OpCompositeConstructReplicateEXT %mat2x4 %one
%last = OpCompositeExtract %float %broadcast 1 3
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpCompositeConstructReplicateEXT"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpCompositeConstruct %_arr_float_uint_8"));
}

TEST_F(HwLowerToStandardTest,
       LowersMixedScalarAndOrdinaryVectorCompositeConstruct) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%f0 = OpUndef %float
%f1 = OpUndef %float
%f2 = OpUndef %float
%f3 = OpUndef %float
%f4 = OpUndef %float
%f5 = OpUndef %float
%f6 = OpUndef %float
%f7 = OpUndef %float
%v3float = OpTypeVector %float 3
%v4float = OpTypeVector %float 4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%three = OpCompositeConstruct %v3float %f1 %f2 %f3
%four = OpCompositeConstruct %v4float %f4 %f5 %f6 %f7
%mixed = OpCompositeConstruct %vec8 %f0 %three %four
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpTypeArray %v4float %uint_2"));
  EXPECT_EQ(7u, CountSubstring(disassembly, "OpCompositeExtract %float"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpCompositeConstruct %_arr_v4float_uint_2"));
}

TEST_F(HwLowerToStandardTest,
       LowersSingleOrdinaryVectorConstituentWithoutBroadcastLoop) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%ordinary = OpUndef %v4float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%copy = OpCompositeConstruct %vec4 %ordinary
%last = OpCompositeExtract %float %copy 3
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u, 1024ull,
      2u, 1024ull);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpCompositeConstruct %_arr_v4float_uint_1"));
}

TEST_F(HwLowerToStandardTest,
       LowersPackedMatrixAndVectorSpecConstantComposite) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%s0 = OpSpecConstant %float 0
%s1 = OpSpecConstant %float 1
%s2 = OpSpecConstant %float 2
%s3 = OpSpecConstant %float 3
%s4 = OpSpecConstant %float 4
%s5 = OpSpecConstant %float 5
%s6 = OpSpecConstant %float 6
%s7 = OpSpecConstant %float 7
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%vec = OpTypeCooperativeVectorHW %float %uint_8
%matrix_value = OpSpecConstantComposite %mat %s0
%vector_value = OpSpecConstantComposite %vec %s0 %s1 %s2 %s3 %s4 %s5 %s6 %s7
%void = OpTypeVoid
%fn = OpTypeFunction %void
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
  // The four identical matrix packs share one specialized vec4 constant;
  // the vector contributes two distinct packs.
  EXPECT_EQ(3u,
            CountSubstring(disassembly, "OpSpecConstantComposite %v4float"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpSpecConstantComposite %_arr_v4float_uint_4"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpSpecConstantComposite %_arr_v4float_uint_2"));
}

TEST_F(HwLowerToStandardTest, DefaultPreferPackedFallsBackToScalarF16Matrix) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %half %uint_10
; CHECK: {{%\w+}} = OpCompositeConstruct %_arr_half_uint_10 {{%\w+}} {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%uint_10 = OpConstant %uint 10
%half = OpTypeFloat 16
%h0 = OpUndef %half
%h1 = OpUndef %half
%h2 = OpUndef %half
%h3 = OpUndef %half
%h4 = OpUndef %half
%h5 = OpUndef %half
%h6 = OpUndef %half
%h7 = OpUndef %half
%h8 = OpUndef %half
%h9 = OpUndef %half
%mat2x5 = OpTypeCooperativeMatrixHW %half %uint_2 %uint_5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%m = OpCompositeConstruct %mat2x5 %h0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %half 4"));
}

TEST_F(HwLowerToStandardTest,
       ForceScalarModeLowersAlignedF16AndF32TypesToScalar) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %half %uint_16
; CHECK: OpTypeArray %float %uint_8
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%uint_16 = OpConstant %uint 16
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%mat2x8 = OpTypeCooperativeMatrixHW %half %uint_2 %uint_8
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%m = OpConstantNull %mat2x8
%v = OpConstantNull %vec8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(
      text, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %half 4"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %float 4"));
}

TEST_F(HwLowerToStandardTest, LowerCompositeExtractPackedF16Matrix) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: {{%\w+}} = OpCompositeExtract %half {{%\w+}} 7 2
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%mat8x8 = OpTypeCooperativeMatrixHW %half %uint_8 %uint_8
%m = OpUndef %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%x = OpCompositeExtract %half %m 3 6
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeExtractPackedF32Matrix) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: {{%\w+}} = OpCompositeExtract %float {{%\w+}} 7 2
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%m = OpUndef %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%x = OpCompositeExtract %float %m 3 6
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeExtractScalarF32Matrix) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: {{%\w+}} = OpCompositeExtract %float {{%\w+}} 14
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%mat3x5 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_5
%m = OpUndef %mat3x5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%x = OpCompositeExtract %float %m 2 4
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeExtractPackedF16Vector) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: {{%\w+}} = OpCompositeExtract %half {{%\w+}} 1 2
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%vec8 = OpTypeCooperativeVectorHW %half %uint_8
%v = OpUndef %vec8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%x = OpCompositeExtract %half %v 6
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeExtractPackedF32Vector) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: {{%\w+}} = OpCompositeExtract %float {{%\w+}} 1 2
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%v = OpUndef %vec8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%x = OpCompositeExtract %float %v 6
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerCompositeExtractScalarF32Vector) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: {{%\w+}} = OpCompositeExtract %float {{%\w+}} 4
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%vec5 = OpTypeCooperativeVectorHW %float %uint_5
%v = OpUndef %vec5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%x = OpCompositeExtract %float %v 4
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest,
       DefaultPreferPackedFallsBackToScalarF16ExtractAndNull) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %half %uint_10
; CHECK: {{%\w+}} = OpConstantNull %_arr_half_uint_10
; CHECK: {{%\w+}} = OpCompositeExtract %half {{%\w+}} 9
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%uint_10 = OpConstant %uint 10
%half = OpTypeFloat 16
%mat2x5 = OpTypeCooperativeMatrixHW %half %uint_2 %uint_5
%m = OpConstantNull %mat2x5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%x = OpCompositeExtract %half %m 1 4
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %half 4"));
}

TEST_F(HwLowerToStandardTest, LowerConstantNullPackedMatrix) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: {{%\w+}} = OpConstantNull %_arr_v4half_uint_16
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%uint_16 = OpConstant %uint 16
%half = OpTypeFloat 16
%mat8x8 = OpTypeCooperativeMatrixHW %half %uint_8 %uint_8
%m = OpConstantNull %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowerUndefPackedVector) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: {{%\w+}} = OpUndef %_arr_v4half_uint_2
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%vec8 = OpTypeCooperativeVectorHW %half %uint_8
%v = OpUndef %vec8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowersConstantCompositeWithUndefConstituents) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%undef = OpUndef %float
%mat1x4 = OpTypeCooperativeMatrixHW %float %uint_1 %uint_4
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%matrix_value = OpConstantComposite %mat1x4 %undef
%vector_value = OpConstantComposite %vec4 %undef %undef %undef %undef
%void = OpTypeVoid
%fn = OpTypeFunction %void
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpConstantComposite %v4float"));
  EXPECT_EQ(2u, CountSubstring(disassembly,
                               "OpConstantComposite %_arr_v4float_uint_1"));
}

TEST_F(HwLowerToStandardTest,
       LowersReplicatedConstantAtCompositeConstituentLimit) {
  struct Case {
    const char* opcode;
    const char* constituent;
    const char* lowered_opcode;
  };
  const std::vector<Case> cases = {
      {"OpConstantCompositeReplicateEXT", "OpConstant %float 1",
       "OpConstantComposite"},
      {"OpSpecConstantCompositeReplicateEXT",
       "OpSpecConstant %float 1065353216", "OpSpecConstantComposite"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.opcode);
    const std::string text = std::string(R"(
OpCapability Shader
OpCapability ReplicatedCompositesEXT
OpCapability CooperativeVectorHW
OpExtension "SPV_EXT_replicated_composites"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_65532 = OpConstant %uint 65532
%float = OpTypeFloat 32
%one = )") + test_case.constituent +
                             R"(
%vec65532 = OpTypeCooperativeVectorHW %float %uint_65532
%value = )" + test_case.opcode +
                             R"( %vec65532 %one
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

    auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
        text, true, true, HwLowerToStandardPass::LoweringMode::kForceScalar,
        HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 70000u,
        16777216ull, 65532u, 4096ull);
    const std::string& disassembly = std::get<0>(result);
    EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
    ExpectNoHwOrCoopMatrix(disassembly);
    EXPECT_EQ(0u, CountSubstring(disassembly, test_case.opcode));
    EXPECT_EQ(
        1u, CountSubstring(disassembly, std::string(test_case.lowered_opcode) +
                                            " %_arr_float_uint_65532"));
  }
}

TEST_F(HwLowerToStandardTest,
       LowersPackedCooperativeVectorSpecConstantCompositeReplicate) {
  const std::string text = R"(
OpCapability Shader
OpCapability ReplicatedCompositesEXT
OpCapability CooperativeVectorHW
OpExtension "SPV_EXT_replicated_composites"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%one = OpSpecConstant %float 1065353216
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%value = OpSpecConstantCompositeReplicateEXT %vec8 %one
%void = OpTypeVoid
%fn = OpTypeFunction %void
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
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpSpecConstantCompositeReplicateEXT"));
  EXPECT_EQ(1u,
            CountSubstring(disassembly, "OpSpecConstantComposite %v4float"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpSpecConstantComposite %_arr_v4float_uint_2"));
}

TEST_F(HwLowerToStandardTest, LowersHwFunctionParameterAndCall) {
  const std::string text = R"(
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
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%fn_mat = OpTypeFunction %void %mat
%helper = OpFunction %void None %fn_mat
%param = OpFunctionParameter %mat
%helper_entry = OpLabel
%copy = OpCopyObject %mat %param
OpReturn
OpFunctionEnd
%main = OpFunction %void None %fn
%entry = OpLabel
%call = OpFunctionCall %void %helper %a
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpFunctionParameter %_arr_v4float_uint_4"));
  EXPECT_NE(std::string::npos, disassembly.find("OpFunctionCall %void"));
}

TEST_F(HwLowerToStandardTest,
       DeduplicatesFunctionTypesAfterCooperativeTypeReplacement) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%vec = OpTypeCooperativeVectorHW %float %uint_5
%array = OpTypeArray %float %uint_5
%vec_fn = OpTypeFunction %vec %vec
%array_fn = OpTypeFunction %array %array
%void_fn = OpTypeFunction %void
%vec_value = OpUndef %vec
%array_value = OpUndef %array
%vec_helper = OpFunction %vec None %vec_fn
%vec_param = OpFunctionParameter %vec
%vec_entry = OpLabel
OpReturnValue %vec_param
OpFunctionEnd
%array_helper = OpFunction %array None %array_fn
%array_param = OpFunctionParameter %array
%array_entry = OpLabel
OpReturnValue %array_param
OpFunctionEnd
%main = OpFunction %void None %void_fn
%entry = OpLabel
%vec_call = OpFunctionCall %vec %vec_helper %vec_value
%array_call = OpFunctionCall %array %array_helper %array_value
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  SetTargetEnv(SPV_ENV_UNIVERSAL_1_3);
  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(
                    disassembly,
                    "OpTypeFunction %_arr_float_uint_5 %_arr_float_uint_5"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpTypeFunction"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpFunctionParameter"));
}

TEST_F(HwLowerToStandardTest, LowersNestedHwFunctionReturnAndFunctionPointer) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_5
%mat_array = OpTypeArray %mat %uint_2
%Payload = OpTypeStruct %mat_array %uint
%_ptr_Function_Payload = OpTypePointer Function %Payload
%void = OpTypeVoid
%main_fn = OpTypeFunction %void
%helper_fn = OpTypeFunction %Payload %Payload %_ptr_Function_Payload
%helper = OpFunction %Payload None %helper_fn
%value = OpFunctionParameter %Payload
%destination = OpFunctionParameter %_ptr_Function_Payload
%helper_entry = OpLabel
%old = OpLoad %Payload %destination
OpStore %destination %value
OpReturnValue %old
OpFunctionEnd
%main = OpFunction %void None %main_fn
%entry = OpLabel
%scratch = OpVariable %_ptr_Function_Payload Function
%m = OpUndef %mat
%mat_pair = OpCompositeConstruct %mat_array %m %m
%payload = OpCompositeConstruct %Payload %mat_pair %uint_0
OpStore %scratch %payload
%call = OpFunctionCall %Payload %helper %payload %scratch
%out = OpCompositeExtract %mat %call 0 1
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpFunctionParameter"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypePointer Function"));
  EXPECT_NE(std::string::npos, disassembly.find("OpFunctionCall"));
  EXPECT_NE(std::string::npos, disassembly.find("OpReturnValue"));
}

TEST_F(HwLowerToStandardTest, LowersLoopCarriedHwPhi) {
  const std::string text = R"(
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
%a = OpUndef %mat
%bool = OpTypeBool
%false = OpConstantFalse %bool
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %header
%header = OpLabel
%p = OpPhi %mat %a %entry %next %continue
OpLoopMerge %merge %continue None
OpBranchConditional %false %body %merge
%body = OpLabel
%sum = OpFAdd %mat %p %a
OpBranch %continue
%continue = OpLabel
%next = OpCopyObject %mat %sum
OpBranch %header
%merge = OpLabel
%first = OpCompositeExtract %float %p 0 0
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpPhi %_arr_v4float_uint_4"));
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 1u);
  EXPECT_GT(CountSubstring(disassembly, "OpFAdd %v4float"), 0u);
}

TEST_F(HwLowerToStandardTest,
       LowersNestedPrivateCooperativeVariableAndPointerParameter) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%vec = OpTypeCooperativeVectorHW %float %uint_5
%vec_array = OpTypeArray %vec %uint_2
%Payload = OpTypeStruct %vec_array %uint
%_ptr_Private_Payload = OpTypePointer Private %Payload
%Payload_null = OpConstantNull %Payload
%global = OpVariable %_ptr_Private_Payload Private %Payload_null
%helper_fn = OpTypeFunction %Payload %_ptr_Private_Payload
%main_fn = OpTypeFunction %void
%helper = OpFunction %Payload None %helper_fn
%pointer = OpFunctionParameter %_ptr_Private_Payload
%helper_entry = OpLabel
%loaded = OpLoad %Payload %pointer
%value = OpCompositeExtract %vec %loaded 0 1
%doubled = OpFAdd %vec %value %value
%updated = OpCompositeInsert %Payload %doubled %loaded 0 1
OpStore %pointer %updated
OpReturnValue %updated
OpFunctionEnd
%main = OpFunction %void None %main_fn
%entry = OpLabel
%result = OpFunctionCall %Payload %helper %global
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  SetTargetEnv(SPV_ENV_UNIVERSAL_1_3);
  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeArray"), 1u);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeStruct"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpTypePointer Private"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpConstantNull"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpVariable"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpFunctionParameter"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpLoad"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpStore"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpFunctionCall"), 0u);
}

TEST_F(HwLowerToStandardTest, LowersSelectOnHwValue) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_5
%a = OpUndef %mat
%b = OpUndef %mat
%bool = OpTypeBool
%true = OpConstantTrue %bool
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%selected = OpSelect %mat %true %a %b
%element = OpCompositeExtract %float %selected 1 2
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpSelect"), 0u);
}

TEST_F(HwLowerToStandardTest,
       PackedSelectLoopUsesHoistedVectorConditionInSpirv13) {
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
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%uint_7 = OpConstant %uint 7
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec = OpTypeCooperativeVectorHW %float %uint_8
%a = OpUndef %vec
%b = OpUndef %vec
%main = OpFunction %void None %fn
%entry = OpLabel
%selected = OpSelect %vec %true %a %b
%last = OpCompositeExtract %float %selected 7
OpReturn
OpFunctionEnd
)";

  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  SetTargetEnv(SPV_ENV_UNIVERSAL_1_3);
  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u, 1024ull,
      4u, 1024ull);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeVector %bool 4"));
  const size_t splat = disassembly.find("OpCompositeConstruct %v4bool");
  const size_t loop_merge = disassembly.find("OpLoopMerge");
  ASSERT_NE(std::string::npos, splat);
  ASSERT_NE(std::string::npos, loop_merge);
  EXPECT_LT(splat, loop_merge);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpSelect %v4float"));
}

TEST_F(HwLowerToStandardTest, LowersCompositeInsertOnHwValue) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%one = OpConstant %float 1
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_5
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%inserted = OpCompositeInsert %mat %one %a 1 2
%element = OpCompositeExtract %float %inserted 1 2
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCompositeInsert"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %float %uint_10"));
  EXPECT_NE(std::string::npos,
            disassembly.find("OpCompositeInsert %_arr_float_uint_10 %float_1"));
}

TEST_F(HwLowerToStandardTest, LowersCooperativeVectorSsaAndCompositeOps) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec = OpTypeCooperativeVectorHW %float %uint_8
%a = OpUndef %vec
%b = OpUndef %vec
%bool = OpTypeBool
%true = OpConstantTrue %bool
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpSelectionMerge %merge None
OpBranchConditional %true %left %right
%left = OpLabel
OpBranch %merge
%right = OpLabel
OpBranch %merge
%merge = OpLabel
%phi = OpPhi %vec %a %left %b %right
%selected = OpSelect %vec %true %phi %a
%inserted = OpCompositeInsert %vec %one %selected 6
%element = OpCompositeExtract %float %inserted 6
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpTypeArray %v4float %uint_2"));
  EXPECT_GE(CountSubstring(disassembly, "OpPhi %_arr_v4float_uint_2"), 1u);
  EXPECT_GT(CountSubstring(disassembly, "OpSelect"), 0u);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCompositeInsert"));
  EXPECT_NE(
      std::string::npos,
      disassembly.find("OpCompositeInsert %_arr_v4float_uint_2 %float_1"));
}

TEST_F(HwLowerToStandardTest,
       LowersNestedStructArrayCompositePathsIntoHwValues) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%one = OpConstant %float 1
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%vec = OpTypeCooperativeVectorHW %float %uint_8
%mat_array = OpTypeArray %mat %uint_2
%Inner = OpTypeStruct %mat_array %vec
%Outer = OpTypeStruct %uint %Inner
%matrix_value = OpUndef %mat
%vector_value = OpUndef %vec
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%matrices = OpCompositeConstruct %mat_array %matrix_value %matrix_value
%inner_value = OpCompositeConstruct %Inner %matrices %vector_value
%outer_value = OpCompositeConstruct %Outer %uint_0 %inner_value
%matrix_element = OpCompositeExtract %float %outer_value 1 0 1 1 6
%vector_element = OpCompositeExtract %float %outer_value 1 1 6
%matrix_inserted = OpCompositeInsert %Outer %one %outer_value 1 0 0 0 5
%vector_inserted = OpCompositeInsert %Outer %one %matrix_inserted 1 1 5
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCompositeExtract %float"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCompositeInsert"));
  EXPECT_NE(std::string::npos, disassembly.find(" 1 0 1 3 2"));
  EXPECT_NE(std::string::npos, disassembly.find(" 1 1 1 2"));
  EXPECT_NE(std::string::npos, disassembly.find(" 1 0 0 1 1"));
  EXPECT_NE(std::string::npos, disassembly.find(" 1 1 1 1"));
}

TEST_F(HwLowerToStandardTest,
       LowersDynamicFunctionAccessChainIntoPackedHwValues) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%one = OpConstant %float 1
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_8
%vec = OpTypeCooperativeVectorHW %float %uint_8
%_ptr_Function_mat = OpTypePointer Function %mat
%_ptr_Function_vec = OpTypePointer Function %vec
%_ptr_Function_float = OpTypePointer Function %float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%matrix_var = OpVariable %_ptr_Function_mat Function
%vector_var = OpVariable %_ptr_Function_vec Function
%index = OpIAdd %uint %uint_1 %uint_2
%matrix_ptr = OpAccessChain %_ptr_Function_float %matrix_var %index
%vector_ptr = OpAccessChain %_ptr_Function_float %vector_var %index
OpStore %matrix_ptr %one
OpStore %vector_ptr %one
%matrix_element = OpLoad %float %matrix_ptr
%vector_element = OpLoad %float %vector_ptr
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpUDiv %uint"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpUMod %uint"));
  EXPECT_EQ(2u,
            CountSubstring(disassembly, "OpAccessChain %_ptr_Function_float"));
}

TEST_F(HwLowerToStandardTest,
       LowersPackedDynamicInsertAndExtractWithUint64Index) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability Int64
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%ulong = OpTypeInt 64 0
%ulong_0 = OpConstant %ulong 0
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%a = OpUndef %vec4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%inserted = OpVectorInsertDynamic %vec4 %a %one %ulong_0
%extracted = OpVectorExtractDynamic %float %inserted %ulong_0
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpUDiv %ulong"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpUMod %ulong"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpVectorInsertDynamic %v4float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpVectorExtractDynamic %float"));
}

TEST_F(HwLowerToStandardTest,
       DoesNotReuseUnrelatedSpecConstantForLoweredArrayLength) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %spec_six SpecId 0
%uint = OpTypeInt 32 0
%spec_six = OpSpecConstant %uint 6
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_3
%value = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%element = OpCompositeExtract %float %value 0 0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos, disassembly.find("OpSpecConstant %uint 6"));
  EXPECT_NE(std::string::npos, disassembly.find("OpConstant %uint 6"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %float %uint_6"));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
