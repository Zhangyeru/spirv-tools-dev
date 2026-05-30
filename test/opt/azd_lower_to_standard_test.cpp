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

#include <string>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using AzdLowerToStandardTest = PassTest<::testing::Test>;

size_t CountSubstring(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

void ExpectNoAzdOrCoopMatrix(const std::string& text) {
  EXPECT_EQ(0u, CountSubstring(text, "AZD"));
  EXPECT_EQ(0u, CountSubstring(text, "CooperativeMatrixKHR"));
  EXPECT_EQ(0u, CountSubstring(text, "OpTypeCooperativeMatrixKHR"));
  EXPECT_EQ(0u, CountSubstring(text, "OpCooperativeMatrix"));
}

void ExpectTiledCounts(const std::string& text, size_t fmul, size_t fadd,
                       size_t extract) {
  ExpectNoAzdOrCoopMatrix(text);
  EXPECT_EQ(fmul, CountSubstring(text, "OpFMul"));
  EXPECT_EQ(fadd, CountSubstring(text, "OpFAdd"));
  EXPECT_EQ(extract, CountSubstring(text, "OpCompositeExtract"));
}

TEST_F(AzdLowerToStandardTest, NonAzdModuleIsUnchanged) {
  const std::string text = R"(
; CHECK: OpCapability Shader
; CHECK-NOT: AZD
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

  auto result = SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(AzdLowerToStandardTest, LowersMatmul4x4x4F32Tiled) {
  const std::string text = R"(
; CHECK-NOT: AZD
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeArray %float %uint_16
; CHECK: OpFMul
; CHECK: OpFAdd
; CHECK: OpCompositeConstruct
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpSourceExtension "GL_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_16 = OpConstant %uint 16
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%a = OpUndef %mat
%b = OpUndef %mat
%c = OpConstantNull %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddAZD %mat %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
  ExpectTiledCounts(std::get<0>(result), 64, 64, 64);
}

TEST_F(AzdLowerToStandardTest, LowersMatmul3x5x4F32TiledTail) {
  const std::string text = R"(
; CHECK-NOT: AZD
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeArray %float %uint_12
; CHECK: OpTypeArray %float %uint_20
; CHECK: OpTypeArray %float %uint_15
; CHECK: OpFMul
; CHECK: OpFAdd
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_4 = OpConstant %uint 4
%uint_5 = OpConstant %uint 5
%uint_12 = OpConstant %uint 12
%uint_15 = OpConstant %uint 15
%uint_20 = OpConstant %uint 20
%float = OpTypeFloat 32
%mat3x4 = OpTypeCooperativeMatrixAZD %float %uint_3 %uint_4
%mat4x5 = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_5
%mat3x5 = OpTypeCooperativeMatrixAZD %float %uint_3 %uint_5
%a = OpUndef %mat3x4
%b = OpUndef %mat4x5
%c = OpConstantNull %mat3x5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddAZD %mat3x5 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
  ExpectTiledCounts(std::get<0>(result), 60, 60, 79);
}

TEST_F(AzdLowerToStandardTest, LowersMatmul2x4x4F32ExactTile) {
  const std::string text = R"(
; CHECK-NOT: AZD
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeArray %float %uint_8
; CHECK: OpTypeArray %float %uint_16
; CHECK: OpFMul
; CHECK: OpFAdd
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%uint_16 = OpConstant %uint 16
%float = OpTypeFloat 32
%mat2x4 = OpTypeCooperativeMatrixAZD %float %uint_2 %uint_4
%mat4x4 = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%a = OpUndef %mat2x4
%b = OpUndef %mat4x4
%c = OpConstantNull %mat2x4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddAZD %mat2x4 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
  ExpectTiledCounts(std::get<0>(result), 32, 32, 32);
}

TEST_F(AzdLowerToStandardTest, SynthesizesArrayLengthConstant) {
  const std::string text = R"(
; CHECK-NOT: AZD
; CHECK-NOT: CooperativeMatrixKHR
; CHECK: OpConstant %uint 6
; CHECK: OpTypeArray %float
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixAZD %float %uint_2 %uint_3
%undef = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
}

TEST_F(AzdLowerToStandardTest, LowersMatrixLoadStoreF32) {
  const std::string text = R"(
; CHECK-NOT: AZD
; CHECK-NOT: CooperativeMatrixKHR
; CHECK: OpTypeArray %float %uint_16
; CHECK: OpLoad %float
; CHECK: OpCompositeConstruct
; CHECK: OpCompositeExtract %float
; CHECK: OpStore
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_16 = OpConstant %uint 16
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%float_4 = OpConstant %float 4
%v2float = OpTypeVector %float 2
%shape = OpConstantComposite %v2float %float_4 %float_4
%offset = OpConstantComposite %v2float %float_0 %float_0
%mat = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %buf %int_0
%m = OpCooperativeMatrixLoadAZD %mat %base %shape %offset %int_0
OpCooperativeMatrixStoreAZD %base %m %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
}

TEST_F(AzdLowerToStandardTest, LowersVectorMatrixMul8x8F32Tile4) {
  const std::string text = R"(
; CHECK-NOT: AZD
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeArray %float %uint_8
; CHECK: OpTypeArray %float %uint_64
; CHECK: OpFMul
; CHECK: OpFAdd
OpCapability Shader
OpCapability CooperativeVectorAZD
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_cooperative_vector"
OpExtension "SPV_AZD_neural_matrix"
OpSourceExtension "GL_AZD_cooperative_vector"
OpSourceExtension "GL_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%uint_64 = OpConstant %uint 64
%float = OpTypeFloat 32
%vec = OpTypeCooperativeVectorAZD %float %uint_8
%mat = OpTypeCooperativeMatrixAZD %float %uint_8 %uint_8
%x = OpUndef %vec
%w = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulAZD %vec %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
  ExpectTiledCounts(std::get<0>(result), 64, 64, 80);
}

TEST_F(AzdLowerToStandardTest, LowersVectorMatrixMulAdd8x8F32Tile4WithBias) {
  const std::string text = R"(
; CHECK-NOT: AZD
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeArray %float %uint_8
; CHECK: OpTypeArray %float %uint_64
; CHECK: OpFMul
; CHECK: OpFAdd
OpCapability Shader
OpCapability CooperativeVectorAZD
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_cooperative_vector"
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%uint_64 = OpConstant %uint 64
%float = OpTypeFloat 32
%vec = OpTypeCooperativeVectorAZD %float %uint_8
%mat = OpTypeCooperativeMatrixAZD %float %uint_8 %uint_8
%x = OpUndef %vec
%w = OpUndef %mat
%bias = OpConstantNull %vec
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulAddAZD %vec %x %w %bias
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
  ExpectTiledCounts(std::get<0>(result), 64, 64, 88);
}

TEST_F(AzdLowerToStandardTest, LowersVectorMatrixMulTailNNotMultipleOf4) {
  const std::string text = R"(
; CHECK-NOT: AZD
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeArray %float %uint_8
; CHECK: OpTypeArray %float %uint_10
; CHECK: OpTypeArray %float %uint_80
; CHECK: OpFMul
; CHECK: OpFAdd
OpCapability Shader
OpCapability CooperativeVectorAZD
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_cooperative_vector"
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%uint_10 = OpConstant %uint 10
%uint_80 = OpConstant %uint 80
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorAZD %float %uint_8
%vec10 = OpTypeCooperativeVectorAZD %float %uint_10
%mat10x8 = OpTypeCooperativeMatrixAZD %float %uint_10 %uint_8
%x = OpUndef %vec8
%w = OpUndef %mat10x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulAZD %vec10 %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<AzdLowerToStandardPass>(text, true);
  ExpectTiledCounts(std::get<0>(result), 80, 80, 104);
}

TEST_F(AzdLowerToStandardTest, FailsForInvalidMatmulShape) {
  const std::string text = R"(
; CHECK: AZD cooperative matrix multiply shapes do not match
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%mat4x8 = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_8
%mat4x4 = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%a = OpUndef %mat4x8
%b = OpUndef %mat4x4
%c = OpUndef %mat4x4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddAZD %mat4x4 %a %b %c
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<AzdLowerToStandardPass>(text);
}

TEST_F(AzdLowerToStandardTest, FailsForInvalidVectorMatmulShape) {
  const std::string text = R"(
; CHECK: AZD cooperative vector matrix multiply shapes do not match
OpCapability Shader
OpCapability CooperativeVectorAZD
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_cooperative_vector"
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorAZD %float %uint_4
%mat8x4 = OpTypeCooperativeMatrixAZD %float %uint_8 %uint_4
%x = OpUndef %vec4
%w = OpUndef %mat8x4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulAZD %vec4 %x %w
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<AzdLowerToStandardPass>(text);
}

TEST_F(AzdLowerToStandardTest, FailsForAzdFunctionParameter) {
  const std::string text = R"(
; CHECK: AZD cooperative values across function boundaries are not supported
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%fn_mat = OpTypeFunction %void %mat
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
%helper = OpFunction %void None %fn_mat
%param = OpFunctionParameter %mat
%helper_entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<AzdLowerToStandardPass>(text);
}

TEST_F(AzdLowerToStandardTest, FailsForAzdFunctionReturn) {
  const std::string text = R"(
; CHECK: AZD cooperative values across function boundaries are not supported
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%a = OpUndef %mat
%fn_mat = OpTypeFunction %mat
%main = OpFunction %mat None %fn_mat
%entry = OpLabel
OpReturnValue %a
OpFunctionEnd
)";

  SinglePassRunAndFail<AzdLowerToStandardPass>(text);
}

TEST_F(AzdLowerToStandardTest, FailsForUnsupportedAzdPhi) {
  const std::string text = R"(
; CHECK: AZD cooperative OpPhi is not supported
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %merge
%merge = OpLabel
%p = OpPhi %mat %a %entry
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<AzdLowerToStandardPass>(text);
}

TEST_F(AzdLowerToStandardTest, FailsForPointerEscapeOrNonFunctionAzdPointer) {
  const std::string text = R"(
; CHECK: AZD cooperative values may only be stored in Function variables before lowering
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%_ptr_StorageBuffer_mat = OpTypePointer StorageBuffer %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<AzdLowerToStandardPass>(text);
}

TEST_F(AzdLowerToStandardTest, FailsForUnsupportedMatrixReduce) {
  const std::string text = R"(
; CHECK: OpCooperativeMatrixReduceAZD is not supported
OpCapability Shader
OpCapability CooperativeMatrixAZD
OpExtension "SPV_AZD_neural_matrix"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixAZD %float %uint_4 %uint_4
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%r = OpCooperativeMatrixReduceAZD %mat %a %uint_0 %uint_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<AzdLowerToStandardPass>(text);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
