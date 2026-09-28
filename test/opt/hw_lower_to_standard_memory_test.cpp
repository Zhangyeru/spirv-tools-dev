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

#include <array>
#include <set>
#include <sstream>
#include <vector>

#include "test/opt/hw_lower_to_standard_test_utils.h"

namespace spvtools {
namespace opt {
namespace {

TEST_F(HwLowerToStandardTest, LowersMatrixLoadStoreF32) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK: OpTypeVector %float 2
; CHECK: OpTypeArray %v2float %uint_8
; CHECK-DAG: OpLoopMerge
; CHECK-DAG: OpLoad %float
; CHECK-DAG: OpCompositeConstruct
; CHECK-DAG: OpCompositeExtract %float
; CHECK-DAG: OpFunctionCall %v2float
; CHECK-DAG: OpFunctionCall %void
; CHECK-DAG: OpCopyObject
; CHECK-DAG: OpStore
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
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
%mat = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
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
%m = OpCooperativeMatrixLoadHW %mat %base %shape %offset %int_0
OpCooperativeMatrixStoreHW %base %m %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  EXPECT_EQ(2u, CountSubstring(std::get<0>(result), "OpLoopMerge"));
  EXPECT_EQ(2u, CountSubstring(std::get<0>(result), "OpFunctionCall"));
  EXPECT_EQ(0u, CountSubstring(std::get<0>(result), " Volatile"));
}

TEST_F(HwLowerToStandardTest, LoopCarriedSameRootPackedMemoryUsesVolatile) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK: OpLoad %float {{%\w+}} Volatile
; CHECK: OpStore {{%\w+}} {{%\w+}} Volatile
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
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
%initial = OpCooperativeVectorLoadHW %vec4 %base %uint_0
OpCooperativeVectorStoreHW %base %uint_0 %initial
OpBranch %loop
%loop = OpLabel
%index = OpPhi %uint %uint_0 %entry %next %continue
%condition = OpULessThan %bool %index %uint_2
OpLoopMerge %merge %continue None
OpBranchConditional %condition %body %merge
%body = OpLabel
%value = OpCooperativeVectorLoadHW %vec4 %base %uint_0
OpCooperativeVectorStoreHW %base %uint_0 %value
OpBranch %continue
%continue = OpLabel
%next = OpIAdd %uint %index %uint_1
OpBranch %loop
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  const auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
  EXPECT_EQ(4u, CountSubstring(std::get<0>(result), " Volatile"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixSubregionThroughNestedFixedArrays) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK: OpUDiv %uint
; CHECK: OpUMod %uint
; CHECK: OpAccessChain %_ptr_StorageBuffer_uint {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: OpLoad %uint
; CHECK: OpBitcast %float
; CHECK: OpBitcast %uint
; CHECK: OpStore {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %inner ArrayStride 16
OpDecorate %outer ArrayStride 32
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_6 = OpConstant %uint 6
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v4uint = OpTypeVector %uint 4
%shape_type = OpTypeVector %uint 2
%shape = OpConstantComposite %shape_type %uint_6 %uint_8
%offset = OpConstantComposite %shape_type %uint_2 %uint_3
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_3
%inner = OpTypeArray %v4uint %uint_2
%outer = OpTypeArray %inner %uint_6
%Buf = OpTypeStruct %outer
%ptr_buffer = OpTypePointer StorageBuffer %Buf
%ptr_outer = OpTypePointer StorageBuffer %outer
%buf = OpVariable %ptr_buffer StorageBuffer
%main = OpFunction %void None %func
%entry = OpLabel
%base = OpAccessChain %ptr_outer %buf %int_0
%value = OpCooperativeMatrixLoadHW %mat %base %shape %offset %int_0
OpCooperativeMatrixStoreHW %base %value %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  const auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
  EXPECT_GE(CountSubstring(std::get<0>(result), "OpUDiv %uint"), 2u);
  EXPECT_GE(CountSubstring(std::get<0>(result), "OpUMod %uint"), 2u);
}

TEST_F(HwLowerToStandardTest,
       LowersDynamicPackedVectorThroughNestedArrayVectorLeafBitcasts) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK: OpUDiv %uint
; CHECK: OpUMod %uint
; CHECK: OpAccessChain %_ptr_StorageBuffer_uint {{%\w+}} {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: OpLoad %uint
; CHECK: OpBitcast %float
; CHECK: OpBitcast %uint
; CHECK: OpStore {{%\w+}} {{%\w+}}
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %inner ArrayStride 16
OpDecorate %outer ArrayStride 32
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%void = OpTypeVoid
%func = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v4uint = OpTypeVector %uint 4
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%inner = OpTypeArray %v4uint %uint_2
%outer = OpTypeRuntimeArray %inner
%Buf = OpTypeStruct %outer
%ptr_buffer = OpTypePointer StorageBuffer %Buf
%ptr_outer = OpTypePointer StorageBuffer %outer
%buf = OpVariable %ptr_buffer StorageBuffer
%main = OpFunction %void None %func
%entry = OpLabel
%base = OpAccessChain %ptr_outer %buf %int_0
%dynamic_offset = OpIAdd %uint %uint_4 %uint_1
%value = OpCooperativeVectorLoadHW %vec4 %base %dynamic_offset
OpCooperativeVectorStoreHW %base %dynamic_offset %value
OpReturn
OpFunctionEnd
)";

  const auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
  EXPECT_GE(CountSubstring(std::get<0>(result), "OpBitcast %float"), 2u);
  EXPECT_GE(CountSubstring(std::get<0>(result), "OpBitcast %uint"), 2u);
}

TEST_F(HwLowerToStandardTest, MatrixLoadPreservesAlignedMemoryAccess) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpLoad %float {{%\w+}} Aligned 4
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
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
%mat = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
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
%m = OpCooperativeMatrixLoadHW %mat %base %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedAlignedMemoryAccess(
      this, text, spv::Op::OpCooperativeMatrixLoadHW);
  ExpectNoHwOrCoopMatrix(result);
  EXPECT_EQ(0u, CountSubstring(result, "Aligned 16"));
  EXPECT_EQ(2u, CountSubstring(result, "Aligned 4"));
}

TEST_F(HwLowerToStandardTest, MatrixStorePreservesAlignedMemoryAccess) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpStore {{%\w+}} {{%\w+}} Aligned 4
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
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
%mat = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%m = OpUndef %mat
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
OpCooperativeMatrixStoreHW %base %m %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedAlignedMemoryAccess(
      this, text, spv::Op::OpCooperativeMatrixStoreHW);
  ExpectNoHwOrCoopMatrix(result);
  EXPECT_EQ(0u, CountSubstring(result, "Aligned 16"));
  EXPECT_EQ(2u, CountSubstring(result, "Aligned 4"));
}

TEST_F(HwLowerToStandardTest, VectorLoadPreservesAlignedMemoryAccess) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpLoad %float {{%\w+}} Aligned 4
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
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
%v = OpCooperativeVectorLoadHW %vec8 %base %int_0
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedAlignedMemoryAccess(
      this, text, spv::Op::OpCooperativeVectorLoadHW);
  ExpectNoHwOrCoopMatrix(result);
  EXPECT_EQ(0u, CountSubstring(result, "Aligned 16"));
  EXPECT_EQ(2u, CountSubstring(result, "Aligned 4"));
}

TEST_F(HwLowerToStandardTest, VectorStorePreservesAlignedMemoryAccess) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpStore {{%\w+}} {{%\w+}} Aligned 4
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%v = OpUndef %vec8
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
OpCooperativeVectorStoreHW %base %int_0 %v
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedAlignedMemoryAccess(
      this, text, spv::Op::OpCooperativeVectorStoreHW);
  ExpectNoHwOrCoopMatrix(result);
  EXPECT_EQ(0u, CountSubstring(result, "Aligned 16"));
  EXPECT_EQ(2u, CountSubstring(result, "Aligned 4"));
}

TEST_F(HwLowerToStandardTest, VectorLoadPreservesFullMemoryAccessOperands) {
  const std::string text = R"(
OpCapability Shader
OpCapability VulkanMemoryModel
OpCapability MemoryAccessAliasingINTEL
OpCapability CooperativeVectorHW
OpExtension "SPV_KHR_vulkan_memory_model"
OpExtension "SPV_INTEL_memory_access_aliasing"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical VulkanKHR
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%domain = OpAliasDomainDeclINTEL
%alias_scope = OpAliasScopeDeclINTEL %domain
%noalias_scope = OpAliasScopeDeclINTEL %domain
%alias_list = OpAliasScopeListDeclINTEL %alias_scope
%noalias_list = OpAliasScopeListDeclINTEL %noalias_scope
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %buf %int_0
%v = OpCooperativeVectorLoadHW %vec4 %base %int_0
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedFullMemoryAccess(
      this, text, spv::Op::OpCooperativeVectorLoadHW, true);
  ExpectNoHwOrCoopMatrix(result);
  const std::string access =
      "Volatile|Aligned|Nontemporal|MakePointerVisible|NonPrivatePointer|"
      "AliasScopeINTELMask|NoAliasINTELMask";
  EXPECT_EQ(4u, CountSubstring(result, access)) << result;
  EXPECT_EQ(1u, CountSubstring(result, access + " 16")) << result;
  EXPECT_EQ(1u, CountSubstring(result, access + " 8")) << result;
  EXPECT_EQ(2u, CountSubstring(result, access + " 4")) << result;
  EXPECT_EQ(1u, CountSubstring(result, "OpFunctionEnd")) << result;
}

TEST_F(HwLowerToStandardTest, VectorStorePreservesFullMemoryAccessOperands) {
  const std::string text = R"(
OpCapability Shader
OpCapability VulkanMemoryModel
OpCapability MemoryAccessAliasingINTEL
OpCapability CooperativeVectorHW
OpExtension "SPV_KHR_vulkan_memory_model"
OpExtension "SPV_INTEL_memory_access_aliasing"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical VulkanKHR
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%v = OpUndef %vec4
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%domain = OpAliasDomainDeclINTEL
%alias_scope = OpAliasScopeDeclINTEL %domain
%noalias_scope = OpAliasScopeDeclINTEL %domain
%alias_list = OpAliasScopeListDeclINTEL %alias_scope
%noalias_list = OpAliasScopeListDeclINTEL %noalias_scope
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %buf %int_0
OpCooperativeVectorStoreHW %base %int_0 %v
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedFullMemoryAccess(
      this, text, spv::Op::OpCooperativeVectorStoreHW, false);
  ExpectNoHwOrCoopMatrix(result);
  const std::string access =
      "Volatile|Aligned|Nontemporal|MakePointerAvailable|NonPrivatePointer|"
      "AliasScopeINTELMask|NoAliasINTELMask";
  EXPECT_EQ(4u, CountSubstring(result, access)) << result;
  EXPECT_EQ(1u, CountSubstring(result, access + " 16")) << result;
  EXPECT_EQ(1u, CountSubstring(result, access + " 8")) << result;
  EXPECT_EQ(2u, CountSubstring(result, access + " 4")) << result;
  EXPECT_EQ(1u, CountSubstring(result, "OpFunctionEnd")) << result;
}

TEST_F(HwLowerToStandardTest, LocalAliasListsPreventPackedVectorMemoryHelpers) {
  const std::string text = R"(
OpCapability Shader
OpCapability MemoryAccessAliasingINTEL
OpCapability CooperativeVectorHW
OpExtension "SPV_INTEL_memory_access_aliasing"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%domain = OpAliasDomainDeclINTEL
%alias_scope = OpAliasScopeDeclINTEL %domain
%noalias_scope = OpAliasScopeDeclINTEL %domain
%alias_list = OpAliasScopeListDeclINTEL %alias_scope
%noalias_list = OpAliasScopeListDeclINTEL %noalias_scope
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %buf %int_0
%v = OpCooperativeVectorLoadHW %vec8 %base %int_0
OpCooperativeVectorStoreHW %base %int_0 %v
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedAliasMemoryAccess(this, text);
  ExpectNoHwOrCoopMatrix(result);
  EXPECT_EQ(16u, CountSubstring(result, "AliasScopeINTELMask|NoAliasINTELMask"))
      << result;
  EXPECT_EQ(2u, CountSubstring(result, "OpAliasScopeListDeclINTEL")) << result;
  EXPECT_EQ(1u, CountSubstring(result, "OpFunctionEnd")) << result;
}

TEST_F(HwLowerToStandardTest, ScalarLoadDerivesAlignmentFromConstantOffset) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_int ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%vec4 = OpTypeCooperativeVectorHW %int %uint_4
%_runtimearr_int = OpTypeRuntimeArray %int
%Buf = OpTypeStruct %_runtimearr_int
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_int = OpTypePointer StorageBuffer %_runtimearr_int
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_int %buf %int_0
%v = OpCooperativeVectorLoadHW %vec4 %base %int_0
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedAlignedMemoryAccess(
      this, text, spv::Op::OpCooperativeVectorLoadHW);
  ExpectNoHwOrCoopMatrix(result);
  EXPECT_EQ(1u, CountSubstring(result, "Aligned 16"));
  EXPECT_EQ(1u, CountSubstring(result, "Aligned 8"));
  EXPECT_EQ(2u, CountSubstring(result, "Aligned 4"));
}

TEST_F(HwLowerToStandardTest, ScalarLoadUsesNaturalAlignmentForDynamicOffset) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_int ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%vec4 = OpTypeCooperativeVectorHW %int %uint_4
%_runtimearr_int = OpTypeRuntimeArray %int
%Buf = OpTypeStruct %_runtimearr_int
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_int = OpTypePointer StorageBuffer %_runtimearr_int
%_ptr_Function_int = OpTypePointer Function %int
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%offset_var = OpVariable %_ptr_Function_int Function %int_0
%offset = OpLoad %int %offset_var
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_int %buf %int_0
%v = OpCooperativeVectorLoadHW %vec4 %base %offset
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedAlignedMemoryAccess(
      this, text, spv::Op::OpCooperativeVectorLoadHW);
  ExpectNoHwOrCoopMatrix(result);
  EXPECT_EQ(0u, CountSubstring(result, "Aligned 16"));
  EXPECT_EQ(0u, CountSubstring(result, "Aligned 8"));
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 4"));
}

TEST_F(HwLowerToStandardTest,
       PhysicalStorageBufferLoadPreservesRequiredAlignment) {
  const std::string text = R"(
OpCapability Shader
OpCapability Int64
OpCapability PhysicalStorageBufferAddresses
OpCapability CooperativeVectorHW
OpExtension "SPV_EXT_physical_storage_buffer"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_int ArrayStride 4
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%ulong = OpTypeInt 64 0
%ulong_4096 = OpConstant %ulong 4096
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%vec4 = OpTypeCooperativeVectorHW %int %uint_4
%_runtimearr_int = OpTypeRuntimeArray %int
%_ptr_PhysicalStorageBuffer__runtimearr_int = OpTypePointer PhysicalStorageBuffer %_runtimearr_int
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpConvertUToPtr %_ptr_PhysicalStorageBuffer__runtimearr_int %ulong_4096
%v = OpCooperativeVectorLoadHW %vec4 %base %int_0
OpReturn
OpFunctionEnd
)";

  const std::string result = RunWithInjectedAlignedMemoryAccess(
      this, text, spv::Op::OpCooperativeVectorLoadHW);
  ExpectNoHwOrCoopMatrix(result);
  EXPECT_NE(std::string::npos,
            result.find("OpTypePointer PhysicalStorageBuffer"));
  EXPECT_EQ(1u, CountSubstring(result, "Aligned 16"));
  EXPECT_EQ(1u, CountSubstring(result, "Aligned 8"));
  EXPECT_EQ(2u, CountSubstring(result, "Aligned 4"));
}

TEST_F(HwLowerToStandardTest,
       PhysicalStorageBufferLoadWithoutAlignmentFailsLowering) {
  const std::string text = R"(
; CHECK: PhysicalStorageBuffer HW access requires an Aligned memory operand
OpCapability Shader
OpCapability Int64
OpCapability PhysicalStorageBufferAddresses
OpCapability CooperativeVectorHW
OpExtension "SPV_EXT_physical_storage_buffer"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_int ArrayStride 4
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%ulong = OpTypeInt 64 0
%ulong_4096 = OpConstant %ulong 4096
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%vec4 = OpTypeCooperativeVectorHW %int %uint_4
%_runtimearr_int = OpTypeRuntimeArray %int
%_ptr_PhysicalStorageBuffer__runtimearr_int = OpTypePointer PhysicalStorageBuffer %_runtimearr_int
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpConvertUToPtr %_ptr_PhysicalStorageBuffer__runtimearr_int %ulong_4096
%v = OpCooperativeVectorLoadHW %vec4 %base %int_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, ForceScalarModeLowersF16VectorLoadStoreToScalar) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %half %uint_8
; CHECK: OpLoad %half
; CHECK: OpStore
; CHECK-NOT: OpTypeVector %half 2
; CHECK-NOT: OpFunctionCall
; CHECK-NOT: OpLoopMerge
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_half ArrayStride 2
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%half = OpTypeFloat 16
%vec8 = OpTypeCooperativeVectorHW %half %uint_8
%_runtimearr_half = OpTypeRuntimeArray %half
%Buf = OpTypeStruct %_runtimearr_half
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_half = OpTypePointer StorageBuffer %_runtimearr_half
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %buf %int_0
%v = OpCooperativeVectorLoadHW %vec8 %base %int_0
OpCooperativeVectorStoreHW %base %int_0 %v
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(
      text, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(8u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_EQ(8u, CountSubstring(disassembly, "OpStore"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %half 2"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFunctionCall"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
}

TEST_F(HwLowerToStandardTest, ForceScalarModeLowersF16MatrixLoadStoreToScalar) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %half %uint_16
; CHECK: OpLoad %half
; CHECK: OpStore
; CHECK-NOT: OpTypeVector %half 2
; CHECK-NOT: OpFunctionCall
; CHECK-NOT: OpLoopMerge
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_half ArrayStride 2
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
%half = OpTypeFloat 16
%mat4x4 = OpTypeCooperativeMatrixHW %half %uint_4 %uint_4
%_runtimearr_half = OpTypeRuntimeArray %half
%Buf = OpTypeStruct %_runtimearr_half
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_half = OpTypePointer StorageBuffer %_runtimearr_half
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %buf %int_0
%m = OpCooperativeMatrixLoadHW %mat4x4 %base %shape %offset %int_0
OpCooperativeMatrixStoreHW %base %m %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(
      text, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(16u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_EQ(16u, CountSubstring(disassembly, "OpStore"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %half 2"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFunctionCall"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
}

TEST_F(HwLowerToStandardTest, LowersVectorLoadStoreF16Packed) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 2
; CHECK: OpTypeArray %v2half %uint_4
; CHECK-DAG: OpLoopMerge
; CHECK-DAG: OpLoad %half
; CHECK-DAG: OpLoad %v2half
; CHECK-DAG: OpCompositeExtract %half
; CHECK-DAG: OpFunctionCall %v2half
; CHECK-DAG: OpFunctionCall %void
; CHECK-DAG: OpStore
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_half ArrayStride 2
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%half = OpTypeFloat 16
%vec8 = OpTypeCooperativeVectorHW %half %uint_8
%_runtimearr_half = OpTypeRuntimeArray %half
%Buf = OpTypeStruct %_runtimearr_half
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_half = OpTypePointer StorageBuffer %_runtimearr_half
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %buf %int_0
%v = OpCooperativeVectorLoadHW %vec8 %base %int_0
OpCooperativeVectorStoreHW %base %int_0 %v
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
  EXPECT_EQ(2u, CountSubstring(std::get<0>(result), "OpLoopMerge"));
  EXPECT_EQ(2u, CountSubstring(std::get<0>(result), "OpFunctionCall"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixLoadStoreF16PackedRowMajor) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 2
; CHECK: OpTypeArray %v2half %uint_16
; CHECK-DAG: OpLoopMerge
; CHECK-DAG: OpLoad %half
; CHECK-DAG: OpLoad %v2half
; CHECK-DAG: OpCompositeExtract %half
; CHECK-DAG: OpFunctionCall %v2half
; CHECK-DAG: OpFunctionCall %void
; CHECK-DAG: OpStore
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_half ArrayStride 2
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%half = OpTypeFloat 16
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_4 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%mat4x8 = OpTypeCooperativeMatrixHW %half %uint_4 %uint_8
%_runtimearr_half = OpTypeRuntimeArray %half
%Buf = OpTypeStruct %_runtimearr_half
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_half = OpTypePointer StorageBuffer %_runtimearr_half
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %buf %int_0
%m = OpCooperativeMatrixLoadHW %mat4x8 %base %shape %offset %int_0
OpCooperativeMatrixStoreHW %base %m %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
  EXPECT_EQ(2u, CountSubstring(std::get<0>(result), "OpLoopMerge"));
  EXPECT_EQ(2u, CountSubstring(std::get<0>(result), "OpFunctionCall"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixLoadStoreF16PackedColumnMajor) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 2
; CHECK: OpTypeArray %v2half %uint_16
; CHECK: OpLoad %half
; CHECK: OpCompositeConstruct %v2half
; CHECK: OpCompositeExtract %v2half
; CHECK: OpCompositeExtract %half
; CHECK: OpStore
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_half ArrayStride 2
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%half = OpTypeFloat 16
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_4 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%mat4x8 = OpTypeCooperativeMatrixHW %half %uint_4 %uint_8
%_runtimearr_half = OpTypeRuntimeArray %half
%Buf = OpTypeStruct %_runtimearr_half
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_half = OpTypePointer StorageBuffer %_runtimearr_half
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %buf %int_0
%m = OpCooperativeMatrixLoadHW %mat4x8 %base %shape %offset %int_1
OpCooperativeMatrixStoreHW %base %m %shape %offset %int_1
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest,
       PreservesFunctionVariableAcrossLoopBlocksDuringLowering) {
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
%vec = OpTypeCooperativeVectorHW %float %uint_8
%initial = OpUndef %vec
%delta = OpUndef %vec
%_ptr_Function_vec = OpTypePointer Function %vec
%bool = OpTypeBool
%false = OpConstantFalse %bool
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%carried = OpVariable %_ptr_Function_vec Function
OpStore %carried %initial
OpBranch %header
%header = OpLabel
%current = OpLoad %vec %carried
OpLoopMerge %merge %continue None
OpBranchConditional %false %body %merge
%body = OpLabel
%updated = OpFAdd %vec %current %delta
OpStore %carried %updated
OpBranch %continue
%continue = OpLabel
%continued = OpLoad %vec %carried
OpStore %carried %continued
OpBranch %header
%merge = OpLabel
%after = OpLoad %vec %carried
%element = OpCompositeExtract %float %after 3
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec2,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 0, 0, 2);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpTypePointer Function %_arr_v2float_uint_4"));
  EXPECT_GE(CountSubstring(disassembly, "OpLoad %_arr_v2float_uint_4"), 2u);
  EXPECT_GE(CountSubstring(disassembly, "OpStore"), 2u);
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 2u);
  EXPECT_GT(CountSubstring(disassembly, "OpFAdd %v2float"), 0u);
}

TEST_F(HwLowerToStandardTest,
       FunctionVariableForwardingRespectsAccessChainAliasStores) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%zero = OpConstantNull %vec4
%_ptr_Function_vec4 = OpTypePointer Function %vec4
%_ptr_Function_float = OpTypePointer Function %float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%var = OpVariable %_ptr_Function_vec4 Function
OpStore %var %zero
%element_ptr = OpAccessChain %_ptr_Function_float %var %uint_0
OpStore %element_ptr %one
%loaded = OpLoad %vec4 %var
%element = OpVectorExtractDynamic %float %loaded %uint_0
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpLoad %_arr_v2float_uint_2"));
  EXPECT_NE(std::string::npos, disassembly.find("OpStore %"));
  EXPECT_NE(std::string::npos, disassembly.find("%float_1"));
}

TEST_F(HwLowerToStandardTest, FunctionVariableForwardingPreservesVolatileLoad) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%zero = OpConstantNull %vec4
%_ptr_Function_vec4 = OpTypePointer Function %vec4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%var = OpVariable %_ptr_Function_vec4 Function
OpStore %var %zero
%loaded = OpLoad %vec4 %var Volatile
%element = OpVectorExtractDynamic %float %loaded %uint_0
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpLoad %_arr_v2float_uint_2"));
  EXPECT_EQ(1u, CountSubstring(disassembly, " Volatile"));
}

// A memory lane and a cooperative component have independent types. In
// particular, changing the component width must not change pointer arithmetic
// or cause a native-vector memory access to bypass the scalar conversion.
struct MemoryScalarType {
  const char* name;
  uint32_t width;
  bool floating;
  bool signed_integer;
  const char* capability;
};

const MemoryScalarType kMemoryF16 = {"f16", 16, true, false, "Float16"};
const MemoryScalarType kMemoryF32 = {"f32", 32, true, false, ""};
const MemoryScalarType kMemoryF64 = {"f64", 64, true, false, "Float64"};
const MemoryScalarType kMemoryI8 = {"i8", 8, false, true, "Int8"};
const MemoryScalarType kMemoryU8 = {"u8", 8, false, false, "Int8"};
const MemoryScalarType kMemoryI16 = {"i16", 16, false, true, "Int16"};
const MemoryScalarType kMemoryU16 = {"u16", 16, false, false, "Int16"};
const MemoryScalarType kMemoryI32 = {"i32", 32, false, true, ""};
const MemoryScalarType kMemoryU32 = {"u32", 32, false, false, ""};
const MemoryScalarType kMemoryI64 = {"i64", 64, false, true, "Int64"};
const MemoryScalarType kMemoryU64 = {"u64", 64, false, false, "Int64"};

struct MemoryConversionStep {
  spv::Op opcode;
  const MemoryScalarType* result_type;
};

struct MemoryConversionCase {
  const char* name;
  const MemoryScalarType* memory_type;
  const MemoryScalarType* cooperative_type;
  std::vector<MemoryConversionStep> load_steps;
  std::vector<MemoryConversionStep> store_steps;
};

// Expected instruction chains are explicit: widening signed input followed by
// an unsigned destination must sign-extend first, whereas the reverse store
// must zero-extend/truncate its unsigned source before changing signedness.
const MemoryConversionCase kMemoryConversionCases[] = {
    {"F32_F16",
     &kMemoryF32,
     &kMemoryF16,
     {{spv::Op::OpFConvert, &kMemoryF16}},
     {{spv::Op::OpFConvert, &kMemoryF32}}},
    {"F16_F32",
     &kMemoryF16,
     &kMemoryF32,
     {{spv::Op::OpFConvert, &kMemoryF32}},
     {{spv::Op::OpFConvert, &kMemoryF16}}},
    {"F64_F32",
     &kMemoryF64,
     &kMemoryF32,
     {{spv::Op::OpFConvert, &kMemoryF32}},
     {{spv::Op::OpFConvert, &kMemoryF64}}},
    {"I8_I16",
     &kMemoryI8,
     &kMemoryI16,
     {{spv::Op::OpSConvert, &kMemoryI16}},
     {{spv::Op::OpSConvert, &kMemoryI8}}},
    {"U8_U16",
     &kMemoryU8,
     &kMemoryU16,
     {{spv::Op::OpUConvert, &kMemoryU16}},
     {{spv::Op::OpUConvert, &kMemoryU8}}},
    {"I16_I32",
     &kMemoryI16,
     &kMemoryI32,
     {{spv::Op::OpSConvert, &kMemoryI32}},
     {{spv::Op::OpSConvert, &kMemoryI16}}},
    {"U16_U32",
     &kMemoryU16,
     &kMemoryU32,
     {{spv::Op::OpUConvert, &kMemoryU32}},
     {{spv::Op::OpUConvert, &kMemoryU16}}},
    {"I64_I32",
     &kMemoryI64,
     &kMemoryI32,
     {{spv::Op::OpSConvert, &kMemoryI32}},
     {{spv::Op::OpSConvert, &kMemoryI64}}},
    {"U64_U32",
     &kMemoryU64,
     &kMemoryU32,
     {{spv::Op::OpUConvert, &kMemoryU32}},
     {{spv::Op::OpUConvert, &kMemoryU64}}},
    {"I8_U16",
     &kMemoryI8,
     &kMemoryU16,
     {{spv::Op::OpSConvert, &kMemoryI16}, {spv::Op::OpBitcast, &kMemoryU16}},
     {{spv::Op::OpUConvert, &kMemoryU8}, {spv::Op::OpBitcast, &kMemoryI8}}},
    {"U8_I16",
     &kMemoryU8,
     &kMemoryI16,
     {{spv::Op::OpUConvert, &kMemoryU16}, {spv::Op::OpBitcast, &kMemoryI16}},
     {{spv::Op::OpSConvert, &kMemoryI8}, {spv::Op::OpBitcast, &kMemoryU8}}},
    {"I16_U8",
     &kMemoryI16,
     &kMemoryU8,
     {{spv::Op::OpSConvert, &kMemoryI8}, {spv::Op::OpBitcast, &kMemoryU8}},
     {{spv::Op::OpUConvert, &kMemoryU16}, {spv::Op::OpBitcast, &kMemoryI16}}},
    {"U16_I8",
     &kMemoryU16,
     &kMemoryI8,
     {{spv::Op::OpUConvert, &kMemoryU8}, {spv::Op::OpBitcast, &kMemoryI8}},
     {{spv::Op::OpSConvert, &kMemoryI16}, {spv::Op::OpBitcast, &kMemoryU16}}},
    {"I64_U32",
     &kMemoryI64,
     &kMemoryU32,
     {{spv::Op::OpSConvert, &kMemoryI32}, {spv::Op::OpBitcast, &kMemoryU32}},
     {{spv::Op::OpUConvert, &kMemoryU64}, {spv::Op::OpBitcast, &kMemoryI64}}},
    {"U64_I32",
     &kMemoryU64,
     &kMemoryI32,
     {{spv::Op::OpUConvert, &kMemoryU32}, {spv::Op::OpBitcast, &kMemoryI32}},
     {{spv::Op::OpSConvert, &kMemoryI64}, {spv::Op::OpBitcast, &kMemoryU64}}},
    {"I16_F32",
     &kMemoryI16,
     &kMemoryF32,
     {{spv::Op::OpConvertSToF, &kMemoryF32}},
     {{spv::Op::OpConvertFToS, &kMemoryI16}}},
    {"U16_F32",
     &kMemoryU16,
     &kMemoryF32,
     {{spv::Op::OpConvertUToF, &kMemoryF32}},
     {{spv::Op::OpConvertFToU, &kMemoryU16}}},
    {"I64_F32",
     &kMemoryI64,
     &kMemoryF32,
     {{spv::Op::OpConvertSToF, &kMemoryF32}},
     {{spv::Op::OpConvertFToS, &kMemoryI64}}},
    {"U64_F32",
     &kMemoryU64,
     &kMemoryF32,
     {{spv::Op::OpConvertUToF, &kMemoryF32}},
     {{spv::Op::OpConvertFToU, &kMemoryU64}}},
    {"F64_I32",
     &kMemoryF64,
     &kMemoryI32,
     {{spv::Op::OpConvertFToS, &kMemoryI32}},
     {{spv::Op::OpConvertSToF, &kMemoryF64}}},
    {"F64_U32",
     &kMemoryF64,
     &kMemoryU32,
     {{spv::Op::OpConvertFToU, &kMemoryU32}},
     {{spv::Op::OpConvertUToF, &kMemoryF64}}},
    {"F32_I16",
     &kMemoryF32,
     &kMemoryI16,
     {{spv::Op::OpConvertFToS, &kMemoryI16}},
     {{spv::Op::OpConvertSToF, &kMemoryF32}}},
    {"F32_U16",
     &kMemoryF32,
     &kMemoryU16,
     {{spv::Op::OpConvertFToU, &kMemoryU16}},
     {{spv::Op::OpConvertUToF, &kMemoryF32}}},
    {"U32_F32Bitcast",
     &kMemoryU32,
     &kMemoryF32,
     {{spv::Op::OpBitcast, &kMemoryF32}},
     {{spv::Op::OpBitcast, &kMemoryU32}}},
    {"I16_F16Bitcast",
     &kMemoryI16,
     &kMemoryF16,
     {{spv::Op::OpBitcast, &kMemoryF16}},
     {{spv::Op::OpBitcast, &kMemoryI16}}},
    {"I8_U8Bitcast",
     &kMemoryI8,
     &kMemoryU8,
     {{spv::Op::OpBitcast, &kMemoryU8}},
     {{spv::Op::OpBitcast, &kMemoryI8}}},
    {"F32Identity", &kMemoryF32, &kMemoryF32, {}, {}},
    {"I16Identity", &kMemoryI16, &kMemoryI16, {}, {}},
};

std::string MakeMemoryConversionModule(const MemoryConversionCase& test,
                                       bool matrix, bool nested = false,
                                       bool column_major = false,
                                       bool storage_only = false) {
  std::ostringstream text;
  std::set<std::string> capabilities = {"Shader"};
  std::set<std::string> extensions = {"SPV_HW_neural_shader"};
  const std::array<const MemoryScalarType*, 3> types = {
      &kMemoryU32, test.memory_type, test.cooperative_type};
  for (const auto* type : types) {
    if (type->capability[0] && !(storage_only && type == test.memory_type)) {
      capabilities.insert(type->capability);
    }
  }
  capabilities.insert(matrix ? "CooperativeMatrixHW" : "CooperativeVectorHW");
  if (test.memory_type->width == 8) {
    capabilities.insert("StorageBuffer8BitAccess");
    extensions.insert("SPV_KHR_8bit_storage");
  } else if (test.memory_type->width == 16) {
    capabilities.insert("StorageBuffer16BitAccess");
    extensions.insert("SPV_KHR_16bit_storage");
  }
  for (const auto& capability : capabilities)
    text << "OpCapability " << capability << '\n';
  for (const auto& extension : extensions)
    text << "OpExtension \"" << extension << "\"\n";
  text << R"(OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
)";
  const uint32_t bytes = test.memory_type->width / 8;
  if (nested) {
    text << "OpDecorate %inner ArrayStride " << 4 * bytes << '\n';
    text << "OpDecorate %outer ArrayStride " << 8 * bytes << '\n';
  } else {
    text << "OpDecorate %outer ArrayStride " << bytes << '\n';
  }
  text << R"(OpMemberDecorate %Buffer 0 Offset 0
OpDecorate %Buffer Block
OpDecorate %buffer DescriptorSet 0
OpDecorate %buffer Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
)";
  std::set<const MemoryScalarType*> emitted;
  for (const auto* type : types) {
    if (!emitted.insert(type).second) continue;
    text << '%' << type->name << " = OpType"
         << (type->floating ? "Float " : "Int ") << type->width;
    if (!type->floating) text << ' ' << (type->signed_integer ? 1 : 0);
    text << '\n';
  }
  text << R"(%zero = OpConstant %u32 0
%one = OpConstant %u32 1
%two = OpConstant %u32 2
%four = OpConstant %u32 4
%eight = OpConstant %u32 8
%v2uint = OpTypeVector %u32 2
%shape = OpConstantComposite %v2uint %eight %eight
)";
  text << "%cooperative = OpTypeCooperative" << (matrix ? "Matrix" : "Vector")
       << "HW %" << test.cooperative_type->name
       << (matrix ? " %two %two\n" : " %four\n");
  if (nested) {
    text << "%leaf_vector = OpTypeVector %" << test.memory_type->name << " 4\n";
    text << "%inner = OpTypeArray %leaf_vector %two\n";
    text << "%outer = OpTypeRuntimeArray %inner\n";
  } else {
    text << "%outer = OpTypeRuntimeArray %" << test.memory_type->name << '\n';
  }
  text << R"(%Buffer = OpTypeStruct %outer
%ptr_buffer = OpTypePointer StorageBuffer %Buffer
%ptr_outer = OpTypePointer StorageBuffer %outer
%buffer = OpVariable %ptr_buffer StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %ptr_outer %buffer %zero
%dynamic_offset = OpIAdd %u32 %one %one
%offset = OpCompositeConstruct %v2uint %dynamic_offset %one
)";
  if (matrix) {
    const char* layout = column_major ? "%one" : "%zero";
    text << "%value = OpCooperativeMatrixLoadHW %cooperative %base %shape "
            "%offset "
         << layout << '\n';
    text << "OpCooperativeMatrixStoreHW %base %value %shape %offset " << layout
         << '\n';
  } else {
    text << "%value = OpCooperativeVectorLoadHW %cooperative %base "
            "%dynamic_offset\n";
    text << "OpCooperativeVectorStoreHW %base %dynamic_offset %value\n";
  }
  text << "OpReturn\nOpFunctionEnd\n";
  return text.str();
}

class HwLowerToStandardMemoryConversionTest
    : public PassTest<::testing::TestWithParam<MemoryConversionCase>> {
 protected:
  bool IsScalarType(uint32_t type_id, const MemoryScalarType& expected) {
    const auto* type = context()->get_def_use_mgr()->GetDef(type_id);
    if (!type ||
        type->opcode() !=
            (expected.floating ? spv::Op::OpTypeFloat : spv::Op::OpTypeInt) ||
        type->GetSingleWordInOperand(0) != expected.width) {
      return false;
    }
    return expected.floating || type->GetSingleWordInOperand(1) ==
                                    (expected.signed_integer ? 1u : 0u);
  }

  bool HasConversionUsers(uint32_t value,
                          const std::vector<MemoryConversionStep>& steps,
                          size_t index = 0) {
    if (index == steps.size()) return true;
    bool found = false;
    context()->get_def_use_mgr()->ForEachUser(value, [&](Instruction* user) {
      if (user->opcode() == steps[index].opcode &&
          IsScalarType(user->type_id(), *steps[index].result_type) &&
          user->GetSingleWordInOperand(0) == value) {
        found |= HasConversionUsers(user->result_id(), steps, index + 1);
      }
    });
    return found;
  }

  bool HasConversionDefinitions(uint32_t value,
                                const std::vector<MemoryConversionStep>& steps,
                                const MemoryScalarType& source_type) {
    auto* def_use = context()->get_def_use_mgr();
    for (auto step = steps.rbegin(); step != steps.rend(); ++step) {
      const auto* definition = def_use->GetDef(value);
      if (!definition || definition->opcode() != step->opcode ||
          !IsScalarType(definition->type_id(), *step->result_type)) {
        return false;
      }
      value = definition->GetSingleWordInOperand(0);
    }
    const auto* source = def_use->GetDef(value);
    return source && IsScalarType(source->type_id(), source_type);
  }

  void CheckMemoryConversionChains(const MemoryConversionCase& test) {
    size_t load_count = 0;
    size_t store_count = 0;
    auto* def_use = context()->get_def_use_mgr();
    context()->module()->ForEachInst([&](Instruction* inst) {
      if (inst->opcode() != spv::Op::OpLoad &&
          inst->opcode() != spv::Op::OpStore) {
        return;
      }
      const auto* pointer = def_use->GetDef(inst->GetSingleWordInOperand(0));
      if (!pointer) return;
      const auto* pointer_type = def_use->GetDef(pointer->type_id());
      if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer ||
          pointer_type->GetSingleWordInOperand(0) !=
              static_cast<uint32_t>(spv::StorageClass::StorageBuffer)) {
        return;
      }
      // Even packed cooperative values must access one original buffer lane at
      // a time. This also catches a conversion skipped by a pointer helper.
      EXPECT_TRUE(IsScalarType(pointer_type->GetSingleWordInOperand(1),
                               *test.memory_type));
      if (inst->opcode() == spv::Op::OpLoad) {
        ++load_count;
        EXPECT_TRUE(IsScalarType(inst->type_id(), *test.memory_type));
        EXPECT_TRUE(HasConversionUsers(inst->result_id(), test.load_steps));
      } else {
        ++store_count;
        EXPECT_TRUE(HasConversionDefinitions(inst->GetSingleWordInOperand(1),
                                             test.store_steps,
                                             *test.cooperative_type));
      }
    });
    EXPECT_GT(load_count, 0u);
    EXPECT_GT(store_count, 0u);
  }

  std::string RunMemoryConversionCase(const MemoryConversionCase& test,
                                      const std::string& text,
                                      HwLowerToStandardPass::LoweringMode mode,
                                      uint32_t max_unrolled_elements = 4096) {
    constexpr auto env = SPV_ENV_UNIVERSAL_1_3;
    SpirvTools tools(env);
    tools.SetMessageConsumer(consumer());
    std::vector<uint32_t> binary;
    if (!tools.Assemble(text, &binary) || !tools.Validate(binary)) {
      ADD_FAILURE() << "Invalid original HW memory module:\n" << text;
      return {};
    }
    SetTargetEnv(env);
    const auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
        text, true, true, mode,
        HwLowerToStandardPass::CompletenessMode::kExtensionFree, 1048576u,
        16777216ull, max_unrolled_elements, 4096ull);
    EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
    if (std::get<1>(result) == Pass::Status::Failure) return {};
    const auto& disassembly = std::get<0>(result);
    ExpectNoHwOrCoopMatrix(disassembly);
    EXPECT_EQ(std::string::npos, disassembly.find("ReplicateEXT"));
    CheckMemoryConversionChains(test);
    return disassembly;
  }
};

TEST_P(HwLowerToStandardMemoryConversionTest,
       LoadAndStoreUseScalarConversionInBothLoweringModes) {
  for (const bool matrix : {false, true}) {
    for (const auto mode :
         {HwLowerToStandardPass::LoweringMode::kPreferPackedVec2,
          HwLowerToStandardPass::LoweringMode::kForceScalar}) {
      SCOPED_TRACE(matrix ? "matrix" : "vector");
      SCOPED_TRACE(static_cast<int>(mode));
      RunMemoryConversionCase(
          GetParam(), MakeMemoryConversionModule(GetParam(), matrix), mode);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ScalarPairs, HwLowerToStandardMemoryConversionTest,
    ::testing::ValuesIn(kMemoryConversionCases),
    [](const ::testing::TestParamInfo<MemoryConversionCase>& info) {
      return info.param.name;
    });

TEST_F(HwLowerToStandardMemoryConversionTest,
       CrossWidthNestedVectorLeafPreservesMatrixSubregionAndDynamicOffset) {
  const auto& test = kMemoryConversionCases[0];  // f32 memory, f16 values.
  for (const bool matrix : {false, true}) {
    for (const bool column_major : {false, true}) {
      for (const auto mode :
           {HwLowerToStandardPass::LoweringMode::kPreferPackedVec2,
            HwLowerToStandardPass::LoweringMode::kForceScalar}) {
        SCOPED_TRACE(matrix ? "matrix" : "vector");
        SCOPED_TRACE(column_major ? "column major" : "row major");
        SCOPED_TRACE(static_cast<int>(mode));
        const auto text =
            MakeMemoryConversionModule(test, matrix, true, column_major);
        const auto disassembly = RunMemoryConversionCase(test, text, mode);
        EXPECT_NE(
            std::string::npos,
            disassembly.find("OpDecorate %_arr_v4float_uint_2 ArrayStride 16"));
        EXPECT_GT(CountSubstring(disassembly, "OpUDiv %uint"), 0u);
        EXPECT_GT(CountSubstring(disassembly, "OpUMod %uint"), 0u);
        EXPECT_EQ(0u, CountSubstring(disassembly, "OpFunctionCall"));
      }
    }
  }
}

TEST_F(HwLowerToStandardMemoryConversionTest,
       CrossWidthStructuredLoopsConvertEachMemoryLane) {
  for (const auto* test :
       {&kMemoryConversionCases[0], &kMemoryConversionCases[9]}) {
    for (const bool matrix : {false, true}) {
      for (const auto mode :
           {HwLowerToStandardPass::LoweringMode::kPreferPackedVec2,
            HwLowerToStandardPass::LoweringMode::kForceScalar}) {
        SCOPED_TRACE(test->name);
        SCOPED_TRACE(matrix ? "matrix" : "vector");
        SCOPED_TRACE(static_cast<int>(mode));
        const auto disassembly = RunMemoryConversionCase(
            *test, MakeMemoryConversionModule(*test, matrix, true), mode,
            /*max_unrolled_elements=*/2u);
        EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
        EXPECT_EQ(0u, CountSubstring(disassembly, "OpFunctionCall"));
      }
    }
  }
}

TEST_F(HwLowerToStandardMemoryConversionTest,
       CrossWidthStorageOnlyLeafPromotesRequiredArithmeticCapability) {
  for (const auto* test :
       {&kMemoryConversionCases[1], &kMemoryConversionCases[5],
        &kMemoryConversionCases[3]}) {
    for (const bool matrix : {false, true}) {
      for (const auto mode :
           {HwLowerToStandardPass::LoweringMode::kPreferPackedVec2,
            HwLowerToStandardPass::LoweringMode::kForceScalar}) {
        SCOPED_TRACE(test->name);
        SCOPED_TRACE(matrix ? "matrix" : "vector");
        SCOPED_TRACE(static_cast<int>(mode));
        const std::string capability =
            std::string("OpCapability ") + test->memory_type->capability + "\n";
        const auto text =
            MakeMemoryConversionModule(*test, matrix, false, false, true);
        ASSERT_EQ(0u, CountSubstring(text, capability));
        const auto disassembly = RunMemoryConversionCase(*test, text, mode);
        EXPECT_EQ(1u, CountSubstring(disassembly, capability));
      }
    }
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
