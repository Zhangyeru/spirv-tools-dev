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

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
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

}  // namespace
}  // namespace opt
}  // namespace spvtools
