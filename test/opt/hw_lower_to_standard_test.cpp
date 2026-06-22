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

using HwLowerToStandardTest = PassTest<::testing::Test>;

size_t CountSubstring(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

void ExpectNoHwOrCoopMatrix(const std::string& text) {
  EXPECT_EQ(0u, CountSubstring(text, "CooperativeMatrixHW"));
  EXPECT_EQ(0u, CountSubstring(text, "CooperativeVectorHW"));
  EXPECT_EQ(0u, CountSubstring(text, "SPV_HW_neural_shader"));
  EXPECT_EQ(0u, CountSubstring(text, "GL_HW_neural_shader"));
  EXPECT_EQ(0u, CountSubstring(text, "CooperativeMatrixKHR"));
  EXPECT_EQ(0u, CountSubstring(text, "OpTypeCooperativeMatrixKHR"));
  EXPECT_EQ(0u, CountSubstring(text, "OpCooperativeMatrix"));
}

void ExpectPackedVec4Math(const std::string& text,
                          const std::string& component_name) {
  ExpectNoHwOrCoopMatrix(text);
  EXPECT_NE(std::string::npos,
            text.find("OpTypeVector " + component_name + " 4"));
  EXPECT_GT(CountSubstring(text, "OpTypeArray %v4"), 0u);
  EXPECT_GT(CountSubstring(text, "OpExtInst %v4"), 0u);
  EXPECT_GT(CountSubstring(text, " Fma "), 0u);
  EXPECT_EQ(0u, CountSubstring(text, "OpVectorTimesScalar"));
}

void ExpectPackedVec4MatmulPattern(const std::string& text,
                                   const std::string& component_name) {
  ExpectNoHwOrCoopMatrix(text);
  EXPECT_NE(std::string::npos,
            text.find("OpTypeVector " + component_name + " 4"));
  EXPECT_GT(CountSubstring(text, "OpTypeArray %v4"), 0u);
  EXPECT_GT(CountSubstring(text, "OpExtInst %v4"), 0u);
  EXPECT_GT(CountSubstring(text, " Fma "), 0u);
  EXPECT_GT(CountSubstring(text, "OpCompositeExtract " + component_name), 0u);
  EXPECT_GT(CountSubstring(text, "OpFunctionCall"), 0u);
  EXPECT_GT(CountSubstring(text, "OpReturnValue"), 0u);
  EXPECT_EQ(0u, CountSubstring(text, "OpVectorTimesScalar"));
}

void ExpectScalarFallbackMath(const std::string& text,
                              const std::string& component_name = "%float") {
  ExpectNoHwOrCoopMatrix(text);
  EXPECT_GT(CountSubstring(text, "OpExtInst " + component_name), 0u);
  EXPECT_GT(CountSubstring(text, " Fma "), 0u);
  EXPECT_EQ(0u, CountSubstring(text, "OpFMul"));
  EXPECT_EQ(0u, CountSubstring(text, "OpFAdd " + component_name));
  EXPECT_EQ(0u, CountSubstring(text, "OpVectorTimesScalar"));
}

std::string RunWithInjectedAlignedMemoryAccess(HwLowerToStandardTest* test,
                                               const std::string& text,
                                               spv::Op opcode) {
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_3, test->consumer(), text,
                             SpirvTools::kDefaultAssembleOption);
  EXPECT_NE(nullptr, context.get()) << "Assembling failed for shader:\n"
                                    << text << std::endl;
  if (!context) return std::string();

  bool injected = false;
  context->module()->ForEachInst([opcode, &injected](Instruction* inst) {
    if (injected || inst->opcode() != opcode) return;
    inst->AddOperand({SPV_OPERAND_TYPE_MEMORY_ACCESS,
                      {static_cast<uint32_t>(spv::MemoryAccessMask::Aligned)}});
    inst->AddOperand({SPV_OPERAND_TYPE_LITERAL_INTEGER, {16}});
    injected = true;
  });
  EXPECT_TRUE(injected);
  if (!injected) return std::string();

  HwLowerToStandardPass pass;
  pass.SetMessageConsumer(test->consumer());
  const Pass::Status status = pass.Run(context.get());
  EXPECT_EQ(Pass::Status::SuccessWithChange, status);
  if (status == Pass::Status::Failure) return std::string();

  std::vector<uint32_t> binary;
  context->module()->ToBinary(&binary, /* skip_nop = */ true);

  spv_context spv_context = spvContextCreate(SPV_ENV_UNIVERSAL_1_3);
  spv_diagnostic diagnostic = nullptr;
  spv_const_binary_t spv_binary = {binary.data(), binary.size()};
  const spv_result_t error = spvValidateWithOptions(
      spv_context, test->ValidatorOptions(), &spv_binary, &diagnostic);
  EXPECT_EQ(SPV_SUCCESS, error);
  if (error != SPV_SUCCESS) spvDiagnosticPrint(diagnostic);
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(spv_context);

  std::string disassembly;
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  EXPECT_TRUE(tools.Disassemble(binary, &disassembly,
                                SpirvTools::kDefaultDisassembleOption));
  return disassembly;
}

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

TEST_F(HwLowerToStandardTest, LowersMatmul4x4x4F32Tiled) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_4
; CHECK: OpExtInst %v4float
; CHECK-SAME: Fma
; CHECK: OpCompositeConstruct
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpSourceExtension "GL_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_16 = OpConstant %uint 16
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%a = OpUndef %mat
%b = OpUndef %mat
%c = OpConstantNull %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4MatmulPattern(std::get<0>(result), "%float");
}

TEST_F(HwLowerToStandardTest, ReusesPackedVec4MatmulPatternFunction) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpFunctionCall
; CHECK: OpFunctionCall
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
%b = OpUndef %mat
%c = OpConstantNull %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d0 = OpCooperativeMatrixMulAddHW %mat %a %b %c
%d1 = OpCooperativeMatrixMulAddHW %mat %d0 %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& output = std::get<0>(result);
  ExpectPackedVec4MatmulPattern(output, "%float");
  EXPECT_EQ(2u, CountSubstring(output, "OpFunctionCall"));
  EXPECT_EQ(4u, CountSubstring(output, "OpExtInst %v4float"));
  EXPECT_EQ(3u, CountSubstring(output, "OpFunctionParameter"));
}

TEST_F(HwLowerToStandardTest, LowersMatmul3x5x4F32TiledTail) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_3
; CHECK: OpTypeArray %float %uint_20
; CHECK: OpTypeArray %float %uint_15
; CHECK: OpExtInst %float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
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
%mat3x4 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_4
%mat4x5 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_5
%mat3x5 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_5
%a = OpUndef %mat3x4
%b = OpUndef %mat4x5
%c = OpConstantNull %mat3x5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat3x5 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectScalarFallbackMath(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowersMatmul2x4x4F32ExactTile) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_2
; CHECK: OpTypeArray %v4float %uint_4
; CHECK: OpExtInst %v4float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%uint_16 = OpConstant %uint 16
%float = OpTypeFloat 32
%mat2x4 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_4
%mat4x4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%a = OpUndef %mat2x4
%b = OpUndef %mat4x4
%c = OpConstantNull %mat2x4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat2x4 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4MatmulPattern(std::get<0>(result), "%float");
}

TEST_F(HwLowerToStandardTest, LowersMatmul3x5x8F32PackedKTail) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeArray %float %uint_15
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_10
; CHECK: OpTypeArray %v4float %uint_6
; CHECK: OpExtInst %v4float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_5 = OpConstant %uint 5
%uint_6 = OpConstant %uint 6
%uint_8 = OpConstant %uint 8
%uint_10 = OpConstant %uint 10
%uint_15 = OpConstant %uint 15
%float = OpTypeFloat 32
%mat3x5 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_5
%mat5x8 = OpTypeCooperativeMatrixHW %float %uint_5 %uint_8
%mat3x8 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_8
%a = OpUndef %mat3x5
%b = OpUndef %mat5x8
%c = OpConstantNull %mat3x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat3x8 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4MatmulPattern(std::get<0>(result), "%float");
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

TEST_F(HwLowerToStandardTest, LowersMatrixLoadStoreF32) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_4
; CHECK-DAG: OpLoopMerge
; CHECK-DAG: OpLoad %float
; CHECK-DAG: OpCompositeConstruct
; CHECK-DAG: OpCompositeExtract %float
; CHECK-DAG: OpFunctionCall %v4float
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

TEST_F(HwLowerToStandardTest, LowersVectorMatrixMul8x8F32Tile4) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_2
; CHECK: OpTypeArray %v4float %uint_16
; CHECK: OpExtInst %v4float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpSourceExtension "GL_HW_neural_shader"
OpSourceExtension "GL_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%uint_64 = OpConstant %uint 64
%float = OpTypeFloat 32
%vec = OpTypeCooperativeVectorHW %float %uint_8
%mat = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%x = OpUndef %vec
%w = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4Math(std::get<0>(result), "%float");
}

TEST_F(HwLowerToStandardTest, LowersVectorMatrixMulAdd8x8F32Tile4WithBias) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_2
; CHECK: OpTypeArray %v4float %uint_16
; CHECK: OpExtInst %v4float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%uint_64 = OpConstant %uint 64
%float = OpTypeFloat 32
%vec = OpTypeCooperativeVectorHW %float %uint_8
%mat = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%x = OpUndef %vec
%w = OpUndef %mat
%bias = OpConstantNull %vec
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulAddHW %vec %x %w %bias
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4Math(std::get<0>(result), "%float");
}

TEST_F(HwLowerToStandardTest, LowersVectorMatrixMulTailNNotMultipleOf4) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_2
; CHECK: OpTypeArray %float %uint_10
; CHECK: OpTypeArray %float %uint_80
; CHECK: OpExtInst %float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%uint_10 = OpConstant %uint 10
%uint_80 = OpConstant %uint 80
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%vec10 = OpTypeCooperativeVectorHW %float %uint_10
%mat8x10 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_10
%x = OpUndef %vec8
%w = OpUndef %mat8x10
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec10 %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectScalarFallbackMath(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest, LowersVectorMatrixMulF32PackedKTail) {
  const std::string text = R"(
; CHECK-NOT: CooperativeMatrixHW
; CHECK-NOT: CooperativeVectorHW
; CHECK-NOT: CooperativeMatrixKHR
; CHECK-NOT: OpTypeCooperativeMatrixKHR
; CHECK-NOT: OpCooperativeMatrix
; CHECK: OpTypeArray %float %uint_5
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_2
; CHECK: OpTypeArray %v4float %uint_10
; CHECK: OpExtInst %float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_5 = OpConstant %uint 5
%uint_8 = OpConstant %uint 8
%uint_40 = OpConstant %uint 40
%float = OpTypeFloat 32
%vec5 = OpTypeCooperativeVectorHW %float %uint_5
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat5x8 = OpTypeCooperativeMatrixHW %float %uint_5 %uint_8
%x = OpUndef %vec5
%w = OpUndef %mat5x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpReturn
OpFunctionEnd
  )";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& output = std::get<0>(result);
  ExpectScalarFallbackMath(output, "%float");
  EXPECT_NE(std::string::npos, output.find("OpTypeVector %float 4"));
  EXPECT_GT(CountSubstring(output, "OpTypeArray %v4"), 0u);
}

TEST_F(HwLowerToStandardTest,
       VectorMatrixMulPointerPathBlockedByInterveningFunctionStore) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpFunctionCall %_arr_v4float_uint_2
; CHECK-NOT: OpFunctionParameter %_ptr_Function
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%zero = OpConstantNull %vec8
%_ptr_Function_vec8 = OpTypePointer Function %vec8
%_ptr_Function_mat8x8 = OpTypePointer Function %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xvar = OpVariable %_ptr_Function_vec8 Function
%wvar = OpVariable %_ptr_Function_mat8x8 Function
%x = OpLoad %vec8 %xvar
OpStore %xvar %zero
%w = OpLoad %mat8x8 %wvar
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpFunctionCall %_arr_v4float_uint_2"));
  EXPECT_EQ(std::string::npos,
            disassembly.find("OpFunctionParameter %_ptr_Function"));
}

TEST_F(HwLowerToStandardTest,
       VectorMatrixMulPointerPathBlockedByVolatileFunctionLoad) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpFunctionCall %_arr_v4float_uint_2
; CHECK-NOT: OpFunctionParameter %_ptr_Function
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%_ptr_Function_vec8 = OpTypePointer Function %vec8
%_ptr_Function_mat8x8 = OpTypePointer Function %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xvar = OpVariable %_ptr_Function_vec8 Function
%wvar = OpVariable %_ptr_Function_mat8x8 Function
%x = OpLoad %vec8 %xvar Volatile
%w = OpLoad %mat8x8 %wvar
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpFunctionCall %_arr_v4float_uint_2"));
  EXPECT_EQ(std::string::npos,
            disassembly.find("OpFunctionParameter %_ptr_Function"));
}

TEST_F(HwLowerToStandardTest,
       FusedVectorMatmulStoreBlockedByInterveningStorageStore) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpFunctionCall %_arr_v4float_uint_2
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %buf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %base %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %base %shape %offset %int_0
%elem0 = OpAccessChain %_ptr_StorageBuffer_float %base %int_0
OpStore %elem0 %float_1
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpCooperativeVectorStoreHW %base %int_0 %y
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpFunctionCall %_arr_v4float_uint_2"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulElidesSeparateLoadLoops) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability CooperativeVectorHW
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x4 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_4
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x4 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulHW %vec4 %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(12u, CountSubstring(disassembly, "OpFAdd %float"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulAddElidesSeparateLoadLoops) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability CooperativeVectorHW
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x4 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_4
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x4 %wbase %shape %offset %int_0
%bias = OpCooperativeVectorLoadHW %vec4 %bbase %int_0
%y = OpCooperativeVectorMatrixMulAddHW %vec4 %x %w %bias
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(12u, CountSubstring(disassembly, "OpFAdd %float"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulElidesSeparateLoadLoopsFromUniform) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_float ArrayStride 4
OpDecorate %_arr_float_uint_32 ArrayStride 4
OpMemberDecorate %StorageBuf 0 Offset 0
OpMemberDecorate %UniformBuf 0 Offset 0
OpDecorate %StorageBuf Block
OpDecorate %UniformBuf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%uint_16 = OpConstant %uint 16
%uint_32 = OpConstant %uint 32
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_16 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%vec16 = OpTypeCooperativeVectorHW %float %uint_16
%mat16x8 = OpTypeCooperativeMatrixHW %float %uint_16 %uint_8
%_runtimearr_float = OpTypeRuntimeArray %float
%_arr_float_uint_32 = OpTypeArray %float %uint_32
%StorageBuf = OpTypeStruct %_runtimearr_float
%UniformBuf = OpTypeStruct %_arr_float_uint_32
%_ptr_StorageBuffer_StorageBuf = OpTypePointer StorageBuffer %StorageBuf
%_ptr_Uniform_UniformBuf = OpTypePointer Uniform %UniformBuf
%xbuf = OpVariable %_ptr_StorageBuffer_StorageBuf StorageBuffer
%wbuf = OpVariable %_ptr_Uniform_UniformBuf Uniform
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%_ptr_Uniform__arr_float_uint_32 = OpTypePointer Uniform %_arr_float_uint_32
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_Uniform__arr_float_uint_32 %wbuf %int_0
%x = OpCooperativeVectorLoadHW %vec16 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat16x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(12u, CountSubstring(disassembly, "OpFAdd %float"));
}

TEST_F(HwLowerToStandardTest, LowersVectorMatrixMulAddWithConstantBias) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability CooperativeVectorHW
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%float_n3q = OpConstant %float -0.75
%float_n2q = OpConstant %float -0.5
%float_n1q = OpConstant %float -0.25
%float_0 = OpConstant %float 0
%float_1q = OpConstant %float 0.25
%float_2q = OpConstant %float 0.5
%float_3q = OpConstant %float 0.75
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x4 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_4
%bias = OpConstantComposite %vec4 %float_n3q %float_n2q %float_n1q %float_0
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x4 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulAddHW %vec4 %x %w %bias
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 2u);
  EXPECT_EQ(12u, CountSubstring(disassembly, "OpFAdd %float"));
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(4u, CountSubstring(disassembly, "OpConstantComposite %v4float"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixMulAddWithConstantMatrix) {
  const std::string text = R"(
; CHECK-NOT: HW
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
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%float_2 = OpConstant %float 2
%float_3 = OpConstant %float 3
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_4 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%mat4x4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%abuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat4x4 %abase %shape %offset %int_0
%b = OpCooperativeMatrixLoadHW %mat4x4 %bbase %shape %offset %int_0
%c = OpCompositeConstruct %mat4x4
  %float_0 %float_1 %float_2 %float_3
  %float_1 %float_2 %float_3 %float_0
  %float_2 %float_3 %float_0 %float_1
  %float_3 %float_0 %float_1 %float_2
%d = OpCooperativeMatrixMulAddHW %mat4x4 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  ExpectPackedVec4MatmulPattern(disassembly, "%float");
}

TEST_F(HwLowerToStandardTest, DirectMatmulAddElidesSeparateLoadLoops) {
  const std::string text = R"(
; CHECK-NOT: HW
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
%uint_16 = OpConstant %uint 16
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_16 %uint_16
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%mat16x16 = OpTypeCooperativeMatrixHW %float %uint_16 %uint_16
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%abuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%cbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat16x16 %abase %shape %offset %int_0
%b = OpCooperativeMatrixLoadHW %mat16x16 %bbase %shape %offset %int_0
%c = OpCooperativeMatrixLoadHW %mat16x16 %cbase %shape %offset %int_0
%d = OpCooperativeMatrixMulAddHW %mat16x16 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(6u, CountSubstring(disassembly, "OpFunctionCall %v4float"));
  EXPECT_GE(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_64"),
            1u);
}

TEST_F(HwLowerToStandardTest, FusedMatrixMatmulStoreStreamsToSsbo) {
  const std::string text = R"(
; CHECK-NOT: HW
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%abuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%cbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%dbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%dbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %dbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat8x8 %abase %shape %offset %int_0
%b = OpCooperativeMatrixLoadHW %mat8x8 %bbase %shape %offset %int_0
%c = OpCooperativeMatrixLoadHW %mat8x8 %cbase %shape %offset %int_0
%d = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %c
OpCooperativeMatrixStoreHW %dbase %d %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  // Fused path emits a void function with 3 nested loops (row, col_pack,
  // k_pack) and NO matmul-pattern function returning the result array.
  EXPECT_EQ(3u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpTypeFunction %_arr_v4float_uint_16"));
}

TEST_F(HwLowerToStandardTest,
       FusedMatrixMatmulStoreBlockedByInterveningStorageStore) {
  const std::string text = R"(
; CHECK-NOT: HW
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%float_1 = OpConstant %float 1
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%abuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%cbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%dbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%dbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %dbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat8x8 %abase %shape %offset %int_0
%b = OpCooperativeMatrixLoadHW %mat8x8 %bbase %shape %offset %int_0
%c = OpCooperativeMatrixLoadHW %mat8x8 %cbase %shape %offset %int_0
%d = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %c
%elem0 = OpAccessChain %_ptr_StorageBuffer_float %dbase %int_0
OpStore %elem0 %float_1
OpCooperativeMatrixStoreHW %dbase %d %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  // Fusion blocked: the non-fused matmul-pattern function (returning the
  // result array) is still emitted.
  EXPECT_NE(std::string::npos,
            disassembly.find("OpTypeFunction %_arr_v4float_uint_16"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulFallsBackWhenLoadResultIsReused) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability CooperativeVectorHW
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
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%outbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%outbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %outbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpCooperativeVectorStoreHW %outbase %int_0 %x
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(5u, CountSubstring(disassembly, "OpLoopMerge"));
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
%m = OpCompositeConstruct %mat2x8 %h0 %h1 %h2 %h3 %h4 %h5 %h6 %h7 %h8 %h9 %h10 %h11 %h12 %h13 %h14 %h15
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
%m = OpCompositeConstruct %mat2x8 %f0 %f1 %f2 %f3 %f4 %f5 %f6 %f7 %f8 %f9 %f10 %f11 %f12 %f13 %f14 %f15
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
%m = OpCompositeConstruct %mat3x5 %f0 %f1 %f2 %f3 %f4 %f5 %f6 %f7 %f8 %f9 %f10 %f11 %f12 %f13 %f14
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
%m = OpCompositeConstruct %mat2x5 %h0 %h1 %h2 %h3 %h4 %h5 %h6 %h7 %h8 %h9
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
; CHECK: [[PACK:%\w+]] = OpCompositeExtract %v4half {{%\w+}} 7
; CHECK: {{%\w+}} = OpCompositeExtract %half [[PACK]] 2
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
; CHECK: [[PACK:%\w+]] = OpCompositeExtract %v4float {{%\w+}} 7
; CHECK: {{%\w+}} = OpCompositeExtract %float [[PACK]] 2
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
; CHECK: [[PACK:%\w+]] = OpCompositeExtract %v4half {{%\w+}} 1
; CHECK: {{%\w+}} = OpCompositeExtract %half [[PACK]] 2
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
; CHECK: [[PACK:%\w+]] = OpCompositeExtract %v4float {{%\w+}} 1
; CHECK: {{%\w+}} = OpCompositeExtract %float [[PACK]] 2
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

TEST_F(HwLowerToStandardTest, MatrixLoadPreservesAlignedMemoryAccess) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpLoad %float {{%\w+}} Aligned 16
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
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 16"));
}

TEST_F(HwLowerToStandardTest, MatrixStorePreservesAlignedMemoryAccess) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpStore {{%\w+}} {{%\w+}} Aligned 16
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
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 16"));
}

TEST_F(HwLowerToStandardTest, VectorLoadPreservesAlignedMemoryAccess) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpLoad %float {{%\w+}} Aligned 16
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
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 16"));
}

TEST_F(HwLowerToStandardTest, VectorStorePreservesAlignedMemoryAccess) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpStore {{%\w+}} {{%\w+}} Aligned 16
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
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 16"));
}

TEST_F(HwLowerToStandardTest,
       ForceScalarModeLowersF16VectorLoadStoreToScalar) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %half %uint_8
; CHECK: OpLoad %half
; CHECK: OpStore
; CHECK-NOT: OpTypeVector %half 4
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %half 4"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFunctionCall"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
}

TEST_F(HwLowerToStandardTest,
       ForceScalarModeLowersF16MatrixLoadStoreToScalar) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %half %uint_16
; CHECK: OpLoad %half
; CHECK: OpStore
; CHECK-NOT: OpTypeVector %half 4
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %half 4"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFunctionCall"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
}

TEST_F(HwLowerToStandardTest, LowersVectorLoadStoreF16Packed) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 4
; CHECK: OpTypeArray %v4half %uint_2
; CHECK-DAG: OpLoopMerge
; CHECK-DAG: OpLoad %half
; CHECK-DAG: OpLoad %v4half
; CHECK-DAG: OpCompositeExtract %half
; CHECK-DAG: OpFunctionCall %v4half
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
; CHECK: OpTypeVector %half 4
; CHECK: OpTypeArray %v4half %uint_8
; CHECK-DAG: OpLoopMerge
; CHECK-DAG: OpLoad %half
; CHECK-DAG: OpLoad %v4half
; CHECK-DAG: OpCompositeExtract %half
; CHECK-DAG: OpFunctionCall %v4half
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
; CHECK: OpTypeVector %half 4
; CHECK: OpTypeArray %v4half %uint_8
; CHECK: OpLoad %half
; CHECK: OpCompositeConstruct %v4half
; CHECK: OpCompositeExtract %v4half
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

TEST_F(HwLowerToStandardTest, LowersVectorMatrixMul16x64F16Packed) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 4
; CHECK: OpExtInst %v4half
; CHECK-SAME: Fma
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_16 = OpConstant %uint 16
%uint_64 = OpConstant %uint 64
%half = OpTypeFloat 16
%vec16 = OpTypeCooperativeVectorHW %half %uint_16
%vec64 = OpTypeCooperativeVectorHW %half %uint_64
%mat16x64 = OpTypeCooperativeMatrixHW %half %uint_16 %uint_64
%x = OpUndef %vec16
%w = OpUndef %mat16x64
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec64 %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectPackedVec4Math(disassembly, "%half");
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(12u, CountSubstring(disassembly, "OpFAdd %half"));
}

TEST_F(HwLowerToStandardTest, LowersVectorMatrixMulAdd16x64F16Packed) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 4
; CHECK: OpExtInst %v4half
; CHECK-SAME: Fma
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_16 = OpConstant %uint 16
%uint_64 = OpConstant %uint 64
%half = OpTypeFloat 16
%vec16 = OpTypeCooperativeVectorHW %half %uint_16
%vec64 = OpTypeCooperativeVectorHW %half %uint_64
%mat16x64 = OpTypeCooperativeMatrixHW %half %uint_16 %uint_64
%x = OpUndef %vec16
%w = OpUndef %mat16x64
%bias = OpConstantNull %vec64
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulAddHW %vec64 %x %w %bias
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectPackedVec4Math(disassembly, "%half");
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(12u, CountSubstring(disassembly, "OpFAdd %half"));
}

TEST_F(HwLowerToStandardTest, LowersMatrixMulAdd4x16x8F16Packed) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 4
; CHECK: OpExtInst %v4half
; CHECK-SAME: Fma
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%uint_16 = OpConstant %uint 16
%half = OpTypeFloat 16
%mat4x16 = OpTypeCooperativeMatrixHW %half %uint_4 %uint_16
%mat16x8 = OpTypeCooperativeMatrixHW %half %uint_16 %uint_8
%mat4x8 = OpTypeCooperativeMatrixHW %half %uint_4 %uint_8
%a = OpUndef %mat4x16
%b = OpUndef %mat16x8
%c = OpConstantNull %mat4x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat4x8 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4MatmulPattern(std::get<0>(result), "%half");
}

TEST_F(HwLowerToStandardTest, LowersMatrixMulAdd16x16x16F16Packed) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 4
; CHECK: OpExtInst %v4half
; CHECK-SAME: Fma
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_16 = OpConstant %uint 16
%half = OpTypeFloat 16
%mat16x16 = OpTypeCooperativeMatrixHW %half %uint_16 %uint_16
%a = OpUndef %mat16x16
%b = OpUndef %mat16x16
%c = OpConstantNull %mat16x16
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat16x16 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4MatmulPattern(std::get<0>(result), "%half");
}

TEST_F(HwLowerToStandardTest, LowersPackedF16MatrixMulAdd8x8) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 16
; CHECK: OpTypeVector %half 4
; CHECK: OpTypeArray %v4half %uint_16
; CHECK: OpExtInst %v4half
; CHECK-SAME: Fma
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
%a = OpUndef %mat8x8
%b = OpUndef %mat8x8
%c = OpConstantNull %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4MatmulPattern(std::get<0>(result), "%half");
}

TEST_F(HwLowerToStandardTest, LowersPackedF32MatrixMulAdd8x8) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeFloat 32
; CHECK: OpTypeVector %float 4
; CHECK: OpTypeArray %v4float %uint_16
; CHECK: OpExtInst %v4float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_8 = OpConstant %uint 8
%uint_16 = OpConstant %uint 16
%float = OpTypeFloat 32
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%a = OpUndef %mat8x8
%b = OpUndef %mat8x8
%c = OpConstantNull %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectPackedVec4MatmulPattern(std::get<0>(result), "%float");
}

TEST_F(HwLowerToStandardTest, F32MatrixMulAddUsesScalarFallbackWhenUnaligned) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %float %uint_15
; CHECK: OpTypeArray %float %uint_9
; CHECK: OpExtInst %float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_5 = OpConstant %uint 5
%uint_9 = OpConstant %uint 9
%uint_15 = OpConstant %uint 15
%float = OpTypeFloat 32
%mat3x5 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_5
%mat5x3 = OpTypeCooperativeMatrixHW %float %uint_5 %uint_3
%mat3x3 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_3
%a = OpUndef %mat3x5
%b = OpUndef %mat5x3
%c = OpConstantNull %mat3x3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat3x3 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectScalarFallbackMath(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest,
       F32VectorMatrixMulUsesScalarFallbackWhenUnaligned) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %float %uint_5
; CHECK: OpTypeArray %float %uint_3
; CHECK: OpTypeArray %float %uint_15
; CHECK: OpExtInst %float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_5 = OpConstant %uint 5
%uint_15 = OpConstant %uint 15
%float = OpTypeFloat 32
%vec5 = OpTypeCooperativeVectorHW %float %uint_5
%vec3 = OpTypeCooperativeVectorHW %float %uint_3
%mat5x3 = OpTypeCooperativeMatrixHW %float %uint_5 %uint_3
%x = OpUndef %vec5
%w = OpUndef %mat5x3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulHW %vec3 %x %w
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectScalarFallbackMath(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest,
       F32VectorMatrixMulAddUsesScalarFallbackWhenUnaligned) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %float %uint_5
; CHECK: OpTypeArray %float %uint_3
; CHECK: OpTypeArray %float %uint_15
; CHECK: OpExtInst %float
; CHECK-SAME: Fma
OpCapability Shader
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_5 = OpConstant %uint 5
%uint_15 = OpConstant %uint 15
%float = OpTypeFloat 32
%vec5 = OpTypeCooperativeVectorHW %float %uint_5
%vec3 = OpTypeCooperativeVectorHW %float %uint_3
%mat5x3 = OpTypeCooperativeMatrixHW %float %uint_5 %uint_3
%x = OpUndef %vec5
%w = OpUndef %mat5x3
%bias = OpConstantNull %vec3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpCooperativeVectorMatrixMulAddHW %vec3 %x %w %bias
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  ExpectScalarFallbackMath(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest,
       ForceScalarModeUsesScalarFallbackForAlignedF16MatrixMulAdd) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpTypeArray %half %uint_16
; CHECK: OpExtInst %half
; CHECK-SAME: Fma
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_16 = OpConstant %uint 16
%half = OpTypeFloat 16
%mat4x4 = OpTypeCooperativeMatrixHW %half %uint_4 %uint_4
%a = OpUndef %mat4x4
%b = OpUndef %mat4x4
%c = OpConstantNull %mat4x4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat4x4 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(
      text, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  ExpectScalarFallbackMath(disassembly, "%half");
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %half 4"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpFunctionCall"));
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

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
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

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
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

TEST_F(HwLowerToStandardTest, FailsForHwFunctionParameter) {
  const std::string text = R"(
; CHECK: HW cooperative values across function boundaries are not supported
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

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsForHwFunctionReturn) {
  const std::string text = R"(
; CHECK: HW cooperative values across function boundaries are not supported
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
%fn_mat = OpTypeFunction %mat
%main = OpFunction %mat None %fn_mat
%entry = OpLabel
OpReturnValue %a
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsForUnsupportedHwPhi) {
  const std::string text = R"(
; CHECK: HW cooperative OpPhi is not supported
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
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %merge
%merge = OpLabel
%p = OpPhi %mat %a %entry
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsForPointerEscapeOrNonFunctionHwPointer) {
  const std::string text = R"(
; CHECK: HW cooperative values may only be stored in Function variables before lowering
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
; CHECK: HW cooperative values may only be stored in Function variables before lowering
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
; CHECK: HW cooperative values may only be stored in Function variables before lowering
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

TEST_F(HwLowerToStandardTest, FailsForUnsupportedMatrixReduce) {
  const std::string text = R"(
; CHECK: OpCooperativeMatrixReduceHW is not supported
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%r = OpCooperativeMatrixReduceHW %mat %a %uint_0 %uint_0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
