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

bool HasHalfVec4WithZeroTail(const std::string& disassembly,
                             size_t minimum_zero_lanes) {
  size_t zero_instruction_pos = disassembly.find(" = OpConstantNull %half");
  if (zero_instruction_pos == std::string::npos) {
    zero_instruction_pos = disassembly.find(" = OpConstant %half 0");
  }
  if (zero_instruction_pos == std::string::npos) return false;

  size_t zero_id_begin = disassembly.rfind('\n', zero_instruction_pos);
  zero_id_begin = zero_id_begin == std::string::npos ? 0 : zero_id_begin + 1;
  const size_t first_id_character =
      disassembly.find_first_not_of(" \t", zero_id_begin);
  if (first_id_character == std::string::npos ||
      first_id_character >= zero_instruction_pos) {
    return false;
  }
  const std::string zero_id = disassembly.substr(
      first_id_character, zero_instruction_pos - first_id_character);
  if (zero_id.empty()) return false;

  size_t line_begin = 0;
  while (line_begin < disassembly.size()) {
    const size_t line_end = disassembly.find('\n', line_begin);
    const std::string line = disassembly.substr(
        line_begin, line_end == std::string::npos ? std::string::npos
                                                  : line_end - line_begin);
    if (line.find("OpCompositeConstruct %v4half") != std::string::npos &&
        CountSubstring(line, " " + zero_id) >= minimum_zero_lanes) {
      return true;
    }
    if (line_end == std::string::npos) break;
    line_begin = line_end + 1;
  }
  return false;
}

TEST_F(HwLowerToStandardTest, LowersMatmulWithCollidingLoweredMatrixTypes) {
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
%uint_3 = OpConstant %uint 3
%float = OpTypeFloat 32
%mat2x3 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_3
%mat3x2 = OpTypeCooperativeMatrixHW %float %uint_3 %uint_2
%mat2x2 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%a = OpUndef %mat2x3
%b = OpUndef %mat3x2
%c = OpUndef %mat2x2
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpCooperativeMatrixMulAddHW %mat2x2 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(12u, CountSubstring(disassembly, " Fma ")) << disassembly;
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

TEST_F(HwLowerToStandardTest, LowersChainedPackedVec4MatmulsDirectly) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpFunctionCall
; CHECK: OpFunctionCall
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d0 FPFastMathMode Fast
OpDecorate %d1 FPFastMathMode Fast
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
  EXPECT_EQ(8u, CountSubstring(output, "OpExtInst %v4float"));
  EXPECT_EQ(6u, CountSubstring(output, "OpFunctionParameter"));
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
; CHECK: OpExtInst %float
; CHECK-SAME: Fma
; CHECK: OpCompositeConstruct %v4float
; CHECK-NOT: OpFunctionCall
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
  const std::string& output = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(output);
  EXPECT_NE(std::string::npos, output.find("OpTypeArray %float %uint_15"));
  EXPECT_NE(std::string::npos, output.find("OpTypeVector %float 4"));
  EXPECT_NE(std::string::npos, output.find("OpTypeArray %v4float %uint_10"));
  EXPECT_NE(std::string::npos, output.find("OpTypeArray %v4float %uint_6"));
  EXPECT_GT(CountSubstring(output, "OpExtInst %float"), 0u);
  EXPECT_GT(CountSubstring(output, " Fma "), 0u);
  EXPECT_GT(CountSubstring(output, "OpCompositeConstruct %v4float"), 0u);
  EXPECT_EQ(0u, CountSubstring(output, "OpFunctionCall"));
  EXPECT_EQ(0u, CountSubstring(output, "OpReturnValue"));
  EXPECT_EQ(0u, CountSubstring(output, "OpVectorTimesScalar"));
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
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
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
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
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
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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

TEST_F(HwLowerToStandardTest, FusedVectorMatmulStoreStreamsToSsbo) {
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
%ybuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%ybase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %ybuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpCooperativeVectorStoreHW %ybase %int_0 %y
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_2"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpDot %float"));
  EXPECT_EQ(4u, CountSubstring(disassembly, " Fma "));
  EXPECT_EQ(12u, CountSubstring(disassembly, "OpFAdd %float"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpUDiv"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpUMod"));
}

TEST_F(HwLowerToStandardTest,
       FusedVectorMatmulAddWithConstantBiasStoreStreamsToSsbo) {
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
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%bias = OpConstantComposite %vec8 %float_0 %float_1 %float_0 %float_1 %float_0 %float_1 %float_0 %float_1
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%ybuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%ybase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %ybuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulAddHW %vec8 %x %w %bias
OpCooperativeVectorStoreHW %ybase %int_0 %y
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_2"));
  EXPECT_EQ(4u, CountSubstring(disassembly, "OpDot %float"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpFAdd %v4float"));
}

TEST_F(HwLowerToStandardTest,
       VectorMatmulAddBitcastConstantBiasFallsBackSafely) {
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
%uint_1 = OpConstant %uint 1
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%uvec8 = OpTypeCooperativeVectorHW %uint %uint_8
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%bias_bits = OpConstantComposite %uvec8 %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %uint_1 %uint_1
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%ybuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%ybase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %ybuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%bias = OpBitcast %vec8 %bias_bits
%y = OpCooperativeVectorMatrixMulAddHW %vec8 %x %w %bias
OpCooperativeVectorStoreHW %ybase %int_0 %y
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos, disassembly.find("OpBitcast %v4float"));
}

TEST_F(HwLowerToStandardTest,
       VectorMatmulAddAliasedInputOutputFallsBackSafely) {
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
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%bias = OpConstantComposite %vec8 %float_0 %float_1 %float_0 %float_1 %float_0 %float_1 %float_0 %float_1
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xybuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xybase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xybuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xybase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulAddHW %vec8 %x %w %bias
OpCooperativeVectorStoreHW %xybase %int_0 %y
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_2"),
            0u);
}

TEST_F(HwLowerToStandardTest, VectorMatmulAddAliasedRootsFallsBackSafely) {
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
OpDecorate %xbuf Aliased
OpDecorate %ybuf Aliased
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x8 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8
%bias = OpConstantComposite %vec8 %float_0 %float_1 %float_0 %float_1 %float_0 %float_1 %float_0 %float_1
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%ybuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%ybase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %ybuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulAddHW %vec8 %x %w %bias
OpCooperativeVectorStoreHW %ybase %int_0 %y
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_2"),
            0u);
}

TEST_F(HwLowerToStandardTest, FusedVectorMatmulStoreRespectsFPFastMathDefault) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpExecutionModeId %main FPFastMathDefault %float %uint_0
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
%ybuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%ybase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %ybuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpCooperativeVectorStoreHW %ybase %int_0 %y
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_2"),
            0u);
}

TEST_F(HwLowerToStandardTest,
       FusedVectorMatmulStoreTreatsMissingDefaultTypeAsNoFlags) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability Float16
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpExecutionModeId %main FPFastMathDefault %half %uint_allow_reassoc
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %Buf 0 Offset 0
OpDecorate %Buf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_8 = OpConstant %uint 8
%uint_allow_reassoc = OpConstant %uint 131072
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%half = OpTypeFloat 16
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
%ybuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%ybase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %ybuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
OpCooperativeVectorStoreHW %ybase %int_0 %y
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndMatch<HwLowerToStandardPass>(text, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_2"),
            0u);
}

TEST_F(HwLowerToStandardTest,
       FusedVectorMatmulStoreBlockedByInterveningStorageStore) {
  const std::string text = R"(
; CHECK-NOT: HW
; CHECK: OpFunctionCall %_arr_v4float_uint_2
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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

TEST_F(HwLowerToStandardTest, DirectVectorMatrixMulElidesSeparateLoadLoops) {
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
       DirectMixedPrecisionVectorMatrixMulAdd10x64HandlesKTail) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_half ArrayStride 2
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %HalfBuf 0 Offset 0
OpMemberDecorate %FloatBuf 0 Offset 0
OpDecorate %HalfBuf Block
OpDecorate %FloatBuf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_10 = OpConstant %uint 10
%uint_64 = OpConstant %uint 64
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%v4half = OpTypeVector %half 4
%v4float = OpTypeVector %float 4
%shape = OpConstantComposite %v2uint %uint_10 %uint_64
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%input_type = OpTypeCooperativeVectorHW %half %uint_10
%result_type = OpTypeCooperativeVectorHW %float %uint_64
%matrix_load_type = OpTypeCooperativeMatrixHW %half %uint_10 %uint_64 MatrixUseAHW
%matrix_mul_type = OpTypeCooperativeMatrixHW %half %uint_10 %uint_64 MatrixUseBHW
%_runtimearr_half = OpTypeRuntimeArray %half
%_runtimearr_float = OpTypeRuntimeArray %float
%HalfBuf = OpTypeStruct %_runtimearr_half
%FloatBuf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_HalfBuf = OpTypePointer StorageBuffer %HalfBuf
%_ptr_StorageBuffer_FloatBuf = OpTypePointer StorageBuffer %FloatBuf
%_ptr_Function_matrix_mul_type = OpTypePointer Function %matrix_mul_type
%input_buf = OpVariable %_ptr_StorageBuffer_HalfBuf StorageBuffer
%matrix_buf = OpVariable %_ptr_StorageBuffer_HalfBuf StorageBuffer
%bias_buf = OpVariable %_ptr_StorageBuffer_FloatBuf StorageBuffer
%_ptr_StorageBuffer__runtimearr_half = OpTypePointer StorageBuffer %_runtimearr_half
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%matrix_copy_0 = OpVariable %_ptr_Function_matrix_mul_type Function
%matrix_copy_1 = OpVariable %_ptr_Function_matrix_mul_type Function
%input_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %input_buf %int_0
%matrix_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %matrix_buf %int_0
%bias_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bias_buf %int_0
%input = OpCooperativeVectorLoadHW %input_type %input_base %uint_0
%matrix_load = OpCooperativeMatrixLoadHW %matrix_load_type %matrix_base %shape %offset %int_0
%matrix = OpBitcast %matrix_mul_type %matrix_load
OpStore %matrix_copy_0 %matrix
%matrix_copy_value_0 = OpLoad %matrix_mul_type %matrix_copy_0
OpStore %matrix_copy_1 %matrix_copy_value_0
%matrix_arg = OpLoad %matrix_mul_type %matrix_copy_1
%bias = OpCooperativeVectorLoadHW %result_type %bias_base %uint_0
%result = OpCooperativeVectorMatrixMulAddHW %result_type %input %matrix_arg %bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpFunctionCall %_arr_v4float_uint_16"),
            0u);
  EXPECT_GT(CountSubstring(disassembly, "OpFConvert %v4float"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v4float"), 0u);
  EXPECT_EQ(8u, CountSubstring(disassembly, " Fma ")) << disassembly;
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge")) << disassembly;
  EXPECT_TRUE(HasHalfVec4WithZeroTail(disassembly, 2u)) << disassembly;
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoad %_arr_v4half_uint_160"));
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4half_uint_160"));
}

TEST_F(HwLowerToStandardTest,
       DirectMixedPrecisionVectorMatrixMulAdd64x16Aligned) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_half ArrayStride 2
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %HalfBuf 0 Offset 0
OpMemberDecorate %FloatBuf 0 Offset 0
OpDecorate %HalfBuf Block
OpDecorate %FloatBuf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_16 = OpConstant %uint 16
%uint_24 = OpConstant %uint 24
%uint_64 = OpConstant %uint 64
%uint_70 = OpConstant %uint 70
%uint_256 = OpConstant %uint 256
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%v4half = OpTypeVector %half 4
%v4float = OpTypeVector %float 4
%shape = OpConstantComposite %v2uint %uint_70 %uint_24
%offset = OpConstantComposite %v2uint %uint_2 %uint_3
%input_type = OpTypeCooperativeVectorHW %half %uint_64
%result_type = OpTypeCooperativeVectorHW %float %uint_16
%matrix_load_type = OpTypeCooperativeMatrixHW %half %uint_64 %uint_16 MatrixUseAHW
%matrix_mul_type = OpTypeCooperativeMatrixHW %half %uint_64 %uint_16 MatrixUseBHW
%_runtimearr_half = OpTypeRuntimeArray %half
%_runtimearr_float = OpTypeRuntimeArray %float
%HalfBuf = OpTypeStruct %_runtimearr_half
%FloatBuf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_HalfBuf = OpTypePointer StorageBuffer %HalfBuf
%_ptr_StorageBuffer_FloatBuf = OpTypePointer StorageBuffer %FloatBuf
%input_buf = OpVariable %_ptr_StorageBuffer_HalfBuf StorageBuffer
%matrix_buf = OpVariable %_ptr_StorageBuffer_HalfBuf StorageBuffer
%bias_buf = OpVariable %_ptr_StorageBuffer_FloatBuf StorageBuffer
%_ptr_StorageBuffer__runtimearr_half = OpTypePointer StorageBuffer %_runtimearr_half
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%input_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %input_buf %int_0
%matrix_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %matrix_buf %int_0
%bias_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bias_buf %int_0
%input = OpCooperativeVectorLoadHW %input_type %input_base %uint_0
%matrix_load = OpCooperativeMatrixLoadHW %matrix_load_type %matrix_base %shape %offset %int_0
%matrix = OpBitcast %matrix_mul_type %matrix_load
%bias = OpCooperativeVectorLoadHW %result_type %bias_base %uint_0
%result = OpCooperativeVectorMatrixMulAddHW %result_type %input %matrix %bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpFunctionCall %_arr_v4float_uint_4"),
            0u);
  EXPECT_GT(CountSubstring(disassembly, "OpFConvert %v4float"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v4float"), 0u);
  EXPECT_EQ(4u, CountSubstring(disassembly, " Fma ")) << disassembly;
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge")) << disassembly;
  EXPECT_GT(CountSubstring(disassembly, "OpUDiv %uint"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpUMod %uint"), 0u);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoad %_arr_v4half_uint_256"));
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4half_uint_256"));
}

TEST_F(HwLowerToStandardTest,
       MatrixUseBitcastWithVolatileForwardCopyKeepsFallback) {
  const std::string base_text = R"(
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
__VOLATILE_DECORATION__
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_8 = OpConstant %uint 8
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%v4float = OpTypeVector %float 4
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec = OpTypeCooperativeVectorHW %float %uint_8
%matrix_load_type = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixUseAHW
%matrix_mul_type = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixUseBHW
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%_ptr_Function_matrix_mul_type = OpTypePointer Function %matrix_mul_type
%input_buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%matrix_buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bias_buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%volatile_copy = OpVariable %_ptr_Function_matrix_mul_type Function
%input_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %input_buf %int_0
%matrix_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %matrix_buf %int_0
%bias_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bias_buf %int_0
%input = OpCooperativeVectorLoadHW %vec %input_base %uint_0
%matrix_load = OpCooperativeMatrixLoadHW %matrix_load_type %matrix_base %shape %offset %int_0
%matrix = OpBitcast %matrix_mul_type %matrix_load
OpStore %volatile_copy %matrix__MEMORY_ACCESS__
%volatile_value = OpLoad %matrix_mul_type %volatile_copy__MEMORY_ACCESS__
%bias = OpCooperativeVectorLoadHW %vec %bias_base %uint_0
%result = OpCooperativeVectorMatrixMulAddHW %vec %input %matrix %bias
OpReturn
OpFunctionEnd
)";

  struct VolatileCase {
    const char* name;
    const char* decoration;
    const char* memory_access;
    size_t minimum_volatile_count;
  };
  const VolatileCase cases[] = {
      {"memory operands", "", " Volatile", 2u},
      {"variable decoration", "OpDecorate %volatile_copy Volatile", "", 1u},
  };
  auto replace_all = [](std::string* text, const std::string& from,
                        const std::string& to) {
    size_t position = 0;
    while ((position = text->find(from, position)) != std::string::npos) {
      text->replace(position, from.size(), to);
      position += to.size();
    }
  };

  for (const VolatileCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    std::string text = base_text;
    replace_all(&text, "__VOLATILE_DECORATION__", test_case.decoration);
    replace_all(&text, "__MEMORY_ACCESS__", test_case.memory_access);
    auto result =
        SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
    const std::string& disassembly = std::get<0>(result);
    EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
    ExpectNoHwOrCoopMatrix(disassembly);
    EXPECT_GE(CountSubstring(disassembly, " Volatile"),
              test_case.minimum_volatile_count);
    EXPECT_GT(CountSubstring(disassembly,
                             "OpFunctionParameter %_arr_v4float_uint_16"),
              0u);
  }
}

TEST_F(HwLowerToStandardTest,
       DirectMixedPrecisionVectorMatrixMulAdd16x3HandlesNTail) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %_runtimearr_half ArrayStride 2
OpDecorate %_runtimearr_float ArrayStride 4
OpMemberDecorate %HalfBuf 0 Offset 0
OpMemberDecorate %FloatBuf 0 Offset 0
OpDecorate %HalfBuf Block
OpDecorate %FloatBuf Block
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_8 = OpConstant %uint 8
%uint_16 = OpConstant %uint 16
%uint_20 = OpConstant %uint 20
%uint_48 = OpConstant %uint 48
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%v4half = OpTypeVector %half 4
%v4float = OpTypeVector %float 4
%shape = OpConstantComposite %v2uint %uint_20 %uint_8
%offset = OpConstantComposite %v2uint %uint_2 %uint_3
%input_type = OpTypeCooperativeVectorHW %half %uint_16
%result_type = OpTypeCooperativeVectorHW %float %uint_3
%matrix_load_type = OpTypeCooperativeMatrixHW %half %uint_16 %uint_3 MatrixUseAHW
%matrix_mul_type = OpTypeCooperativeMatrixHW %half %uint_16 %uint_3 MatrixUseBHW
%_runtimearr_half = OpTypeRuntimeArray %half
%_runtimearr_float = OpTypeRuntimeArray %float
%HalfBuf = OpTypeStruct %_runtimearr_half
%FloatBuf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_HalfBuf = OpTypePointer StorageBuffer %HalfBuf
%_ptr_StorageBuffer_FloatBuf = OpTypePointer StorageBuffer %FloatBuf
%input_buf = OpVariable %_ptr_StorageBuffer_HalfBuf StorageBuffer
%matrix_buf = OpVariable %_ptr_StorageBuffer_HalfBuf StorageBuffer
%bias_buf = OpVariable %_ptr_StorageBuffer_FloatBuf StorageBuffer
%_ptr_StorageBuffer__runtimearr_half = OpTypePointer StorageBuffer %_runtimearr_half
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%input_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %input_buf %int_0
%matrix_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_half %matrix_buf %int_0
%bias_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bias_buf %int_0
%input = OpCooperativeVectorLoadHW %input_type %input_base %uint_0
%matrix_load = OpCooperativeMatrixLoadHW %matrix_load_type %matrix_base %shape %offset %int_0
%matrix = OpBitcast %matrix_mul_type %matrix_load
%bias = OpCooperativeVectorLoadHW %result_type %bias_base %uint_0
%result = OpCooperativeVectorMatrixMulAddHW %result_type %input %matrix %bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpFunctionCall %_arr_float_uint_3"),
            0u);
  EXPECT_GT(CountSubstring(disassembly, "OpFConvert %v4float"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v4float"), 0u);
  EXPECT_EQ(3u, CountSubstring(disassembly, " Fma ")) << disassembly;
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpLoopMerge")) << disassembly;
  EXPECT_GT(CountSubstring(disassembly, "OpUDiv %uint"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpUMod %uint"), 0u);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoad %_arr_half_uint_48"));
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_half_uint_48"));
}

TEST_F(HwLowerToStandardTest,
       ReverseMatrixUseBitcastKeepsVectorMatrixMulAddFallback) {
  const std::string text = R"(
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
%uint_16 = OpConstant %uint 16
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%float = OpTypeFloat 32
%v2uint = OpTypeVector %uint 2
%v4float = OpTypeVector %float 4
%shape = OpConstantComposite %v2uint %uint_8 %uint_8
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec = OpTypeCooperativeVectorHW %float %uint_8
%matrix_load_type = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixUseBHW
%matrix_mul_type = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixUseAHW
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%input_buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%matrix_buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bias_buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%input_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %input_buf %int_0
%matrix_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %matrix_buf %int_0
%bias_base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bias_buf %int_0
%input = OpCooperativeVectorLoadHW %vec %input_base %uint_0
%matrix_load = OpCooperativeMatrixLoadHW %matrix_load_type %matrix_base %shape %offset %int_0
%matrix = OpBitcast %matrix_mul_type %matrix_load
%bias = OpCooperativeVectorLoadHW %vec %bias_base %uint_0
%result = OpCooperativeVectorMatrixMulAddHW %vec %input %matrix %bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(
      CountSubstring(disassembly, "OpFunctionParameter %_arr_v4float_uint_16"),
      0u);
  EXPECT_GT(CountSubstring(disassembly, "OpLoad %_arr_v4float_uint_16"), 0u);
}

TEST_F(HwLowerToStandardTest,
       MixedPrecisionLoopDirectIgnoresMatmulUnrollLimit) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_10 = OpConstant %uint 10
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%input_type = OpTypeCooperativeVectorHW %half %uint_10
%result_type = OpTypeCooperativeVectorHW %float %uint_3
%matrix_type = OpTypeCooperativeMatrixHW %half %uint_10 %uint_3 MatrixUseBHW
%input = OpUndef %input_type
%matrix = OpUndef %matrix_type
%bias = OpConstantNull %result_type
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpCooperativeVectorMatrixMulAddHW %result_type %input %matrix %bias
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u, 1024ull,
      1024u, 16ull);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpFunctionCall %_arr_float_uint_3"),
            0u);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpLoopMerge")) << disassembly;
  EXPECT_GT(CountSubstring(disassembly, "OpFConvert %v4float"), 0u);
  EXPECT_EQ(6u, CountSubstring(disassembly, " Fma ")) << disassembly;
  EXPECT_TRUE(HasHalfVec4WithZeroTail(disassembly, 2u)) << disassembly;
}

TEST_F(HwLowerToStandardTest, DirectVectorMatrixMulAddElidesSeparateLoadLoops) {
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
       DirectVectorMatrixMulAddReluValueInputKeepsMatrixBiasDirect) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %h0 FPFastMathMode Fast
OpDecorate %y FPFastMathMode Fast
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
%shape8x4 = OpConstantComposite %v2uint %uint_8 %uint_4
%shape4x4 = OpConstantComposite %v2uint %uint_4 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x4 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_4
%mat4x4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%zero = OpConstantNull %vec4
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%w0buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%b0buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%w1buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%b1buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%_ptr_Function_vec8 = OpTypePointer Function %vec8
%_ptr_Function_vec4 = OpTypePointer Function %vec4
%_ptr_Function_mat8x4 = OpTypePointer Function %mat8x4
%_ptr_Function_mat4x4 = OpTypePointer Function %mat4x4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xvar = OpVariable %_ptr_Function_vec8 Function
%w0var = OpVariable %_ptr_Function_mat8x4 Function
%b0var = OpVariable %_ptr_Function_vec4 Function
%hvar = OpVariable %_ptr_Function_vec4 Function
%w1var = OpVariable %_ptr_Function_mat4x4 Function
%b1var = OpVariable %_ptr_Function_vec4 Function
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%w0base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %w0buf %int_0
%b0base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %b0buf %int_0
%w1base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %w1buf %int_0
%b1base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %b1buf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
OpStore %xvar %x
%xv = OpLoad %vec8 %xvar
%w0 = OpCooperativeMatrixLoadHW %mat8x4 %w0base %shape8x4 %offset %int_0
OpStore %w0var %w0
%w0v = OpLoad %mat8x4 %w0var
%b0 = OpCooperativeVectorLoadHW %vec4 %b0base %int_0
OpStore %b0var %b0
%b0v = OpLoad %vec4 %b0var
%h0 = OpCooperativeVectorMatrixMulAddHW %vec4 %xv %w0v %b0v
OpStore %hvar %h0
%hload = OpLoad %vec4 %hvar
%relu = OpExtInst %vec4 %1 FMax %hload %zero
OpStore %hvar %relu
%w1 = OpCooperativeMatrixLoadHW %mat4x4 %w1base %shape4x4 %offset %int_0
OpStore %w1var %w1
%w1v = OpLoad %mat4x4 %w1var
%b1 = OpCooperativeVectorLoadHW %vec4 %b1base %int_0
OpStore %b1var %b1
%b1v = OpLoad %vec4 %b1var
%reluv = OpLoad %vec4 %hvar
%y = OpCooperativeVectorMatrixMulAddHW %vec4 %reluv %w1v %b1v
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_4"));
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_1"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulElidesSeparateLoadLoopsFromUniform) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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
%c = OpCompositeConstruct %mat4x4 %float_0
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

TEST_F(HwLowerToStandardTest, DirectMatmulAddElidesFunctionValueStaging) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode AllowReassoc
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
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%_ptr_Function_mat8x8 = OpTypePointer Function %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%avar = OpVariable %_ptr_Function_mat8x8 Function
%bvar = OpVariable %_ptr_Function_mat8x8 Function
%cvar = OpVariable %_ptr_Function_mat8x8 Function
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat8x8 %abase %shape %offset %int_0
OpStore %avar %a
%av = OpLoad %mat8x8 %avar
%b = OpCooperativeMatrixLoadHW %mat8x8 %bbase %shape %offset %int_0
OpStore %bvar %b
%bv = OpLoad %mat8x8 %bvar
%c = OpCooperativeMatrixLoadHW %mat8x8 %cbase %shape %offset %int_0
OpStore %cvar %c
%cv = OpLoad %mat8x8 %cvar
%d = OpCooperativeMatrixMulAddHW %mat8x8 %av %bv %cv
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_16"));
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
  EXPECT_EQ(
      0u, CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_16"));
}

TEST_F(HwLowerToStandardTest,
       FusedMatrixMatmulStoreRespectsExplicitNonReassocMode) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode NotNaN
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
  EXPECT_GT(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_16"),
            0u);
  EXPECT_GT(CountSubstring(disassembly, "FPFastMathMode NotNaN"), 0u);
}

TEST_F(HwLowerToStandardTest,
       FusedMatrixMatmulStoreElidesFunctionValueStaging) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode Fast
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
%_ptr_Function_mat8x8 = OpTypePointer Function %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%avar = OpVariable %_ptr_Function_mat8x8 Function
%bvar = OpVariable %_ptr_Function_mat8x8 Function
%cvar = OpVariable %_ptr_Function_mat8x8 Function
%dvar = OpVariable %_ptr_Function_mat8x8 Function
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%dbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %dbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat8x8 %abase %shape %offset %int_0
OpStore %avar %a
%av = OpLoad %mat8x8 %avar
%b = OpCooperativeMatrixLoadHW %mat8x8 %bbase %shape %offset %int_0
OpStore %bvar %b
%bv = OpLoad %mat8x8 %bvar
%c = OpCooperativeMatrixLoadHW %mat8x8 %cbase %shape %offset %int_0
OpStore %cvar %c
%cv = OpLoad %mat8x8 %cvar
%d = OpCooperativeMatrixMulAddHW %mat8x8 %av %bv %cv
OpStore %dvar %d
%dv = OpLoad %mat8x8 %dvar
OpCooperativeMatrixStoreHW %dbase %dv %shape %offset %int_0
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_16"));
}

TEST_F(HwLowerToStandardTest,
       FusedMatrixMatmulStoreBlockedByInterveningStorageStore) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode Fast
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
  // The matmul is lowered to a function call (either via fusion or the
  // direct path).
  EXPECT_NE(std::string::npos, disassembly.find("OpFunctionCall"));
}

TEST_F(HwLowerToStandardTest, GeneratedFusedStoreBlocksAliasingBiasLoadMotion) {
  const std::string text = R"(
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode Fast
OpDecorate %y FPFastMathMode Fast
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
%abuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%cbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%dbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%dbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %dbuf %int_0
%bias = OpCooperativeVectorLoadHW %vec8 %dbase %uint_0
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat8x8 %abase %shape %offset %int_0
%b = OpCooperativeMatrixLoadHW %mat8x8 %bbase %shape %offset %int_0
%c = OpCooperativeMatrixLoadHW %mat8x8 %cbase %shape %offset %int_0
%d = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %c
OpCooperativeMatrixStoreHW %dbase %d %shape %offset %int_0
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %uint_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulAddHW %vec8 %x %w %bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_2"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulKeepsDirectPathWhenLoadIsSharedWithHwStore) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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
  EXPECT_EQ(4u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_2"));
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_16"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulFallsBackWhenLoadIsSharedWithCompositeExtract) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w = OpCooperativeMatrixLoadHW %mat8x8 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulHW %vec8 %x %w
%x0 = OpCompositeExtract %float %x 0
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos,
            disassembly.find("OpFunctionParameter %_arr_v4float_uint_2"));
}

TEST_F(HwLowerToStandardTest,
       DirectMatmulAddKeepsDirectPathWhenLoadsAreSharedWithOtherHwOp) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d0 FPFastMathMode Fast
OpDecorate %d1 FPFastMathMode Fast
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
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat8x8 %abase %shape %offset %int_0
%b = OpCooperativeMatrixLoadHW %mat8x8 %bbase %shape %offset %int_0
%c = OpCooperativeMatrixLoadHW %mat8x8 %cbase %shape %offset %int_0
%d0 = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %c
%d1 = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_16"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulAddKeepsDirectPathWhenBiasLoadIsShared) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y0 FPFastMathMode Fast
OpDecorate %y1 FPFastMathMode Fast
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
%w0buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%w1buf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%w0base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %w0buf %int_0
%w1base = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %w1buf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%w0 = OpCooperativeMatrixLoadHW %mat8x4 %w0base %shape %offset %int_0
%w1 = OpCooperativeMatrixLoadHW %mat8x4 %w1base %shape %offset %int_0
%bias = OpCooperativeVectorLoadHW %vec4 %bbase %int_0
%y0 = OpCooperativeVectorMatrixMulAddHW %vec4 %x %w0 %bias
%y1 = OpCooperativeVectorMatrixMulAddHW %vec4 %x %w1 %bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_1"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulAddKeepsXBDirectWhenMatrixFeedsEarlierMatmuls) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %mat0 FPFastMathMode Fast
OpDecorate %mat1 FPFastMathMode Fast
OpDecorate %vec0 FPFastMathMode Fast
OpDecorate %vec1 FPFastMathMode Fast
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
%matA = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixUseAHW
%matB = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixUseBHW
%matAcc = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixAccumulatorHW
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%abuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%cbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%biasbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%_ptr_Function_matA = OpTypePointer Function %matA
%_ptr_Function_matAcc = OpTypePointer Function %matAcc
%_ptr_Function_vec8 = OpTypePointer Function %vec8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%tempArgA = OpVariable %_ptr_Function_matA Function
%a = OpVariable %_ptr_Function_matA Function
%tempArgB = OpVariable %_ptr_Function_matA Function
%b = OpVariable %_ptr_Function_matA Function
%tempArgC = OpVariable %_ptr_Function_matAcc Function
%c = OpVariable %_ptr_Function_matAcc Function
%tempArgD0 = OpVariable %_ptr_Function_matAcc Function
%d0 = OpVariable %_ptr_Function_matAcc Function
%tempArgX = OpVariable %_ptr_Function_vec8 Function
%x = OpVariable %_ptr_Function_vec8 Function
%tempArgBias = OpVariable %_ptr_Function_vec8 Function
%bias = OpVariable %_ptr_Function_vec8 Function
%tempArgY0 = OpVariable %_ptr_Function_vec8 Function
%y0 = OpVariable %_ptr_Function_vec8 Function
%tempArgY1 = OpVariable %_ptr_Function_vec8 Function
%y1 = OpVariable %_ptr_Function_vec8 Function
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%biasbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %biasbuf %int_0
%aload = OpCooperativeMatrixLoadHW %matA %abase %shape %offset %int_0
OpStore %tempArgA %aload
%aval = OpLoad %matA %tempArgA
OpStore %a %aval
%bload = OpCooperativeMatrixLoadHW %matA %bbase %shape %offset %int_0
OpStore %tempArgB %bload
%bval = OpLoad %matA %tempArgB
OpStore %b %bval
%cload = OpCooperativeMatrixLoadHW %matAcc %cbase %shape %offset %int_0
OpStore %tempArgC %cload
%cval = OpLoad %matAcc %tempArgC
OpStore %c %cval
%a0 = OpLoad %matA %a
%b0 = OpLoad %matA %b
%c0 = OpLoad %matAcc %c
%b0cast = OpBitcast %matB %b0
%mat0 = OpCooperativeMatrixMulAddHW %matAcc %a0 %b0cast %c0
OpStore %tempArgD0 %mat0
%d0val = OpLoad %matAcc %tempArgD0
OpStore %d0 %d0val
%a1 = OpLoad %matA %a
%b1 = OpLoad %matA %b
%c1 = OpLoad %matAcc %d0
%b1cast = OpBitcast %matB %b1
%mat1 = OpCooperativeMatrixMulAddHW %matAcc %a1 %b1cast %c1
%xload = OpCooperativeVectorLoadHW %vec8 %xbase %uint_0
OpStore %tempArgX %xload
%xval = OpLoad %vec8 %tempArgX
OpStore %x %xval
%biasload = OpCooperativeVectorLoadHW %vec8 %biasbase %uint_0
OpStore %tempArgBias %biasload
%biasval = OpLoad %vec8 %tempArgBias
OpStore %bias %biasval
%x0 = OpLoad %vec8 %x
%bm0 = OpLoad %matA %b
%bias0 = OpLoad %vec8 %bias
%vec0 = OpCooperativeVectorMatrixMulAddHW %vec8 %x0 %bm0 %bias0
OpStore %tempArgY0 %vec0
%y0val = OpLoad %vec8 %tempArgY0
OpStore %y0 %y0val
%x1 = OpLoad %vec8 %x
%bm1 = OpLoad %matA %b
%bias1 = OpLoad %vec8 %bias
%vec1 = OpCooperativeVectorMatrixMulAddHW %vec8 %x1 %bm1 %bias1
OpStore %tempArgY1 %vec1
%y1val = OpLoad %vec8 %tempArgY1
OpStore %y1 %y1val
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(
      1u, CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_2\n"));
  EXPECT_EQ(2u,
            CountSubstring(disassembly, "OpFunctionCall %_arr_v4float_uint_2"));
}

TEST_F(HwLowerToStandardTest,
       DirectMatmulAddKeepsABDirectWhenAccumulatorComesFromEarlierMatmul) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d0 FPFastMathMode Fast
OpDecorate %d1 FPFastMathMode Fast
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
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%_ptr_Function_mat8x8 = OpTypePointer Function %mat8x8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%dvar = OpVariable %_ptr_Function_mat8x8 Function
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat8x8 %abase %shape %offset %int_0
%b = OpCooperativeMatrixLoadHW %mat8x8 %bbase %shape %offset %int_0
%c = OpCooperativeMatrixLoadHW %mat8x8 %cbase %shape %offset %int_0
%d0 = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %c
OpStore %dvar %d0
%d0v = OpLoad %mat8x8 %dvar
%d1 = OpCooperativeMatrixMulAddHW %mat8x8 %a %b %d0v
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_16"));
}

TEST_F(HwLowerToStandardTest,
       DirectVectorMatrixMulAddKeepsXBDirectAcrossInterveningVectorStore) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %mat0 FPFastMathMode Fast
OpDecorate %vec0 FPFastMathMode Fast
OpDecorate %vec1 FPFastMathMode Fast
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
%matA = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixUseAHW
%matB = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixUseBHW
%matAcc = OpTypeCooperativeMatrixHW %float %uint_8 %uint_8 MatrixAccumulatorHW
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%abuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%cbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%outbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%_ptr_Function_matA = OpTypePointer Function %matA
%_ptr_Function_matAcc = OpTypePointer Function %matAcc
%_ptr_Function_vec8 = OpTypePointer Function %vec8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%tempArgA = OpVariable %_ptr_Function_matA Function
%a = OpVariable %_ptr_Function_matA Function
%tempArgB = OpVariable %_ptr_Function_matA Function
%b = OpVariable %_ptr_Function_matA Function
%tempArgC = OpVariable %_ptr_Function_matAcc Function
%c = OpVariable %_ptr_Function_matAcc Function
%tempArgD0 = OpVariable %_ptr_Function_matAcc Function
%d0 = OpVariable %_ptr_Function_matAcc Function
%tempArgX = OpVariable %_ptr_Function_vec8 Function
%x = OpVariable %_ptr_Function_vec8 Function
%tempArgBias = OpVariable %_ptr_Function_vec8 Function
%bias = OpVariable %_ptr_Function_vec8 Function
%tempArgY0 = OpVariable %_ptr_Function_vec8 Function
%y0 = OpVariable %_ptr_Function_vec8 Function
%tempArgY1 = OpVariable %_ptr_Function_vec8 Function
%y1 = OpVariable %_ptr_Function_vec8 Function
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%outbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %outbuf %int_0
%aload = OpCooperativeMatrixLoadHW %matA %abase %shape %offset %int_0
OpStore %tempArgA %aload
%aval = OpLoad %matA %tempArgA
OpStore %a %aval
%bload = OpCooperativeMatrixLoadHW %matA %bbase %shape %offset %int_0
OpStore %tempArgB %bload
%bval = OpLoad %matA %tempArgB
OpStore %b %bval
%cload = OpCooperativeMatrixLoadHW %matAcc %cbase %shape %offset %int_0
OpStore %tempArgC %cload
%cval = OpLoad %matAcc %tempArgC
OpStore %c %cval
%a0 = OpLoad %matA %a
%b0 = OpLoad %matA %b
%c0 = OpLoad %matAcc %c
%b0cast = OpBitcast %matB %b0
%mat0 = OpCooperativeMatrixMulAddHW %matAcc %a0 %b0cast %c0
OpStore %tempArgD0 %mat0
%d0val = OpLoad %matAcc %tempArgD0
OpStore %d0 %d0val
%xload = OpCooperativeVectorLoadHW %vec8 %xbase %uint_0
OpStore %tempArgX %xload
%xval = OpLoad %vec8 %tempArgX
OpStore %x %xval
%biasload = OpCooperativeVectorLoadHW %vec8 %outbase %uint_0
OpStore %tempArgBias %biasload
%biasval = OpLoad %vec8 %tempArgBias
OpStore %bias %biasval
%x0 = OpLoad %vec8 %x
%bm0 = OpLoad %matA %b
%bias0 = OpLoad %vec8 %bias
%vec0 = OpCooperativeVectorMatrixMulAddHW %vec8 %x0 %bm0 %bias0
OpStore %tempArgY0 %vec0
%y0val = OpLoad %vec8 %tempArgY0
OpStore %y0 %y0val
OpCooperativeVectorStoreHW %outbase %uint_0 %y0val
%biasreload = OpCooperativeVectorLoadHW %vec8 %outbase %uint_0
OpStore %tempArgBias %biasreload
%biasreloadval = OpLoad %vec8 %tempArgBias
OpStore %bias %biasreloadval
%x1 = OpLoad %vec8 %x
%bm1 = OpLoad %matA %b
%bias1 = OpLoad %vec8 %bias
%vec1 = OpCooperativeVectorMatrixMulAddHW %vec8 %x1 %bm1 %bias1
OpStore %tempArgY1 %vec1
%y1val = OpLoad %vec8 %tempArgY1
OpStore %y1 %y1val
OpCooperativeVectorStoreHW %outbase %uint_0 %y1val
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_2"));
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_16"));
  EXPECT_EQ(2u,
            CountSubstring(disassembly, "OpFunctionCall %_arr_v4float_uint_2"));
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

TEST_F(HwLowerToStandardTest, DirectMatmulWithConstantB) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode Fast
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
%const_b = OpConstantComposite %mat4x4 %float_1
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%abuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%cbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%abase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %abuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%a = OpCooperativeMatrixLoadHW %mat4x4 %abase %shape %offset %int_0
%c = OpCooperativeMatrixLoadHW %mat4x4 %cbase %shape %offset %int_0
%d = OpCooperativeMatrixMulAddHW %mat4x4 %a %const_b %c
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  // Direct path should be used: the matmul function has no parameters
  // (the constant is accessed via OpAccessChain inside the function)
  EXPECT_GE(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_4"),
            1u);
  // Should have FMA operations
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
  // Should have nested loops (row, col, k)
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 3u);
  // Should have OpAccessChain to access the constant matrix
  EXPECT_GT(CountSubstring(disassembly, "OpAccessChain"), 0u);
}

TEST_F(HwLowerToStandardTest, DirectMatmulWithConstantA) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode Fast
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
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_4 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%mat4x4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%const_a = OpConstantComposite %mat4x4 %float_1
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%bbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%cbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%bbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %bbuf %int_0
%cbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %cbuf %int_0
%b = OpCooperativeMatrixLoadHW %mat4x4 %bbase %shape %offset %int_0
%c = OpCooperativeMatrixLoadHW %mat4x4 %cbase %shape %offset %int_0
%d = OpCooperativeMatrixMulAddHW %mat4x4 %const_a %b %c
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  // Direct path should be used: the matmul function has no parameters
  // (the constant is accessed via OpAccessChain inside the function)
  EXPECT_GE(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_4"),
            1u);
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 3u);
  // Should have OpAccessChain to access the constant matrix
  EXPECT_GT(CountSubstring(disassembly, "OpAccessChain"), 0u);
}

TEST_F(HwLowerToStandardTest, DirectMatmulWithConstantC) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode Fast
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
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_4 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%mat4x4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%const_c = OpConstantComposite %mat4x4 %float_0
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
%d = OpCooperativeMatrixMulAddHW %mat4x4 %a %b %const_c
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  // Direct path should be used: the matmul function has no parameters
  // (the constant is accessed via OpAccessChain inside the function)
  EXPECT_GE(CountSubstring(disassembly, "OpTypeFunction %_arr_v4float_uint_4"),
            1u);
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 3u);
  // Should have OpAccessChain to access the constant matrix
  EXPECT_GT(CountSubstring(disassembly, "OpAccessChain"), 0u);
}

TEST_F(HwLowerToStandardTest, DirectVectorMatmulWithConstantMatrix) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x4 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_4
%const_w = OpConstantComposite %mat8x4 %float_1
%const_bias = OpConstantComposite %vec4 %float_0 %float_0 %float_0 %float_0
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%xbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%xbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %xbuf %int_0
%x = OpCooperativeVectorLoadHW %vec8 %xbase %int_0
%y = OpCooperativeVectorMatrixMulAddHW %vec4 %x %const_w %const_bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  // Direct path should be used: the matmul function has no parameters
  // (the constant is accessed via OpAccessChain inside the function)
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
  // Should have nested loops
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 2u);
  // Should have OpAccessChain to access the constant
  EXPECT_GT(CountSubstring(disassembly, "OpAccessChain"), 0u);
}

TEST_F(HwLowerToStandardTest, DirectVectorMatmulWithConstantInput) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y FPFastMathMode Fast
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
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_8 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%mat8x4 = OpTypeCooperativeMatrixHW %float %uint_8 %uint_4
%const_x = OpConstantComposite %vec8
  %float_1 %float_1 %float_1 %float_1 %float_1 %float_1 %float_1 %float_1
%const_bias = OpConstantComposite %vec4 %float_0 %float_0 %float_0 %float_0
%_runtimearr_float = OpTypeRuntimeArray %float
%Buf = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_Buf = OpTypePointer StorageBuffer %Buf
%wbuf = OpVariable %_ptr_StorageBuffer_Buf StorageBuffer
%_ptr_StorageBuffer__runtimearr_float = OpTypePointer StorageBuffer %_runtimearr_float
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%wbase = OpAccessChain %_ptr_StorageBuffer__runtimearr_float %wbuf %int_0
%w = OpCooperativeMatrixLoadHW %mat8x4 %wbase %shape %offset %int_0
%y = OpCooperativeVectorMatrixMulAddHW %vec4 %const_x %w %const_bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  // Direct path should be used: the matmul function has no parameters
  // (the constant is accessed via OpAccessChain inside the function)
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
  // Should have nested loops
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 2u);
  // Should have OpAccessChain to access the constant
  EXPECT_GT(CountSubstring(disassembly, "OpAccessChain"), 0u);
}

TEST_F(HwLowerToStandardTest,
       LowMatmulUnrollLimitUsesStructuredMixedPrecisionLoops) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability Float16
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d FPFastMathMode Fast
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%a_type = OpTypeCooperativeMatrixHW %half %uint_2 %uint_4
%b_type = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%c_type = OpTypeCooperativeMatrixHW %float %uint_2 %uint_4
%a = OpUndef %a_type
%b = OpUndef %b_type
%c = OpUndef %c_type
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %c_type %a %b %c
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u, 1024ull,
      1u, 1ull);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_GT(CountSubstring(disassembly, "OpFConvert %float"), 0u);
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
  EXPECT_GE(CountSubstring(disassembly, "FPFastMathMode Fast"), 2u);
}

TEST_F(HwLowerToStandardTest,
       CreatesTrueZeroForFloat32AndInt32MatmulAccumulators) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%int = OpTypeInt 32 1
%int_nonzero = OpConstant %int 1
%float = OpTypeFloat 32
%float_nonzero = OpConstant %float 1
%ivec = OpTypeCooperativeVectorHW %int %uint_2
%imat = OpTypeCooperativeMatrixHW %int %uint_2 %uint_2
%fvec = OpTypeCooperativeVectorHW %float %uint_2
%fmat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%ix = OpUndef %ivec
%im = OpUndef %imat
%fx = OpUndef %fvec
%fm = OpUndef %fmat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%iy = OpCooperativeVectorMatrixMulHW %ivec %ix %im
%fy = OpCooperativeVectorMatrixMulHW %fvec %fx %fm
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u, 1024ull,
      1024u, 1ull);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(4u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpConstantNull %int"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpConstantNull %float"));
  EXPECT_GT(CountSubstring(disassembly, "OpIMul %int"), 0u);
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwLowerToStandardTest, FastMathModesUseDistinctMatmulHelpers) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability FloatControls2
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %m0 FPFastMathMode NotNaN
OpDecorate %m1 FPFastMathMode NotInf
OpDecorate %v0 FPFastMathMode NotNaN
OpDecorate %v1 FPFastMathMode NotInf
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%a_type = OpTypeCooperativeMatrixHW %float %uint_2 %uint_4
%b_type = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%x_type = OpTypeCooperativeVectorHW %float %uint_4
%a = OpUndef %a_type
%b = OpUndef %b_type
%c = OpUndef %a_type
%x = OpUndef %x_type
%bias = OpUndef %x_type
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%m0 = OpCooperativeMatrixMulAddHW %a_type %a %b %c
%m1 = OpCooperativeMatrixMulAddHW %a_type %a %b %c
%v0 = OpCooperativeVectorMatrixMulAddHW %x_type %x %b %bias
%v1 = OpCooperativeVectorMatrixMulAddHW %x_type %x %b %bias
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "FPFastMathMode NotNaN"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "FPFastMathMode NotInf"), 0u);
}

TEST_F(HwLowerToStandardTest, LowersPackedF32MatrixReduceAllAxesAndOperations) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_3 %uint_8
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%row_add = OpCooperativeMatrixReduceHW %mat %a %uint_0 %uint_0
%row_min = OpCooperativeMatrixReduceHW %mat %a %uint_0 %uint_1
%row_max = OpCooperativeMatrixReduceHW %mat %a %uint_0 %uint_2
%column_add = OpCooperativeMatrixReduceHW %mat %a %uint_1 %uint_0
%column_min = OpCooperativeMatrixReduceHW %mat %a %uint_1 %uint_1
%column_max = OpCooperativeMatrixReduceHW %mat %a %uint_1 %uint_2
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeVector %float 4"));
  EXPECT_NE(std::string::npos,
            disassembly.find("OpTypeArray %v4float %uint_6"));
  EXPECT_GT(CountSubstring(disassembly, "OpFAdd"), 0u);
  EXPECT_GT(CountSubstring(disassembly, " FMin "), 0u);
  EXPECT_GT(CountSubstring(disassembly, " FMax "), 0u);
}

TEST_F(HwLowerToStandardTest, LowersPackedF16MatrixReduce) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%half = OpTypeFloat 16
%mat = OpTypeCooperativeMatrixHW %half %uint_2 %uint_8
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%row_max = OpCooperativeMatrixReduceHW %mat %a %uint_0 %uint_2
%column_min = OpCooperativeMatrixReduceHW %mat %a %uint_1 %uint_1
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeVector %half 4"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %v4half %uint_4"));
  EXPECT_GT(CountSubstring(disassembly, " FMin "), 0u);
  EXPECT_GT(CountSubstring(disassembly, " FMax "), 0u);
}

TEST_F(HwLowerToStandardTest, ForceScalarLowersMatrixReduceWithTail) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_3 %uint_5
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%row_add = OpCooperativeMatrixReduceHW %mat %a %uint_0 %uint_0
%column_max = OpCooperativeMatrixReduceHW %mat %a %uint_1 %uint_2
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %float 4"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeArray %float %uint_15"));
  EXPECT_GT(CountSubstring(disassembly, "OpFAdd %float"), 0u);
  EXPECT_GT(CountSubstring(disassembly, " FMax "), 0u);
}

TEST_F(HwLowerToStandardTest, LowReduceUnrollLimitUsesThreeStructuredLoops) {
  const std::string text = R"(
; CHECK-NOT: HW
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_4
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%row = OpCooperativeMatrixReduceHW %mat %a %uint_0 %uint_0
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u, 1024ull,
      1u, 1ull);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(3u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_GT(CountSubstring(disassembly, "OpFAdd %float"), 0u);
}

TEST_F(HwLowerToStandardTest, LowersSignedAndUnsignedIntegerMatrixReduce) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_3 = OpConstant %uint 3
%int = OpTypeInt 32 1
%smat = OpTypeCooperativeMatrixHW %int %uint_2 %uint_3
%umat = OpTypeCooperativeMatrixHW %uint %uint_2 %uint_3
%signed_value = OpUndef %smat
%unsigned_value = OpUndef %umat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%row_smin = OpCooperativeMatrixReduceHW %smat %signed_value %uint_0 %uint_1
%column_smax = OpCooperativeMatrixReduceHW %smat %signed_value %uint_1 %uint_2
%row_umin = OpCooperativeMatrixReduceHW %umat %unsigned_value %uint_0 %uint_1
%column_umax = OpCooperativeMatrixReduceHW %umat %unsigned_value %uint_1 %uint_2
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, " SMin "), 0u);
  EXPECT_GT(CountSubstring(disassembly, " SMax "), 0u);
  EXPECT_GT(CountSubstring(disassembly, " UMin "), 0u);
  EXPECT_GT(CountSubstring(disassembly, " UMax "), 0u);
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 4"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %uint 4"));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
