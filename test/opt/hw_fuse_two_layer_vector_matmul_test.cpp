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

#include "source/opt/hw_fuse_two_layer_vector_matmul_pass.h"
#include "source/opt/hw_lower_to_standard_pass.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using HwFuseTwoLayerVectorMatmulTest = PassTest<::testing::Test>;

size_t CountSubstring(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

bool ReplaceOnce(std::string* text, const std::string& from,
                 const std::string& to) {
  if (!text) return false;
  const size_t pos = text->find(from);
  if (pos == std::string::npos) return false;
  text->replace(pos, from.size(), to);
  return true;
}

enum class IntermediateKind {
  kDirect,
  kSelect,
  kCrossBlock,
};

std::string MakePureMulChain(
    uint32_t middle_length, uint32_t output_length, bool use_float16 = true,
    IntermediateKind intermediate = IntermediateKind::kDirect) {
  const std::string component_capability =
      use_float16 ? "OpCapability Float16\n" : "";
  const std::string component_type = use_float16
                                         ? "%component = OpTypeFloat 16\n"
                                         : "%component = OpTypeFloat 32\n";
  const std::string middle_id = "%uint_" + std::to_string(middle_length);
  const std::string output_id = "%uint_" + std::to_string(output_length);

  std::string bridge;
  std::string second_input = "%hidden";
  if (intermediate == IntermediateKind::kSelect) {
    bridge = "%selected = OpSelect %vec_middle %false %hidden %hidden\n";
    second_input = "%selected";
  } else if (intermediate == IntermediateKind::kCrossBlock) {
    bridge = "OpBranch %second_block\n%second_block = OpLabel\n";
  }

  return "OpCapability Shader\n" + component_capability +
         "OpCapability CooperativeVectorHW\n"
         "OpCapability CooperativeMatrixHW\n"
         "OpExtension \"SPV_HW_neural_shader\"\n"
         "OpMemoryModel Logical GLSL450\n"
         "OpEntryPoint GLCompute %main \"main\"\n"
         "OpExecutionMode %main LocalSize 1 1 1\n"
         "%uint = OpTypeInt 32 0\n"
         "%uint_3 = OpConstant %uint 3\n" +
         middle_id + " = OpConstant %uint " + std::to_string(middle_length) +
         "\n" + output_id + " = OpConstant %uint " +
         std::to_string(output_length) + "\n" + component_type +
         "%bool = OpTypeBool\n"
         "%false = OpConstantFalse %bool\n"
         "%vec_input = OpTypeCooperativeVectorHW %component %uint_3\n"
         "%vec_middle = OpTypeCooperativeVectorHW %component " +
         middle_id +
         "\n"
         "%vec_output = OpTypeCooperativeVectorHW %component " +
         output_id +
         "\n"
         "%mat_first = OpTypeCooperativeMatrixHW %component %uint_3 " +
         middle_id +
         "\n"
         "%mat_second = OpTypeCooperativeMatrixHW %component " +
         middle_id + " " + output_id +
         "\n"
         "%input = OpUndef %vec_input\n"
         "%matrix0 = OpUndef %mat_first\n"
         "%matrix1 = OpUndef %mat_second\n"
         "%void = OpTypeVoid\n"
         "%fn = OpTypeFunction %void\n"
         "%main = OpFunction %void None %fn\n"
         "%entry = OpLabel\n"
         "%hidden = OpCooperativeVectorMatrixMulHW %vec_middle %input "
         "%matrix0\n" +
         bridge + "%out = OpCooperativeVectorMatrixMulHW %vec_output " +
         second_input +
         " %matrix1\n"
         "OpReturn\n"
         "OpFunctionEnd\n";
}

std::string MakePureMulChainWithK(uint32_t input_length, uint32_t middle_length,
                                  uint32_t output_length) {
  const std::string input_id = "%uint_" + std::to_string(input_length);
  const std::string middle_id = "%uint_" + std::to_string(middle_length);
  const std::string output_id = "%uint_" + std::to_string(output_length);

  return "OpCapability Shader\n"
         "OpCapability Float16\n"
         "OpCapability CooperativeVectorHW\n"
         "OpCapability CooperativeMatrixHW\n"
         "OpExtension \"SPV_HW_neural_shader\"\n"
         "OpMemoryModel Logical GLSL450\n"
         "OpEntryPoint GLCompute %main \"main\"\n"
         "OpExecutionMode %main LocalSize 1 1 1\n"
         "%uint = OpTypeInt 32 0\n" +
         input_id + " = OpConstant %uint " + std::to_string(input_length) +
         "\n" + middle_id + " = OpConstant %uint " +
         std::to_string(middle_length) + "\n" + output_id +
         " = OpConstant %uint " + std::to_string(output_length) +
         "\n"
         "%component = OpTypeFloat 16\n"
         "%vec_input = OpTypeCooperativeVectorHW %component " +
         input_id +
         "\n"
         "%vec_middle = OpTypeCooperativeVectorHW %component " +
         middle_id +
         "\n"
         "%vec_output = OpTypeCooperativeVectorHW %component " +
         output_id +
         "\n"
         "%mat_first = OpTypeCooperativeMatrixHW %component " +
         input_id + " " + middle_id +
         "\n"
         "%mat_second = OpTypeCooperativeMatrixHW %component " +
         middle_id + " " + output_id +
         "\n"
         "%input = OpUndef %vec_input\n"
         "%matrix0 = OpUndef %mat_first\n"
         "%matrix1 = OpUndef %mat_second\n"
         "%void = OpTypeVoid\n"
         "%fn = OpTypeFunction %void\n"
         "%main = OpFunction %void None %fn\n"
         "%entry = OpLabel\n"
         "%hidden = OpCooperativeVectorMatrixMulHW %vec_middle %input "
         "%matrix0\n"
         "%out = OpCooperativeVectorMatrixMulHW %vec_output %hidden "
         "%matrix1\n"
         "OpReturn\n"
         "OpFunctionEnd\n";
}

std::string MakeMulAddChain(uint32_t middle_length, uint32_t output_length,
                            bool first_has_bias, bool second_has_bias,
                            bool insert_relu) {
  const std::string middle_id = "%uint_" + std::to_string(middle_length);
  const std::string output_id = "%uint_" + std::to_string(output_length);

  const std::string first_opcode = first_has_bias
                                       ? "OpCooperativeVectorMatrixMulAddHW"
                                       : "OpCooperativeVectorMatrixMulHW";
  const std::string first_bias_suffix = first_has_bias ? " %bias0" : "";

  const std::string second_opcode = second_has_bias
                                        ? "OpCooperativeVectorMatrixMulAddHW"
                                        : "OpCooperativeVectorMatrixMulHW";
  const std::string second_bias_suffix = second_has_bias ? " %bias1" : "";

  std::string bias_decls;
  if (first_has_bias) bias_decls += "%bias0 = OpUndef %vec_middle\n";
  if (second_has_bias) bias_decls += "%bias1 = OpUndef %vec_output\n";

  std::string ext_inst_import;
  std::string relu_bridge;
  std::string second_input = "%hidden";
  if (insert_relu) {
    ext_inst_import = "%glsl = OpExtInstImport \"GLSL.std.450\"\n";
    relu_bridge =
        "%zero = OpConstantNull %vec_middle\n"
        "%relu = OpExtInst %vec_middle %glsl FMax %hidden %zero\n";
    second_input = "%relu";
  }

  return "OpCapability Shader\n"
         "OpCapability Float16\n"
         "OpCapability CooperativeVectorHW\n"
         "OpCapability CooperativeMatrixHW\n"
         "OpExtension \"SPV_HW_neural_shader\"\n" +
         ext_inst_import +
         "OpMemoryModel Logical GLSL450\n"
         "OpEntryPoint GLCompute %main \"main\"\n"
         "OpExecutionMode %main LocalSize 1 1 1\n"
         "%uint = OpTypeInt 32 0\n"
         "%uint_3 = OpConstant %uint 3\n" +
         middle_id + " = OpConstant %uint " + std::to_string(middle_length) +
         "\n" + output_id + " = OpConstant %uint " +
         std::to_string(output_length) +
         "\n"
         "%component = OpTypeFloat 16\n"
         "%vec_input = OpTypeCooperativeVectorHW %component %uint_3\n"
         "%vec_middle = OpTypeCooperativeVectorHW %component " +
         middle_id +
         "\n"
         "%vec_output = OpTypeCooperativeVectorHW %component " +
         output_id +
         "\n"
         "%mat_first = OpTypeCooperativeMatrixHW %component %uint_3 " +
         middle_id +
         "\n"
         "%mat_second = OpTypeCooperativeMatrixHW %component " +
         middle_id + " " + output_id +
         "\n"
         "%input = OpUndef %vec_input\n"
         "%matrix0 = OpUndef %mat_first\n"
         "%matrix1 = OpUndef %mat_second\n" +
         bias_decls +
         "%void = OpTypeVoid\n"
         "%fn = OpTypeFunction %void\n"
         "%main = OpFunction %void None %fn\n"
         "%entry = OpLabel\n"
         "%hidden = " +
         first_opcode + " %vec_middle %input %matrix0" + first_bias_suffix +
         "\n" + relu_bridge + "%out = " + second_opcode + " %vec_output " +
         second_input + " %matrix1" + second_bias_suffix +
         "\n"
         "OpReturn\n"
         "OpFunctionEnd\n";
}

std::string MakeMultiLayerMulAddChain(bool include_odd_tail) {
  const std::string prefix = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %vec4 "vec4"
OpName %vec8 "vec8"
OpName %vec48 "vec48"
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%uint_48 = OpConstant %uint 48
%half = OpTypeFloat 16
%vec4 = OpTypeCooperativeVectorHW %half %uint_4
%vec8 = OpTypeCooperativeVectorHW %half %uint_8
%vec48 = OpTypeCooperativeVectorHW %half %uint_48
%mat8x4 = OpTypeCooperativeMatrixHW %half %uint_8 %uint_4
%mat8x48 = OpTypeCooperativeMatrixHW %half %uint_8 %uint_48
%mat48x4 = OpTypeCooperativeMatrixHW %half %uint_48 %uint_4
%mat48x8 = OpTypeCooperativeMatrixHW %half %uint_48 %uint_8
%ptr_fn_vec4 = OpTypePointer Function %vec4
%ptr_fn_vec8 = OpTypePointer Function %vec8
%ptr_fn_vec48 = OpTypePointer Function %vec48
%input = OpUndef %vec8
%matrix0 = OpUndef %mat8x48
%matrix1 = OpUndef %mat48x8
%matrix2 = OpUndef %mat8x48
%matrix3_4 = OpUndef %mat48x4
%matrix3_8 = OpUndef %mat48x8
%matrix4 = OpUndef %mat8x4
%bias0 = OpUndef %vec48
%bias1 = OpUndef %vec8
%bias2 = OpUndef %vec48
%bias3_4 = OpUndef %vec4
%bias3_8 = OpUndef %vec8
%bias4 = OpUndef %vec4
%zero4 = OpConstantNull %vec4
%zero8 = OpConstantNull %vec8
%zero48 = OpConstantNull %vec48
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%raw0_var = OpVariable %ptr_fn_vec48 Function
%act0_var = OpVariable %ptr_fn_vec48 Function
%raw1_var = OpVariable %ptr_fn_vec8 Function
%act1_var = OpVariable %ptr_fn_vec8 Function
%raw2_var = OpVariable %ptr_fn_vec48 Function
%act2_var = OpVariable %ptr_fn_vec48 Function
%raw3_4_var = OpVariable %ptr_fn_vec4 Function
%raw3_8_var = OpVariable %ptr_fn_vec8 Function
%act3_var = OpVariable %ptr_fn_vec8 Function
%raw4_var = OpVariable %ptr_fn_vec4 Function
%layer0 = OpCooperativeVectorMatrixMulAddHW %vec48 %input %matrix0 %bias0
OpStore %raw0_var %layer0
%raw0 = OpLoad %vec48 %raw0_var
%relu0 = OpExtInst %vec48 %glsl FMax %raw0 %zero48
OpStore %act0_var %relu0
%act0 = OpLoad %vec48 %act0_var
%layer1 = OpCooperativeVectorMatrixMulAddHW %vec8 %act0 %matrix1 %bias1
OpStore %raw1_var %layer1
%raw1 = OpLoad %vec8 %raw1_var
%relu1 = OpExtInst %vec8 %glsl FMax %raw1 %zero8
OpStore %act1_var %relu1
%act1 = OpLoad %vec8 %act1_var
%layer2 = OpCooperativeVectorMatrixMulAddHW %vec48 %act1 %matrix2 %bias2
OpStore %raw2_var %layer2
%raw2 = OpLoad %vec48 %raw2_var
%relu2 = OpExtInst %vec48 %glsl FMax %raw2 %zero48
OpStore %act2_var %relu2
%act2 = OpLoad %vec48 %act2_var
)";

  const std::string four_layer_tail = R"(
%layer3 = OpCooperativeVectorMatrixMulAddHW %vec4 %act2 %matrix3_4 %bias3_4
OpStore %raw3_4_var %layer3
%raw3_4 = OpLoad %vec4 %raw3_4_var
%relu3 = OpExtInst %vec4 %glsl FMax %raw3_4 %zero4
OpReturn
OpFunctionEnd
)";

  const std::string five_layer_tail = R"(
%layer3 = OpCooperativeVectorMatrixMulAddHW %vec8 %act2 %matrix3_8 %bias3_8
OpStore %raw3_8_var %layer3
%raw3_8 = OpLoad %vec8 %raw3_8_var
%relu3 = OpExtInst %vec8 %glsl FMax %raw3_8 %zero8
OpStore %act3_var %relu3
%act3 = OpLoad %vec8 %act3_var
%layer4 = OpCooperativeVectorMatrixMulAddHW %vec4 %act3 %matrix4 %bias4
OpStore %raw4_var %layer4
%raw4 = OpLoad %vec4 %raw4_var
%relu4 = OpExtInst %vec4 %glsl FMax %raw4 %zero4
OpReturn
OpFunctionEnd
)";

  return prefix + (include_odd_tail ? five_layer_tail : four_layer_tail);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FusesPureMulWithAllThreeTails) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability FloatControls2
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_KHR_float_controls2"
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %hidden FPFastMathMode NotNaN
OpDecorate %out FPFastMathMode NotInf
OpName %vec7 "vec7"
OpName %hidden "hidden"
OpName %out "out"
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
%input = OpUndef %vec3
%matrix0 = OpUndef %mat3x18
%matrix1 = OpUndef %mat18x7
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%hidden = OpCooperativeVectorMatrixMulHW %vec18 %input %matrix0
%out = OpCooperativeVectorMatrixMulHW %vec7 %hidden %matrix1
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCompositeConstruct %vec7"));
  // The disassembler does not preserve the symbolic ID of the GLSL import.
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v2half"), 0u);
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
  EXPECT_GT(CountSubstring(disassembly, "FPFastMathMode NotNaN"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "FPFastMathMode NotInf"), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsNoContractionOnFirstLayer) {
  std::string text = MakePureMulChain(18, 7);
  ASSERT_TRUE(ReplaceOnce(&text, "OpExecutionMode %main LocalSize 1 1 1\n",
                          "OpExecutionMode %main LocalSize 1 1 1\n"
                          "OpDecorate %hidden NoContraction\n"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, false);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "NoContraction"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsNoContractionOnRelu) {
  std::string text =
      MakeMulAddChain(18, 7, /*first_has_bias=*/true, /*second_has_bias=*/true,
                      /*insert_relu=*/true);
  ASSERT_TRUE(ReplaceOnce(&text, "OpExecutionMode %main LocalSize 1 1 1\n",
                          "OpExecutionMode %main LocalSize 1 1 1\n"
                          "OpDecorate %relu NoContraction\n"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, false);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u,
            CountSubstring(disassembly, "OpCooperativeVectorMatrixMulAddHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, " FMax "));
  EXPECT_EQ(1u, CountSubstring(disassembly, "NoContraction"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsNoContractionOnSecondLayer) {
  std::string text = MakePureMulChain(18, 7);
  ASSERT_TRUE(ReplaceOnce(&text, "OpExecutionMode %main LocalSize 1 1 1\n",
                          "OpExecutionMode %main LocalSize 1 1 1\n"
                          "OpDecorate %out NoContraction\n"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, false);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "NoContraction"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FusesSingleHiddenLaneFinalSplit) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChain(17, 7), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

std::string MakeStreamingMatrixChain() {
  return R"(
OpCapability Shader
OpCapability Float16
OpCapability StorageBuffer16BitAccess
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %matrix0 "matrix0"
OpName %matrix1 "matrix1"
OpDecorate %arr54 ArrayStride 4
OpDecorate %arr126 ArrayStride 4
OpDecorate %Block0 Block
OpDecorate %Block1 Block
OpMemberDecorate %Block0 0 Offset 0
OpMemberDecorate %Block1 0 Offset 0
OpDecorate %buffer0 DescriptorSet 0
OpDecorate %buffer0 Binding 0
OpDecorate %buffer1 DescriptorSet 0
OpDecorate %buffer1 Binding 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_7 = OpConstant %uint 7
%uint_18 = OpConstant %uint 18
%uint_54 = OpConstant %uint 54
%uint_126 = OpConstant %uint 126
%half = OpTypeFloat 16
%vec3 = OpTypeCooperativeVectorHW %half %uint_3
%vec7 = OpTypeCooperativeVectorHW %half %uint_7
%vec18 = OpTypeCooperativeVectorHW %half %uint_18
%mat3x18 = OpTypeCooperativeMatrixHW %half %uint_3 %uint_18
%mat18x7 = OpTypeCooperativeMatrixHW %half %uint_18 %uint_7
%arr54 = OpTypeArray %half %uint_54
%arr126 = OpTypeArray %half %uint_126
%Block0 = OpTypeStruct %arr54
%Block1 = OpTypeStruct %arr126
%ptr_sb_Block0 = OpTypePointer StorageBuffer %Block0
%ptr_sb_Block1 = OpTypePointer StorageBuffer %Block1
%ptr_sb_arr54 = OpTypePointer StorageBuffer %arr54
%ptr_sb_arr126 = OpTypePointer StorageBuffer %arr126
%buffer0 = OpVariable %ptr_sb_Block0 StorageBuffer
%buffer1 = OpVariable %ptr_sb_Block1 StorageBuffer
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_3 = OpConstant %int 3
%int_7 = OpConstant %int 7
%int_18 = OpConstant %int 18
%v2int = OpTypeVector %int 2
%shape0 = OpConstantComposite %v2int %int_3 %int_18
%shape1 = OpConstantComposite %v2int %int_18 %int_7
%offset = OpConstantComposite %v2int %int_0 %int_0
%input = OpUndef %vec3
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%matrix0_ptr = OpAccessChain %ptr_sb_arr54 %buffer0 %int_0
%matrix0 = OpCooperativeMatrixLoadHW %mat3x18 %matrix0_ptr %shape0 %offset %int_0 Aligned 16
%hidden = OpCooperativeVectorMatrixMulHW %vec18 %input %matrix0
%matrix1_ptr = OpAccessChain %ptr_sb_arr126 %buffer1 %int_0
%matrix1 = OpCooperativeMatrixLoadHW %mat18x7 %matrix1_ptr %shape1 %offset %int_0 Aligned 16
%out = OpCooperativeVectorMatrixMulHW %vec7 %hidden %matrix1
OpReturn
OpFunctionEnd
)";
}

std::string MakeStreamingBiasChain() {
  return R"(
OpCapability Shader
OpCapability Float16
OpCapability StorageBuffer16BitAccess
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_KHR_16bit_storage"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %bias0_source "bias0_source"
OpName %bias1_source "bias1_source"
OpName %bias0 "bias0"
OpDecorate %arr32 ArrayStride 2
OpDecorate %BiasBlock Block
OpMemberDecorate %BiasBlock 0 Offset 0
OpDecorate %bias0_buffer DescriptorSet 0
OpDecorate %bias0_buffer Binding 0
OpDecorate %bias1_buffer DescriptorSet 0
OpDecorate %bias1_buffer Binding 1
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_7 = OpConstant %uint 7
%uint_17 = OpConstant %uint 17
%uint_32 = OpConstant %uint 32
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%int_4 = OpConstant %int 4
%int_8 = OpConstant %int 8
%half = OpTypeFloat 16
%vec3 = OpTypeCooperativeVectorHW %half %uint_3
%vec7 = OpTypeCooperativeVectorHW %half %uint_7
%vec17 = OpTypeCooperativeVectorHW %half %uint_17
%mat3x17 = OpTypeCooperativeMatrixHW %half %uint_3 %uint_17
%mat17x7 = OpTypeCooperativeMatrixHW %half %uint_17 %uint_7
%arr32 = OpTypeArray %half %uint_32
%BiasBlock = OpTypeStruct %arr32
%ptr_sb_BiasBlock = OpTypePointer StorageBuffer %BiasBlock
%ptr_sb_arr32 = OpTypePointer StorageBuffer %arr32
%ptr_fn_vec17 = OpTypePointer Function %vec17
%ptr_fn_vec7 = OpTypePointer Function %vec7
%bias0_buffer = OpVariable %ptr_sb_BiasBlock StorageBuffer
%bias1_buffer = OpVariable %ptr_sb_BiasBlock StorageBuffer
%input = OpUndef %vec3
%matrix0 = OpUndef %mat3x17
%matrix1 = OpUndef %mat17x7
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%bias0_temp = OpVariable %ptr_fn_vec17 Function
%bias0_value = OpVariable %ptr_fn_vec17 Function
%bias1_temp = OpVariable %ptr_fn_vec7 Function
%bias1_value = OpVariable %ptr_fn_vec7 Function
%bias0_base = OpAccessChain %ptr_sb_arr32 %bias0_buffer %int_0
%bias0_source = OpCooperativeVectorLoadHW %vec17 %bias0_base %int_8 Aligned 16
OpStore %bias0_temp %bias0_source
%bias0_temp_load = OpLoad %vec17 %bias0_temp
OpStore %bias0_value %bias0_temp_load
%bias0 = OpLoad %vec17 %bias0_value
%hidden = OpCooperativeVectorMatrixMulAddHW %vec17 %input %matrix0 %bias0
%bias1_base = OpAccessChain %ptr_sb_arr32 %bias1_buffer %int_0
%bias1_source = OpCooperativeVectorLoadHW %vec7 %bias1_base %int_4 Aligned 8
OpStore %bias1_temp %bias1_source
%bias1_temp_load = OpLoad %vec7 %bias1_temp
OpStore %bias1_value %bias1_temp_load
%bias1 = OpLoad %vec7 %bias1_value
%out = OpCooperativeVectorMatrixMulAddHW %vec7 %hidden %matrix1 %bias1
OpReturn
OpFunctionEnd
)";
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, StreamsPrivateRowMajorMatrixLoads) {
  const std::string text = MakeStreamingMatrixChain();

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeMatrixLoadHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(180u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpCompositeExtract %half %matrix"));
  EXPECT_GT(CountSubstring(disassembly, "Aligned 16"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "Aligned 4"), 0u);
  EXPECT_EQ(0u, CountSubstring(disassembly, "Aligned 2"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       StreamsPrivateBiasLoadsThroughFunctionTransport) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakeStreamingBiasChain(), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpCooperativeVectorMatrixMulAddHW"));
  EXPECT_EQ(24u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoad %vec"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpStore"));
  EXPECT_GT(CountSubstring(disassembly, "Aligned 16"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "Aligned 8"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "Aligned 4"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "Aligned 2"), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       KeepsSharedFirstBiasAggregateWithoutDisablingSecondDirectBias) {
  std::string text = MakeStreamingBiasChain();
  const size_t store_pos = text.find("OpStore %bias0_temp %bias0_source");
  ASSERT_NE(std::string::npos, store_pos);
  text.insert(store_pos, "%extra = OpCompositeExtract %half %bias0_source 0\n");

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(7u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_GT(CountSubstring(disassembly, "OpCompositeExtract %half %bias0"), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       KeepsSharedSecondBiasAggregateWithoutDisablingFirstDirectBias) {
  std::string text = MakeStreamingBiasChain();
  const size_t store_pos = text.find("OpStore %bias1_temp %bias1_source");
  ASSERT_NE(std::string::npos, store_pos);
  text.insert(store_pos, "%extra = OpCompositeExtract %half %bias1_source 0\n");

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(17u, CountSubstring(disassembly, "OpLoad %half"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, PropagatesMovableBiasMemoryOperands) {
  std::string text = MakeStreamingBiasChain();
  ASSERT_TRUE(
      ReplaceOnce(&text, "%int_8 Aligned 16", "%int_8 Nontemporal|Aligned 16"));
  ASSERT_TRUE(
      ReplaceOnce(&text, "%int_4 Aligned 8", "%int_4 Nontemporal|Aligned 8"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(24u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_EQ(24u, CountSubstring(disassembly, "Nontemporal"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       KeepsEarlierBiasAggregateAcrossModuleStore) {
  std::string text = MakeStreamingBiasChain();
  ASSERT_TRUE(ReplaceOnce(&text, "%half = OpTypeFloat 16\n",
                          "%half = OpTypeFloat 16\n"
                          "%half_0 = OpConstant %half 0\n"));
  ASSERT_TRUE(
      ReplaceOnce(&text, "%ptr_sb_arr32 = OpTypePointer StorageBuffer %arr32\n",
                  "%ptr_sb_arr32 = OpTypePointer StorageBuffer %arr32\n"
                  "%ptr_sb_half = OpTypePointer StorageBuffer %half\n"));
  const size_t hidden_pos =
      text.find("%hidden = OpCooperativeVectorMatrixMulAddHW");
  ASSERT_NE(std::string::npos, hidden_pos);
  text.insert(hidden_pos,
              "%bias0_element = OpAccessChain %ptr_sb_half %bias0_base "
              "%int_0\n"
              "OpStore %bias0_element %half_0\n");

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(7u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_GT(CountSubstring(disassembly, "OpStore"), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       KeepsBiasSourceThatIsAlsoTheFirstInputAggregate) {
  std::string text = MakeStreamingBiasChain();
  ASSERT_TRUE(ReplaceOnce(
      &text, "%vec3 = OpTypeCooperativeVectorHW %half %uint_3\n", ""));
  ASSERT_TRUE(ReplaceOnce(
      &text, "%mat3x17 = OpTypeCooperativeMatrixHW %half %uint_3 %uint_17",
      "%mat3x17 = OpTypeCooperativeMatrixHW %half %uint_17 %uint_17"));
  ASSERT_TRUE(ReplaceOnce(&text, "%input = OpUndef %vec3\n", ""));
  ASSERT_TRUE(ReplaceOnce(
      &text,
      "%hidden = OpCooperativeVectorMatrixMulAddHW %vec17 %input %matrix0 "
      "%bias0",
      "%hidden = OpCooperativeVectorMatrixMulAddHW %vec17 %bias0 %matrix0 "
      "%bias0"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(7u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_GT(CountSubstring(disassembly, "OpCompositeExtract %half %bias0"), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       KeepsOutOfBoundsFixedArrayBiasAggregate) {
  std::string text = MakeStreamingBiasChain();
  ASSERT_TRUE(ReplaceOnce(&text, "%int_8 = OpConstant %int 8",
                          "%int_8 = OpConstant %int 20"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(7u, CountSubstring(disassembly, "OpLoad %half"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       KeepsVolatileFirstBiasWithoutDisablingSecondDirectBias) {
  std::string text = MakeStreamingBiasChain();
  ASSERT_TRUE(
      ReplaceOnce(&text, "%int_8 Aligned 16", "%int_8 Volatile|Aligned 16"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(7u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_GT(CountSubstring(disassembly, "Volatile|Aligned 16"), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, KeepsBiasLoadsAcrossHwBarrier) {
  std::string text = MakeStreamingBiasChain();
  const size_t second_pos =
      text.find("%out = OpCooperativeVectorMatrixMulAddHW");
  ASSERT_NE(std::string::npos, second_pos);
  text.insert(second_pos,
              "OpBarrierArriveHW %int_0 %int_1\n"
              "OpBarrierWaitHW %int_0 %int_1\n");

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorLoadHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoad %half"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierArriveHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierWaitHW"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       KeepsAggregateMatrixLoadsAcrossHwBarrier) {
  std::string text = MakeStreamingMatrixChain();
  const size_t second_pos = text.find("%out = OpCooperativeVectorMatrixMulHW");
  ASSERT_NE(std::string::npos, second_pos);
  text.insert(second_pos,
              "OpBarrierArriveHW %int_0 %int_1\n"
              "OpBarrierWaitHW %int_0 %int_1\n");

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeMatrixLoadHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierArriveHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierWaitHW"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       KeepsAliasedMatrixLoadAcrossModuleStore) {
  std::string text = MakeStreamingMatrixChain();
  ASSERT_TRUE(ReplaceOnce(&text, "%half = OpTypeFloat 16\n",
                          "%half = OpTypeFloat 16\n"
                          "%half_0 = OpConstant %half 0\n"));
  ASSERT_TRUE(ReplaceOnce(
      &text, "%ptr_sb_arr126 = OpTypePointer StorageBuffer %arr126\n",
      "%ptr_sb_arr126 = OpTypePointer StorageBuffer %arr126\n"
      "%ptr_sb_half = OpTypePointer StorageBuffer %half\n"));
  const size_t first_pos =
      text.find("%hidden = OpCooperativeVectorMatrixMulHW");
  ASSERT_NE(std::string::npos, first_pos);
  text.insert(first_pos,
              "%matrix0_element = OpAccessChain %ptr_sb_half %matrix0_ptr "
              "%int_0\n"
              "OpStore %matrix0_element %half_0\n");

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCooperativeMatrixLoadHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpStore "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       FallsBackBeforeFlattenedDirectIndexOverflowsUint32) {
  std::string text = MakeStreamingMatrixChain();
  ASSERT_TRUE(ReplaceOnce(&text, "%int_18 = OpConstant %int 18\n",
                          "%int_18 = OpConstant %int 18\n"
                          "%int_n4 = OpConstant %int -4\n"
                          "%int_n7 = OpConstant %int -7\n"));
  ASSERT_TRUE(ReplaceOnce(
      &text, "%offset = OpConstantComposite %v2int %int_0 %int_0\n",
      "%offset = OpConstantComposite %v2int %int_0 %int_0\n"
      "%shape_huge = OpConstantComposite %v2int %int_n4 %int_18\n"
      "%offset_huge = OpConstantComposite %v2int %int_n7 %int_0\n"));
  ASSERT_TRUE(
      ReplaceOnce(&text, "%matrix0_ptr %shape0 %offset %int_0 Aligned 16",
                  "%matrix0_ptr %shape_huge %offset_huge %int_0 Aligned 16"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCooperativeMatrixLoadHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_GT(CountSubstring(disassembly, "OpCompositeExtract %half %matrix0"),
            0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       FusesMulAddZeroMaxThroughFunctionTransport) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %vec7 "vec7"
OpName %hidden "hidden"
OpName %relu "relu"
OpName %out "out"
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
%_ptr_Function_vec18 = OpTypePointer Function %vec18
%input = OpUndef %vec3
%matrix0 = OpUndef %mat3x18
%matrix1 = OpUndef %mat18x7
%bias0 = OpUndef %vec18
%bias1 = OpUndef %vec7
%zero = OpConstantNull %vec18
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%hidden_var = OpVariable %_ptr_Function_vec18 Function
%relu_var = OpVariable %_ptr_Function_vec18 Function
%hidden = OpCooperativeVectorMatrixMulAddHW %vec18 %input %matrix0 %bias0
OpStore %hidden_var %hidden
%hidden_load = OpLoad %vec18 %hidden_var
%relu = OpExtInst %vec18 %glsl FMax %hidden_load %zero
OpStore %relu_var %relu
%relu_load = OpLoad %vec18 %relu_var
%out = OpCooperativeVectorMatrixMulAddHW %vec7 %relu_load %matrix1 %bias1
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpCooperativeVectorMatrixMulAddHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpExtInst %vec18"));
  EXPECT_EQ(9u, CountSubstring(disassembly, " FMax "));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoad"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpStore"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCompositeConstruct %vec7"));
  // Match the stable result type and opcode instead of the import ID.
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v2half"), 0u);
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       FusesFourLayersAsTwoDisjointPairsWithIndependentBudgets) {
  // Each pair is within the 768-MAC cap, while the full four-layer chain is
  // not.  This also exercises the candidate snapshot after the first pair is
  // rewritten and its two original matmuls are killed.
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakeMultiLayerMulAddChain(/*include_odd_tail=*/false), true, true,
      768ull);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpCooperativeVectorMatrixMulAddHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCompositeConstruct %vec8"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpCompositeConstruct %vec4"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpExtInst %vec48"));
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v2half"), 0u);
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FusesTwoPairsAndKeepsOddFifthLayer) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakeMultiLayerMulAddChain(/*include_odd_tail=*/true), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(1u,
            CountSubstring(disassembly, "OpCooperativeVectorMatrixMulAddHW"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCompositeConstruct %vec8"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpExtInst %vec48"));
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v2half"), 0u);
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsNonZeroMaxOperand) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpName %out "out"
%uint = OpTypeInt 32 0
%uint_3 = OpConstant %uint 3
%uint_7 = OpConstant %uint 7
%uint_18 = OpConstant %uint 18
%half = OpTypeFloat 16
%half_1 = OpConstant %half 1
%vec3 = OpTypeCooperativeVectorHW %half %uint_3
%vec7 = OpTypeCooperativeVectorHW %half %uint_7
%vec18 = OpTypeCooperativeVectorHW %half %uint_18
%mat3x18 = OpTypeCooperativeMatrixHW %half %uint_3 %uint_18
%mat18x7 = OpTypeCooperativeMatrixHW %half %uint_18 %uint_7
%nonzero = OpConstantComposite %vec18
    %half_1 %half_1 %half_1 %half_1 %half_1 %half_1
    %half_1 %half_1 %half_1 %half_1 %half_1 %half_1
    %half_1 %half_1 %half_1 %half_1 %half_1 %half_1
%input = OpUndef %vec3
%matrix0 = OpUndef %mat3x18
%matrix1 = OpUndef %mat18x7
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%hidden = OpCooperativeVectorMatrixMulHW %vec18 %input %matrix0
%relu = OpExtInst %vec18 %glsl FMax %hidden %nonzero
%out = OpCooperativeVectorMatrixMulHW %vec7 %relu %matrix1
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, " FMax "));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsIntermediateWithExtraUser) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
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
%input = OpUndef %vec3
%matrix0 = OpUndef %mat3x18
%matrix1 = OpUndef %mat18x7
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%hidden = OpCooperativeVectorMatrixMulHW %vec18 %input %matrix0
%out = OpCooperativeVectorMatrixMulHW %vec7 %hidden %matrix1
%extra = OpCooperativeVectorMatrixMulHW %vec7 %hidden %matrix1
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(3u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RespectsUnrolledMacThreshold) {
  const std::string text = R"(
OpCapability Shader
OpCapability Float16
OpCapability CooperativeVectorHW
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
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
%input = OpUndef %vec3
%matrix0 = OpUndef %mat3x18
%matrix1 = OpUndef %mat18x7
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%hidden = OpCooperativeVectorMatrixMulHW %vec18 %input %matrix0
%out = OpCooperativeVectorMatrixMulHW %vec7 %hidden %matrix1
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true, 1ull);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsMiddleDimensionAtSplitWidth) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChain(16, 7), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsOutputWiderThanSplitWidth) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChain(18, 17), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsFloat32Chain) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChain(18, 7, false), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsIntermediateSelect) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChain(18, 7, true, IntermediateKind::kSelect), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpSelect"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsCrossBasicBlockChain) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChain(18, 7, true, IntermediateKind::kCrossBlock), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpLabel"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FullScalarLoweringSkipsTwoLayerFusion) {
  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      MakePureMulChain(18, 7), true, true,
      HwLowerToStandardPass::LoweringMode::kForceScalar);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  // The pre-fusion pass always materializes an f16vec2 type.  Force-scalar
  // lowering handles the original two operations independently and therefore
  // must not introduce that vector type.
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpTypeVector"));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest,
       PreferPackedLoweringConsumesFusedPairsAndOddTail) {
  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      MakeMultiLayerMulAddChain(/*include_odd_tail=*/true), true, true,
      HwLowerToStandardPass::LoweringMode::kPreferPackedVec4);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "CooperativeVectorHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "CooperativeMatrixHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "SPV_HW_neural_shader"));
  EXPECT_GT(CountSubstring(disassembly, "OpTypeVector %half 2"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeVector %half 4"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v2half"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v4half"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpFunctionCall"), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FusesMulAddFirstPureMulSecond) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakeMulAddChain(18, 7, /*first_has_bias=*/true,
                      /*second_has_bias=*/false, /*insert_relu=*/false),
      true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpCooperativeVectorMatrixMulAddHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FusesPureMulFirstMulAddSecond) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakeMulAddChain(18, 7, /*first_has_bias=*/false,
                      /*second_has_bias=*/true, /*insert_relu=*/false),
      true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpCooperativeVectorMatrixMulAddHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FusesMulAddPairWithoutRelu) {
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakeMulAddChain(18, 7, /*first_has_bias=*/true,
                      /*second_has_bias=*/true, /*insert_relu=*/false),
      true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u,
            CountSubstring(disassembly, "OpCooperativeVectorMatrixMulAddHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " FMax "));
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsVolatileMemoryOperand) {
  std::string text = MakeStreamingMatrixChain();
  // Add Volatile to the first matrix load's memory operands.
  ASSERT_TRUE(
      ReplaceOnce(&text, "%int_0 Aligned 16", "%int_0 Volatile|Aligned 16"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  // The volatile load cannot be streamed — aggregate load preserved.
  EXPECT_GT(CountSubstring(disassembly, "OpCooperativeMatrixLoadHW"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpCompositeExtract %half"), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, AcceptsExactMacBudgetBoundary) {
  // K=112, N=32, P=16: K*N + N*P = 3584 + 512 = 4096 = default max.
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChainWithK(112, 32, 16), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsOneMacOverBudget) {
  // K=113, N=32, P=16: K*N + N*P = 3616 + 512 = 4128 > 4096.
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChainWithK(113, 32, 16), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FusesLargeHiddenDimMultipleSplits) {
  // K=3, N=64, P=7: 4 splits of 16. K*N+N*P = 192+448 = 640.
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChain(64, 7), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, FusesSingleOutputLane) {
  // K=3, N=18, P=1.
  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      MakePureMulChain(18, 1), true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u);
}

TEST_F(HwFuseTwoLayerVectorMatmulTest, RejectsSpecConstantShape) {
  std::string text = MakePureMulChain(18, 7);
  ASSERT_TRUE(ReplaceOnce(&text, "%uint_18 = OpConstant %uint 18",
                          "%uint_18 = OpSpecConstant %uint 18"));

  auto result = SinglePassRunAndDisassemble<HwFuseTwoLayerVectorMatmulPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithoutChange, std::get<1>(result));
  EXPECT_EQ(2u, CountSubstring(disassembly, "OpCooperativeVectorMatrixMulHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, " Fma "));
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
