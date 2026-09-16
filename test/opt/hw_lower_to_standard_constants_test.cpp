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

std::string ConstantModule(
    const std::string& type, uint32_t length, const std::string& extra = "",
    const std::string& body = "%use = OpCopyObject %T %value\n",
    const std::string& scalar = "OpConstant %float 1") {
  return R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpCapability ReplicatedCompositesEXT
OpExtension "SPV_HW_neural_shader"
OpExtension "SPV_EXT_replicated_composites"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%one_u = OpConstant %uint 1
%float = OpTypeFloat 32
%n = OpConstant %uint )" +
         std::to_string(length) + "\n%one = " + scalar + "\n%T = " + type + R"(
%value = OpConstantCompositeReplicateEXT %T %one
)" + extra +
         R"(
%main = OpFunction %void None %fn
%entry = OpLabel
)" + body +
         "OpReturn\nOpFunctionEnd\n";
}

class HwLargeConstantTest : public HwLowerToStandardTest {
 protected:
  std::string Lower(const std::string& text, bool scalar = false) {
    SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
    std::vector<uint32_t> binary;
    EXPECT_TRUE(tools.Assemble(text, &binary));
    EXPECT_TRUE(tools.Validate(binary));
    auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
        text, true, true,
        scalar ? HwLowerToStandardPass::LoweringMode::kForceScalar
               : HwLowerToStandardPass::LoweringMode::kPreferPackedVec2,
        HwLowerToStandardPass::CompletenessMode::kExtensionFree);
    EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
    ExpectNoHwOrCoopMatrix(std::get<0>(result));
    EXPECT_EQ(0u, CountSubstring(std::get<0>(result), "ReplicateEXT"));
    return std::get<0>(result);
  }
};

TEST_F(HwLargeConstantTest, PreservesSmallConstantsAndLoopsAboveWordLimit) {
  for (bool scalar : {false, true}) {
    for (uint32_t n : {65532u, 65533u, 131066u}) {
      for (const char* type :
           {"OpTypeArray %float %n", "OpTypeCooperativeVectorHW %float %n",
            "OpTypeCooperativeMatrixHW %float %one_u %n"}) {
        SCOPED_TRACE(type);
        SCOPED_TRACE(n);
        SCOPED_TRACE(scalar);
        const std::string out = Lower(ConstantModule(type, n), scalar);
        const bool packed =
            !scalar && n % 2 == 0 &&
            std::string(type).find("Cooperative") != std::string::npos;
        EXPECT_EQ((packed ? n / 2 : n) > 65532,
                  out.find("OpLoopMerge") != std::string::npos);
      }
    }
  }
}

TEST_F(HwLargeConstantTest, LargeConstantFunctionAndPrivateInitializers) {
  for (const char* storage : {"Function", "Private"}) {
    for (const char* type :
         {"OpTypeArray %float %n", "OpTypeCooperativeVectorHW %float %n"}) {
      std::string declarations =
          std::string("%ptr = OpTypePointer ") + storage + " %T\n";
      std::string variable =
          std::string("%var = OpVariable %ptr ") + storage + " %value\n";
      std::string body = "%loaded = OpLoad %T %var\n";
      if (std::string(storage) == "Function")
        body = variable + body;
      else
        declarations += variable;
      const std::string out =
          Lower(ConstantModule(type, 65533, declarations, body));
      EXPECT_NE(std::string::npos, out.find("OpLoopMerge"));
      EXPECT_NE(std::string::npos, out.find("OpStore"));
    }
  }
}

TEST_F(HwLargeConstantTest, NestedParentConstantAndLiteralSpecExtract) {
  const std::string extra = R"(
%Pair = OpTypeStruct %T %float
%pair = OpConstantComposite %Pair %value %one
%extracted = OpSpecConstantOp %float CompositeExtract %value 65532
)";
  const std::string out =
      Lower(ConstantModule("OpTypeArray %float %n", 65533, extra,
                           "%copy = OpCopyObject %Pair %pair\n%x = "
                           "OpCopyObject %float %extracted\n"));
  EXPECT_NE(std::string::npos, out.find("OpLoopMerge"));
  EXPECT_EQ(0u, CountSubstring(out, "OpSpecConstantOp"));
}

TEST_F(HwLargeConstantTest, RuntimeOrdinaryRepeatUsesLoopAtOriginalSite) {
  std::string text = ConstantModule("OpTypeArray %float %n", 65533, "",
                                    "%x = OpCopyObject %float %one\n%runtime = "
                                    "OpCompositeConstructReplicateEXT %T %x\n");
  const std::string out = Lower(text);
  EXPECT_NE(std::string::npos, out.find("OpLoopMerge"));
}

TEST_F(HwLargeConstantTest, ZeroAndUndefinedLargeValues) {
  for (const char* scalar : {"OpConstantNull %float", "OpUndef %float"}) {
    const auto out =
        Lower(ConstantModule("OpTypeArray %float %n", 65533, "",
                             "%use = OpCopyObject %T %value\n", scalar));
    EXPECT_NE(std::string::npos, out.find("OpLoopMerge"));
  }
}

TEST_F(HwLargeConstantTest,
       DefaultElementPolicyAllowsMoreThanOneMillionElements) {
  std::string text =
      ConstantModule("OpTypeCooperativeVectorHW %float %n", 1048577, "",
                     "%sum = OpFAdd %T %value %value\n");
  const auto out = Lower(text, true);
  EXPECT_NE(std::string::npos, out.find("OpLoopMerge"));
  EXPECT_LT(out.size(), 30000u);
}

TEST_F(HwLargeConstantTest, DefaultMacPolicyAllowsMoreThanSixteenMillionMacs) {
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
%float = OpTypeFloat 32
%n = OpConstant %uint 256
%k = OpConstant %uint 257
%A = OpTypeCooperativeMatrixHW %float %n %k
%B = OpTypeCooperativeMatrixHW %float %k %n
%C = OpTypeCooperativeMatrixHW %float %n %n
%a = OpConstantNull %A
%b = OpConstantNull %B
%c = OpConstantNull %C
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpCooperativeMatrixMulAddHW %C %a %b %c
OpReturn
OpFunctionEnd
)";
  const auto out = Lower(text, true);
  EXPECT_NE(std::string::npos, out.find("OpLoopMerge"));
  EXPECT_LT(out.size(), 30000u);
}

TEST_F(HwLargeConstantTest, SharedPrivateInitializerRunsAtBothEntryPoints) {
  std::string text = ConstantModule("OpTypeArray %float %n", 65533,
                                    "%ptr = OpTypePointer Private %T\n%global "
                                    "= OpVariable %ptr Private %value\n",
                                    "%read = OpFunctionCall %void %reader\n");
  const std::string entry = "OpEntryPoint GLCompute %main \"main\"";
  text.replace(text.find(entry), entry.size(),
               entry + "\nOpEntryPoint GLCompute %other \"other\"");
  const std::string mode = "OpExecutionMode %main LocalSize 1 1 1";
  text.replace(text.find(mode), mode.size(),
               mode + "\nOpExecutionMode %other LocalSize 1 1 1");
  text +=
      "%other = OpFunction %void None %fn\n%other_entry = OpLabel\n"
      "%read_other = OpFunctionCall %void %reader\nOpReturn\nOpFunctionEnd\n"
      "%reader = OpFunction %void None %fn\n%reader_entry = OpLabel\n"
      "%shared_read = OpLoad %T %global\nOpReturn\nOpFunctionEnd\n";
  const std::string out = Lower(text);
  EXPECT_EQ(2u, CountSubstring(out, "OpEntryPoint"));
  // Two entry-point stores plus the shared helper's element store.
  EXPECT_EQ(3u, CountSubstring(out, "OpStore"));
  EXPECT_EQ(1u, CountSubstring(out, "OpLoopMerge"));
}

TEST_F(HwLargeConstantTest, LargeConstantDominatesPhiAndCallArguments) {
  std::string text = ConstantModule(
      "OpTypeCooperativeVectorHW %float %n", 65533,
      "%bool = OpTypeBool\n%true = OpConstantTrue %bool\n%take_type = "
      "OpTypeFunction %T %T\n",
      "OpSelectionMerge %merge None\nOpBranchConditional %true %left %right\n"
      "%left = OpLabel\nOpBranch %merge\n%right = OpLabel\nOpBranch %merge\n"
      "%merge = OpLabel\n%phi = OpPhi %T %value %left %value %right\n"
      "%call = OpFunctionCall %T %take %phi\n");
  text +=
      "%take = OpFunction %T None %take_type\n%p = OpFunctionParameter %T\n"
      "%take_entry = OpLabel\nOpReturnValue %p\nOpFunctionEnd\n";
  const auto out = Lower(text);
  EXPECT_NE(std::string::npos, out.find("OpPhi"));
  EXPECT_EQ(1u, CountSubstring(out, "OpLoopMerge"));
}

TEST_F(HwLargeConstantTest, OutputInitializerKeepsDeclaredArrayAndInterface) {
  std::string text = ConstantModule("OpTypeArray %float %n", 65533,
                                    "%ptr = OpTypePointer Output %T\n%output = "
                                    "OpVariable %ptr Output %value\n",
                                    "");
  const std::string ep = "OpEntryPoint GLCompute %main \"main\"";
  text.replace(text.find(ep), ep.size(),
               "OpEntryPoint Fragment %main \"main\" %output");
  const std::string mode = "OpExecutionMode %main LocalSize 1 1 1";
  text.replace(
      text.find(mode), mode.size(),
      "OpExecutionMode %main OriginUpperLeft\nOpDecorate %output Location 0");
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  SpirvTools tools(SPV_ENV_VULKAN_1_1);
  std::vector<uint32_t> binary;
  ASSERT_TRUE(tools.Assemble(text, &binary));
  ASSERT_TRUE(tools.Validate(binary));
  const auto out = Lower(text);
  EXPECT_NE(std::string::npos, out.find("OpTypeArray %float %uint_65533"));
  EXPECT_EQ(1u, CountSubstring(out, "OpLoopMerge"));
}

TEST_F(HwLargeConstantTest, ParentContainingLargeCooperativeConstant) {
  const std::string text =
      ConstantModule("OpTypeCooperativeVectorHW %float %n", 65533,
                     "%Parent = OpTypeStruct %T %T\n%parent = "
                     "OpConstantComposite %Parent %value %value\n",
                     "%copy = OpCopyObject %Parent %parent\n");
  const auto out = Lower(text);
  EXPECT_NE(std::string::npos, out.find("OpLoopMerge"));
}

TEST_F(HwLargeConstantTest, PreservesSpecializedScalarDataInLargeConstant) {
  std::string text = ConstantModule(
      "OpTypeCooperativeVectorHW %float %n", 65533, "",
      "%copy = OpCopyObject %T %value\n", "OpSpecConstant %float 1");
  const std::string from = "OpConstantCompositeReplicateEXT";
  text.replace(text.find(from), from.size(),
               "OpSpecConstantCompositeReplicateEXT");
  const auto out = Lower(text);
  EXPECT_NE(std::string::npos, out.find("OpSpecConstant %float 1"));
  EXPECT_EQ(1u, CountSubstring(out, "OpLoopMerge"));
}

TEST_F(HwLargeConstantTest, MatrixBroadcastConstantLiteralExtract) {
  const auto out = Lower(ConstantModule(
      "OpTypeCooperativeMatrixHW %float %one_u %n", 65533,
      "%extracted = OpSpecConstantOp %float CompositeExtract %value 0 65532\n",
      "%copy = OpCopyObject %float %extracted\n"));
  EXPECT_EQ(0u, CountSubstring(out, "OpSpecConstantOp"));
}

TEST_F(HwLargeConstantTest,
       DynamicExtractReadsMaterializedStorageWithoutArrayCopies) {
  for (uint32_t n : {65533u, 131066u}) {
    const auto out = Lower(ConstantModule(
        "OpTypeCooperativeVectorHW %float %n", n, "",
        "%index = OpCopyObject %uint %one_u\n"
        "%first = OpVectorExtractDynamic %float %value %index\n"
        "%second = OpVectorExtractDynamic %float %value %index\n"));
    EXPECT_EQ(1u, CountSubstring(out, "OpLoopMerge"));
    EXPECT_EQ(0u, CountSubstring(out, "OpLoad %_arr_"));
    EXPECT_EQ(0u, CountSubstring(out, "OpReturnValue"));
  }
}

TEST_F(HwLargeConstantTest,
       PrivateInitializerReachedThroughGlobalPointerAlias) {
  std::string text = ConstantModule(
      "OpTypeArray %float %n", 65533,
      "%P = OpTypePointer Private %T\n%PP = OpTypePointer Private %P\n"
      "%global = OpVariable %P Private %value\n%alias = OpVariable %PP Private "
      "%global\n",
      "%pointer = OpLoad %P %alias\n%read = OpLoad %T %pointer\n");
  text.insert(0, "OpCapability VariablePointers\n");
  text.insert(text.find("OpExtension"),
              "OpExtension \"SPV_KHR_variable_pointers\"\n");
  SpirvTools tools(SPV_ENV_VULKAN_1_1);
  std::vector<uint32_t> binary;
  ASSERT_TRUE(tools.Assemble(text, &binary));
  ASSERT_TRUE(tools.Validate(binary));
  SetTargetEnv(SPV_ENV_VULKAN_1_1);
  const auto out = Lower(text);
  EXPECT_EQ(1u, CountSubstring(out, "OpLoopMerge"));
  EXPECT_EQ(2u, CountSubstring(out, "OpStore"));
}

TEST_F(HwLargeConstantTest, RejectsIndexRepresentationOverflow) {
  std::string text =
      ConstantModule("OpTypeCooperativeMatrixHW %float %n %n", 65536);
  text = "; CHECK: element count exceeds uint32 index representation\n" + text;
  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
