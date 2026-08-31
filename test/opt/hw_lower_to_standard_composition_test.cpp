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
#include <string>
#include <vector>

#include "test/opt/hw_lower_to_standard_test_utils.h"

namespace spvtools {
namespace opt {
namespace {

class HwLowerToStandardCompositionTest : public HwLowerToStandardTest {
 protected:
  std::string RunCompositionCase(const std::string& text, spv_target_env env,
                                 HwLowerToStandardPass::LoweringMode mode,
                                 uint32_t max_unrolled_elements = 4096u) {
    SpirvTools tools(env);
    std::vector<uint32_t> original_binary;
    if (!tools.Assemble(text, &original_binary)) {
      ADD_FAILURE() << "failed to assemble composition input";
      return {};
    }
    if (!tools.Validate(original_binary)) {
      ADD_FAILURE() << "composition input is not valid SPIR-V";
      return {};
    }

    SetTargetEnv(env);
    auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
        text, true, true, mode,
        HwLowerToStandardPass::CompletenessMode::kExtensionFree, 1048576u,
        16777216ull, max_unrolled_elements, 4096ull);
    EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
    const std::string& disassembly = std::get<0>(result);
    ExpectNoHwOrCoopMatrix(disassembly);
    return disassembly;
  }

  static constexpr std::array<HwLowerToStandardPass::LoweringMode, 2>
      kLoweringModes = {
          HwLowerToStandardPass::LoweringMode::kPreferPackedVec2,
          HwLowerToStandardPass::LoweringMode::kForceScalar,
      };
};

constexpr std::array<HwLowerToStandardPass::LoweringMode, 2>
    HwLowerToStandardCompositionTest::kLoweringModes;

TEST_F(HwLowerToStandardCompositionTest,
       LowersDeepPrivateAggregateThroughPointerToPointer) {
  const std::string text = R"(
OpCapability Shader
OpCapability VariablePointers
OpCapability CooperativeVectorHW
OpExtension "SPV_KHR_variable_pointers"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec_array = OpTypeArray %vec4 %uint_2
%Inner = OpTypeStruct %uint %vec_array
%inner_array = OpTypeArray %Inner %uint_2
%Middle = OpTypeStruct %inner_array
%middle_array = OpTypeArray %Middle %uint_2
%Outer = OpTypeStruct %middle_array
%outer_null = OpConstantNull %Outer
%ptr_private_outer = OpTypePointer Private %Outer
%ptr_private_float = OpTypePointer Private %float
%ptr_function_private_ptr = OpTypePointer Function %ptr_private_outer
%global_a = OpVariable %ptr_private_outer Private %outer_null
%global_b = OpVariable %ptr_private_outer Private %outer_null
%main = OpFunction %void None %fn
%entry = OpLabel
%pointer_slot = OpVariable %ptr_function_private_ptr Function
OpCopyMemory %global_b %global_a
OpStore %pointer_slot %global_b
%base = OpLoad %ptr_private_outer %pointer_slot
%dynamic_index = OpIAdd %uint %uint_0 %uint_1
%element_ptr = OpAccessChain %ptr_private_float %base %uint_0 %dynamic_index %uint_0 %dynamic_index %uint_1 %dynamic_index %dynamic_index
OpStore %element_ptr %one
%element = OpLoad %float %element_ptr
OpReturn
OpFunctionEnd
)";

  for (const auto mode : kLoweringModes) {
    SCOPED_TRACE(mode == HwLowerToStandardPass::LoweringMode::kForceScalar
                     ? "force-scalar"
                     : "prefer-packed");
    const std::string result =
        RunCompositionCase(text, SPV_ENV_UNIVERSAL_1_3, mode);
    EXPECT_EQ(1u, CountSubstring(result, "OpCopyMemory"));
    EXPECT_EQ(1u, CountSubstring(result, "OpTypePointer Function"));
    EXPECT_EQ(1u, CountSubstring(result, "OpAccessChain"));
    if (mode == HwLowerToStandardPass::LoweringMode::kPreferPackedVec2) {
      EXPECT_EQ(1u, CountSubstring(result, "OpUDiv %uint"));
      EXPECT_EQ(1u, CountSubstring(result, "OpUMod %uint"));
      EXPECT_NE(std::string::npos, result.find("OpTypeVector %float 2"));
    } else {
      EXPECT_EQ(0u, CountSubstring(result, "OpTypeVector %float 2"));
    }
  }
}

TEST_F(HwLowerToStandardCompositionTest,
       LowersWorkgroupBackedInterleavedCoreAndHwOperations) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %scratch
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_4 = OpConstant %uint 4
%uint_16 = OpConstant %uint 16
%float = OpTypeFloat 32
%one = OpConstant %float 1
%v2uint = OpTypeVector %uint 2
%shape = OpConstantComposite %v2uint %uint_4 %uint_4
%offset = OpConstantComposite %v2uint %uint_0 %uint_0
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%mat4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%bias = OpConstantNull %vec4
%float_array = OpTypeArray %float %uint_16
%ptr_workgroup_array = OpTypePointer Workgroup %float_array
%scratch = OpVariable %ptr_workgroup_array Workgroup
%main = OpFunction %void None %fn
%entry = OpLabel
%input = OpCooperativeVectorLoadHW %vec4 %scratch %uint_0
%weights = OpCooperativeMatrixLoadHW %mat4 %scratch %shape %offset %uint_0
%product = OpCooperativeVectorMatrixMulAddHW %vec4 %input %weights %bias
%sum = OpFAdd %vec4 %product %input
%selected = OpSelect %vec4 %true %sum %product
%inserted = OpVectorInsertDynamic %vec4 %selected %one %uint_1
%element = OpVectorExtractDynamic %float %inserted %uint_1
%scaled = OpVectorTimesScalar %vec4 %inserted %element
OpCooperativeVectorStoreHW %scratch %uint_0 %scaled
OpReturn
OpFunctionEnd
)";

  for (const auto mode : kLoweringModes) {
    SCOPED_TRACE(mode == HwLowerToStandardPass::LoweringMode::kForceScalar
                     ? "force-scalar"
                     : "prefer-packed");
    const std::string result =
        RunCompositionCase(text, SPV_ENV_VULKAN_1_3, mode);
    EXPECT_EQ(1u, CountSubstring(result, "OpEntryPoint GLCompute"));
    EXPECT_GT(CountSubstring(result, "Workgroup"), 0u);
    EXPECT_GT(CountSubstring(result, "OpLoad %float"), 0u);
    EXPECT_GT(CountSubstring(result, "OpStore"), 0u);
    EXPECT_GT(CountSubstring(result, "OpFAdd"), 0u);
    EXPECT_GT(CountSubstring(result, "OpFMul"), 0u);
    EXPECT_GT(CountSubstring(result, "OpSelect"), 0u);
  }
}

TEST_F(HwLowerToStandardCompositionTest,
       LowersSharedAggregateReturnAcrossMultipleEntryPointsAndCallSites) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main_a "main_a" %global
OpEntryPoint GLCompute %main_b "main_b" %global
OpExecutionMode %main_a LocalSize 1 1 1
OpExecutionMode %main_b LocalSize 1 1 1
%void = OpTypeVoid
%void_fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec_array = OpTypeArray %vec4 %uint_2
%Payload = OpTypeStruct %vec_array %uint
%payload_null = OpConstantNull %Payload
%ptr_private_payload = OpTypePointer Private %Payload
%global = OpVariable %ptr_private_payload Private %payload_null
%payload_fn = OpTypeFunction %Payload %Payload
%leaf = OpFunction %Payload None %payload_fn
%leaf_value = OpFunctionParameter %Payload
%leaf_entry = OpLabel
%vectors = OpCompositeExtract %vec_array %leaf_value 0
%left = OpCompositeExtract %vec4 %vectors 0
%right = OpCompositeExtract %vec4 %vectors 1
%sum = OpFAdd %vec4 %left %right
%updated = OpCompositeInsert %Payload %sum %leaf_value 0 1
OpReturnValue %updated
OpFunctionEnd
%middle = OpFunction %Payload None %payload_fn
%middle_value = OpFunctionParameter %Payload
%middle_entry = OpLabel
%middle_call_0 = OpFunctionCall %Payload %leaf %middle_value
%middle_call_1 = OpFunctionCall %Payload %leaf %middle_call_0
OpReturnValue %middle_call_1
OpFunctionEnd
%main_a = OpFunction %void None %void_fn
%main_a_entry = OpLabel
%main_a_value = OpLoad %Payload %global
%main_a_call_0 = OpFunctionCall %Payload %middle %main_a_value
%main_a_call_1 = OpFunctionCall %Payload %middle %main_a_call_0
OpStore %global %main_a_call_1
OpReturn
OpFunctionEnd
%main_b = OpFunction %void None %void_fn
%main_b_entry = OpLabel
%main_b_value = OpLoad %Payload %global
%main_b_call = OpFunctionCall %Payload %middle %main_b_value
OpStore %global %main_b_call
OpReturn
OpFunctionEnd
)";

  for (const auto mode : kLoweringModes) {
    SCOPED_TRACE(mode == HwLowerToStandardPass::LoweringMode::kForceScalar
                     ? "force-scalar"
                     : "prefer-packed");
    const std::string result =
        RunCompositionCase(text, SPV_ENV_VULKAN_1_3, mode);
    EXPECT_EQ(2u, CountSubstring(result, "OpEntryPoint GLCompute"));
    EXPECT_EQ(5u, CountSubstring(result, "OpFunctionCall"));
    EXPECT_EQ(2u, CountSubstring(result, "OpReturnValue"));
    EXPECT_GT(CountSubstring(result, "OpTypePointer Private"), 0u);
    EXPECT_GT(CountSubstring(result, "OpTypeStruct"), 0u);
  }
}

TEST_F(HwLowerToStandardCompositionTest,
       LowersNestedStructuredCfgWithLoopSwitchPhiAndDynamicIndexing) {
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
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
%uint_2 = OpConstant %uint 2
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%a = OpUndef %vec8
%b = OpUndef %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
OpBranch %loop_header
%loop_header = OpLabel
%loop_value = OpPhi %vec8 %a %entry %next_value %continue
%iteration = OpPhi %uint %uint_0 %entry %next_iteration %continue
%loop_condition = OpULessThan %bool %iteration %uint_2
OpLoopMerge %loop_merge %continue None
OpBranchConditional %loop_condition %switch_header %loop_merge
%switch_header = OpLabel
OpSelectionMerge %switch_merge None
OpSwitch %iteration %case_exit 0 %case_add 1 %case_sub
%case_add = OpLabel
%added = OpFAdd %vec8 %loop_value %b
OpBranch %switch_merge
%case_sub = OpLabel
%subtracted = OpFSub %vec8 %loop_value %b
OpBranch %switch_merge
%case_exit = OpLabel
OpReturn
%switch_merge = OpLabel
%case_value = OpPhi %vec8 %added %case_add %subtracted %case_sub
%selected = OpSelect %vec8 %true %case_value %loop_value
OpSelectionMerge %if_merge None
OpBranchConditional %true %if_then %if_else
%if_then = OpLabel
%inserted = OpVectorInsertDynamic %vec8 %selected %one %iteration
OpBranch %if_merge
%if_else = OpLabel
%negated = OpFNegate %vec8 %selected
OpBranch %if_merge
%if_merge = OpLabel
%merged = OpPhi %vec8 %inserted %if_then %negated %if_else
%element = OpVectorExtractDynamic %float %merged %iteration
OpBranch %continue
%continue = OpLabel
%next_value = OpCopyObject %vec8 %merged
%next_iteration = OpIAdd %uint %iteration %uint_1
OpBranch %loop_header
%loop_merge = OpLabel
%last = OpVectorExtractDynamic %float %loop_value %uint_0
OpReturn
OpFunctionEnd
)";

  for (const auto mode : kLoweringModes) {
    SCOPED_TRACE(mode == HwLowerToStandardPass::LoweringMode::kForceScalar
                     ? "force-scalar"
                     : "prefer-packed");
    const std::string result = RunCompositionCase(
        text, SPV_ENV_VULKAN_1_3, mode, /*max_unrolled_elements=*/2u);
    EXPECT_GT(CountSubstring(result, "OpLoopMerge"), 1u);
    EXPECT_EQ(1u, CountSubstring(result, "OpSwitch"));
    EXPECT_GE(CountSubstring(result, "OpSelectionMerge"), 2u);
    EXPECT_GT(CountSubstring(result, "OpPhi"), 2u);
  }
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
