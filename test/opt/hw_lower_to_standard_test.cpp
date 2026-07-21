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
#include <string>
#include <utility>
#include <vector>

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

std::string MakeAsymmetricHwElementwiseModule(
    const std::string& diagnostic, const std::string& instruction) {
  return std::string("\n; CHECK: ") + diagnostic + R"(
OpCapability Shader
OpCapability Float16
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
%uint_4 = OpConstant %uint 4
%half = OpTypeFloat 16
%float = OpTypeFloat 32
%one = OpConstant %float 1
%v2half = OpTypeVector %half 2
%v2float = OpTypeVector %float 2
%v4half = OpTypeVector %half 4
%v4float = OpTypeVector %float 4
%m2half = OpTypeMatrix %v2half 2
%m2float = OpTypeMatrix %v2float 2
%hvec4 = OpTypeCooperativeVectorHW %half %uint_4
%fvec4 = OpTypeCooperativeVectorHW %float %uint_4
%hmat2 = OpTypeCooperativeMatrixHW %half %uint_2 %uint_2
%fmat2 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%ordinary_half = OpUndef %v4half
%ordinary_float = OpUndef %v4float
%ordinary_half_matrix = OpUndef %m2half
%ordinary_float_matrix = OpUndef %m2float
%hw_half = OpUndef %hvec4
%hw_float = OpUndef %fvec4
%hw_half_matrix = OpUndef %hmat2
%hw_float_matrix = OpUndef %fmat2
%main = OpFunction %void None %fn
%entry = OpLabel
)" + instruction + R"(
OpReturn
OpFunctionEnd
)";
}

void ExpectFailureWithMissingExtInstOperand(HwLowerToStandardTest* test,
                                            const std::string& text) {
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_3, test->consumer(), text,
                             SpirvTools::kDefaultAssembleOption);
  ASSERT_NE(nullptr, context.get()) << "Assembling failed for shader:\n"
                                    << text;

  Instruction* ext_inst = nullptr;
  context->module()->ForEachInst([&ext_inst](Instruction* inst) {
    if (!ext_inst && inst->opcode() == spv::Op::OpExtInst) ext_inst = inst;
  });
  ASSERT_NE(nullptr, ext_inst);
  ASSERT_GE(ext_inst->NumInOperands(), 1u);
  ext_inst->RemoveInOperand(ext_inst->NumInOperands() - 1);
  context->get_def_use_mgr()->AnalyzeInstUse(ext_inst);

  std::ostringstream errors;
  HwLowerToStandardPass pass;
  pass.SetMessageConsumer(
      [&errors](spv_message_level_t, const char*, const spv_position_t&,
                const char* message) { errors << message << '\n'; });
  EXPECT_EQ(Pass::Status::Failure, pass.Run(context.get()));
  EXPECT_NE(std::string::npos,
            errors.str().find(
                "invalid HW cooperative vector GLSL.std.450 operand count"))
      << errors.str();
}

void ExpectSingleElementwiseLoop(const std::string& text) {
  EXPECT_EQ(1u, CountSubstring(text, "OpLoopMerge")) << text;
  EXPECT_EQ(1u, CountSubstring(text, "OpULessThan")) << text;
  EXPECT_EQ(1u, CountSubstring(text, "OpBranchConditional")) << text;
  EXPECT_GE(CountSubstring(text, "OpAccessChain"), 2u) << text;
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

std::string RunWithInjectedFullMemoryAccess(HwLowerToStandardTest* test,
                                            const std::string& text,
                                            spv::Op opcode, bool is_load) {
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_3, test->consumer(), text,
                             SpirvTools::kDefaultAssembleOption);
  EXPECT_NE(nullptr, context.get()) << "Assembling failed for shader:\n"
                                    << text << std::endl;
  if (!context) return std::string();

  uint32_t device_scope_id = 0;
  std::vector<uint32_t> alias_list_ids;
  context->module()->ForEachInst([&device_scope_id,
                                  &alias_list_ids](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpConstant && inst->NumInOperands() == 1 &&
        inst->GetSingleWordInOperand(0) == 1) {
      device_scope_id = inst->result_id();
    } else if (inst->opcode() == spv::Op::OpAliasScopeListDeclINTEL) {
      alias_list_ids.push_back(inst->result_id());
    }
  });
  EXPECT_NE(0u, device_scope_id);
  EXPECT_GE(alias_list_ids.size(), 2u);
  if (device_scope_id == 0 || alias_list_ids.size() < 2) return std::string();

  bool injected = false;
  context->module()->ForEachInst([opcode, is_load, device_scope_id,
                                  &alias_list_ids,
                                  &injected](Instruction* inst) {
    if (injected || inst->opcode() != opcode) return;
    uint32_t mask = uint32_t(spv::MemoryAccessMask::Volatile) |
                    uint32_t(spv::MemoryAccessMask::Aligned) |
                    uint32_t(spv::MemoryAccessMask::Nontemporal) |
                    uint32_t(spv::MemoryAccessMask::NonPrivatePointer) |
                    uint32_t(spv::MemoryAccessMask::AliasScopeINTELMask) |
                    uint32_t(spv::MemoryAccessMask::NoAliasINTELMask);
    mask |= is_load ? uint32_t(spv::MemoryAccessMask::MakePointerVisible)
                    : uint32_t(spv::MemoryAccessMask::MakePointerAvailable);
    inst->AddOperand({SPV_OPERAND_TYPE_MEMORY_ACCESS, {mask}});
    inst->AddOperand({SPV_OPERAND_TYPE_LITERAL_INTEGER, {16}});
    inst->AddOperand({SPV_OPERAND_TYPE_SCOPE_ID, {device_scope_id}});
    inst->AddOperand({SPV_OPERAND_TYPE_ID, {alias_list_ids[0]}});
    inst->AddOperand({SPV_OPERAND_TYPE_ID, {alias_list_ids[1]}});
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
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::string disassembly;
  EXPECT_TRUE(tools.Disassemble(binary, &disassembly,
                                SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));
  return disassembly;
}

std::string RunWithInjectedAliasMemoryAccess(HwLowerToStandardTest* test,
                                             const std::string& text) {
  auto context = BuildModule(SPV_ENV_UNIVERSAL_1_3, test->consumer(), text,
                             SpirvTools::kDefaultAssembleOption);
  EXPECT_NE(nullptr, context.get()) << "Assembling failed for shader:\n"
                                    << text << std::endl;
  if (!context) return std::string();

  std::vector<uint32_t> alias_list_ids;
  context->module()->ForEachInst([&alias_list_ids](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpAliasScopeListDeclINTEL) {
      alias_list_ids.push_back(inst->result_id());
    }
  });
  EXPECT_GE(alias_list_ids.size(), 2u);
  if (alias_list_ids.size() < 2) return std::string();

  uint32_t injected = 0;
  context->module()->ForEachInst([&alias_list_ids,
                                  &injected](Instruction* inst) {
    if (inst->opcode() != spv::Op::OpCooperativeVectorLoadHW &&
        inst->opcode() != spv::Op::OpCooperativeVectorStoreHW) {
      return;
    }
    const uint32_t mask = uint32_t(spv::MemoryAccessMask::AliasScopeINTELMask) |
                          uint32_t(spv::MemoryAccessMask::NoAliasINTELMask);
    inst->AddOperand({SPV_OPERAND_TYPE_MEMORY_ACCESS, {mask}});
    inst->AddOperand({SPV_OPERAND_TYPE_ID, {alias_list_ids[0]}});
    inst->AddOperand({SPV_OPERAND_TYPE_ID, {alias_list_ids[1]}});
    ++injected;
  });
  EXPECT_EQ(2u, injected);
  if (injected != 2) return std::string();

  HwLowerToStandardPass pass;
  pass.SetMessageConsumer(test->consumer());
  const Pass::Status status = pass.Run(context.get());
  EXPECT_EQ(Pass::Status::SuccessWithChange, status);
  if (status == Pass::Status::Failure) return std::string();

  std::vector<uint32_t> binary;
  context->module()->ToBinary(&binary, /* skip_nop = */ true);
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::string disassembly;
  EXPECT_TRUE(tools.Disassemble(binary, &disassembly,
                                SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));
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

TEST_F(HwLowerToStandardTest, PreservesSharedExtensionForBarrierOps) {
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
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%vec4 = OpTypeCooperativeVectorHW %uint %uint_4
%value = OpUndef %vec4
%main = OpFunction %void None %fn
%entry = OpLabel
OpBarrierArriveHW %int_0 %int_1
OpBarrierWaitHW %int_0 %int_1
OpReturn
OpFunctionEnd
)";

  auto result =
      SinglePassRunAndDisassemble<HwLowerToStandardPass>(text, true, true);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "CooperativeVectorHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "SPV_HW_neural_shader"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierArriveHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierWaitHW"));
}

TEST_F(HwLowerToStandardTest,
       CooperativeOnlyModeLowersValueAndPreservesNonCooperativeHwOpcode) {
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
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%int_1 = OpConstant %int 1
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%a = OpUndef %vec4
%b = OpUndef %vec4
%main = OpFunction %void None %fn
%entry = OpLabel
%sum = OpFAdd %vec4 %a %b
OpBarrierArriveHW %int_0 %int_1
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  EXPECT_EQ(0u, CountSubstring(disassembly, "CooperativeVectorHW"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "SPV_HW_neural_shader"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpBarrierArriveHW"));
  EXPECT_GT(CountSubstring(disassembly, "OpFAdd %v4float"), 0u);
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFConvert %v4float"));
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v4float"));
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFMul %v4float"));
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 4"));
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 4"));
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 4"));
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 4"));
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFConvert %v4float"));
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 4"));
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v4float"));
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %float 4"));
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
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %int 4"));
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
  ExpectSingleElementwiseLoop(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %float"));
  EXPECT_EQ(std::string::npos, disassembly.find("OpTypeVector %float 4"));
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpConvertSToF %v4float"));
  EXPECT_NE(std::string::npos, disassembly.find("OpTypeVector %int 4"));
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpExtInst %v4float"));
}

TEST_F(HwLowerToStandardTest, FailsUnsupportedCooperativeVectorExtInst) {
  const std::string text = R"(
; CHECK: unsupported HW cooperative vector GLSL.std.450 opcode
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
%y = OpExtInst %vec8 %glsl Sin %zero
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, FailsMismatchedCooperativeVectorExtInstLength) {
  const std::string text = R"(
; CHECK: invalid HW cooperative vector OpExtInst operand
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
%uint_4 = OpConstant %uint 4
%uint_8 = OpConstant %uint 8
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%vec8 = OpTypeCooperativeVectorHW %float %uint_8
%short = OpConstantNull %vec4
%wide = OpConstantNull %vec8
%main = OpFunction %void None %fn
%entry = OpLabel
%y = OpExtInst %vec8 %glsl FMax %wide %short
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsMatrixExtInstResult) {
  const std::string text = R"(
; CHECK: HW OpExtInst requires a cooperative vector result
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%float = OpTypeFloat 32
%mat2x2 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%zero = OpConstantNull %mat2x2
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpExtInst %mat2x2 %glsl FMax %zero %zero
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest,
       RejectsCooperativeVectorExtInstComponentTypeMismatch) {
  const std::string text = R"(
; CHECK: invalid HW cooperative vector OpExtInst operand
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
%fvec8 = OpTypeCooperativeVectorHW %float %uint_8
%uvec8 = OpTypeCooperativeVectorHW %uint %uint_8
%fzero = OpConstantNull %fvec8
%uzero = OpConstantNull %uvec8
%main = OpFunction %void None %fn
%entry = OpLabel
%result = OpExtInst %fvec8 %glsl FMax %fzero %uzero
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest,
       RejectsCooperativeVectorExtInstInvalidArity) {
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
%result = OpExtInst %vec8 %glsl FMax %zero %zero
OpReturn
OpFunctionEnd
)";

  ExpectFailureWithMissingExtInstOperand(this, text);
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v4float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFNegate %v4float"));
}

TEST_F(HwLowerToStandardTest,
       RejectsConversionUnlessInputAndResultAreCooperative) {
  const std::vector<std::string> instructions = {
      "%bad = OpFConvert %fvec4 %ordinary_half",
      "%bad = OpFConvert %v4float %hw_half",
      "%bad = OpFConvert %fmat2 %ordinary_half_matrix",
      "%bad = OpFConvert %m2float %hw_half_matrix",
  };
  for (const std::string& instruction : instructions) {
    SCOPED_TRACE(instruction);
    SinglePassRunAndFail<HwLowerToStandardPass>(
        MakeAsymmetricHwElementwiseModule("unsupported HW conversion",
                                          instruction));
  }
}

TEST_F(HwLowerToStandardTest,
       RejectsArithmeticUnlessInputAndResultAreCooperative) {
  const std::vector<std::string> instructions = {
      "%bad = OpFNegate %fvec4 %ordinary_float",
      "%bad = OpFNegate %v4float %hw_float",
      "%bad = OpFNegate %fmat2 %ordinary_float_matrix",
      "%bad = OpFNegate %m2float %hw_float_matrix",
  };
  for (const std::string& instruction : instructions) {
    SCOPED_TRACE(instruction);
    SinglePassRunAndFail<HwLowerToStandardPass>(
        MakeAsymmetricHwElementwiseModule("unsupported HW arithmetic",
                                          instruction));
  }
}

TEST_F(HwLowerToStandardTest,
       RejectsScaleUnlessInputAndResultAreCooperative) {
  const std::vector<std::string> instructions = {
      "%bad = OpVectorTimesScalar %fvec4 %ordinary_float %one",
      "%bad = OpVectorTimesScalar %v4float %hw_float %one",
      "%bad = OpMatrixTimesScalar %fmat2 %ordinary_float_matrix %one",
      "%bad = OpMatrixTimesScalar %m2float %hw_float_matrix %one",
  };
  for (const std::string& instruction : instructions) {
    SCOPED_TRACE(instruction);
    SinglePassRunAndFail<HwLowerToStandardPass>(
        MakeAsymmetricHwElementwiseModule("unsupported HW scale operation",
                                          instruction));
  }
}

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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v4float"));
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpConstantCompositeReplicateEXT"));
  EXPECT_GT(CountSubstring(disassembly, "OpConstantComposite %v4half"), 0u);
  EXPECT_GT(CountSubstring(disassembly, "OpExtInst %v4half"), 0u);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpTypeFunction %_arr_v4float_uint_16"));
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

TEST_F(HwLowerToStandardTest,
       GeneratedFusedStoreBlocksAliasingBiasLoadMotion) {
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(
                    disassembly,
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(1u, CountSubstring(disassembly,
                               "OpTypeFunction %_arr_v4float_uint_2\n"));
  EXPECT_EQ(2u, CountSubstring(disassembly,
                               "OpFunctionCall %_arr_v4float_uint_2"));
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
  const std::string& disassembly = std::get<0>(result);
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_2"));
  EXPECT_EQ(0u, CountSubstring(disassembly,
                               "OpFunctionParameter %_arr_v4float_uint_16"));
  EXPECT_EQ(2u, CountSubstring(disassembly,
                               "OpFunctionCall %_arr_v4float_uint_2"));
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

TEST_F(HwLowerToStandardTest,
       LowersVectorReplicateInPackedAndScalarLoopPaths) {
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
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u,
      1024ull, 8u, 1024ull);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(unrolled));
  ExpectNoHwOrCoopMatrix(std::get<0>(unrolled));
  EXPECT_EQ(0u, CountSubstring(std::get<0>(unrolled), "OpLoopMerge"));
  EXPECT_EQ(2u,
            CountSubstring(std::get<0>(unrolled),
                           "OpCompositeConstruct %v4float"));

  auto looped = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kForceScalar,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u,
      1024ull, 4u, 1024ull);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(looped));
  ExpectNoHwOrCoopMatrix(std::get<0>(looped));
  EXPECT_EQ(1u, CountSubstring(std::get<0>(looped), "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(std::get<0>(looped), "OpULessThan"));
  EXPECT_EQ(1u,
            CountSubstring(std::get<0>(looped), "OpBranchConditional"));
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
  EXPECT_EQ(1u, CountSubstring(
                    disassembly, "OpCompositeConstruct %_arr_float_uint_8"));
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
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u,
      1024ull, 2u, 1024ull);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge"));
  EXPECT_EQ(1u, CountSubstring(
                    disassembly, "OpCompositeConstruct %_arr_v4float_uint_1"));
}

TEST_F(HwLowerToStandardTest,
       RejectsPlainVectorCompositeConstructScalarBroadcast) {
  const std::string text = R"(
; CHECK: HW vector OpCompositeConstruct operand count is invalid
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%bad = OpCompositeConstruct %vec4 %one
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
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
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 4"));
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
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 4"));
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
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 4"));
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
  EXPECT_EQ(4u, CountSubstring(result, "Aligned 4"));
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

TEST_F(HwLowerToStandardTest, ForceScalarModeLowersF16MatrixLoadStoreToScalar) {
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

  SinglePassRunAndFail<HwLowerToStandardPass>(
      text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1048576u,
      65536ull, 4096u, 4096ull);
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

  SinglePassRunAndFail<HwLowerToStandardPass>(
      text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1048576u,
      65536ull, 4096u, 4096ull);
}

TEST_F(HwLowerToStandardTest,
       LowersConstantCompositeWithUndefConstituents) {
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
  EXPECT_EQ(1u,
            CountSubstring(disassembly, "OpConstantComposite %v4float"));
  EXPECT_EQ(2u, CountSubstring(
                    disassembly,
                    "OpConstantComposite %_arr_v4float_uint_1"));
}

TEST_F(HwLowerToStandardTest,
       RejectsNestedVectorConstantCompositeConstituent) {
  const std::string text = R"(
; CHECK: unsupported HW OpConstantComposite operand
OpCapability Shader
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%one = OpConstant %float 1
%v2float = OpTypeVector %float 2
%pair = OpConstantComposite %v2float %one %one
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%bad = OpConstantComposite %vec4 %pair %pair %pair %pair
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
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
%one = )") + test_case.constituent + R"(
%vec65532 = OpTypeCooperativeVectorHW %float %uint_65532
%value = )" + test_case.opcode + R"( %vec65532 %one
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
    EXPECT_EQ(1u, CountSubstring(
                      disassembly,
                      std::string(test_case.lowered_opcode) +
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
  EXPECT_EQ(0u, CountSubstring(
                    disassembly, "OpSpecConstantCompositeReplicateEXT"));
  EXPECT_EQ(1u,
            CountSubstring(disassembly, "OpSpecConstantComposite %v4float"));
  EXPECT_EQ(1u, CountSubstring(
                    disassembly,
                    "OpSpecConstantComposite %_arr_v4float_uint_2"));
}

TEST_F(HwLowerToStandardTest,
       RejectsReplicatedConstantThatCannotBeSerializedAfterLowering) {
  const std::vector<std::string> opcodes = {
      "OpConstantCompositeReplicateEXT",
      "OpSpecConstantCompositeReplicateEXT",
  };
  for (const std::string& opcode : opcodes) {
    SCOPED_TRACE(opcode);
    const std::string text = std::string(R"(
; CHECK: HW replicated constant exceeds the SPIR-V composite operand limit
OpCapability Shader
OpCapability ReplicatedCompositesEXT
OpCapability CooperativeVectorHW
OpExtension "SPV_EXT_replicated_composites"
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_65533 = OpConstant %uint 65533
%float = OpTypeFloat 32
%one = OpConstant %float 1
%vec65533 = OpTypeCooperativeVectorHW %float %uint_65533
%value = )") + opcode + R"( %vec65533 %one
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

    SinglePassRunAndFail<HwLowerToStandardPass>(
        text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
        HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 70000u,
        16777216ull, 70000u, 4096ull);
  }
}

TEST_F(HwLowerToStandardTest,
       RejectsDirectConstantCompositesAboveConstituentLimit) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"OpConstantComposite", "OpConstant %float 1"},
      {"OpSpecConstantComposite", "OpSpecConstant %float 1065353216"},
  };
  for (const auto& test_case : cases) {
    const std::string& opcode = test_case.first;
    SCOPED_TRACE(opcode);
    const std::string text = std::string(R"(
; CHECK: HW constant composite exceeds the SPIR-V composite operand limit
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_65533 = OpConstant %uint 65533
%float = OpTypeFloat 32
%one = )") + test_case.second + R"(
%mat1x65533 = OpTypeCooperativeMatrixHW %float %uint_1 %uint_65533
%value = )" + opcode + R"( %mat1x65533 %one
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
)";

    SinglePassRunAndFail<HwLowerToStandardPass>(
        text, HwLowerToStandardPass::LoweringMode::kForceScalar,
        HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 70000u,
        16777216ull, 65532u, 4096ull);
  }
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
  EXPECT_EQ(1u,
            CountSubstring(
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
            disassembly.find("OpTypePointer Function %_arr_v4float_uint_2"));
  EXPECT_GE(CountSubstring(disassembly, "OpLoad %_arr_v4float_uint_2"), 2u);
  EXPECT_GE(CountSubstring(disassembly, "OpStore"), 2u);
  EXPECT_GE(CountSubstring(disassembly, "OpLoopMerge"), 2u);
  EXPECT_GT(CountSubstring(disassembly, "OpFAdd %v4float"), 0u);
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpLoad %_arr_v4float_uint_1"));
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
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpLoad %_arr_v4float_uint_1"));
  EXPECT_EQ(1u, CountSubstring(disassembly, " Volatile"));
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

TEST_F(HwLowerToStandardTest, FailsForPointerEscapeOrNonFunctionHwPointer) {
  const std::string text = R"(
; CHECK: HW cooperative values may only be stored in Function or Private variables before lowering
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
; CHECK: HW cooperative values may only be stored in Function or Private variables before lowering
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
; CHECK: HW cooperative values may only be stored in Function or Private variables before lowering
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true);
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
      HwLowerToStandardPass::CompletenessMode::kCooperativeOnly, 1024u,
      1024ull, 4u, 1024ull);
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

TEST_F(HwLowerToStandardTest, ExtensionFreeModeLowersCooperativeModule) {
  const std::string text = R"(
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%uint_1 = OpConstant %uint 1
%uint_5 = OpConstant %uint 5
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_1 %uint_5
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%element = OpCompositeExtract %float %a 0 4
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      text, true, true, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kExtensionFree);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(std::get<0>(result));
}

TEST_F(HwLowerToStandardTest,
       ExtensionFreeModeRejectsEveryUnloweredHwOpcode) {
  struct UnsupportedHwOpcodeCase {
    const char* name;
    const char* declarations;
    const char* body;
  };
  const UnsupportedHwOpcodeCase cases[] = {
      {"OpTypeTensorMapHW", "%tensor_map = OpTypeTensorMapHW 1\n", ""},
      {"OpCpAsyncTensorGlobalSharedHW", "",
       "OpCpAsyncTensorGlobalSharedHW 1 %uint_0 %uint_0 %uint_0\n"},
      {"OpCpAsyncCommitGroupHW", "", "OpCpAsyncCommitGroupHW\n"},
      {"OpCpAsyncWaitGroupHW", "", "OpCpAsyncWaitGroupHW 0\n"},
      {"OpBarrierArriveHW", "",
       "OpBarrierArriveHW %uint_0 %uint_1\n"},
      {"OpBarrierWaitHW", "", "OpBarrierWaitHW %uint_0 %uint_1\n"},
      {"OpShuffleIndexHW", "",
       "%result = OpShuffleIndexHW %uint %uint_1 %uint_0\n"},
      {"OpBytePermuteHW", "",
       "%result = OpBytePermuteHW %uint %uint_1 %uint_1 %uint_0\n"},
      {"OpShuffleFillDownHW", "",
       "%result = OpShuffleFillDownHW %uint %uint_1 %uint_0 %uint_1\n"},
  };

  for (const UnsupportedHwOpcodeCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    const std::string text = std::string(R"(
; CHECK: extension-free HW lowering has no equivalent lowering for this HW opcode
OpCapability Shader
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_1 = OpConstant %uint 1
)") + test_case.declarations + R"(
%main = OpFunction %void None %fn
%entry = OpLabel
)" + test_case.body + R"(
OpReturn
OpFunctionEnd
)";

    SinglePassRunAndFail<HwLowerToStandardPass>(
        text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
        HwLowerToStandardPass::CompletenessMode::kExtensionFree);
  }
}

TEST_F(HwLowerToStandardTest, ExtensionFreeModeRejectsRelregOperand) {
  const std::string text = R"(
; CHECK: extension-free HW lowering has no equivalent lowering for this HW operand
OpCapability Shader
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%main = OpFunction %void None %fn
%entry = OpLabel
OpSelectionMerge %merge Relreg
OpBranchConditional %true %then %merge
%then = OpLabel
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(
      text, HwLowerToStandardPass::LoweringMode::kPreferPackedVec4,
      HwLowerToStandardPass::CompletenessMode::kExtensionFree);
}

TEST_F(HwLowerToStandardTest, RejectsNoContractionMatrixMatmul) {
  const std::string text = R"(
; CHECK: NoContraction HW cooperative matrix multiply cannot be lowered
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %d NoContraction
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%mat4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%a = OpUndef %mat4
%b = OpUndef %mat4
%c = OpUndef %mat4
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%d = OpCooperativeMatrixMulAddHW %mat4 %a %b %c
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsNoContractionVectorMatmul) {
  const std::string text = R"(
; CHECK: NoContraction HW cooperative vector matrix multiply cannot be lowered
OpCapability Shader
OpCapability CooperativeMatrixHW
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %y NoContraction
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%float = OpTypeFloat 32
%vec4 = OpTypeCooperativeVectorHW %float %uint_4
%mat4 = OpTypeCooperativeMatrixHW %float %uint_4 %uint_4
%x = OpUndef %vec4
%w = OpUndef %mat4
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

TEST_F(HwLowerToStandardTest, RejectsSpecConstantCooperativeShape) {
  const std::string text = R"(
; CHECK: HW cooperative matrix specialization-constant shape is not supported
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %rows SpecId 0
OpDecorate %cols SpecId 1
%uint = OpTypeInt 32 0
%rows = OpSpecConstant %uint 3
%cols = OpSpecConstant %uint 5
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %rows %cols
%a = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%element = OpCompositeExtract %float %a 0 0
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsFloat64CooperativeComponentType) {
  const std::string text = R"(
; CHECK: HW cooperative matrix component type must be 16/32-bit floating-point or 8/16/32-bit integer
OpCapability Shader
OpCapability Float64
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%double = OpTypeFloat 64
%mat = OpTypeCooperativeMatrixHW %double %uint_2 %uint_2
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsSignedInt64CooperativeComponentType) {
  const std::string text = R"(
; CHECK: HW cooperative vector component type must be 16/32-bit floating-point or 8/16/32-bit integer
OpCapability Shader
OpCapability Int64
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%long = OpTypeInt 64 1
%vec = OpTypeCooperativeVectorHW %long %uint_2
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsUnsignedInt64CooperativeComponentType) {
  const std::string text = R"(
; CHECK: HW cooperative matrix component type must be 16/32-bit floating-point or 8/16/32-bit integer
OpCapability Shader
OpCapability Int64
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%uint_2 = OpConstant %uint 2
%ulong = OpTypeInt 64 0
%mat = OpTypeCooperativeMatrixHW %ulong %uint_2 %uint_2
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsSpecConstantMatrixLayout) {
  const std::string text = R"(
; CHECK: HW cooperative matrix memory access cannot be lowered
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %layout SpecId 0
OpDecorate %data ArrayStride 4
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%uint_4 = OpConstant %uint 4
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%layout = OpSpecConstant %int 0
%v2int = OpTypeVector %int 2
%shape = OpConstantComposite %v2int %int_0 %int_0
%offset = OpConstantComposite %v2int %int_0 %int_0
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%data = OpTypeRuntimeArray %float
%_ptr_StorageBuffer_data = OpTypePointer StorageBuffer %data
%buffer = OpVariable %_ptr_StorageBuffer_data StorageBuffer
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%value = OpCooperativeMatrixLoadHW %mat %buffer %shape %offset %layout
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

TEST_F(HwLowerToStandardTest, RejectsSpecConstantMatrixReduceControl) {
  const std::string text = R"(
; CHECK: invalid OpCooperativeMatrixReduceHW
OpCapability Shader
OpCapability CooperativeMatrixHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %axis SpecId 0
%uint = OpTypeInt 32 0
%axis = OpSpecConstant %uint 0
%add = OpConstant %uint 0
%uint_2 = OpConstant %uint 2
%float = OpTypeFloat 32
%mat = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%input = OpUndef %mat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%value = OpCooperativeMatrixReduceHW %mat %input %axis %add
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
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
  EXPECT_EQ(7u, CountSubstring(disassembly, "OpFConvert %v4"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFAdd %v4float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFSub %v4half"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFMul %v4float"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFDiv %v4half"));
  EXPECT_EQ(1u, CountSubstring(disassembly, "OpFNegate %v4float"));
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

TEST_F(HwLowerToStandardTest, RejectsMixedWidthUnsignedDivision) {
  const std::string text = R"(
; CHECK: OpUDiv requires matching HW integer component bit widths
OpCapability Shader
OpCapability Int16
OpCapability CooperativeVectorHW
OpExtension "SPV_HW_neural_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn_ty = OpTypeFunction %void
%ushort = OpTypeInt 16 0
%uint = OpTypeInt 32 0
%uint_5 = OpConstant %uint 5
%u16vec = OpTypeCooperativeVectorHW %ushort %uint_5
%u32vec = OpTypeCooperativeVectorHW %uint %uint_5
%u16 = OpUndef %u16vec
%u32 = OpUndef %u32vec
%main = OpFunction %void None %fn_ty
%entry = OpLabel
%div = OpUDiv %u32vec %u16 %u32
OpReturn
OpFunctionEnd
)";

  // The current validator deliberately accepts this HW form even though a
  // standard OpUDiv requires matching operand/result types.  Refuse to guess
  // whether narrowing occurs before or after division.
  SpirvTools tools(SPV_ENV_UNIVERSAL_1_3);
  std::vector<uint32_t> original_binary;
  ASSERT_TRUE(tools.Assemble(text, &original_binary));
  ASSERT_TRUE(tools.Validate(original_binary));

  SinglePassRunAndFail<HwLowerToStandardPass>(text);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
