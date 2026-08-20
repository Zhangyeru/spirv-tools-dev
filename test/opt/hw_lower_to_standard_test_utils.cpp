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

#include <sstream>
#include <utility>
#include <vector>

namespace spvtools {
namespace opt {

size_t CountSubstring(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

std::string MakeAsymmetricHwElementwiseModule(const std::string& diagnostic,
                                              const std::string& instruction) {
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
%ordinary_v4half = OpTypeVector %half 4
%ordinary_v4float = OpTypeVector %float 4
%m2half = OpTypeMatrix %v2half 2
%m2float = OpTypeMatrix %v2float 2
%hvec4 = OpTypeCooperativeVectorHW %half %uint_4
%fvec4 = OpTypeCooperativeVectorHW %float %uint_4
%hmat2 = OpTypeCooperativeMatrixHW %half %uint_2 %uint_2
%fmat2 = OpTypeCooperativeMatrixHW %float %uint_2 %uint_2
%ordinary_half = OpUndef %ordinary_v4half
%ordinary_float = OpUndef %ordinary_v4float
%ordinary_half_matrix = OpUndef %m2half
%ordinary_float_matrix = OpUndef %m2float
%hw_half = OpUndef %hvec4
%hw_float = OpUndef %fvec4
%hw_half_matrix = OpUndef %hmat2
%hw_float_matrix = OpUndef %fmat2
%main = OpFunction %void None %fn
%entry = OpLabel
)" + instruction +
         R"(
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

void ExpectPackedVec2Math(const std::string& text,
                          const std::string& component_name) {
  ExpectNoHwOrCoopMatrix(text);
  EXPECT_NE(std::string::npos,
            text.find("OpTypeVector " + component_name + " 2"));
  EXPECT_GT(CountSubstring(text, "OpTypeArray %v2"), 0u);
  EXPECT_GT(CountSubstring(text, "OpExtInst %v2"), 0u);
  EXPECT_GT(CountSubstring(text, " Fma "), 0u);
  EXPECT_EQ(0u, CountSubstring(text, "OpVectorTimesScalar"));
}

void ExpectPackedVec2MatmulPattern(const std::string& text,
                                   const std::string& component_name) {
  ExpectNoHwOrCoopMatrix(text);
  EXPECT_NE(std::string::npos,
            text.find("OpTypeVector " + component_name + " 2"));
  EXPECT_GT(CountSubstring(text, "OpTypeArray %v2"), 0u);
  EXPECT_GT(CountSubstring(text, "OpExtInst %v2"), 0u);
  EXPECT_GT(CountSubstring(text, " Fma "), 0u);
  EXPECT_GT(CountSubstring(text, "OpCompositeExtract " + component_name), 0u);
  EXPECT_GT(CountSubstring(text, "OpFunctionCall"), 0u);
  EXPECT_GT(CountSubstring(text, "OpReturnValue"), 0u);
  EXPECT_EQ(0u, CountSubstring(text, "OpVectorTimesScalar"));
}

void ExpectScalarFallbackMath(const std::string& text,
                              const std::string& component_name) {
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

}  // namespace opt
}  // namespace spvtools
