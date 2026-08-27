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

// Tests for the non-direct packed vec2 lowering path: scenarios where the
// direct dispatch (TryLowerDirect*) cannot handle a matmul and the generic
// packed vec2 lowering takes over.

#include <string>

#include "source/opt/hw_lower_to_standard_pass.h"
#include "test/opt/pass_fixture.h"

namespace spvtools {
namespace opt {
namespace {

using HwLowerNonDirectPackedTest = PassTest<::testing::Test>;

size_t CountSubstring(const std::string& text, const std::string& needle) {
  size_t count = 0;
  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++count;
    pos += needle.size();
  }
  return count;
}

void ExpectNoHwOrCoopMatrix(const std::string& disassembly) {
  EXPECT_EQ(0u, CountSubstring(disassembly, "CooperativeMatrixHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "CooperativeVectorHW"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeMatrix"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpCooperativeVector"));
  EXPECT_EQ(0u, CountSubstring(disassembly, "SPV_HW_neural_shader"));
}

// ──────────────────────────────────────────────────────────────────
// Condition 4: CanUseDirectVectorMatrixMul returns false.
//
// The direct path requires same-component (f16/f16/f16 or f32/f32/f32)
// or f16×f16→f32 mixed precision.  An f16 input × f32 matrix → f32
// result combination is valid SPIR-V for the HW lowering pass (both
// component types are float-domain and the accumulator is not narrower)
// but CanUseDirectVectorMatrixMul rejects it because input and matrix
// have different component types.  The matmul falls through to the
// generic packed vec2 path.
// ──────────────────────────────────────────────────────────────────

// Helper: build SPIR-V assembly for a vector-matrix multiply with
// mixed component types.  |input_k| is the input length (= matrix rows),
// |output_n| is the result length (= matrix columns).
// MAC count = input_k * output_n.
// When MAC ≤ 4096 the generic path produces an unrolled packed vec2
// helper; when MAC > 4096 it produces a rolled structured-loop helper.
std::string MakeMixedPrecisionVecMatMul(uint32_t input_k, uint32_t output_n) {
  const std::string k_id = "%uint_" + std::to_string(input_k);
  const std::string n_id = "%uint_" + std::to_string(output_n);

  return "OpCapability Shader\n"
         "OpCapability Float16\n"
         "OpCapability CooperativeVectorHW\n"
         "OpCapability CooperativeMatrixHW\n"
         "OpExtension \"SPV_HW_neural_shader\"\n"
         "OpMemoryModel Logical GLSL450\n"
         "OpEntryPoint GLCompute %main \"main\"\n"
         "OpExecutionMode %main LocalSize 1 1 1\n"
         "%uint = OpTypeInt 32 0\n" +
         k_id + " = OpConstant %uint " + std::to_string(input_k) + "\n" + n_id +
         " = OpConstant %uint " + std::to_string(output_n) +
         "\n"
         "%half = OpTypeFloat 16\n"
         "%float = OpTypeFloat 32\n"
         // Input vector: f16
         "%vec_input = OpTypeCooperativeVectorHW %half " +
         k_id +
         "\n"
         // Matrix: f32  (rows=K, cols=N)
         "%mat = OpTypeCooperativeMatrixHW %float " +
         k_id + " " + n_id +
         "\n"
         // Result vector: f32
         "%vec_result = OpTypeCooperativeVectorHW %float " +
         n_id +
         "\n"
         "%input = OpUndef %vec_input\n"
         "%matrix = OpUndef %mat\n"
         "%void = OpTypeVoid\n"
         "%fn = OpTypeFunction %void\n"
         "%main = OpFunction %void None %fn\n"
         "%entry = OpLabel\n"
         "%result = OpCooperativeVectorMatrixMulHW %vec_result %input "
         "%matrix\n"
         "OpReturn\n"
         "OpFunctionEnd\n";
}

// Helper: build SPIR-V assembly for a matrix multiply-add with mixed
// component types (f16 A × f32 B → f32 result, with f32 C).
// MAC count = M * K * N.
std::string MakeMixedPrecisionMatMulAdd(uint32_t m, uint32_t k, uint32_t n) {
  return "OpCapability Shader\n"
         "OpCapability Float16\n"
         "OpCapability CooperativeVectorHW\n"
         "OpCapability CooperativeMatrixHW\n"
         "OpExtension \"SPV_HW_neural_shader\"\n"
         "OpMemoryModel Logical GLSL450\n"
         "OpEntryPoint GLCompute %main \"main\"\n"
         "OpExecutionMode %main LocalSize 1 1 1\n"
         "%uint = OpTypeInt 32 0\n"
         "%m_const = OpConstant %uint " +
         std::to_string(m) +
         "\n"
         "%k_const = OpConstant %uint " +
         std::to_string(k) +
         "\n"
         "%n_const = OpConstant %uint " +
         std::to_string(n) +
         "\n"
         "%half = OpTypeFloat 16\n"
         "%float = OpTypeFloat 32\n"
         // A: f16 (M×K)
         "%mat_a = OpTypeCooperativeMatrixHW %half %m_const %k_const\n"
         // B: f32 (K×N)
         "%mat_b = OpTypeCooperativeMatrixHW %float %k_const %n_const\n"
         // C and result: f32 (M×N)
         "%mat_c = OpTypeCooperativeMatrixHW %float %m_const %n_const\n"
         "%a = OpUndef %mat_a\n"
         "%b = OpUndef %mat_b\n"
         "%c = OpUndef %mat_c\n"
         "%void = OpTypeVoid\n"
         "%fn = OpTypeFunction %void\n"
         "%main = OpFunction %void None %fn\n"
         "%entry = OpLabel\n"
         "%result = OpCooperativeMatrixMulAddHW %mat_c %a %b %c\n"
         "OpReturn\n"
         "OpFunctionEnd\n";
}

// ── Unrolled cases (MAC ≤ 4096) ──

TEST_F(HwLowerNonDirectPackedTest,
       Condition4_DirectUnrolledVecMatMul_F16xF32toF32_4x8) {
  // K=4, N=8 → MAC=32 ≤ 256 → direct unrolled packed vec2.
  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      MakeMixedPrecisionVecMatMul(4, 8), true, true,
      HwLowerToStandardPass::LoweringMode::kPreferPackedVec2);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  // Packed vec2 types present (f32 result uses v2float, f16 input uses v2half).
  EXPECT_GT(CountSubstring(disassembly, "OpTypeVector %float 2"), 0u)
      << disassembly;
  EXPECT_GT(CountSubstring(disassembly, "OpTypeVector %half 2"), 0u)
      << disassembly;
  // Direct packed vec2 path produces Fma with f16→f32 conversion inline.
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u) << disassembly;
  EXPECT_GT(CountSubstring(disassembly, "OpFConvert %v2float"), 0u)
      << disassembly;
  // No structured loops (unrolled).
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge")) << disassembly;
}

TEST_F(HwLowerNonDirectPackedTest,
       Condition4_UnrolledMatMulAdd_F16xF32toF32_4x4x8) {
  // M=4, K=4, N=8 → MAC=128 ≤ 4096 → generic unrolled packed vec2.
  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      MakeMixedPrecisionMatMulAdd(4, 4, 8), true, true,
      HwLowerToStandardPass::LoweringMode::kPreferPackedVec2);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeVector %float 2"), 0u)
      << disassembly;
  // Generic packed vec2 Fma pattern present.
  EXPECT_GT(CountSubstring(disassembly, " Fma "), 0u) << disassembly;
  EXPECT_EQ(0u, CountSubstring(disassembly, "OpLoopMerge")) << disassembly;
}

// ── Rolled cases (MAC > 4096) ──

TEST_F(HwLowerNonDirectPackedTest,
       Condition4_RolledVecMatMul_F16xF32toF32_128x64) {
  // K=128, N=64 → MAC=8192 > 4096 → generic rolled packed vec2 (loop).
  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      MakeMixedPrecisionVecMatMul(128, 64), true, true,
      HwLowerToStandardPass::LoweringMode::kPreferPackedVec2);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  // Packed vec2 types present.
  EXPECT_GT(CountSubstring(disassembly, "OpTypeVector %float 2"), 0u)
      << disassembly;
  // Structured loops present (rolled path).
  EXPECT_GT(CountSubstring(disassembly, "OpLoopMerge"), 0u) << disassembly;
}

TEST_F(HwLowerNonDirectPackedTest,
       Condition4_RolledMatMulAdd_F16xF32toF32_32x32x32) {
  // M=32, K=32, N=32 → MAC=32768 > 4096 → generic rolled packed vec2 (loop).
  auto result = SinglePassRunAndDisassemble<HwLowerToStandardPass>(
      MakeMixedPrecisionMatMulAdd(32, 32, 32), true, true,
      HwLowerToStandardPass::LoweringMode::kPreferPackedVec2);
  const std::string& disassembly = std::get<0>(result);
  EXPECT_EQ(Pass::Status::SuccessWithChange, std::get<1>(result));
  ExpectNoHwOrCoopMatrix(disassembly);
  EXPECT_GT(CountSubstring(disassembly, "OpTypeVector %float 2"), 0u)
      << disassembly;
  EXPECT_GT(CountSubstring(disassembly, "OpLoopMerge"), 0u) << disassembly;
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
