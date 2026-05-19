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

using AzdFixCooperativeMatrixUseTest = PassTest<::testing::Test>;

TEST_F(AzdFixCooperativeMatrixUseTest, FixesMulAddRoles) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_8 MatrixUseAAZD
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_8 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK-DAG: [[a:%\w+]] = OpUndef [[type_a]]
; CHECK-DAG: [[b:%\w+]] = OpUndef [[type_b]]
; CHECK-DAG: [[c:%\w+]] = OpUndef [[type_acc]]
; CHECK: [[d:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[a]] [[b]] [[c]]
               OpCapability Shader
               OpCapability CooperativeMatrixAZD
               OpExtension "SPV_AZD_neural_matrix"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %uint = OpTypeInt 32 0
    %uint_8 = OpConstant %uint 8
   %uint_16 = OpConstant %uint 16
      %float = OpTypeFloat 32
 %old_type_a = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_8
 %old_type_b = OpTypeCooperativeMatrixAZD %float %uint_8 %uint_16
%old_type_acc = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
          %a = OpUndef %old_type_a
          %b = OpUndef %old_type_b
          %c = OpUndef %old_type_acc
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
          %d = OpCooperativeMatrixMulAddAZD %old_type_acc %a %b %c
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

TEST_F(AzdFixCooperativeMatrixUseTest, ResolvesABConflictByUseCount) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseAAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK-DAG: [[x:%\w+]] = OpUndef [[type_a]]
; CHECK-DAG: [[acc:%\w+]] = OpUndef [[type_acc]]
; CHECK: [[d0:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x]] [[acc]] [[acc]]
; CHECK: [[d1:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x]] [[acc]] [[acc]]
; CHECK: [[d2:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[acc]] [[x]] [[acc]]
               OpCapability Shader
               OpCapability CooperativeMatrixAZD
               OpExtension "SPV_AZD_neural_matrix"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %uint = OpTypeInt 32 0
   %uint_16 = OpConstant %uint 16
      %float = OpTypeFloat 32
   %old_type = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
          %x = OpUndef %old_type
        %acc = OpUndef %old_type
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %x %acc %acc
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %x %acc %acc
         %d2 = OpCooperativeMatrixMulAddAZD %old_type %acc %x %acc
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

TEST_F(AzdFixCooperativeMatrixUseTest, AccumulatorUseHasPriority) {
  const std::string text = R"(
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK-DAG: [[x:%\w+]] = OpUndef [[type_acc]]
; CHECK-DAG: [[b:%\w+]] = OpUndef [[type_b]]
; CHECK: [[d:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x]] [[b]] [[x]]
               OpCapability Shader
               OpCapability CooperativeMatrixAZD
               OpExtension "SPV_AZD_neural_matrix"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %uint = OpTypeInt 32 0
   %uint_16 = OpConstant %uint 16
      %float = OpTypeFloat 32
   %old_type = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
          %x = OpUndef %old_type
          %b = OpUndef %old_type
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
          %d = OpCooperativeMatrixMulAddAZD %old_type %x %b %x
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

TEST_F(AzdFixCooperativeMatrixUseTest, RewritesLoadStorePointerTypes) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_8 MatrixUseAAZD
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_8 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK-DAG: [[ptr_a:%\w+]] = OpTypePointer Function [[type_a]]
; CHECK-DAG: [[ptr_b:%\w+]] = OpTypePointer Function [[type_b]]
; CHECK-DAG: [[ptr_acc:%\w+]] = OpTypePointer Function [[type_acc]]
; CHECK-DAG: [[pa:%\w+]] = OpVariable [[ptr_a]] Function
; CHECK-DAG: [[pb:%\w+]] = OpVariable [[ptr_b]] Function
; CHECK-DAG: [[pc:%\w+]] = OpVariable [[ptr_acc]] Function
; CHECK-DAG: [[a:%\w+]] = OpLoad [[type_a]] [[pa]]
; CHECK-DAG: [[b:%\w+]] = OpLoad [[type_b]] [[pb]]
; CHECK-DAG: [[c:%\w+]] = OpLoad [[type_acc]] [[pc]]
; CHECK: [[d:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[a]] [[b]] [[c]]
; CHECK: OpStore [[pc]] [[d]]
               OpCapability Shader
               OpCapability CooperativeMatrixAZD
               OpExtension "SPV_AZD_neural_matrix"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %uint = OpTypeInt 32 0
    %uint_8 = OpConstant %uint 8
   %uint_16 = OpConstant %uint 16
      %float = OpTypeFloat 32
 %old_type_a = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_8
 %old_type_b = OpTypeCooperativeMatrixAZD %float %uint_8 %uint_16
%old_type_acc = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16
  %ptr_old_a = OpTypePointer Function %old_type_a
  %ptr_old_b = OpTypePointer Function %old_type_b
%ptr_old_acc = OpTypePointer Function %old_type_acc
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %pa = OpVariable %ptr_old_a Function
         %pb = OpVariable %ptr_old_b Function
         %pc = OpVariable %ptr_old_acc Function
          %a = OpLoad %old_type_a %pa
          %b = OpLoad %old_type_b %pb
          %c = OpLoad %old_type_acc %pc
          %d = OpCooperativeMatrixMulAddAZD %old_type_acc %a %b %c
               OpStore %pc %d
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

TEST_F(AzdFixCooperativeMatrixUseTest, RewritesNullAccumulatorForMatMul) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseAAZD
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK-DAG: [[a:%\w+]] = OpUndef [[type_a]]
; CHECK-DAG: [[b:%\w+]] = OpUndef [[type_b]]
; CHECK-DAG: [[zero:%\w+]] = OpConstantNull [[type_acc]]
; CHECK: [[d:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[a]] [[b]] [[zero]]
               OpCapability Shader
               OpCapability CooperativeMatrixAZD
               OpExtension "SPV_AZD_neural_matrix"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %uint = OpTypeInt 32 0
   %uint_16 = OpConstant %uint 16
      %float = OpTypeFloat 32
   %old_type = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
          %a = OpUndef %old_type
          %b = OpUndef %old_type
       %zero = OpConstantNull %old_type
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
          %d = OpCooperativeMatrixMulAddAZD %old_type %a %b %zero
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
