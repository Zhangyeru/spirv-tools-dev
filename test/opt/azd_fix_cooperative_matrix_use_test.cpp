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
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK-DAG: [[x:%\w+]] = OpUndef [[type_a]]
; CHECK-DAG: {{%\w+}} = OpUndef [[type_a]]
; CHECK-DAG: {{%\w+}} = OpUndef [[type_b]]
; CHECK-DAG: {{%\w+}} = OpUndef [[type_b]]
; CHECK-DAG: [[c:%\w+]] = OpUndef [[type_acc]]
; CHECK: [[d0:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x]] {{%\w+}} [[c]]
; CHECK: [[d1:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x]] {{%\w+}} [[c]]
; CHECK: [[d2:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] {{%\w+}} [[x]] [[c]]
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
          %a = OpUndef %old_type
         %b0 = OpUndef %old_type
         %b1 = OpUndef %old_type
          %c = OpUndef %old_type
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %x %b0 %c
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %x %b1 %c
         %d2 = OpCooperativeMatrixMulAddAZD %old_type %a %x %c
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

TEST_F(AzdFixCooperativeMatrixUseTest, FailsForOperandAccumulatorPointerUse) {
  const std::string text = R"(
; CHECK: AZD cooperative matrix id {{[0-9]+}} has both OperandAB and Accumulator uses
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
    %ptr_old = OpTypePointer Function %old_type
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %px = OpVariable %ptr_old Function
         %pa = OpVariable %ptr_old Function
         %pb = OpVariable %ptr_old Function
         %pc = OpVariable %ptr_old Function
         %x0 = OpLoad %old_type %px
         %b0 = OpLoad %old_type %pb
         %c0 = OpLoad %old_type %pc
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %x0 %b0 %c0
         %a1 = OpLoad %old_type %pa
         %b1 = OpLoad %old_type %pb
         %x1 = OpLoad %old_type %px
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %a1 %b1 %x1
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndFail<AzdFixCooperativeMatrixUsePass>(text);
}

TEST_F(AzdFixCooperativeMatrixUseTest, FailsForSameIdOperandAccumulatorUse) {
  const std::string text = R"(
; CHECK: AZD cooperative matrix id {{[0-9]+}} has both OperandAB and Accumulator uses
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
          %a = OpUndef %old_type
          %b = OpUndef %old_type
          %c = OpUndef %old_type
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %x %b %c
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %a %b %x
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndFail<AzdFixCooperativeMatrixUsePass>(text);
}

TEST_F(AzdFixCooperativeMatrixUseTest,
       InsertsBitcastForAccumulatorToOperandStoreBoundary) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseAAZD
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK-DAG: [[ptr_a:%\w+]] = OpTypePointer Function [[type_a]]
; CHECK-DAG: [[ptr_b:%\w+]] = OpTypePointer Function [[type_b]]
; CHECK-DAG: [[ptr_acc:%\w+]] = OpTypePointer Function [[type_acc]]
; CHECK: [[d0:%\w+]] = OpCooperativeMatrixMulAddAZD [[type_acc]] {{%\w+}} {{%\w+}} {{%\w+}}
; CHECK: [[converted:%\w+]] = OpBitcast [[type_a]] [[d0]]
; CHECK-NEXT: OpStore [[px:%\w+]] [[converted]]
; CHECK: [[x:%\w+]] = OpLoad [[type_a]] [[px]]
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x]] {{%\w+}} {{%\w+}}
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
    %ptr_old = OpTypePointer Function %old_type
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %pa = OpVariable %ptr_old Function
         %pb = OpVariable %ptr_old Function
         %pc = OpVariable %ptr_old Function
         %px = OpVariable %ptr_old Function
         %a0 = OpLoad %old_type %pa
         %b0 = OpLoad %old_type %pb
         %c0 = OpLoad %old_type %pc
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %a0 %b0 %c0
               OpStore %px %d0
          %x = OpLoad %old_type %px
         %b1 = OpLoad %old_type %pb
         %c1 = OpLoad %old_type %pc
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %x %b1 %c1
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

TEST_F(AzdFixCooperativeMatrixUseTest, InsertsBitcastForStoreBoundary) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseAAZD
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK-DAG: [[ptr_a:%\w+]] = OpTypePointer Function [[type_a]]
; CHECK-DAG: [[ptr_b:%\w+]] = OpTypePointer Function [[type_b]]
; CHECK-DAG: [[ptr_acc:%\w+]] = OpTypePointer Function [[type_acc]]
; CHECK-DAG: [[px:%\w+]] = OpVariable [[ptr_a]] Function
; CHECK-DAG: [[py:%\w+]] = OpVariable [[ptr_acc]] Function
; CHECK: [[x0:%\w+]] = OpLoad [[type_a]] [[px]]
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x0]] {{%\w+}} {{%\w+}}
; CHECK: [[xcopy:%\w+]] = OpLoad [[type_a]] [[px]]
; CHECK: [[converted:%\w+]] = OpBitcast [[type_acc]] [[xcopy]]
; CHECK: OpStore [[py]] [[converted]]
; CHECK: [[y:%\w+]] = OpLoad [[type_acc]] [[py]]
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] {{%\w+}} {{%\w+}} [[y]]
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
    %ptr_old = OpTypePointer Function %old_type
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %px = OpVariable %ptr_old Function
         %py = OpVariable %ptr_old Function
         %pa = OpVariable %ptr_old Function
         %pb = OpVariable %ptr_old Function
         %pc = OpVariable %ptr_old Function
         %x0 = OpLoad %old_type %px
         %b0 = OpLoad %old_type %pb
         %c0 = OpLoad %old_type %pc
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %x0 %b0 %c0
      %xcopy = OpLoad %old_type %px
               OpStore %py %xcopy
         %a1 = OpLoad %old_type %pa
         %b1 = OpLoad %old_type %pb
          %y = OpLoad %old_type %py
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %a1 %b1 %y
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

TEST_F(AzdFixCooperativeMatrixUseTest, TurnsCopyObjectBoundaryIntoBitcast) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseAAZD
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK: [[x:%\w+]] = OpLoad [[type_a]] {{%\w+}}
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x]] {{%\w+}} {{%\w+}}
; CHECK: [[copy:%\w+]] = OpBitcast [[type_acc]] [[x]]
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] {{%\w+}} {{%\w+}} [[copy]]
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
    %ptr_old = OpTypePointer Function %old_type
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %px = OpVariable %ptr_old Function
         %pa = OpVariable %ptr_old Function
         %pb = OpVariable %ptr_old Function
         %pc = OpVariable %ptr_old Function
          %x = OpLoad %old_type %px
         %b0 = OpLoad %old_type %pb
         %c0 = OpLoad %old_type %pc
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %x %b0 %c0
       %copy = OpCopyObject %old_type %x
         %a1 = OpLoad %old_type %pa
         %b1 = OpLoad %old_type %pb
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %a1 %b1 %copy
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

TEST_F(AzdFixCooperativeMatrixUseTest, InsertsPhiEdgeBitcasts) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseAAZD
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK: [[x:%\w+]] = OpUndef [[type_a]]
; CHECK-NEXT: [[y:%\w+]] = OpUndef [[type_a]]
; CHECK: OpSelectionMerge [[merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional %true [[then:%\w+]] [[else:%\w+]]
; CHECK-NEXT: [[then]] = OpLabel
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] [[x]] {{%\w+}} {{%\w+}}
; CHECK: [[x_acc:%\w+]] = OpBitcast [[type_acc]] [[x]]
; CHECK-NEXT: OpBranch [[merge]]
; CHECK-NEXT: [[else]] = OpLabel
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] [[y]] {{%\w+}} {{%\w+}}
; CHECK: [[y_acc:%\w+]] = OpBitcast [[type_acc]] [[y]]
; CHECK-NEXT: OpBranch [[merge]]
; CHECK-NEXT: [[merge]] = OpLabel
; CHECK: [[z:%\w+]] = OpPhi [[type_acc]] [[x_acc]] [[then]] [[y_acc]] [[else]]
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] {{%\w+}} {{%\w+}} [[z]]
               OpCapability Shader
               OpCapability CooperativeMatrixAZD
               OpExtension "SPV_AZD_neural_matrix"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %uint = OpTypeInt 32 0
   %uint_16 = OpConstant %uint 16
      %float = OpTypeFloat 32
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
   %old_type = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
          %x = OpUndef %old_type
          %y = OpUndef %old_type
          %a = OpUndef %old_type
          %b = OpUndef %old_type
          %c = OpUndef %old_type
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
               OpSelectionMerge %merge None
               OpBranchConditional %true %then %else
       %then = OpLabel
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %x %b %c
               OpBranch %merge
       %else = OpLabel
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %y %b %c
               OpBranch %merge
      %merge = OpLabel
          %z = OpPhi %old_type %x %then %y %else
         %d2 = OpCooperativeMatrixMulAddAZD %old_type %a %b %z
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, true);
}

TEST_F(AzdFixCooperativeMatrixUseTest, InsertsSelectOperandBitcasts) {
  const std::string text = R"(
; CHECK-DAG: [[type_a:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseAAZD
; CHECK-DAG: [[type_b:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixUseBAZD
; CHECK-DAG: [[type_acc:%\w+]] = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16 MatrixAccumulatorAZD
; CHECK: [[x:%\w+]] = OpUndef [[type_a]]
; CHECK-NEXT: [[y:%\w+]] = OpUndef [[type_a]]
; CHECK: [[x_acc:%\w+]] = OpBitcast [[type_acc]] [[x]]
; CHECK-NEXT: [[y_acc:%\w+]] = OpBitcast [[type_acc]] [[y]]
; CHECK-NEXT: [[z:%\w+]] = OpSelect [[type_acc]] %true [[x_acc]] [[y_acc]]
; CHECK: {{%\w+}} = OpCooperativeMatrixMulAddAZD [[type_acc]] {{%\w+}} {{%\w+}} [[z]]
               OpCapability Shader
               OpCapability CooperativeMatrixAZD
               OpExtension "SPV_AZD_neural_matrix"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %uint = OpTypeInt 32 0
   %uint_16 = OpConstant %uint 16
      %float = OpTypeFloat 32
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
   %old_type = OpTypeCooperativeMatrixAZD %float %uint_16 %uint_16
       %void = OpTypeVoid
    %fn_void = OpTypeFunction %void
          %x = OpUndef %old_type
          %y = OpUndef %old_type
          %a = OpUndef %old_type
          %b = OpUndef %old_type
          %c = OpUndef %old_type
       %main = OpFunction %void None %fn_void
      %entry = OpLabel
         %d0 = OpCooperativeMatrixMulAddAZD %old_type %x %b %c
         %d1 = OpCooperativeMatrixMulAddAZD %old_type %y %b %c
          %z = OpSelect %old_type %true %x %y
         %d2 = OpCooperativeMatrixMulAddAZD %old_type %a %b %z
               OpReturn
               OpFunctionEnd
)";

  SinglePassRunAndMatch<AzdFixCooperativeMatrixUsePass>(text, false);
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
