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

#ifndef SOURCE_OPT_HW_FUSE_TWO_LAYER_VECTOR_MATMUL_PASS_H_
#define SOURCE_OPT_HW_FUSE_TWO_LAYER_VECTOR_MATMUL_PASS_H_

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

class BasicBlock;
class Instruction;
class InstructionBuilder;

// Fuses a narrow two-layer f16 cooperative vector-matrix chain before the HW
// cooperative types are lowered.  The middle dimension is processed in
// 16-element splits and both dot products are expanded as ordinary f16vec2
// Fma operations.  This pass is intentionally internal to
// HwLowerToStandardPass; it has no public optimizer flag or factory.
class HwFuseTwoLayerVectorMatmulPass : public Pass {
 public:
  explicit HwFuseTwoLayerVectorMatmulPass(
      uint64_t max_unrolled_matmul_macs = 4096)
      : max_unrolled_matmul_macs_(max_unrolled_matmul_macs) {}

  const char* name() const override {
    return "hw-fuse-two-layer-vector-matmul";
  }
  Status Process() override;

 private:
  struct VectorTypeInfo {
    uint32_t type_id = 0;
    uint32_t component_type_id = 0;
    uint32_t length = 0;
  };

  struct MatrixTypeInfo {
    uint32_t type_id = 0;
    uint32_t component_type_id = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
  };

  struct DirectMatrixLoadInfo {
    Instruction* source_load = nullptr;
    uint32_t pointer_id = 0;
    uint32_t storage_class = 0;
    uint32_t shape_rows = 0;
    uint32_t shape_cols = 0;
    uint32_t offset_row = 0;
    uint32_t offset_col = 0;
    uint32_t element_stride_bytes = 0;
    std::vector<Operand> memory_operands;
  };

  struct DirectVectorLoadInfo {
    Instruction* source_load = nullptr;
    uint32_t pointer_id = 0;
    uint32_t storage_class = 0;
    uint32_t offset = 0;
    uint32_t element_stride_bytes = 0;
    std::vector<Operand> memory_operands;
  };

  struct Match {
    Instruction* first = nullptr;
    Instruction* relu = nullptr;
    Instruction* second = nullptr;
    BasicBlock* block = nullptr;
    VectorTypeInfo input;
    VectorTypeInfo middle;
    VectorTypeInfo output;
    MatrixTypeInfo first_matrix;
    MatrixTypeInfo second_matrix;
    DirectMatrixLoadInfo first_direct_matrix;
    DirectMatrixLoadInfo second_direct_matrix;
    DirectVectorLoadInfo first_direct_bias;
    DirectVectorLoadInfo second_direct_bias;
    uint32_t first_fp_fast_math_mode = 0;
    uint32_t relu_fp_fast_math_mode = 0;
    uint32_t second_fp_fast_math_mode = 0;
    std::vector<Instruction*> kill_list;
  };

  bool MatchSecondLayer(Instruction* second, Match* match) const;
  bool RewriteMatch(const Match& match);

  Instruction* TraceFunctionValueSource(Instruction* value, Instruction* before,
                                        std::vector<Instruction*>* chain,
                                        uint32_t depth = 0) const;
  Instruction* FindLastStoreToFunctionPointer(uint32_t pointer_id,
                                              Instruction* before) const;
  bool IsFunctionPointer(uint32_t pointer_id) const;
  bool IsClosedIntermediateChain(Instruction* second,
                                 std::vector<Instruction*>* kill_list) const;
  bool GetDirectMatrixLoad(Instruction* value, Instruction* before,
                           const MatrixTypeInfo& matrix,
                           DirectMatrixLoadInfo* info,
                           std::vector<Instruction*>* chain) const;
  bool GetDirectVectorLoad(Instruction* value, Instruction* before,
                           const VectorTypeInfo& vector,
                           DirectVectorLoadInfo* info,
                           std::vector<Instruction*>* chain) const;
  bool CanMoveDirectLoad(Instruction* load, Instruction* before) const;
  bool MemoryOperandsAreMovable(const Instruction* inst,
                                uint32_t first_memory_operand) const;
  uint32_t GetArrayStride(uint32_t array_type_id) const;
  bool IsModuleVisibleValue(uint32_t id) const;

  bool GetVectorType(uint32_t type_id, VectorTypeInfo* info) const;
  bool GetMatrixType(uint32_t type_id, MatrixTypeInfo* info) const;
  bool GetConstantU32(uint32_t id, uint32_t* value) const;
  bool GetConstantPairU32(uint32_t id, uint32_t* first, uint32_t* second) const;
  bool IsFloat16Type(uint32_t type_id) const;
  bool IsZeroConstant(uint32_t id) const;
  bool IsZeroConstantImpl(uint32_t id,
                          std::unordered_set<uint32_t>* visited) const;
  bool IsVectorMatrixMul(const Instruction* inst) const;
  bool IsVectorMatrixMulAdd(const Instruction* inst) const;
  bool IsGlslFMax(const Instruction* inst) const;
  bool IsIgnorableUser(const Instruction* inst) const;

  uint32_t GetOrCreateGLSLStd450Import();
  uint32_t GetOrCreateVectorType(uint32_t component_type_id,
                                 uint32_t component_count);
  uint32_t GetOrCreateUIntType();
  uint32_t GetOrCreateUIntConstant(uint32_t value);
  uint32_t GetOrCreatePointerType(uint32_t pointee_type_id,
                                  uint32_t storage_class);
  uint32_t GetOrCreateZero(uint32_t type_id);
  uint32_t GetFPFastMathMode(uint32_t result_id) const;
  void ApplyFPFastMathMode(Instruction* inst, uint32_t mode);
  void RemoveFPFastMathMode(uint32_t result_id);

  uint32_t BuildExtract(InstructionBuilder* builder, uint32_t type_id,
                        uint32_t composite_id,
                        const std::vector<uint32_t>& indices);
  uint32_t BuildMatrixElementLoad(InstructionBuilder* builder,
                                  uint32_t component_type_id,
                                  const DirectMatrixLoadInfo& direct,
                                  uint32_t row, uint32_t col);
  uint32_t BuildVectorElementLoad(InstructionBuilder* builder,
                                  uint32_t component_type_id,
                                  const DirectVectorLoadInfo& direct,
                                  uint32_t index);
  uint32_t BuildVec2(InstructionBuilder* builder, uint32_t vec2_type_id,
                     uint32_t lane0, uint32_t lane1);
  uint32_t BuildFma(InstructionBuilder* builder, uint32_t vec2_type_id,
                    uint32_t glsl_import_id, uint32_t lhs, uint32_t rhs,
                    uint32_t accumulator, uint32_t fp_fast_math_mode);
  uint32_t BuildFMaxZero(InstructionBuilder* builder,
                         uint32_t component_type_id, uint32_t glsl_import_id,
                         uint32_t value_id, uint32_t zero_id,
                         uint32_t fp_fast_math_mode);

  void ReportError(const Instruction* inst, const char* message) const;

  uint64_t max_unrolled_matmul_macs_ = 4096;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_HW_FUSE_TWO_LAYER_VECTOR_MATMUL_PASS_H_
