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

#ifndef SOURCE_OPT_HW_LOWER_TO_STANDARD_PASS_H_
#define SOURCE_OPT_HW_LOWER_TO_STANDARD_PASS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

class Function;
class Instruction;
class InstructionBuilder;
class BasicBlock;
struct Operand;

// Lowers HW cooperative matrix/vector types and operations to ordinary SPIR-V
// array code.  By default, f16/f32 values use packed vec4 arrays when naturally
// 4-wide, and scalar arrays otherwise.
class HwLowerToStandardPass : public Pass {
 public:
  enum class LoweringMode {
    kPreferPackedVec4,
    kForceScalar,
  };

  enum class CompletenessMode {
    kCooperativeOnly,
    kExtensionFree,
  };

  explicit HwLowerToStandardPass(
      LoweringMode lowering_mode = LoweringMode::kPreferPackedVec4,
      CompletenessMode completeness_mode = CompletenessMode::kCooperativeOnly,
      uint32_t max_elements = 1048576, uint64_t max_matmul_macs = 16777216,
      uint32_t max_unrolled_elements = 4096,
      uint64_t max_unrolled_matmul_macs = 4096)
      : lowering_mode_(lowering_mode),
        completeness_mode_(completeness_mode),
        max_elements_(max_elements),
        max_matmul_macs_(max_matmul_macs),
        max_unrolled_elements_(max_unrolled_elements),
        max_unrolled_matmul_macs_(max_unrolled_matmul_macs) {}

  const char* name() const override { return "hw-lower-to-standard"; }
  Status Process() override;

 private:
  struct MatrixTypeInfo {
    uint32_t type_id = 0;
    uint32_t component_type_id = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    bool has_matrix_use = false;
    spv::CooperativeMatrixUseHW matrix_use =
        spv::CooperativeMatrixUseHW::MatrixUseAHW;
    uint32_t lowered_type_id = 0;
    bool packed_f16vec4 = false;
    bool packed_f32vec4 = false;
    uint32_t packed_vec4_type_id = 0;
    uint32_t packed_cols = 0;
  };

  struct VectorTypeInfo {
    uint32_t type_id = 0;
    uint32_t component_type_id = 0;
    uint32_t length = 0;
    uint32_t lowered_type_id = 0;
    bool packed_f16vec4 = false;
    bool packed_f32vec4 = false;
    uint32_t packed_vec4_type_id = 0;
    uint32_t packed_length = 0;
  };

  struct ValueLayout {
    uint32_t component_type_id = 0;
    uint32_t piece_type_id = 0;
    uint32_t piece_count = 0;
    bool packed_vec4 = false;
  };

  // Describes a cooperative load that can be folded into a generated direct
  // matmul helper. Matrix-only fields remain zero for vector loads.
  struct DirectLoadSource {
    Instruction* source_load = nullptr;
    Instruction* conversion = nullptr;
    std::vector<Instruction*> chain;
    uint32_t pointer_id = 0;
    uint32_t pointer_type_id = 0;
    uint32_t component_type_id = 0;
    uint32_t conversion_fp_fast_math_mode = 0;
    bool conversion_has_explicit_fp_fast_math_mode = false;
    uint32_t shape_id = 0;
    uint32_t offset_id = 0;
    uint32_t constant_offset = 0;
    uint32_t layout = 0;
    std::vector<Operand> memory_operands;
  };

  enum class ElementwiseLoopKind {
    kConversion,
    kArithmetic,
    kScale,
    kExtInst,
    kSelect,
    kBroadcast,
  };

  bool CollectHwTypes();
  bool MaterializeLoweredTypes();
  bool PreflightExtensionFreeMode() const;
  bool PreflightNoContractionVectorMatmul() const;
  void RecordOriginalHwValueTypes();
  bool EliminateHwFunctionVariables();
  bool LegalizeModule();
  bool LegalizeCooperativeCoreInstruction(Instruction* inst) const;
  bool ValidateCompositeIndices(const Instruction* inst,
                                uint32_t composite_in_operand,
                                uint32_t first_index_in_operand) const;
  bool ValidateAccessChain(const Instruction* inst) const;
  bool PrepareMatmulPatternFunctions();
  bool LowerHwInstructions(std::vector<Instruction*>* to_kill);
  bool ReplaceHwTypeUses();
  bool CleanupHwDeclarations(const std::vector<Instruction*>& to_kill);
  bool FinalHwCheck() const;

  bool LowerMatrixLoad(Instruction* inst);
  bool LowerMatrixStore(Instruction* inst, std::vector<Instruction*>* to_kill);
  bool LowerMatrixMulAdd(Instruction* inst);
  bool LowerMatrixMulAddPackedVec4(Instruction* inst);
  bool LowerMatrixMulAddScalarFallback(Instruction* inst);
  bool LowerMatrixMulAddWithLoop(Instruction* inst);
  bool LowerMatrixLength(Instruction* inst, std::vector<Instruction*>* to_kill);
  bool LowerMatrixReduce(Instruction* inst);
  bool LowerMatrixReduceImpl(Instruction* inst);
  bool LowerMatrixReduceWithLoop(Instruction* inst, uint32_t reduce_axis,
                                 uint32_t combine_op);
  bool LowerVectorLoad(Instruction* inst);
  bool LowerVectorStore(Instruction* inst, std::vector<Instruction*>* to_kill);
  bool TryLowerFusedVectorMatmulStore(Instruction* inst, bool* handled);
  bool TryLowerFusedMatrixMatmulStore(Instruction* inst, bool* handled);
  bool TryLowerDirectMatrixMulAdd(Instruction* inst, bool* handled);
  bool TryLowerDirectVectorMatrixMulPackedVec4(Instruction* inst, bool has_bias,
                                               bool* handled);
  bool LowerVectorMatrixMul(Instruction* inst, bool has_bias);
  bool LowerVectorMatrixMulPackedVec4(Instruction* inst, bool has_bias);
  bool LowerVectorMatrixMulScalarFallback(Instruction* inst, bool has_bias);
  bool LowerVectorMatrixMulWithLoop(Instruction* inst, bool has_bias);
  bool LowerConstantComposite(Instruction* inst);
  bool LowerCompositeConstruct(Instruction* inst);
  bool LowerCompositeExtract(Instruction* inst);
  bool LowerCompositeInsert(Instruction* inst);
  bool LowerSelect(Instruction* inst);
  bool LowerVectorExtractDynamic(Instruction* inst);
  bool LowerVectorInsertDynamic(Instruction* inst);
  bool LowerNullOrUndef(Instruction* inst);
  bool LowerHwBitcast(Instruction* inst);
  bool LowerHwConversion(Instruction* inst);
  bool LowerHwArithmetic(Instruction* inst);
  bool LowerHwScale(Instruction* inst);
  bool LowerExtInstOnCooperativeVector(Instruction* inst);
  bool LowerElementwiseWithLoop(Instruction* inst, ElementwiseLoopKind kind);

  uint32_t GetOrCreateArrayType(uint32_t component_type_id, uint32_t length,
                                Instruction* insert_after);
  uint32_t GetOrCreateVectorType(uint32_t component_type_id,
                                 uint32_t component_count,
                                 Instruction** insert_after);
  uint32_t GetOrCreatePackedArrayType(uint32_t vec4_type_id, uint32_t length,
                                      Instruction* insert_after);
  uint32_t GetOrCreatePointerType(uint32_t pointee_type_id,
                                  spv::StorageClass storage_class);
  uint32_t GetOrCreateVoidType();
  uint32_t GetOrCreateBoolType();
  uint32_t GetOrCreateUIntType();
  uint32_t GetOrCreateIntegerType(uint32_t width, bool is_signed);
  uint32_t GetOrCreateUIntConstant(uint32_t value);
  uint32_t GetOrCreateUIntTypeAfter(Instruction** insert_after);
  uint32_t GetOrCreateUIntConstantAfter(uint32_t value,
                                        Instruction** insert_after);
  uint32_t GetOrCreateConstant(uint32_t type_id, uint32_t value);
  uint32_t GetOrCreateZero(uint32_t type_id);
  uint32_t GetOrCreateCompositeConstant(
      uint32_t type_id, const std::vector<uint32_t>& constituent_ids,
      Instruction** insert_after, spv::Op opcode);
  bool RemapCompositeIndices(Instruction* inst, uint32_t composite_in_operand,
                             uint32_t first_index_in_operand);
  bool LowerAccessChain(Instruction* inst);
  uint32_t BuildPairComponentAsUInt(InstructionBuilder* builder,
                                    Instruction* user, uint32_t pair_id,
                                    uint32_t component_index);
  uint32_t BuildMatrixElementIndex(InstructionBuilder* builder,
                                   Instruction* user,
                                   const MatrixTypeInfo& info,
                                   uint32_t shape_id, uint32_t offset_id,
                                   uint32_t layout, uint32_t row, uint32_t col);
  uint32_t BuildDynamicMatrixElementIndex(InstructionBuilder* builder,
                                          Instruction* user, uint32_t shape_id,
                                          uint32_t offset_id, uint32_t layout,
                                          uint32_t matrix_cols,
                                          uint32_t flat_index_id);
  uint32_t BuildVectorElementIndex(InstructionBuilder* builder,
                                   Instruction* user, uint32_t offset_id,
                                   uint32_t logical_index_id);
  uint32_t BuildElementAccess(InstructionBuilder* builder, Instruction* user,
                              uint32_t pointer_id, uint32_t component_type_id,
                              uint32_t element_index_id);
  uint32_t BuildElementAccessFromPointerType(InstructionBuilder* builder,
                                             uint32_t pointer_type_id,
                                             uint32_t pointer_id,
                                             uint32_t component_type_id,
                                             uint32_t element_index_id);
  Instruction* AddFunctionVariable(Function* function, uint32_t pointer_type_id,
                                   uint32_t initializer_id = 0);
  uint32_t BuildLogicalAggregateLoad(InstructionBuilder* builder,
                                     uint32_t aggregate_pointer_id,
                                     uint32_t component_type_id,
                                     uint32_t packed_vec4_type_id,
                                     uint32_t logical_index_id);
  bool BuildLogicalAggregateStore(InstructionBuilder* builder,
                                  uint32_t aggregate_pointer_id,
                                  uint32_t component_type_id,
                                  uint32_t packed_vec4_type_id,
                                  uint32_t logical_index_id, uint32_t value_id);
  BasicBlock* MakeBasicBlock(uint32_t label_id);
  bool BuildPackedMatrixLoadOuterLoop(
      Instruction* insert_before, const MatrixTypeInfo& info,
      uint32_t pointer_id, uint32_t pointer_type_id, uint32_t shape_id,
      uint32_t offset_id, uint32_t layout,
      const std::vector<Operand>& memory_operands, uint32_t* result_id);
  bool BuildPackedMatrixStoreOuterLoop(
      Instruction* insert_before, const MatrixTypeInfo& info,
      uint32_t pointer_id, uint32_t pointer_type_id, uint32_t object_id,
      uint32_t shape_id, uint32_t offset_id, uint32_t layout,
      const std::vector<Operand>& memory_operands);
  bool BuildPackedVectorLoadOuterLoop(
      Instruction* insert_before, const VectorTypeInfo& info,
      uint32_t pointer_id, uint32_t pointer_type_id, uint32_t offset_id,
      const std::vector<Operand>& memory_operands, uint32_t* result_id);
  bool BuildPackedVectorStoreOuterLoop(
      Instruction* insert_before, const VectorTypeInfo& info,
      uint32_t pointer_id, uint32_t pointer_type_id, uint32_t offset_id,
      uint32_t object_id, const std::vector<Operand>& memory_operands);
  bool BuildScalarMemoryLoop(Instruction* insert_before,
                             uint32_t lowered_type_id,
                             uint32_t component_type_id, uint32_t element_count,
                             uint32_t pointer_id, uint32_t object_id,
                             uint32_t shape_id, uint32_t offset_id,
                             uint32_t matrix_cols, uint32_t layout,
                             bool is_matrix, uint32_t packed_vec4_type_id,
                             const std::vector<Operand>& memory_operands,
                             uint32_t* result_id);
  uint32_t BuildConstantPackedVectorSelectFunction(const VectorTypeInfo& vector,
                                                   uint32_t constant_id);
  uint32_t BuildFusedVectorMatmulStoreFunctionPackedVec4(
      const VectorTypeInfo& result, const VectorTypeInfo& input,
      const MatrixTypeInfo& matrix, uint32_t input_pointer_id,
      uint32_t input_pointer_type_id,
      const std::vector<Operand>& input_memory_operands,
      uint32_t matrix_pointer_id, uint32_t matrix_pointer_type_id,
      uint32_t matrix_shape_id, uint32_t matrix_offset_id,
      const std::vector<Operand>& matrix_memory_operands,
      uint32_t output_pointer_id, uint32_t output_pointer_type_id,
      const std::vector<Operand>& output_memory_operands);
  uint32_t BuildFusedVectorMatmulAddStoreFunctionPackedVec4(
      const VectorTypeInfo& result, const VectorTypeInfo& input,
      const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
      uint32_t bias_constant_id, uint32_t input_pointer_id,
      uint32_t input_pointer_type_id,
      const std::vector<Operand>& input_memory_operands,
      uint32_t matrix_pointer_id, uint32_t matrix_pointer_type_id,
      uint32_t matrix_shape_id, uint32_t matrix_offset_id,
      const std::vector<Operand>& matrix_memory_operands,
      uint32_t output_pointer_id, uint32_t output_pointer_type_id,
      const std::vector<Operand>& output_memory_operands);
  uint32_t BuildFusedMatrixMatmulStoreFunctionPackedVec4(
      const MatrixTypeInfo& result, const MatrixTypeInfo& a,
      const MatrixTypeInfo& b, const MatrixTypeInfo& c, uint32_t a_pointer_id,
      uint32_t a_pointer_type_id, uint32_t a_shape_id, uint32_t a_offset_id,
      const std::vector<Operand>& a_memory_operands, uint32_t b_pointer_id,
      uint32_t b_pointer_type_id, uint32_t b_shape_id, uint32_t b_offset_id,
      const std::vector<Operand>& b_memory_operands, uint32_t c_pointer_id,
      uint32_t c_pointer_type_id, uint32_t c_shape_id, uint32_t c_offset_id,
      const std::vector<Operand>& c_memory_operands, uint32_t output_pointer_id,
      uint32_t output_pointer_type_id, uint32_t output_shape_id,
      uint32_t output_offset_id,
      const std::vector<Operand>& output_memory_operands);
  uint32_t BuildDirectVectorMatmulFunctionPackedVec4(
      const VectorTypeInfo& result, const VectorTypeInfo& input,
      const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
      uint32_t input_pointer_id, uint32_t input_pointer_type_id,
      const std::vector<Operand>& input_memory_operands,
      uint32_t input_constant_id, bool input_is_value,
      uint32_t matrix_pointer_id, uint32_t matrix_pointer_type_id,
      uint32_t matrix_shape_id, uint32_t matrix_offset_id,
      const std::vector<Operand>& matrix_memory_operands,
      uint32_t matrix_constant_id, bool matrix_is_value,
      uint32_t bias_pointer_id, uint32_t bias_pointer_type_id,
      const std::vector<Operand>& bias_memory_operands,
      uint32_t bias_source_component_type_id, uint32_t bias_offset,
      uint32_t bias_conversion_fp_fast_math_mode,
      bool bias_conversion_has_explicit_fp_fast_math_mode,
      uint32_t bias_constant_id, bool bias_is_value,
      const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments = {});
  uint32_t BuildDirectMatmulFunctionPackedVec4(
      const MatrixTypeInfo& result, const MatrixTypeInfo& a,
      const MatrixTypeInfo& b, const MatrixTypeInfo& c, uint32_t a_pointer_id,
      uint32_t a_pointer_type_id, uint32_t a_shape_id, uint32_t a_offset_id,
      const std::vector<Operand>& a_memory_operands, uint32_t a_constant_id,
      bool a_is_value, uint32_t b_pointer_id, uint32_t b_pointer_type_id,
      uint32_t b_shape_id, uint32_t b_offset_id,
      const std::vector<Operand>& b_memory_operands, uint32_t b_constant_id,
      bool b_is_value, uint32_t c_pointer_id, uint32_t c_pointer_type_id,
      uint32_t c_shape_id, uint32_t c_offset_id,
      const std::vector<Operand>& c_memory_operands, uint32_t c_constant_id,
      bool c_is_value,
      const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments = {});
  uint32_t BuildDirectMatrixMatmulFunction(
      const MatrixTypeInfo& result, const MatrixTypeInfo& a,
      const MatrixTypeInfo& b, const MatrixTypeInfo& c, uint32_t a_pointer_id,
      uint32_t a_pointer_type_id, uint32_t a_shape_id, uint32_t a_offset_id,
      const std::vector<Operand>& a_memory_operands, uint32_t a_constant_id,
      bool a_is_value, uint32_t b_pointer_id, uint32_t b_pointer_type_id,
      uint32_t b_shape_id, uint32_t b_offset_id,
      const std::vector<Operand>& b_memory_operands, uint32_t b_constant_id,
      bool b_is_value, uint32_t c_pointer_id, uint32_t c_pointer_type_id,
      uint32_t c_shape_id, uint32_t c_offset_id,
      const std::vector<Operand>& c_memory_operands, uint32_t c_constant_id,
      bool c_is_value,
      const std::vector<std::pair<uint32_t, uint32_t>>& value_arguments = {});
  uint32_t BuildRowMajorMatrixMemoryIndex(InstructionBuilder* builder,
                                          Instruction* user, uint32_t shape_id,
                                          uint32_t offset_id, uint32_t cols,
                                          uint32_t base_id);
  uint32_t BuildCapturedPointer(InstructionBuilder* builder,
                                uint32_t pointer_id);
  bool CanCapturePointer(uint32_t pointer_id) const;
  bool IsModuleVisibleValue(uint32_t id) const;
  uint32_t ExtractCompositeElement(InstructionBuilder* builder,
                                   uint32_t component_type_id,
                                   uint32_t composite_id, uint32_t index);
  uint32_t AddLoad(InstructionBuilder* builder, uint32_t type_id,
                   uint32_t pointer_id,
                   const std::vector<Operand>& memory_operands);
  bool AddStore(InstructionBuilder* builder, uint32_t pointer_id,
                uint32_t object_id, const std::vector<Operand>& memory_operands,
                uint32_t accessed_type_id = 0);
  std::vector<Operand> CopyMemoryOperands(const Instruction* inst,
                                          uint32_t first_in_operand) const;
  bool NormalizeMemoryOperandsForAccess(
      uint32_t pointer_id, uint32_t accessed_type_id,
      const std::vector<Operand>& memory_operands,
      std::vector<Operand>* normalized) const;
  bool GetKnownAccessByteOffset(uint32_t pointer_id, uint32_t accessed_type_id,
                                uint64_t* byte_offset) const;
  bool TryEvaluateConstantU32Expression(uint32_t id, uint32_t* value,
                                        uint32_t depth = 0) const;
  bool TryEvaluateConstantFloat32(uint32_t id, float* value,
                                  uint32_t depth = 0) const;
  uint32_t GetTypeNaturalAlignment(uint32_t type_id) const;
  uint32_t GetArrayStride(uint32_t array_type_id) const;
  uint32_t ExtractVectorScalar(InstructionBuilder* builder,
                               const VectorTypeInfo& info, uint32_t vector_id,
                               uint32_t index);
  uint32_t ExtractMatrixScalar(InstructionBuilder* builder,
                               const MatrixTypeInfo& info, uint32_t matrix_id,
                               uint32_t row, uint32_t col);
  bool DescribeVectorValue(uint32_t value_id, uint32_t expected_length,
                           ValueLayout* layout) const;
  bool DescribeMatrixValue(uint32_t value_id, uint32_t expected_rows,
                           uint32_t expected_cols, ValueLayout* layout) const;
  uint32_t ExtractValuePiece(InstructionBuilder* builder,
                             const ValueLayout& layout, uint32_t value_id,
                             uint32_t piece_index);
  uint32_t ExtractVectorValueScalar(InstructionBuilder* builder,
                                    const ValueLayout& layout,
                                    uint32_t value_id, uint32_t index);
  uint32_t ExtractMatrixValueScalar(InstructionBuilder* builder,
                                    const ValueLayout& layout,
                                    uint32_t value_id, uint32_t cols,
                                    uint32_t row, uint32_t col);
  bool RebuildVectorFromScalars(Instruction* inst, const VectorTypeInfo& info,
                                const std::vector<uint32_t>& scalar_ids);
  bool RebuildMatrixFromScalars(Instruction* inst, const MatrixTypeInfo& info,
                                const std::vector<uint32_t>& scalar_ids);
  bool RebuildAggregateFromScalars(Instruction* inst, uint32_t lowered_type_id,
                                   uint32_t packed_vec4_type_id,
                                   uint32_t packed_piece_count,
                                   uint32_t expected_scalar_count,
                                   const std::vector<uint32_t>& scalar_ids,
                                   const char* error_message);
  uint32_t BuildMatrixRowVector(InstructionBuilder* builder,
                                const MatrixTypeInfo& info, uint32_t matrix_id,
                                uint32_t row, uint32_t col_start,
                                uint32_t vec4_type_id);
  uint32_t BuildMatrixColumnVector(InstructionBuilder* builder,
                                   const MatrixTypeInfo& info,
                                   uint32_t matrix_id, uint32_t row_start,
                                   uint32_t col, uint32_t vec4_type_id);
  uint32_t BuildVectorTimesScalar(InstructionBuilder* builder,
                                  spv::Op scale_opcode, uint32_t vec4_type_id,
                                  uint32_t vector_id, uint32_t scalar_id);
  uint32_t BuildScalarSplat(InstructionBuilder* builder, uint32_t vec4_type_id,
                            uint32_t scalar_id);
  uint32_t BuildFma(InstructionBuilder* builder, uint32_t type_id,
                    uint32_t multiplicand_id, uint32_t multiplier_id,
                    uint32_t addend_id);
  uint32_t BuildMatmulAccumulate(InstructionBuilder* builder,
                                 uint32_t accumulator_type_id,
                                 uint32_t lhs_type_id, uint32_t lhs_id,
                                 uint32_t rhs_type_id, uint32_t rhs_id,
                                 uint32_t accumulator_id);
  uint32_t BuildHorizontalReduce(InstructionBuilder* builder,
                                 uint32_t component_type_id,
                                 uint32_t vector_id);
  uint32_t BuildReduceCombine(InstructionBuilder* builder,
                              uint32_t component_type_id, uint32_t combine_op,
                              uint32_t lhs_id, uint32_t rhs_id);
  bool BuildVectorMatrixMulPatternPackedVec4(
      InstructionBuilder* builder, const VectorTypeInfo& result,
      const VectorTypeInfo& input, const MatrixTypeInfo& matrix,
      const VectorTypeInfo* bias, uint32_t input_id, uint32_t matrix_id,
      uint32_t bias_id, bool has_bias, std::vector<uint32_t>* element_ids);
  bool BuildMatmulPatternPackedVec4(InstructionBuilder* builder,
                                    const MatrixTypeInfo& result,
                                    const MatrixTypeInfo& a,
                                    const MatrixTypeInfo& b,
                                    const MatrixTypeInfo& c, uint32_t a_id,
                                    uint32_t b_id, uint32_t c_id,
                                    std::vector<uint32_t>* element_ids);
  void AddGeneratedFunction(std::unique_ptr<Function> function,
                            uint32_t function_id, bool may_write_memory);
  std::string MemoryOperandsKey(
      const std::vector<Operand>& memory_operands) const;
  uint32_t GetOrCreateFunctionType(uint32_t return_type_id,
                                   const std::vector<uint32_t>& param_type_ids);
  uint32_t GetOrCreatePackedLoadChunkFunction(
      uint32_t pointer_id, uint32_t pointer_type_id, uint32_t component_type_id,
      uint32_t vec4_type_id, const std::vector<Operand>& memory_operands);
  uint32_t GetOrCreatePackedStoreChunkFunction(
      uint32_t pointer_id, uint32_t pointer_type_id, uint32_t component_type_id,
      uint32_t vec4_type_id, const std::vector<Operand>& memory_operands);
  uint32_t GetOrCreateTileWeightFunctionPackedVec4(
      const MatrixTypeInfo& matrix);
  uint32_t GetOrCreateMatmulTileWeightFunctionPackedVec4(
      const MatrixTypeInfo& matrix);
  uint32_t GetOrCreateVectorMatmulPatternFunctionPackedVec4(
      const VectorTypeInfo& result, const VectorTypeInfo& input,
      const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias);
  uint32_t GetOrCreateVectorMatmulPatternPointerFunctionPackedVec4(
      const VectorTypeInfo& result, const VectorTypeInfo& input,
      const MatrixTypeInfo& matrix, const VectorTypeInfo* bias, bool has_bias,
      uint32_t input_pointer_type_id, uint32_t matrix_pointer_type_id,
      uint32_t bias_pointer_type_id);
  uint32_t GetOrCreateMatmulPatternFunctionPackedVec4(
      const MatrixTypeInfo& result, const MatrixTypeInfo& a,
      const MatrixTypeInfo& b, const MatrixTypeInfo& c);
  uint32_t GetFunctionPointerOperandForLoad(Instruction* inst,
                                            uint32_t original_pointee_type_id,
                                            uint32_t lowered_pointee_type_id,
                                            uint32_t* pointer_type_id) const;
  bool IsIgnorableDirectUser(const Instruction* user) const;
  bool ResolveDirectVectorLoad(Instruction* value_inst, Instruction* use,
                               DirectLoadSource* source) const;
  bool ResolveDirectF16ToF32VectorLoad(Instruction* value_inst,
                                       Instruction* use,
                                       DirectLoadSource* source) const;
  bool ResolveDirectMatrixLoad(Instruction* value_inst, Instruction* use,
                               DirectLoadSource* source) const;
  enum class DirectMatrixUseBitcastPolicy {
    kAtoBOnly,
    kAnySameShapeComponent,
  };
  bool IsCompatibleDirectMatrixUseBitcast(
      Instruction* bitcast, DirectMatrixUseBitcastPolicy policy) const;
  bool DirectTraceTypesAreCompatible(Instruction* value_inst, Instruction* use,
                                     DirectMatrixUseBitcastPolicy policy) const;
  bool DirectKillListUsersAreClosed(
      Instruction* current_inst,
      const std::vector<Instruction*>& kill_list) const;
  bool CanMoveLoadToUse(Instruction* load, Instruction* use,
                        bool function_memory,
                        uint32_t first_memory_operand) const;
  bool IsDirectSafeSharedValueUser(
      Instruction* current_inst, Instruction* user,
      const std::unordered_set<Instruction*>& kill_set) const;
  bool IsDirectSafeSharedFunctionPointerUser(
      Instruction* current_inst, Instruction* pointer_user,
      const std::unordered_set<Instruction*>& kill_set) const;
  enum class SharedDirectUserState {
    kNone,
    kSafeLive,
    kUnsafe,
  };
  SharedDirectUserState AnalyzeSharedDirectValueUses(
      Instruction* current_inst, Instruction* value,
      const std::unordered_set<Instruction*>& kill_set,
      std::unordered_set<Instruction*>* keep_alive,
      std::unordered_set<const Instruction*>* visited_values,
      std::unordered_set<const Instruction*>* visited_pointers) const;
  SharedDirectUserState AnalyzeSharedDirectPointerUses(
      Instruction* current_inst, Instruction* pointer,
      Instruction* originating_store,
      const std::unordered_set<Instruction*>& kill_set,
      std::unordered_set<Instruction*>* keep_alive,
      std::unordered_set<const Instruction*>* visited_values,
      std::unordered_set<const Instruction*>* visited_pointers) const;
  bool AnalyzeSharedDirectSourceUsers(
      Instruction* current_inst, Instruction* source_load,
      const std::unordered_set<Instruction*>& kill_set,
      bool* has_live_shared_users,
      std::unordered_set<Instruction*>* keep_alive) const;
  bool AddSharedDirectSourceKillsOrCheckSafe(
      Instruction* current_inst, Instruction* source_load,
      std::vector<Instruction*>* kill_list,
      std::unordered_set<Instruction*>* kill_set) const;
  bool HasLiveSafeSharedDirectSourceUsers(
      Instruction* current_inst, Instruction* source_load,
      const std::unordered_set<Instruction*>& kill_set) const;
  void KeepSharedDirectSourceAlive(
      Instruction* current_inst, Instruction* source_load,
      std::vector<Instruction*>* kill_list,
      std::unordered_set<Instruction*>* kill_set) const;
  bool HasUnsafeMemoryInstructionBetween(Instruction* start, Instruction* end,
                                         bool function_memory) const;
  bool InstructionMayWriteOrOrderMemory(const Instruction* inst,
                                        bool function_memory) const;
  uint32_t GetMemoryPointerOperandId(const Instruction* inst) const;
  uint32_t GetRootModulePointerId(uint32_t pointer_id) const;
  bool IsDisjointModuleMemoryWrite(uint32_t load_pointer_id,
                                   const Instruction* inst) const;
  bool MemoryAccessOperandsAreMovable(const Instruction* inst,
                                      uint32_t first_in_operand) const;
  bool DirectTransportChainIsMovable(
      const std::vector<Instruction*>& chain) const;
  bool GetPointerStorageClass(uint32_t pointer_id,
                              uint32_t* storage_class) const;
  Instruction* TraceFunctionValueSource(Instruction* value_inst,
                                        Instruction* before,
                                        std::vector<Instruction*>* chain,
                                        uint32_t depth = 0) const;
  Instruction* FindLastStoreToFunctionPointer(uint32_t pointer_id,
                                              Instruction* before) const;
  bool IsFunctionPointer(uint32_t pointer_id) const;
  std::string TileWeightFunctionKey(const MatrixTypeInfo& matrix) const;
  std::string MatmulTileWeightFunctionKey(const MatrixTypeInfo& matrix) const;
  std::string VectorMatmulPatternFunctionKey(const VectorTypeInfo& result,
                                             const VectorTypeInfo& input,
                                             const MatrixTypeInfo& matrix,
                                             const VectorTypeInfo* bias,
                                             bool has_bias) const;
  std::string MatmulPatternFunctionKey(const MatrixTypeInfo& result,
                                       const MatrixTypeInfo& a,
                                       const MatrixTypeInfo& b,
                                       const MatrixTypeInfo& c) const;
  bool IsPackedVec4(const MatrixTypeInfo& info) const;
  bool IsPackedVec4(const VectorTypeInfo& info) const;
  bool IsSamePackedVec4Kind(const MatrixTypeInfo& a,
                            const MatrixTypeInfo& b) const;
  bool IsSamePackedVec4Kind(const VectorTypeInfo& a,
                            const VectorTypeInfo& b) const;
  bool CanUsePackedVec4MatrixMulAdd(const MatrixTypeInfo& result,
                                    const MatrixTypeInfo& a,
                                    const MatrixTypeInfo& b,
                                    const MatrixTypeInfo& c) const;
  bool CanUseDirectMatrixMulAdd(const MatrixTypeInfo& result,
                                const MatrixTypeInfo& a,
                                const MatrixTypeInfo& b,
                                const MatrixTypeInfo& c) const;
  bool CanUsePackedVec4VectorMatrixMul(const VectorTypeInfo& result,
                                       const VectorTypeInfo& input,
                                       const MatrixTypeInfo& matrix,
                                       const VectorTypeInfo* bias) const;
  bool CanUseDirectVectorMatrixMul(const VectorTypeInfo& result,
                                   const VectorTypeInfo& input,
                                   const MatrixTypeInfo& matrix,
                                   const VectorTypeInfo* bias) const;
  bool ShouldUsePackedVec4(uint32_t extent) const;
  uint32_t MatrixFlatIndex(const MatrixTypeInfo& info, uint32_t row,
                           uint32_t col) const;
  uint32_t MatrixPackedIndex(const MatrixTypeInfo& info, uint32_t row,
                             uint32_t col_pack) const;
  uint32_t VectorPackedIndex(uint32_t scalar_index) const;
  uint32_t PackedLane(uint32_t scalar_index) const;
  uint32_t GetOrCreateGLSLStd450Import();
  bool GetExplicitFPFastMathMode(uint32_t result_id, uint32_t* mode) const;
  uint32_t GetFPFastMathMode(uint32_t result_id) const;
  void ApplyFPFastMathMode(Instruction* inst, uint32_t mode,
                           bool preserve_none = false);
  void ApplyActiveFPFastMathMode(Instruction* inst);
  bool MatmulAllowsReassociation(const Instruction* inst) const;
  void RemoveFPFastMathMode(uint32_t result_id);

  const MatrixTypeInfo* GetMatrixType(uint32_t type_id) const;
  const MatrixTypeInfo* GetMatrixTypeForValue(const Instruction* value) const;
  const VectorTypeInfo* GetVectorType(uint32_t type_id) const;
  uint32_t GetLoweredType(uint32_t type_id) const;
  uint32_t GetPointerTypeId(uint32_t pointer_id) const;
  uint32_t GetPointeeType(uint32_t pointer_type_id) const;
  bool GetConstantU32(uint32_t id, uint32_t* value) const;
  bool IsFloat16Type(uint32_t type_id) const;
  bool IsFloat32Type(uint32_t type_id) const;
  bool IsHwType(uint32_t type_id) const;
  bool TypeContainsHw(uint32_t type_id) const;
  bool TypeContainsHwImpl(uint32_t type_id,
                          std::unordered_set<uint32_t>* visited) const;
  bool InstructionTouchesHw(const Instruction* inst) const;
  bool HasHwTypeReference(const Instruction* inst) const;
  bool IsHwOpcode(spv::Op opcode) const;
  bool IsAnyHwOpcode(spv::Op opcode) const;
  bool IsHwCapabilityOrExtension(const Instruction* inst) const;
  bool IsAnyHwCapabilityOrExtension(const Instruction* inst) const;
  bool HasHwOperand(const Instruction* inst) const;
  bool RequiresHwNeuralShaderExtension(const Instruction* inst) const;
  bool ModuleRequiresHwNeuralShaderExtension() const;
  bool RemoveExtensionByName(const char* extension_name);
  bool RemoveSourceExtensionByName(const char* extension_name);
  void RebuildAsCompositeConstruct(Instruction* inst, uint32_t type_id,
                                   const std::vector<uint32_t>& element_ids);
  void RebuildAsFunctionCall(Instruction* inst, uint32_t type_id,
                             uint32_t function_id,
                             const std::vector<uint32_t>& argument_ids);
  uint32_t GetOrCreateModuleConstantFromCompositeConstruct(
      Instruction* composite_construct);
  void ReportError(const Instruction* inst, const std::string& message) const;

  std::unordered_map<uint32_t, MatrixTypeInfo> matrix_types_;
  std::unordered_map<uint32_t, VectorTypeInfo> vector_types_;
  std::unordered_map<uint32_t, uint32_t> lowered_types_;
  std::unordered_map<uint32_t, uint32_t> original_hw_value_types_;
  std::unordered_map<std::string, uint32_t> packed_load_chunk_functions_;
  std::unordered_map<std::string, uint32_t> packed_store_chunk_functions_;
  std::unordered_map<std::string, uint32_t> tile_weight_functions_;
  std::unordered_map<std::string, uint32_t> matmul_tile_weight_functions_;
  std::unordered_map<std::string, uint32_t> vector_matmul_pattern_functions_;
  std::unordered_map<std::string, uint32_t> matmul_pattern_functions_;
  std::unordered_set<uint32_t> matmul_pattern_function_ids_;
  std::unordered_set<uint32_t> generated_function_ids_;
  std::unordered_set<uint32_t> read_only_generated_function_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> pending_fp_fast_math_modes_;
  LoweringMode lowering_mode_ = LoweringMode::kPreferPackedVec4;
  CompletenessMode completeness_mode_ = CompletenessMode::kCooperativeOnly;
  uint32_t max_elements_ = 1048576;
  uint64_t max_matmul_macs_ = 16777216;
  uint32_t max_unrolled_elements_ = 4096;
  uint64_t max_unrolled_matmul_macs_ = 4096;
  uint32_t active_fp_fast_math_mode_ = 0;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_HW_LOWER_TO_STANDARD_PASS_H_
