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

#ifndef SOURCE_OPT_AZD_LOWER_TO_STANDARD_PASS_H_
#define SOURCE_OPT_AZD_LOWER_TO_STANDARD_PASS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

class Instruction;
class InstructionBuilder;

// Lowers AZD cooperative matrix/vector types and operations to ordinary SPIR-V
// scalar array code.  The MVP intentionally supports only f32 values with
// single-function SSA/local-variable use.
class AzdLowerToStandardPass : public Pass {
 public:
  const char* name() const override { return "azd-lower-to-standard"; }
  Status Process() override;

 private:
  struct MatrixTypeInfo {
    uint32_t type_id = 0;
    uint32_t component_type_id = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t lowered_type_id = 0;
  };

  struct VectorTypeInfo {
    uint32_t type_id = 0;
    uint32_t component_type_id = 0;
    uint32_t length = 0;
    uint32_t lowered_type_id = 0;
  };

  bool CollectAzdTypes();
  bool LegalizeModule();
  bool LowerAzdInstructions(std::vector<Instruction*>* to_kill);
  bool ReplaceAzdTypeUses();
  bool CleanupAzdDeclarations(const std::vector<Instruction*>& to_kill);
  bool FinalAzdCheck() const;

  bool LowerMatrixLoad(Instruction* inst);
  bool LowerMatrixStore(Instruction* inst, std::vector<Instruction*>* to_kill);
  bool LowerMatrixMulAdd(Instruction* inst);
  bool LowerMatrixLength(Instruction* inst, std::vector<Instruction*>* to_kill);
  bool LowerVectorLoad(Instruction* inst);
  bool LowerVectorStore(Instruction* inst, std::vector<Instruction*>* to_kill);
  bool LowerVectorMatrixMul(Instruction* inst, bool has_bias);
  bool LowerAzdBitcast(Instruction* inst);

  uint32_t GetOrCreateArrayType(uint32_t component_type_id, uint32_t length,
                                Instruction* insert_after);
  uint32_t GetOrCreatePointerType(uint32_t pointee_type_id,
                                  spv::StorageClass storage_class);
  uint32_t GetOrCreateUIntType();
  uint32_t GetOrCreateUIntConstant(uint32_t value);
  uint32_t GetOrCreateUIntTypeAfter(Instruction** insert_after);
  uint32_t GetOrCreateUIntConstantAfter(uint32_t value,
                                        Instruction** insert_after);
  uint32_t GetOrCreateConstant(uint32_t type_id, uint32_t value);
  uint32_t GetOrCreateZero(uint32_t type_id);
  uint32_t BuildPairComponentAsUInt(InstructionBuilder* builder,
                                    Instruction* user, uint32_t pair_id,
                                    uint32_t component_index);
  uint32_t BuildMatrixElementIndex(InstructionBuilder* builder,
                                   Instruction* user,
                                   const MatrixTypeInfo& info,
                                   uint32_t shape_id, uint32_t offset_id,
                                   uint32_t layout, uint32_t row, uint32_t col);
  uint32_t BuildElementAccess(InstructionBuilder* builder, Instruction* user,
                              uint32_t pointer_id, uint32_t component_type_id,
                              uint32_t element_index_id);
  uint32_t ExtractCompositeElement(InstructionBuilder* builder,
                                   uint32_t component_type_id,
                                   uint32_t composite_id, uint32_t index);
  uint32_t MatrixFlatIndex(const MatrixTypeInfo& info, uint32_t row,
                           uint32_t col) const;

  const MatrixTypeInfo* GetMatrixType(uint32_t type_id) const;
  const VectorTypeInfo* GetVectorType(uint32_t type_id) const;
  uint32_t GetLoweredType(uint32_t type_id) const;
  uint32_t GetPointeeType(uint32_t pointer_type_id) const;
  bool GetConstantU32(uint32_t id, uint32_t* value) const;
  bool IsFloat32Type(uint32_t type_id) const;
  bool IsAzdType(uint32_t type_id) const;
  bool TypeContainsAzd(uint32_t type_id) const;
  bool IsAzdOpcode(spv::Op opcode) const;
  bool IsAzdCapabilityOrExtension(const Instruction* inst) const;
  bool RemoveExtensionByName(const char* extension_name);
  bool RemoveSourceExtensionByName(const char* extension_name);
  void RebuildAsCompositeConstruct(Instruction* inst, uint32_t type_id,
                                   const std::vector<uint32_t>& element_ids);
  void ReportError(const Instruction* inst, const std::string& message) const;

  std::unordered_map<uint32_t, MatrixTypeInfo> matrix_types_;
  std::unordered_map<uint32_t, VectorTypeInfo> vector_types_;
  std::unordered_map<uint32_t, uint32_t> lowered_types_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_AZD_LOWER_TO_STANDARD_PASS_H_
