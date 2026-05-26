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

#ifndef SOURCE_OPT_AZD_FIX_COOPERATIVE_MATRIX_USE_PASS_H_
#define SOURCE_OPT_AZD_FIX_COOPERATIVE_MATRIX_USE_PASS_H_

#include <string>
#include <unordered_map>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Fixes the optional use operand on OpTypeCooperativeMatrixAZD types based on
// how cooperative matrix values are used by AZD matrix multiply instructions.
// A/B uses may be mixed and are resolved by use count. Direct OperandAB and
// Accumulator mixing for the same id is rejected instead of being reinterpreted.
class AzdFixCooperativeMatrixUsePass : public Pass {
 public:
  const char* name() const override {
    return "azd-fix-cooperative-matrix-use";
  }
  Status Process() override;

 private:
  struct MatrixUseStat {
    uint32_t left_count = 0;
    uint32_t right_count = 0;
    uint32_t accumulator_count = 0;
  };

  bool IsAzdCooperativeMatrixType(uint32_t type_id) const;
  bool IsAzdCooperativeMatrixTypeWithoutUse(uint32_t type_id) const;
  bool HasRoleConflict(const MatrixUseStat& stat) const;
  void ReportRoleConflict(uint32_t id) const;
  void AddValueUseStat(uint32_t value_id, uint32_t MatrixUseStat::*field);
  void AddUseStats(MatrixUseStat* target, const MatrixUseStat& source) const;
  void AddDefaultUseAStatsForUnclassifiedMatrices();
  spv::CooperativeMatrixUseAZD InferUse(const MatrixUseStat& stat) const;

  uint32_t GetOrCreateAzdCooperativeMatrixTypeWithUse(
      uint32_t old_type_id, spv::CooperativeMatrixUseAZD use);
  uint32_t GetOrCreatePointerType(uint32_t old_pointer_type_id,
                                  uint32_t pointee_type_id);
  uint32_t GetPointerTypePointeeType(uint32_t pointer_type_id) const;
  uint32_t GetPointerPointeeType(uint32_t pointer_id) const;
  uint32_t GetPreferredPointerPointeeType(uint32_t pointer_id,
                                          uint32_t fallback_type_id) const;
  std::string GetPointerAccessKey(uint32_t pointer_id) const;
  uint32_t GetPreferredStoreObjectType(uint32_t pointer_id,
                                       uint32_t fallback_type_id) const;
  uint32_t GetStorePointerPointeeType(uint32_t pointer_id,
                                      uint32_t fallback_type_id) const;
  void CollectFunctionCallPointerRequirements();
  bool CanRewriteValueType(const Instruction* inst) const;
  bool CanRewritePointerPointeeType(const Instruction* inst) const;
  uint32_t GetDesiredValueType(uint32_t value_id, uint32_t fallback_type_id);

  bool ChangeResultType(Instruction* inst, uint32_t new_type_id);
  bool RewriteValueType(uint32_t value_id, uint32_t new_type_id);
  bool RewritePointerPointeeType(uint32_t pointer_id,
                                 uint32_t pointee_type_id);
  bool FixMulAddOperandTypes();
  bool FixCopyObjectTypeMismatches();
  bool FixStoreTypeMismatches();
  bool FixReturnValueTypeMismatches();
  bool TryRewriteStoreObjectType(uint32_t object_id, uint32_t target_type_id);
  uint32_t InsertBitcastBefore(Instruction* insert_before,
                               uint32_t result_type_id, uint32_t object_id);
  uint32_t EnsureValueTypeBefore(Instruction* insert_before,
                                 uint32_t value_id, uint32_t target_type_id);
  uint32_t InsertBitcastOnPhiEdge(Instruction* phi,
                                  uint32_t incoming_operand_index,
                                  uint32_t result_type_id);
  bool FixPhiIncomingTypes(Instruction* phi, uint32_t result_type_id);
  bool FixSelectOperandTypes(Instruction* select, uint32_t result_type_id);

  std::unordered_map<uint32_t, MatrixUseStat> value_use_stats_;
  std::unordered_map<uint32_t, MatrixUseStat> pointer_use_stats_;
  std::unordered_map<std::string, MatrixUseStat> pointer_key_use_stats_;
  std::unordered_map<uint32_t, uint32_t> pointer_preferred_pointees_;
  std::unordered_map<std::string, uint32_t> pointer_key_preferred_pointees_;
  std::unordered_map<uint32_t, uint32_t> pointer_required_pointees_;
  std::unordered_map<uint32_t, uint32_t> processed_value_types_;
  std::unordered_map<uint32_t, uint32_t> processed_pointer_pointees_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_AZD_FIX_COOPERATIVE_MATRIX_USE_PASS_H_
