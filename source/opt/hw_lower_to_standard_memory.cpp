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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>

#include "source/opcode.h"
#include "source/opt/basic_block.h"
#include "source/opt/constants.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/function.h"
#include "source/opt/hw_fuse_two_layer_vector_matmul_pass.h"
#include "source/opt/hw_lower_to_standard_pass.h"
#include "source/opt/hw_lower_to_standard_pass_internal.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/opt/types.h"
#include "source/util/make_unique.h"
#include "spirv/unified1/GLSL.std.450.h"

namespace spvtools {
namespace opt {

using namespace hw_lower_internal;

bool HwLowerToStandardPass::LowerMatrixLoad(Instruction* inst) {
  const MatrixTypeInfo* info = GetMatrixType(inst->type_id());
  if (!info || inst->NumInOperands() < 4) {
    ReportError(inst, "invalid OpCooperativeMatrixLoadHW");
    return false;
  }

  uint32_t layout = 0;
  if (!GetConstantU32(inst->GetSingleWordInOperand(kHwMatrixLoadLayoutInIdx),
                      &layout) ||
      layout > 1) {
    ReportError(inst, "HW matrix load layout must be RowMajor or ColumnMajor");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;
  element_ids.reserve(info->rows * info->cols);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kHwMatrixLoadPointerInIdx);
  const uint32_t shape_id =
      inst->GetSingleWordInOperand(kHwMatrixLoadShapeInIdx);
  const uint32_t offset_id =
      inst->GetSingleWordInOperand(kHwMatrixLoadOffsetInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kHwMatrixLoadMemoryOperandsInIdx);

  const uint32_t element_count = info->rows * info->cols;
  if (element_count > max_unrolled_elements_) {
    uint32_t result_id = 0;
    if (!BuildScalarMemoryLoop(
            inst, info->lowered_type_id, info->component_type_id, element_count,
            pointer_id, 0, shape_id, offset_id, info->cols, layout, true,
            info->packed_vec2_type_id, memory_operands, &result_id)) {
      return false;
    }
    inst->SetOpcode(spv::Op::OpCopyObject);
    inst->SetResultType(info->lowered_type_id);
    inst->SetInOperands({IdOperand(result_id)});
    context()->UpdateDefUse(inst);
    return true;
  }

  if (IsPackedVec2(*info)) {
    if (layout ==
            static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR) &&
        CanCapturePointer(pointer_id) &&
        MemoryAccessOperandsAreMovable(inst,
                                       kHwMatrixLoadMemoryOperandsInIdx)) {
      const uint32_t pointer_type_id = GetPointerTypeId(pointer_id);
      if (pointer_type_id == 0) return false;

      uint32_t result_id = 0;
      if (!BuildPackedMatrixLoadOuterLoop(
              inst, *info, pointer_id, pointer_type_id, shape_id, offset_id,
              layout, memory_operands, &result_id)) {
        return false;
      }
      inst->SetOpcode(spv::Op::OpCopyObject);
      inst->SetResultType(info->lowered_type_id);
      inst->SetInOperands({IdOperand(result_id)});
      context()->UpdateDefUse(inst);
      return true;
    }

    element_ids.reserve(info->rows * info->packed_cols);
    for (uint32_t row = 0; row < info->rows; ++row) {
      for (uint32_t col_pack = 0; col_pack < info->packed_cols; ++col_pack) {
        std::vector<uint32_t> lane_ids;
        lane_ids.reserve(kPackedVec2Width);
        for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
          const uint32_t col = col_pack * kPackedVec2Width + lane;
          const uint32_t index_id = BuildMatrixElementIndex(
              &builder, inst, *info, shape_id, offset_id, layout, row, col);
          const uint32_t elem_ptr_id = BuildElementAccess(
              &builder, inst, pointer_id, info->component_type_id, index_id);
          if (index_id == 0 || elem_ptr_id == 0) return false;
          const uint32_t load_id = AddLoad(&builder, info->component_type_id,
                                           elem_ptr_id, memory_operands);
          if (load_id == 0) return false;
          lane_ids.push_back(load_id);
        }
        Instruction* vec =
            builder.AddCompositeConstruct(info->packed_vec2_type_id, lane_ids);
        if (!vec) return false;
        element_ids.push_back(vec->result_id());
      }
    }

    RebuildAsCompositeConstruct(inst, info->lowered_type_id, element_ids);
    return true;
  }

  element_ids.reserve(info->rows * info->cols);
  for (uint32_t row = 0; row < info->rows; ++row) {
    for (uint32_t col = 0; col < info->cols; ++col) {
      const uint32_t index_id = BuildMatrixElementIndex(
          &builder, inst, *info, shape_id, offset_id, layout, row, col);
      const uint32_t elem_ptr_id = BuildElementAccess(
          &builder, inst, pointer_id, info->component_type_id, index_id);
      if (index_id == 0 || elem_ptr_id == 0) return false;
      const uint32_t load_id = AddLoad(&builder, info->component_type_id,
                                       elem_ptr_id, memory_operands);
      if (load_id == 0) return false;
      element_ids.push_back(load_id);
    }
  }

  RebuildAsCompositeConstruct(inst, info->lowered_type_id, element_ids);
  return true;
}

bool HwLowerToStandardPass::LowerMatrixStore(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 5) {
    ReportError(inst, "invalid OpCooperativeMatrixStoreHW");
    return false;
  }

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwMatrixStoreObjectInIdx));
  const MatrixTypeInfo* info = GetMatrixTypeForValue(object);
  if (!info) {
    ReportError(inst, "invalid HW matrix store object");
    return false;
  }

  uint32_t layout = 0;
  if (!GetConstantU32(inst->GetSingleWordInOperand(kHwMatrixStoreLayoutInIdx),
                      &layout) ||
      layout > 1) {
    ReportError(inst, "HW matrix store layout must be RowMajor or ColumnMajor");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kHwMatrixStorePointerInIdx);
  const uint32_t object_id =
      inst->GetSingleWordInOperand(kHwMatrixStoreObjectInIdx);
  const uint32_t shape_id =
      inst->GetSingleWordInOperand(kHwMatrixStoreShapeInIdx);
  const uint32_t offset_id =
      inst->GetSingleWordInOperand(kHwMatrixStoreOffsetInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kHwMatrixStoreMemoryOperandsInIdx);

  const uint32_t element_count = info->rows * info->cols;
  if (element_count > max_unrolled_elements_) {
    if (!BuildScalarMemoryLoop(
            inst, info->lowered_type_id, info->component_type_id, element_count,
            pointer_id, object_id, shape_id, offset_id, info->cols, layout,
            true, info->packed_vec2_type_id, memory_operands, nullptr)) {
      return false;
    }
    to_kill->push_back(inst);
    return true;
  }

  if (IsPackedVec2(*info)) {
    bool fused = false;
    if (!TryLowerFusedMatrixMatmulStore(inst, &fused)) return false;
    if (fused) return true;

    if (layout ==
            static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR) &&
        CanCapturePointer(pointer_id) &&
        MemoryAccessOperandsAreMovable(inst,
                                       kHwMatrixStoreMemoryOperandsInIdx)) {
      const uint32_t pointer_type_id = GetPointerTypeId(pointer_id);
      if (pointer_type_id == 0) return false;

      if (!BuildPackedMatrixStoreOuterLoop(
              inst, *info, pointer_id, pointer_type_id, object_id, shape_id,
              offset_id, layout, memory_operands)) {
        return false;
      }
      to_kill->push_back(inst);
      return true;
    }

    for (uint32_t row = 0; row < info->rows; ++row) {
      for (uint32_t col_pack = 0; col_pack < info->packed_cols; ++col_pack) {
        const uint32_t vec_id = ExtractCompositeElement(
            &builder, info->packed_vec2_type_id, object_id,
            MatrixPackedIndex(*info, row, col_pack));
        if (vec_id == 0) return false;
        for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
          const uint32_t col = col_pack * kPackedVec2Width + lane;
          const uint32_t value_id = ExtractCompositeElement(
              &builder, info->component_type_id, vec_id, lane);
          const uint32_t index_id = BuildMatrixElementIndex(
              &builder, inst, *info, shape_id, offset_id, layout, row, col);
          const uint32_t elem_ptr_id = BuildElementAccess(
              &builder, inst, pointer_id, info->component_type_id, index_id);
          if (value_id == 0 || index_id == 0 || elem_ptr_id == 0) {
            return false;
          }
          if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands,
                        info->component_type_id)) {
            return false;
          }
        }
      }
    }

    to_kill->push_back(inst);
    return true;
  }

  for (uint32_t row = 0; row < info->rows; ++row) {
    for (uint32_t col = 0; col < info->cols; ++col) {
      const uint32_t flat_index = row * info->cols + col;
      const uint32_t value_id = ExtractCompositeElement(
          &builder, info->component_type_id, object_id, flat_index);
      const uint32_t index_id = BuildMatrixElementIndex(
          &builder, inst, *info, shape_id, offset_id, layout, row, col);
      const uint32_t elem_ptr_id = BuildElementAccess(
          &builder, inst, pointer_id, info->component_type_id, index_id);
      if (value_id == 0 || index_id == 0 || elem_ptr_id == 0) return false;
      if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands,
                    info->component_type_id)) {
        return false;
      }
    }
  }

  to_kill->push_back(inst);
  return true;
}

bool HwLowerToStandardPass::LowerVectorLoad(Instruction* inst) {
  const VectorTypeInfo* info = GetVectorType(inst->type_id());
  if (!info || inst->NumInOperands() < 1) {
    ReportError(inst, "invalid OpCooperativeVectorLoadHW");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  std::vector<uint32_t> element_ids;
  element_ids.reserve(info->length);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kHwVectorLoadPointerInIdx);
  const uint32_t offset_id =
      inst->GetSingleWordInOperand(kHwVectorLoadOffsetInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kHwVectorLoadMemoryOperandsInIdx);

  if (info->length > max_unrolled_elements_) {
    uint32_t result_id = 0;
    if (!BuildScalarMemoryLoop(
            inst, info->lowered_type_id, info->component_type_id, info->length,
            pointer_id, 0, 0, offset_id, 0, 0, false, info->packed_vec2_type_id,
            memory_operands, &result_id)) {
      return false;
    }
    inst->SetOpcode(spv::Op::OpCopyObject);
    inst->SetResultType(info->lowered_type_id);
    inst->SetInOperands({IdOperand(result_id)});
    context()->UpdateDefUse(inst);
    return true;
  }

  if (IsPackedVec2(*info)) {
    const uint32_t pointer_type_id = GetPointerTypeId(pointer_id);
    if (pointer_type_id == 0) return false;

    Instruction* offset = get_def_use_mgr()->GetDef(offset_id);
    Instruction* offset_type =
        offset ? get_def_use_mgr()->GetDef(offset->type_id()) : nullptr;
    const bool has_32_bit_offset =
        offset_type && offset_type->opcode() == spv::Op::OpTypeInt &&
        offset_type->NumInOperands() >= 2 &&
        offset_type->GetSingleWordInOperand(0) == 32;
    if (has_32_bit_offset && CanCapturePointer(pointer_id) &&
        MemoryAccessOperandsAreMovable(inst,
                                       kHwVectorLoadMemoryOperandsInIdx)) {
      uint32_t result_id = 0;
      if (!BuildPackedVectorLoadOuterLoop(inst, *info, pointer_id,
                                          pointer_type_id, offset_id,
                                          memory_operands, &result_id)) {
        return false;
      }
      inst->SetOpcode(spv::Op::OpCopyObject);
      inst->SetResultType(info->lowered_type_id);
      inst->SetInOperands({IdOperand(result_id)});
      context()->UpdateDefUse(inst);
      return true;
    }

    element_ids.reserve(info->packed_length);
    for (uint32_t pack = 0; pack < info->packed_length; ++pack) {
      std::vector<uint32_t> lane_ids;
      lane_ids.reserve(kPackedVec2Width);
      for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
        const uint32_t logical_index_id =
            GetOrCreateUIntConstant(pack * kPackedVec2Width + lane);
        const uint32_t index_id = BuildVectorElementIndex(
            &builder, inst, offset_id, logical_index_id);
        const uint32_t elem_ptr_id = BuildElementAccess(
            &builder, inst, pointer_id, info->component_type_id, index_id);
        if (index_id == 0 || elem_ptr_id == 0) return false;
        const uint32_t load_id = AddLoad(&builder, info->component_type_id,
                                         elem_ptr_id, memory_operands);
        if (load_id == 0) return false;
        lane_ids.push_back(load_id);
      }
      Instruction* vec =
          builder.AddCompositeConstruct(info->packed_vec2_type_id, lane_ids);
      if (!vec) return false;
      element_ids.push_back(vec->result_id());
    }

    RebuildAsCompositeConstruct(inst, info->lowered_type_id, element_ids);
    return true;
  }

  for (uint32_t i = 0; i < info->length; ++i) {
    const uint32_t logical_index_id = GetOrCreateUIntConstant(i);
    const uint32_t index_id =
        BuildVectorElementIndex(&builder, inst, offset_id, logical_index_id);
    const uint32_t elem_ptr_id = BuildElementAccess(
        &builder, inst, pointer_id, info->component_type_id, index_id);
    if (elem_ptr_id == 0) return false;
    const uint32_t load_id = AddLoad(&builder, info->component_type_id,
                                     elem_ptr_id, memory_operands);
    if (load_id == 0) return false;
    element_ids.push_back(load_id);
  }

  RebuildAsCompositeConstruct(inst, info->lowered_type_id, element_ids);
  return true;
}

bool HwLowerToStandardPass::LowerVectorStore(
    Instruction* inst, std::vector<Instruction*>* to_kill) {
  if (inst->NumInOperands() < 2) {
    ReportError(inst, "invalid OpCooperativeVectorStoreHW");
    return false;
  }

  Instruction* object = get_def_use_mgr()->GetDef(
      inst->GetSingleWordInOperand(kHwVectorStoreObjectInIdx));
  const VectorTypeInfo* info =
      object ? GetVectorType(object->type_id()) : nullptr;
  if (!info) {
    ReportError(inst, "invalid HW vector store object");
    return false;
  }

  InstructionBuilder builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t pointer_id =
      inst->GetSingleWordInOperand(kHwVectorStorePointerInIdx);
  const uint32_t object_id =
      inst->GetSingleWordInOperand(kHwVectorStoreObjectInIdx);
  const uint32_t offset_id =
      inst->GetSingleWordInOperand(kHwVectorStoreOffsetInIdx);
  const std::vector<Operand> memory_operands =
      CopyMemoryOperands(inst, kHwVectorStoreMemoryOperandsInIdx);

  if (info->length > max_unrolled_elements_) {
    if (!BuildScalarMemoryLoop(
            inst, info->lowered_type_id, info->component_type_id, info->length,
            pointer_id, object_id, 0, offset_id, 0, 0, false,
            info->packed_vec2_type_id, memory_operands, nullptr)) {
      return false;
    }
    to_kill->push_back(inst);
    return true;
  }

  if (IsPackedVec2(*info)) {
    bool fused = false;
    if (!TryLowerFusedVectorMatmulStore(inst, &fused)) return false;
    if (fused) return true;

    const uint32_t pointer_type_id = GetPointerTypeId(pointer_id);
    if (pointer_type_id == 0) return false;

    Instruction* offset = get_def_use_mgr()->GetDef(offset_id);
    Instruction* offset_type =
        offset ? get_def_use_mgr()->GetDef(offset->type_id()) : nullptr;
    const bool has_32_bit_offset =
        offset_type && offset_type->opcode() == spv::Op::OpTypeInt &&
        offset_type->NumInOperands() >= 2 &&
        offset_type->GetSingleWordInOperand(0) == 32;
    if (has_32_bit_offset && CanCapturePointer(pointer_id) &&
        MemoryAccessOperandsAreMovable(inst,
                                       kHwVectorStoreMemoryOperandsInIdx)) {
      if (!BuildPackedVectorStoreOuterLoop(inst, *info, pointer_id,
                                           pointer_type_id, offset_id,
                                           object_id, memory_operands)) {
        return false;
      }
      to_kill->push_back(inst);
      return true;
    }

    for (uint32_t pack = 0; pack < info->packed_length; ++pack) {
      const uint32_t vec_id = ExtractCompositeElement(
          &builder, info->packed_vec2_type_id, object_id, pack);
      if (vec_id == 0) return false;
      for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
        const uint32_t value_id = ExtractCompositeElement(
            &builder, info->component_type_id, vec_id, lane);
        const uint32_t logical_index_id =
            GetOrCreateUIntConstant(pack * kPackedVec2Width + lane);
        const uint32_t index_id = BuildVectorElementIndex(
            &builder, inst, offset_id, logical_index_id);
        const uint32_t elem_ptr_id = BuildElementAccess(
            &builder, inst, pointer_id, info->component_type_id, index_id);
        if (value_id == 0 || index_id == 0 || elem_ptr_id == 0) return false;
        if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands,
                      info->component_type_id)) {
          return false;
        }
      }
    }

    to_kill->push_back(inst);
    return true;
  }

  for (uint32_t i = 0; i < info->length; ++i) {
    const uint32_t value_id = ExtractCompositeElement(
        &builder, info->component_type_id, object_id, i);
    const uint32_t logical_index_id = GetOrCreateUIntConstant(i);
    const uint32_t index_id =
        BuildVectorElementIndex(&builder, inst, offset_id, logical_index_id);
    const uint32_t elem_ptr_id = BuildElementAccess(
        &builder, inst, pointer_id, info->component_type_id, index_id);
    if (value_id == 0 || elem_ptr_id == 0) return false;
    if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands,
                  info->component_type_id)) {
      return false;
    }
  }

  to_kill->push_back(inst);
  return true;
}

uint32_t HwLowerToStandardPass::BuildPairComponentAsUInt(
    InstructionBuilder* builder, Instruction* user, uint32_t pair_id,
    uint32_t component_index) {
  Instruction* pair = get_def_use_mgr()->GetDef(pair_id);
  if (!pair || pair->type_id() == 0) {
    ReportError(user, "HW matrix shape/offset must be a two-component value");
    return 0;
  }

  Instruction* pair_type = get_def_use_mgr()->GetDef(pair->type_id());
  if (!pair_type || pair_type->opcode() != spv::Op::OpTypeVector ||
      pair_type->GetSingleWordInOperand(1) < 2) {
    ReportError(user, "HW matrix shape/offset must be a two-component vector");
    return 0;
  }

  const uint32_t component_type_id = pair_type->GetSingleWordInOperand(0);
  Instruction* extract = builder->AddCompositeExtract(
      component_type_id, pair_id, {component_index});
  if (!extract) return 0;

  const uint32_t uint_type_id = GetOrCreateUIntType();
  if (component_type_id == uint_type_id) return extract->result_id();

  Instruction* component_type = get_def_use_mgr()->GetDef(component_type_id);
  if (!component_type) return 0;
  if (component_type->opcode() == spv::Op::OpTypeFloat &&
      component_type->GetSingleWordInOperand(0) == 32) {
    Instruction* converted = builder->AddUnaryOp(
        uint_type_id, spv::Op::OpConvertFToU, extract->result_id());
    return converted ? converted->result_id() : 0;
  }
  if (component_type->opcode() == spv::Op::OpTypeInt &&
      component_type->GetSingleWordInOperand(0) == 32) {
    Instruction* converted = builder->AddUnaryOp(
        uint_type_id, spv::Op::OpBitcast, extract->result_id());
    return converted ? converted->result_id() : 0;
  }

  ReportError(user, "HW matrix shape/offset component type is unsupported");
  return 0;
}

uint32_t HwLowerToStandardPass::BuildMatrixElementIndex(
    InstructionBuilder* builder, Instruction* user, const MatrixTypeInfo& info,
    uint32_t shape_id, uint32_t offset_id, uint32_t layout, uint32_t row,
    uint32_t col) {
  const uint32_t shape_rows =
      BuildPairComponentAsUInt(builder, user, shape_id, 0);
  const uint32_t shape_cols =
      BuildPairComponentAsUInt(builder, user, shape_id, 1);
  const uint32_t offset_row =
      BuildPairComponentAsUInt(builder, user, offset_id, 0);
  const uint32_t offset_col =
      BuildPairComponentAsUInt(builder, user, offset_id, 1);
  if (shape_rows == 0 || shape_cols == 0 || offset_row == 0 ||
      offset_col == 0) {
    return 0;
  }

  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t row_const_id = GetOrCreateUIntConstant(row);
  const uint32_t col_const_id = GetOrCreateUIntConstant(col);
  Instruction* global_row = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_row, row_const_id);
  Instruction* global_col = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_col, col_const_id);
  if (!global_row || !global_col) return 0;

  Instruction* major_mul = nullptr;
  if (layout ==
      static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    major_mul = builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                     global_row->result_id(), shape_cols);
    if (!major_mul) return 0;
    Instruction* index =
        builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                             major_mul->result_id(), global_col->result_id());
    return index ? index->result_id() : 0;
  }

  (void)info;
  major_mul = builder->AddBinaryOp(uint_type_id, spv::Op::OpIMul,
                                   global_col->result_id(), shape_rows);
  if (!major_mul) return 0;
  Instruction* index =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                           major_mul->result_id(), global_row->result_id());
  return index ? index->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildDynamicMatrixElementIndex(
    InstructionBuilder* builder, Instruction* user, uint32_t shape_id,
    uint32_t offset_id, uint32_t layout, uint32_t matrix_cols,
    uint32_t flat_index_id) {
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t cols_id = GetOrCreateUIntConstant(matrix_cols);
  if (uint_type_id == 0 || cols_id == 0 || flat_index_id == 0) return 0;
  Instruction* row = builder->AddBinaryOp(uint_type_id, spv::Op::OpUDiv,
                                          flat_index_id, cols_id);
  Instruction* col = builder->AddBinaryOp(uint_type_id, spv::Op::OpUMod,
                                          flat_index_id, cols_id);
  if (!row || !col) return 0;

  const uint32_t shape_rows =
      BuildPairComponentAsUInt(builder, user, shape_id, 0);
  const uint32_t shape_cols =
      BuildPairComponentAsUInt(builder, user, shape_id, 1);
  const uint32_t offset_row =
      BuildPairComponentAsUInt(builder, user, offset_id, 0);
  const uint32_t offset_col =
      BuildPairComponentAsUInt(builder, user, offset_id, 1);
  if (shape_rows == 0 || shape_cols == 0 || offset_row == 0 ||
      offset_col == 0) {
    return 0;
  }
  Instruction* global_row = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_row, row->result_id());
  Instruction* global_col = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_col, col->result_id());
  if (!global_row || !global_col) return 0;
  const bool row_major =
      layout ==
      static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR);
  Instruction* major = builder->AddBinaryOp(
      uint_type_id, spv::Op::OpIMul,
      row_major ? global_row->result_id() : global_col->result_id(),
      row_major ? shape_cols : shape_rows);
  Instruction* index =
      major ? builder->AddBinaryOp(
                  uint_type_id, spv::Op::OpIAdd, major->result_id(),
                  row_major ? global_col->result_id() : global_row->result_id())
            : nullptr;
  return index ? index->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildVectorElementIndex(
    InstructionBuilder* builder, Instruction* user, uint32_t offset_id,
    uint32_t logical_index_id) {
  Instruction* offset = get_def_use_mgr()->GetDef(offset_id);
  Instruction* offset_type =
      offset ? get_def_use_mgr()->GetDef(offset->type_id()) : nullptr;
  Instruction* logical_index = get_def_use_mgr()->GetDef(logical_index_id);
  if (!builder || !offset || !offset_type || !logical_index ||
      offset_type->opcode() != spv::Op::OpTypeInt ||
      offset_type->NumInOperands() < 2 ||
      offset_type->GetSingleWordInOperand(0) == 0 ||
      offset_type->GetSingleWordInOperand(0) > 64) {
    ReportError(user,
                "HW cooperative vector offset must be an integer of at most "
                "64 bits");
    return 0;
  }
  if (offset_type->GetSingleWordInOperand(0) == 32) {
    const uint32_t uint_type_id = GetOrCreateUIntType();
    uint32_t unsigned_offset_id = offset_id;
    if (offset->type_id() != uint_type_id) {
      Instruction* converted =
          builder->AddUnaryOp(uint_type_id, spv::Op::OpBitcast, offset_id);
      if (!converted) return 0;
      unsigned_offset_id = converted->result_id();
    }
    Instruction* index = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpIAdd, unsigned_offset_id, logical_index_id);
    return index ? index->result_id() : 0;
  }
  uint32_t converted_logical_index_id = logical_index_id;
  if (logical_index->type_id() != offset->type_id()) {
    Instruction* logical_type =
        get_def_use_mgr()->GetDef(logical_index->type_id());
    if (!logical_type || logical_type->opcode() != spv::Op::OpTypeInt ||
        logical_type->NumInOperands() < 2) {
      return 0;
    }
    const uint32_t source_width = logical_type->GetSingleWordInOperand(0);
    const uint32_t target_width = offset_type->GetSingleWordInOperand(0);
    const bool target_is_signed = offset_type->GetSingleWordInOperand(1) != 0;
    if (source_width == target_width) {
      Instruction* converted = builder->AddUnaryOp(
          offset->type_id(), spv::Op::OpBitcast, logical_index_id);
      if (!converted) return 0;
      converted_logical_index_id = converted->result_id();
    } else {
      const uint32_t unsigned_target_type_id =
          target_is_signed
              ? GetOrCreateIntegerType(target_width, /*is_signed=*/false)
              : offset->type_id();
      Instruction* converted = builder->AddUnaryOp(
          unsigned_target_type_id, spv::Op::OpUConvert, logical_index_id);
      if (!converted) return 0;
      converted_logical_index_id = converted->result_id();
      if (target_is_signed) {
        Instruction* bitcast = builder->AddUnaryOp(
            offset->type_id(), spv::Op::OpBitcast, converted_logical_index_id);
        if (!bitcast) return 0;
        converted_logical_index_id = bitcast->result_id();
      }
    }
  }
  Instruction* index =
      builder->AddBinaryOp(offset->type_id(), spv::Op::OpIAdd, offset_id,
                           converted_logical_index_id);
  return index ? index->result_id() : 0;
}

uint32_t HwLowerToStandardPass::BuildElementAccess(InstructionBuilder* builder,
                                                   Instruction* user,
                                                   uint32_t pointer_id,
                                                   uint32_t component_type_id,
                                                   uint32_t element_index_id) {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer || pointer->type_id() == 0) {
    ReportError(user, "HW load/store pointer is invalid");
    return 0;
  }
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer->type_id());
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    ReportError(user, "HW load/store pointer must be a pointer");
    return 0;
  }

  const uint32_t pointee_type_id = pointer_type->GetSingleWordInOperand(1);
  Instruction* pointee_type = get_def_use_mgr()->GetDef(pointee_type_id);
  if (!pointee_type) return 0;

  if (pointee_type_id == component_type_id) {
    uint32_t index_value = 0;
    if (GetConstantU32(element_index_id, &index_value) && index_value == 0) {
      return pointer_id;
    }
    ReportError(user,
                "HW scalar pointer load/store only supports element zero");
    return 0;
  }

  if ((pointee_type->opcode() == spv::Op::OpTypeRuntimeArray ||
       pointee_type->opcode() == spv::Op::OpTypeArray) &&
      pointee_type->GetSingleWordInOperand(0) == component_type_id) {
    const uint32_t storage_class = pointer_type->GetSingleWordInOperand(0);
    const uint32_t component_pointer_type_id = GetOrCreatePointerType(
        component_type_id, static_cast<spv::StorageClass>(storage_class));
    Instruction* access = builder->AddAccessChain(
        component_pointer_type_id, pointer_id, {element_index_id});
    return access ? access->result_id() : 0;
  }

  ReportError(user,
              "HW load/store pointer must point to the component or a "
              "component array");
  return 0;
}

uint32_t HwLowerToStandardPass::BuildElementAccessFromPointerType(
    InstructionBuilder* builder, uint32_t pointer_type_id, uint32_t pointer_id,
    uint32_t component_type_id, uint32_t element_index_id) {
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer_type_id);
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer) {
    ReportError(nullptr, "HW load/store pointer must be a pointer");
    return 0;
  }

  const uint32_t pointee_type_id = pointer_type->GetSingleWordInOperand(1);
  Instruction* pointee_type = get_def_use_mgr()->GetDef(pointee_type_id);
  if (!pointee_type) return 0;

  if (pointee_type_id == component_type_id) {
    ReportError(nullptr,
                "HW scalar pointer load/store cannot be chunk-lowered");
    return 0;
  }

  if ((pointee_type->opcode() == spv::Op::OpTypeRuntimeArray ||
       pointee_type->opcode() == spv::Op::OpTypeArray) &&
      pointee_type->GetSingleWordInOperand(0) == component_type_id) {
    const uint32_t storage_class = pointer_type->GetSingleWordInOperand(0);
    const uint32_t component_pointer_type_id = GetOrCreatePointerType(
        component_type_id, static_cast<spv::StorageClass>(storage_class));
    Instruction* access = builder->AddAccessChain(
        component_pointer_type_id, pointer_id, {element_index_id});
    return access ? access->result_id() : 0;
  }

  ReportError(nullptr,
              "HW load/store pointer must point to the component or a "
              "component array");
  return 0;
}

Instruction* HwLowerToStandardPass::AddFunctionVariable(
    Function* function, uint32_t pointer_type_id, uint32_t initializer_id) {
  if (!function || function->begin() == function->end()) return nullptr;

  BasicBlock* entry_block = &*function->begin();
  auto insert_iter = entry_block->begin();
  while (insert_iter != entry_block->end() &&
         insert_iter->opcode() == spv::Op::OpVariable) {
    ++insert_iter;
  }

  const uint32_t var_id = TakeNextId();
  if (var_id == 0) return nullptr;
  std::unique_ptr<Instruction> variable = MakeUnique<Instruction>(
      context(), spv::Op::OpVariable, pointer_type_id, var_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS,
           {uint32_t(spv::StorageClass::Function)}}});
  Instruction* variable_ptr = variable.get();
  insert_iter.InsertBefore(std::move(variable));
  context()->AnalyzeDefUse(variable_ptr);
  context()->set_instr_block(variable_ptr, entry_block);

  if (initializer_id != 0) {
    InstructionBuilder builder(
        context(), entry_block,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    if (!builder.AddStore(var_id, initializer_id)) return nullptr;
  }
  return variable_ptr;
}

uint32_t HwLowerToStandardPass::BuildLogicalAggregateLoad(
    InstructionBuilder* builder, uint32_t aggregate_pointer_id,
    uint32_t component_type_id, uint32_t packed_vec2_type_id,
    uint32_t logical_index_id) {
  if (!builder || aggregate_pointer_id == 0 || component_type_id == 0 ||
      logical_index_id == 0) {
    return 0;
  }
  uint32_t piece_index_id = logical_index_id;
  uint32_t lane_id = 0;
  uint32_t piece_type_id = component_type_id;
  if (packed_vec2_type_id != 0) {
    const uint32_t uint_type_id = GetOrCreateUIntType();
    const uint32_t packed_width_id = GetOrCreateUIntConstant(kPackedVec2Width);
    Instruction* piece_index = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpUDiv, logical_index_id, packed_width_id);
    Instruction* lane = builder->AddBinaryOp(uint_type_id, spv::Op::OpUMod,
                                             logical_index_id, packed_width_id);
    if (!piece_index || !lane) return 0;
    piece_index_id = piece_index->result_id();
    lane_id = lane->result_id();
    piece_type_id = packed_vec2_type_id;
  }
  const uint32_t piece_pointer_type_id =
      GetOrCreatePointerType(piece_type_id, spv::StorageClass::Function);
  Instruction* pointer = builder->AddAccessChain(
      piece_pointer_type_id, aggregate_pointer_id, {piece_index_id});
  Instruction* piece =
      pointer ? builder->AddLoad(piece_type_id, pointer->result_id()) : nullptr;
  if (!piece) return 0;
  if (packed_vec2_type_id == 0) return piece->result_id();
  Instruction* value =
      builder->AddBinaryOp(component_type_id, spv::Op::OpVectorExtractDynamic,
                           piece->result_id(), lane_id);
  return value ? value->result_id() : 0;
}

bool HwLowerToStandardPass::BuildLogicalAggregateStore(
    InstructionBuilder* builder, uint32_t aggregate_pointer_id,
    uint32_t component_type_id, uint32_t packed_vec2_type_id,
    uint32_t logical_index_id, uint32_t value_id) {
  if (!builder || aggregate_pointer_id == 0 || component_type_id == 0 ||
      logical_index_id == 0 || value_id == 0) {
    return false;
  }
  uint32_t piece_index_id = logical_index_id;
  uint32_t lane_id = 0;
  uint32_t piece_type_id = component_type_id;
  if (packed_vec2_type_id != 0) {
    const uint32_t uint_type_id = GetOrCreateUIntType();
    const uint32_t packed_width_id = GetOrCreateUIntConstant(kPackedVec2Width);
    Instruction* piece_index = builder->AddBinaryOp(
        uint_type_id, spv::Op::OpUDiv, logical_index_id, packed_width_id);
    Instruction* lane = builder->AddBinaryOp(uint_type_id, spv::Op::OpUMod,
                                             logical_index_id, packed_width_id);
    if (!piece_index || !lane) return false;
    piece_index_id = piece_index->result_id();
    lane_id = lane->result_id();
    piece_type_id = packed_vec2_type_id;
  }
  const uint32_t piece_pointer_type_id =
      GetOrCreatePointerType(piece_type_id, spv::StorageClass::Function);
  Instruction* pointer = builder->AddAccessChain(
      piece_pointer_type_id, aggregate_pointer_id, {piece_index_id});
  if (!pointer) return false;
  uint32_t stored_id = value_id;
  if (packed_vec2_type_id != 0) {
    Instruction* old_piece =
        builder->AddLoad(piece_type_id, pointer->result_id());
    Instruction* inserted =
        old_piece ? builder->AddTernaryOp(
                        piece_type_id, spv::Op::OpVectorInsertDynamic,
                        old_piece->result_id(), value_id, lane_id)
                  : nullptr;
    stored_id = inserted ? inserted->result_id() : 0;
  }
  return stored_id != 0 && builder->AddStore(pointer->result_id(), stored_id);
}

BasicBlock* HwLowerToStandardPass::MakeBasicBlock(uint32_t label_id) {
  if (label_id == 0) return nullptr;
  BasicBlock* block = new BasicBlock(
      MakeUnique<Instruction>(context(), spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));
  context()->AnalyzeDefUse(block->GetLabelInst());
  context()->set_instr_block(block->GetLabelInst(), block);
  return block;
}

uint32_t HwLowerToStandardPass::BuildRowMajorMatrixMemoryIndex(
    InstructionBuilder* builder, Instruction* user, uint32_t shape_id,
    uint32_t offset_id, uint32_t cols, uint32_t base_id) {
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t cols_id = GetOrCreateUIntConstant(cols);
  if (uint_type_id == 0 || cols_id == 0) return 0;

  const uint32_t shape_cols =
      BuildPairComponentAsUInt(builder, user, shape_id, 1);
  const uint32_t offset_row =
      BuildPairComponentAsUInt(builder, user, offset_id, 0);
  const uint32_t offset_col =
      BuildPairComponentAsUInt(builder, user, offset_id, 1);
  if (shape_cols == 0 || offset_row == 0 || offset_col == 0) return 0;

  Instruction* row =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpUDiv, base_id, cols_id);
  Instruction* col =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpUMod, base_id, cols_id);
  if (!row || !col) return 0;

  Instruction* global_row = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_row, row->result_id());
  Instruction* global_col = builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                                                 offset_col, col->result_id());
  if (!global_row || !global_col) return 0;

  Instruction* major_mul = builder->AddBinaryOp(
      uint_type_id, spv::Op::OpIMul, global_row->result_id(), shape_cols);
  if (!major_mul) return 0;
  Instruction* index =
      builder->AddBinaryOp(uint_type_id, spv::Op::OpIAdd,
                           major_mul->result_id(), global_col->result_id());
  return index ? index->result_id() : 0;
}

bool HwLowerToStandardPass::BuildPackedMatrixLoadOuterLoop(
    Instruction* insert_before, const MatrixTypeInfo& info, uint32_t pointer_id,
    uint32_t pointer_type_id, uint32_t shape_id, uint32_t offset_id,
    uint32_t layout, const std::vector<Operand>& memory_operands,
    uint32_t* result_id) {
  if (!insert_before || !result_id || !IsPackedVec2(info)) return false;
  if (layout !=
      static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    return false;
  }
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t load_function_id = GetOrCreatePackedLoadChunkFunction(
      pointer_id, pointer_type_id, info.component_type_id,
      info.packed_vec2_type_id, memory_operands);
  const uint32_t lowered_function_ptr_type_id =
      GetOrCreatePointerType(info.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec2_function_ptr_type_id = GetOrCreatePointerType(
      info.packed_vec2_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t element_count_id =
      GetOrCreateUIntConstant(info.rows * info.cols);
  if (load_function_id == 0 || lowered_function_ptr_type_id == 0 ||
      vec2_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || packed_width_uint_id == 0 || element_count_id == 0) {
    return false;
  }

  Instruction* result_var =
      AddFunctionVariable(function, lowered_function_ptr_type_id);
  Instruction* base_var =
      AddFunctionVariable(function, uint_function_ptr_type_id);
  if (!result_var || !base_var) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }

  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(base_var->result_id(), zero_uint_id)) {
    return false;
  }
  if (!preheader_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* base_load =
      header_builder.AddLoad(uint_type_id, base_var->result_id());
  if (!base_load) return false;
  Instruction* cond =
      header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                 base_load->result_id(), element_count_id);
  if (!cond) return false;
  if (!header_builder.AddLoopMerge(merge_label_id, continue_label_id)) {
    return false;
  }
  if (!header_builder.AddConditionalBranch(cond->result_id(), body_label_id,
                                           merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t memory_base_id = BuildRowMajorMatrixMemoryIndex(
      &body_builder, insert_before, shape_id, offset_id, info.cols,
      base_load->result_id());
  if (memory_base_id == 0) return false;
  Instruction* vec = body_builder.AddFunctionCall(
      info.packed_vec2_type_id, load_function_id, {memory_base_id});
  if (!vec) return false;
  Instruction* packed_index =
      body_builder.AddBinaryOp(uint_type_id, spv::Op::OpUDiv,
                               base_load->result_id(), packed_width_uint_id);
  if (!packed_index) return false;
  Instruction* result_elem_ptr = body_builder.AddAccessChain(
      vec2_function_ptr_type_id, result_var->result_id(),
      {packed_index->result_id()});
  if (!result_elem_ptr) return false;
  if (!body_builder.AddStore(result_elem_ptr->result_id(), vec->result_id())) {
    return false;
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_i = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, base_load->result_id(),
      packed_width_uint_id);
  if (!next_i) return false;
  if (!continue_builder.AddStore(base_var->result_id(), next_i->result_id())) {
    return false;
  }
  if (!continue_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder merge_builder(
      context(), insert_before,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* result =
      merge_builder.AddLoad(info.lowered_type_id, result_var->result_id());
  if (!result) return false;
  *result_id = result->result_id();
  return true;
}

bool HwLowerToStandardPass::BuildPackedMatrixStoreOuterLoop(
    Instruction* insert_before, const MatrixTypeInfo& info, uint32_t pointer_id,
    uint32_t pointer_type_id, uint32_t object_id, uint32_t shape_id,
    uint32_t offset_id, uint32_t layout,
    const std::vector<Operand>& memory_operands) {
  if (!insert_before || object_id == 0 || !IsPackedVec2(info)) return false;
  if (layout !=
      static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    return false;
  }
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t store_function_id = GetOrCreatePackedStoreChunkFunction(
      pointer_id, pointer_type_id, info.component_type_id,
      info.packed_vec2_type_id, memory_operands);
  const uint32_t lowered_function_ptr_type_id =
      GetOrCreatePointerType(info.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec2_function_ptr_type_id = GetOrCreatePointerType(
      info.packed_vec2_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t element_count_id =
      GetOrCreateUIntConstant(info.rows * info.cols);
  if (store_function_id == 0 || lowered_function_ptr_type_id == 0 ||
      vec2_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || void_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || packed_width_uint_id == 0 ||
      element_count_id == 0) {
    return false;
  }

  Instruction* object_var =
      AddFunctionVariable(function, lowered_function_ptr_type_id);
  Instruction* base_var =
      AddFunctionVariable(function, uint_function_ptr_type_id);
  if (!object_var || !base_var) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }

  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(object_var->result_id(), object_id)) {
    return false;
  }
  if (!preheader_builder.AddStore(base_var->result_id(), zero_uint_id)) {
    return false;
  }
  if (!preheader_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* base_load =
      header_builder.AddLoad(uint_type_id, base_var->result_id());
  if (!base_load) return false;
  Instruction* cond =
      header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                 base_load->result_id(), element_count_id);
  if (!cond) return false;
  if (!header_builder.AddLoopMerge(merge_label_id, continue_label_id)) {
    return false;
  }
  if (!header_builder.AddConditionalBranch(cond->result_id(), body_label_id,
                                           merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* packed_index =
      body_builder.AddBinaryOp(uint_type_id, spv::Op::OpUDiv,
                               base_load->result_id(), packed_width_uint_id);
  if (!packed_index) return false;
  Instruction* value_elem_ptr = body_builder.AddAccessChain(
      vec2_function_ptr_type_id, object_var->result_id(),
      {packed_index->result_id()});
  if (!value_elem_ptr) return false;
  Instruction* vec = body_builder.AddLoad(info.packed_vec2_type_id,
                                          value_elem_ptr->result_id());
  if (!vec) return false;
  const uint32_t memory_base_id = BuildRowMajorMatrixMemoryIndex(
      &body_builder, insert_before, shape_id, offset_id, info.cols,
      base_load->result_id());
  if (memory_base_id == 0) return false;
  if (!body_builder.AddFunctionCall(void_type_id, store_function_id,
                                    {memory_base_id, vec->result_id()})) {
    return false;
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_i = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, base_load->result_id(),
      packed_width_uint_id);
  if (!next_i) return false;
  if (!continue_builder.AddStore(base_var->result_id(), next_i->result_id())) {
    return false;
  }
  if (!continue_builder.AddBranch(header_label_id)) return false;
  return true;
}

bool HwLowerToStandardPass::BuildPackedVectorLoadOuterLoop(
    Instruction* insert_before, const VectorTypeInfo& info, uint32_t pointer_id,
    uint32_t pointer_type_id, uint32_t offset_id,
    const std::vector<Operand>& memory_operands, uint32_t* result_id) {
  if (!insert_before || !result_id || !IsPackedVec2(info)) return false;
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t load_function_id = GetOrCreatePackedLoadChunkFunction(
      pointer_id, pointer_type_id, info.component_type_id,
      info.packed_vec2_type_id, memory_operands);
  const uint32_t lowered_function_ptr_type_id =
      GetOrCreatePointerType(info.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec2_function_ptr_type_id = GetOrCreatePointerType(
      info.packed_vec2_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t element_count_id = GetOrCreateUIntConstant(info.length);
  if (load_function_id == 0 || lowered_function_ptr_type_id == 0 ||
      vec2_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || bool_type_id == 0 ||
      zero_uint_id == 0 || packed_width_uint_id == 0 || element_count_id == 0) {
    return false;
  }

  Instruction* result_var =
      AddFunctionVariable(function, lowered_function_ptr_type_id);
  Instruction* base_var =
      AddFunctionVariable(function, uint_function_ptr_type_id);
  if (!result_var || !base_var) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }

  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(base_var->result_id(), zero_uint_id)) {
    return false;
  }
  if (!preheader_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* base_load =
      header_builder.AddLoad(uint_type_id, base_var->result_id());
  if (!base_load) return false;
  Instruction* cond =
      header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                 base_load->result_id(), element_count_id);
  if (!cond) return false;
  if (!header_builder.AddLoopMerge(merge_label_id, continue_label_id)) {
    return false;
  }
  if (!header_builder.AddConditionalBranch(cond->result_id(), body_label_id,
                                           merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t memory_base_id = BuildVectorElementIndex(
      &body_builder, insert_before, offset_id, base_load->result_id());
  Instruction* vec = memory_base_id ? body_builder.AddFunctionCall(
                                          info.packed_vec2_type_id,
                                          load_function_id, {memory_base_id})
                                    : nullptr;
  if (!vec) return false;
  Instruction* packed_index =
      body_builder.AddBinaryOp(uint_type_id, spv::Op::OpUDiv,
                               base_load->result_id(), packed_width_uint_id);
  if (!packed_index) return false;
  Instruction* result_elem_ptr = body_builder.AddAccessChain(
      vec2_function_ptr_type_id, result_var->result_id(),
      {packed_index->result_id()});
  if (!result_elem_ptr) return false;
  if (!body_builder.AddStore(result_elem_ptr->result_id(), vec->result_id())) {
    return false;
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_i = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, base_load->result_id(),
      packed_width_uint_id);
  if (!next_i) return false;
  if (!continue_builder.AddStore(base_var->result_id(), next_i->result_id())) {
    return false;
  }
  if (!continue_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder merge_builder(
      context(), insert_before,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* result =
      merge_builder.AddLoad(info.lowered_type_id, result_var->result_id());
  if (!result) return false;
  *result_id = result->result_id();
  return true;
}

bool HwLowerToStandardPass::BuildPackedVectorStoreOuterLoop(
    Instruction* insert_before, const VectorTypeInfo& info, uint32_t pointer_id,
    uint32_t pointer_type_id, uint32_t offset_id, uint32_t object_id,
    const std::vector<Operand>& memory_operands) {
  if (!insert_before || object_id == 0 || !IsPackedVec2(info)) return false;
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t store_function_id = GetOrCreatePackedStoreChunkFunction(
      pointer_id, pointer_type_id, info.component_type_id,
      info.packed_vec2_type_id, memory_operands);
  const uint32_t lowered_function_ptr_type_id =
      GetOrCreatePointerType(info.lowered_type_id, spv::StorageClass::Function);
  const uint32_t vec2_function_ptr_type_id = GetOrCreatePointerType(
      info.packed_vec2_type_id, spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_function_ptr_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_uint_id = GetOrCreateUIntConstant(0);
  const uint32_t packed_width_uint_id =
      GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t element_count_id = GetOrCreateUIntConstant(info.length);
  if (store_function_id == 0 || lowered_function_ptr_type_id == 0 ||
      vec2_function_ptr_type_id == 0 || uint_type_id == 0 ||
      uint_function_ptr_type_id == 0 || void_type_id == 0 ||
      bool_type_id == 0 || zero_uint_id == 0 || packed_width_uint_id == 0 ||
      element_count_id == 0) {
    return false;
  }

  Instruction* object_var =
      AddFunctionVariable(function, lowered_function_ptr_type_id);
  Instruction* base_var =
      AddFunctionVariable(function, uint_function_ptr_type_id);
  if (!object_var || !base_var) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }

  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  if (!preheader_builder.AddStore(object_var->result_id(), object_id)) {
    return false;
  }
  if (!preheader_builder.AddStore(base_var->result_id(), zero_uint_id)) {
    return false;
  }
  if (!preheader_builder.AddBranch(header_label_id)) return false;

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* base_load =
      header_builder.AddLoad(uint_type_id, base_var->result_id());
  if (!base_load) return false;
  Instruction* cond =
      header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                 base_load->result_id(), element_count_id);
  if (!cond) return false;
  if (!header_builder.AddLoopMerge(merge_label_id, continue_label_id)) {
    return false;
  }
  if (!header_builder.AddConditionalBranch(cond->result_id(), body_label_id,
                                           merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* packed_index =
      body_builder.AddBinaryOp(uint_type_id, spv::Op::OpUDiv,
                               base_load->result_id(), packed_width_uint_id);
  if (!packed_index) return false;
  Instruction* value_elem_ptr = body_builder.AddAccessChain(
      vec2_function_ptr_type_id, object_var->result_id(),
      {packed_index->result_id()});
  if (!value_elem_ptr) return false;
  Instruction* vec = body_builder.AddLoad(info.packed_vec2_type_id,
                                          value_elem_ptr->result_id());
  if (!vec) return false;
  const uint32_t memory_base_id = BuildVectorElementIndex(
      &body_builder, insert_before, offset_id, base_load->result_id());
  if (memory_base_id == 0 ||
      !body_builder.AddFunctionCall(void_type_id, store_function_id,
                                    {memory_base_id, vec->result_id()})) {
    return false;
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_i = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, base_load->result_id(),
      packed_width_uint_id);
  if (!next_i) return false;
  if (!continue_builder.AddStore(base_var->result_id(), next_i->result_id())) {
    return false;
  }
  if (!continue_builder.AddBranch(header_label_id)) return false;
  return true;
}

bool HwLowerToStandardPass::BuildScalarMemoryLoop(
    Instruction* insert_before, uint32_t lowered_type_id,
    uint32_t component_type_id, uint32_t element_count, uint32_t pointer_id,
    uint32_t object_id, uint32_t shape_id, uint32_t offset_id,
    uint32_t matrix_cols, uint32_t layout, bool is_matrix,
    uint32_t packed_vec2_type_id, const std::vector<Operand>& memory_operands,
    uint32_t* result_id) {
  const bool is_load = object_id == 0;
  if (!insert_before || lowered_type_id == 0 || component_type_id == 0 ||
      element_count == 0 || pointer_id == 0 || (is_load && !result_id)) {
    return false;
  }
  BasicBlock* preheader_block = context()->get_instr_block(insert_before);
  Function* function = preheader_block ? preheader_block->GetParent() : nullptr;
  if (!preheader_block || !function) return false;

  const uint32_t aggregate_pointer_type_id =
      GetOrCreatePointerType(lowered_type_id, spv::StorageClass::Function);
  const uint32_t component_pointer_type_id =
      GetOrCreatePointerType(component_type_id, spv::StorageClass::Function);
  const uint32_t piece_pointer_type_id =
      packed_vec2_type_id == 0
          ? component_pointer_type_id
          : GetOrCreatePointerType(packed_vec2_type_id,
                                   spv::StorageClass::Function);
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t uint_pointer_type_id =
      GetOrCreatePointerType(uint_type_id, spv::StorageClass::Function);
  const uint32_t bool_type_id = GetOrCreateBoolType();
  const uint32_t zero_id = GetOrCreateUIntConstant(0);
  const uint32_t one_id = GetOrCreateUIntConstant(1);
  const uint32_t packed_width_id = GetOrCreateUIntConstant(kPackedVec2Width);
  const uint32_t element_count_id = GetOrCreateUIntConstant(element_count);
  if (aggregate_pointer_type_id == 0 || component_pointer_type_id == 0 ||
      piece_pointer_type_id == 0 || uint_type_id == 0 ||
      uint_pointer_type_id == 0 || bool_type_id == 0 || zero_id == 0 ||
      one_id == 0 || packed_width_id == 0 || element_count_id == 0) {
    return false;
  }

  Instruction* aggregate_variable =
      AddFunctionVariable(function, aggregate_pointer_type_id);
  Instruction* index_variable =
      AddFunctionVariable(function, uint_pointer_type_id);
  if (!aggregate_variable || !index_variable) return false;

  auto split_iter = preheader_block->begin();
  while (split_iter != preheader_block->end() &&
         &*split_iter != insert_before) {
    ++split_iter;
  }
  if (split_iter == preheader_block->end()) return false;

  std::unique_ptr<Instruction> enclosing_loop_merge;
  if (Instruction* loop_merge = preheader_block->GetLoopMergeInst()) {
    enclosing_loop_merge.reset(loop_merge->Clone(context()));
  }

  const uint32_t merge_label_id = TakeNextId();
  const uint32_t header_label_id = TakeNextId();
  const uint32_t body_label_id = TakeNextId();
  const uint32_t continue_label_id = TakeNextId();
  if (merge_label_id == 0 || header_label_id == 0 || body_label_id == 0 ||
      continue_label_id == 0) {
    return false;
  }
  BasicBlock* merge_block =
      preheader_block->SplitBasicBlock(context(), merge_label_id, split_iter);
  BasicBlock* header_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(header_label_id)),
      preheader_block);
  BasicBlock* body_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(body_label_id)), header_block);
  BasicBlock* continue_block = function->InsertBasicBlockAfter(
      std::unique_ptr<BasicBlock>(MakeBasicBlock(continue_label_id)),
      body_block);
  if (!merge_block || !header_block || !body_block || !continue_block) {
    return false;
  }
  if (enclosing_loop_merge) {
    Instruction* moved_loop_merge = merge_block->GetLoopMergeInst();
    if (!moved_loop_merge) return false;
    context()->KillInst(moved_loop_merge);
  }

  InstructionBuilder preheader_builder(
      context(), preheader_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t aggregate_initializer_id =
      is_load && packed_vec2_type_id != 0 ? GetOrCreateZero(lowered_type_id)
                                          : object_id;
  if (((!is_load || packed_vec2_type_id != 0) &&
       (aggregate_initializer_id == 0 ||
        !preheader_builder.AddStore(aggregate_variable->result_id(),
                                    aggregate_initializer_id))) ||
      !preheader_builder.AddStore(index_variable->result_id(), zero_id) ||
      (enclosing_loop_merge &&
       !preheader_builder.AddInstruction(std::move(enclosing_loop_merge))) ||
      !preheader_builder.AddBranch(header_label_id)) {
    return false;
  }

  InstructionBuilder header_builder(
      context(), header_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* index =
      header_builder.AddLoad(uint_type_id, index_variable->result_id());
  Instruction* condition =
      index ? header_builder.AddBinaryOp(bool_type_id, spv::Op::OpULessThan,
                                         index->result_id(), element_count_id)
            : nullptr;
  if (!index || !condition ||
      !header_builder.AddLoopMerge(merge_label_id, continue_label_id) ||
      !header_builder.AddConditionalBranch(condition->result_id(),
                                           body_label_id, merge_label_id)) {
    return false;
  }

  InstructionBuilder body_builder(
      context(), body_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  const uint32_t memory_index_id =
      is_matrix ? BuildDynamicMatrixElementIndex(
                      &body_builder, insert_before, shape_id, offset_id, layout,
                      matrix_cols, index->result_id())
                : BuildVectorElementIndex(&body_builder, insert_before,
                                          offset_id, index->result_id());
  const uint32_t external_pointer_id =
      BuildElementAccess(&body_builder, insert_before, pointer_id,
                         component_type_id, memory_index_id);
  uint32_t piece_index_id = index->result_id();
  uint32_t lane_id = 0;
  if (packed_vec2_type_id != 0) {
    Instruction* piece_index = body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpUDiv, index->result_id(), packed_width_id);
    Instruction* lane = body_builder.AddBinaryOp(
        uint_type_id, spv::Op::OpUMod, index->result_id(), packed_width_id);
    if (!piece_index || !lane) return false;
    piece_index_id = piece_index->result_id();
    lane_id = lane->result_id();
  }
  Instruction* aggregate_pointer = body_builder.AddAccessChain(
      piece_pointer_type_id, aggregate_variable->result_id(), {piece_index_id});
  if (memory_index_id == 0 || external_pointer_id == 0 || !aggregate_pointer) {
    return false;
  }
  if (is_load) {
    const uint32_t value_id = AddLoad(&body_builder, component_type_id,
                                      external_pointer_id, memory_operands);
    uint32_t stored_id = value_id;
    if (value_id != 0 && packed_vec2_type_id != 0) {
      Instruction* old_piece = body_builder.AddLoad(
          packed_vec2_type_id, aggregate_pointer->result_id());
      Instruction* inserted =
          old_piece ? body_builder.AddTernaryOp(
                          packed_vec2_type_id, spv::Op::OpVectorInsertDynamic,
                          old_piece->result_id(), value_id, lane_id)
                    : nullptr;
      stored_id = inserted ? inserted->result_id() : 0;
    }
    if (stored_id == 0 ||
        !body_builder.AddStore(aggregate_pointer->result_id(), stored_id)) {
      return false;
    }
  } else {
    Instruction* piece = body_builder.AddLoad(
        packed_vec2_type_id != 0 ? packed_vec2_type_id : component_type_id,
        aggregate_pointer->result_id());
    uint32_t value_id = piece ? piece->result_id() : 0;
    if (piece && packed_vec2_type_id != 0) {
      Instruction* extracted = body_builder.AddBinaryOp(
          component_type_id, spv::Op::OpVectorExtractDynamic,
          piece->result_id(), lane_id);
      value_id = extracted ? extracted->result_id() : 0;
    }
    if (value_id == 0 || !AddStore(&body_builder, external_pointer_id, value_id,
                                   memory_operands, component_type_id)) {
      return false;
    }
  }
  if (!body_builder.AddBranch(continue_label_id)) return false;

  InstructionBuilder continue_builder(
      context(), continue_block,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  Instruction* next_index = continue_builder.AddBinaryOp(
      uint_type_id, spv::Op::OpIAdd, index->result_id(), one_id);
  if (!next_index ||
      !continue_builder.AddStore(index_variable->result_id(),
                                 next_index->result_id()) ||
      !continue_builder.AddBranch(header_label_id)) {
    return false;
  }

  if (is_load) {
    InstructionBuilder merge_builder(
        context(), insert_before,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    Instruction* result =
        merge_builder.AddLoad(lowered_type_id, aggregate_variable->result_id());
    if (!result) return false;
    *result_id = result->result_id();
  }
  return true;
}

uint32_t HwLowerToStandardPass::BuildCapturedPointer(
    InstructionBuilder* builder, uint32_t pointer_id) {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer) return 0;

  if (pointer->opcode() == spv::Op::OpVariable &&
      context()->get_instr_block(pointer) == nullptr) {
    return pointer_id;
  }

  if (pointer->opcode() != spv::Op::OpAccessChain &&
      pointer->opcode() != spv::Op::OpInBoundsAccessChain) {
    return 0;
  }

  const uint32_t base_pointer_id = pointer->GetSingleWordInOperand(0);
  const uint32_t captured_base_id =
      BuildCapturedPointer(builder, base_pointer_id);
  if (captured_base_id == 0) return 0;

  std::vector<uint32_t> index_ids;
  index_ids.reserve(pointer->NumInOperands() - 1);
  for (uint32_t i = 1; i < pointer->NumInOperands(); ++i) {
    const uint32_t index_id = pointer->GetSingleWordInOperand(i);
    if (!IsModuleVisibleValue(index_id)) return 0;
    index_ids.push_back(index_id);
  }

  Instruction* access =
      builder->AddAccessChain(pointer->type_id(), captured_base_id, index_ids);
  return access ? access->result_id() : 0;
}

bool HwLowerToStandardPass::CanCapturePointer(uint32_t pointer_id) const {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer) return false;

  if (pointer->opcode() == spv::Op::OpVariable &&
      context()->get_instr_block(pointer) == nullptr) {
    return true;
  }

  if (pointer->opcode() != spv::Op::OpAccessChain &&
      pointer->opcode() != spv::Op::OpInBoundsAccessChain) {
    return false;
  }

  if (!CanCapturePointer(pointer->GetSingleWordInOperand(0))) {
    return false;
  }

  for (uint32_t i = 1; i < pointer->NumInOperands(); ++i) {
    if (!IsModuleVisibleValue(pointer->GetSingleWordInOperand(i))) {
      return false;
    }
  }
  return true;
}

bool HwLowerToStandardPass::IsModuleVisibleValue(uint32_t id) const {
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  if (!inst) return false;
  return context()->get_instr_block(inst) == nullptr;
}

uint32_t HwLowerToStandardPass::ExtractCompositeElement(
    InstructionBuilder* builder, uint32_t component_type_id,
    uint32_t composite_id, uint32_t index) {
  Instruction* extract =
      builder->AddCompositeExtract(component_type_id, composite_id, {index});
  return extract ? extract->result_id() : 0;
}

uint32_t HwLowerToStandardPass::AddLoad(
    InstructionBuilder* builder, uint32_t type_id, uint32_t pointer_id,
    const std::vector<Operand>& memory_operands) {
  std::vector<Operand> normalized_memory_operands;
  if (!NormalizeMemoryOperandsForAccess(pointer_id, type_id, memory_operands,
                                        &normalized_memory_operands)) {
    return 0;
  }
  std::vector<Operand> operands;
  operands.reserve(1 + normalized_memory_operands.size());
  operands.push_back(IdOperand(pointer_id));
  operands.insert(operands.end(), normalized_memory_operands.begin(),
                  normalized_memory_operands.end());
  std::unique_ptr<Instruction> load = MakeUnique<Instruction>(
      context(), spv::Op::OpLoad, type_id, TakeNextId(), operands);
  Instruction* added = builder->AddInstruction(std::move(load));
  return added ? added->result_id() : 0;
}

bool HwLowerToStandardPass::AddStore(
    InstructionBuilder* builder, uint32_t pointer_id, uint32_t object_id,
    const std::vector<Operand>& memory_operands, uint32_t accessed_type_id) {
  if (accessed_type_id == 0 && !memory_operands.empty()) {
    Instruction* object = get_def_use_mgr()->GetDef(object_id);
    if (!object || object->type_id() == 0) {
      ReportError(nullptr, "cannot determine HW store object type");
      return false;
    }
    accessed_type_id = object->type_id();
  }
  std::vector<Operand> normalized_memory_operands;
  if (!NormalizeMemoryOperandsForAccess(pointer_id, accessed_type_id,
                                        memory_operands,
                                        &normalized_memory_operands)) {
    return false;
  }
  std::vector<Operand> operands;
  operands.reserve(2 + normalized_memory_operands.size());
  operands.push_back(IdOperand(pointer_id));
  operands.push_back(IdOperand(object_id));
  operands.insert(operands.end(), normalized_memory_operands.begin(),
                  normalized_memory_operands.end());
  std::unique_ptr<Instruction> store =
      MakeUnique<Instruction>(context(), spv::Op::OpStore, 0, 0, operands);
  return builder->AddInstruction(std::move(store)) != nullptr;
}

std::vector<Operand> HwLowerToStandardPass::CopyMemoryOperands(
    const Instruction* inst, uint32_t first_in_operand) const {
  std::vector<Operand> operands;
  if (!inst) return operands;
  if (inst->NumInOperands() <= first_in_operand) {
    uint32_t storage_class = 0;
    const uint32_t pointer_id =
        inst->NumInOperands() == 0 ? 0 : inst->GetSingleWordInOperand(0);
    if (GetPointerStorageClass(pointer_id, &storage_class) &&
        storage_class ==
            static_cast<uint32_t>(spv::StorageClass::PhysicalStorageBuffer)) {
      ReportError(inst,
                  "PhysicalStorageBuffer HW access requires an Aligned "
                  "memory operand");
      return {{SPV_OPERAND_TYPE_MEMORY_ACCESS,
               {std::numeric_limits<uint32_t>::max()}}};
    }
    return operands;
  }
  operands.reserve(inst->NumInOperands() - first_in_operand);
  for (uint32_t i = first_in_operand; i < inst->NumInOperands(); ++i) {
    operands.push_back(inst->GetInOperand(i));
  }
  return operands;
}

bool HwLowerToStandardPass::NormalizeMemoryOperandsForAccess(
    uint32_t pointer_id, uint32_t accessed_type_id,
    const std::vector<Operand>& memory_operands,
    std::vector<Operand>* normalized) const {
  if (!normalized) return false;
  normalized->clear();

  uint32_t storage_class = 0;
  const bool has_storage_class =
      GetPointerStorageClass(pointer_id, &storage_class);
  const bool physical_storage =
      has_storage_class &&
      storage_class ==
          static_cast<uint32_t>(spv::StorageClass::PhysicalStorageBuffer);
  if (memory_operands.empty()) {
    if (physical_storage) {
      ReportError(nullptr,
                  "PhysicalStorageBuffer lowered access requires Aligned");
      return false;
    }
    return true;
  }

  const Operand& access = memory_operands.front();
  if (access.type != SPV_OPERAND_TYPE_MEMORY_ACCESS ||
      access.words.size() != 1) {
    ReportError(nullptr, "malformed HW memory access operand");
    return false;
  }

  const uint32_t mask = access.words[0];
  const uint32_t aligned = uint32_t(spv::MemoryAccessMask::Aligned);
  const uint32_t make_available =
      uint32_t(spv::MemoryAccessMask::MakePointerAvailable);
  const uint32_t make_visible =
      uint32_t(spv::MemoryAccessMask::MakePointerVisible);
  const uint32_t alias_scope =
      uint32_t(spv::MemoryAccessMask::AliasScopeINTELMask);
  const uint32_t no_alias = uint32_t(spv::MemoryAccessMask::NoAliasINTELMask);
  const uint32_t known_mask =
      uint32_t(spv::MemoryAccessMask::Volatile) | aligned |
      uint32_t(spv::MemoryAccessMask::Nontemporal) | make_available |
      make_visible | uint32_t(spv::MemoryAccessMask::NonPrivatePointer) |
      alias_scope | no_alias;
  if ((mask & ~known_mask) != 0) {
    ReportError(nullptr, "unsupported HW memory access operand mask");
    return false;
  }

  size_t expected_operands = 1;
  expected_operands += (mask & aligned) != 0;
  expected_operands += (mask & make_available) != 0;
  expected_operands += (mask & make_visible) != 0;
  expected_operands += (mask & alias_scope) != 0;
  expected_operands += (mask & no_alias) != 0;
  if (memory_operands.size() != expected_operands) {
    ReportError(nullptr, "malformed HW memory access operand parameters");
    return false;
  }
  if (physical_storage && (mask & aligned) == 0) {
    ReportError(nullptr,
                "PhysicalStorageBuffer lowered access requires Aligned");
    return false;
  }

  *normalized = memory_operands;
  if ((mask & aligned) == 0) return true;
  Operand& alignment_operand = (*normalized)[1];
  if (alignment_operand.words.size() != 1) {
    ReportError(nullptr, "malformed Aligned memory operand");
    return false;
  }
  const uint32_t base_alignment = alignment_operand.words[0];
  if (base_alignment == 0 || (base_alignment & (base_alignment - 1)) != 0) {
    ReportError(nullptr, "Aligned memory operand must be a power of two");
    return false;
  }

  const uint32_t natural_alignment = GetTypeNaturalAlignment(accessed_type_id);
  if (natural_alignment == 0) {
    ReportError(nullptr, "cannot determine lowered memory access alignment");
    return false;
  }

  uint64_t byte_offset = 0;
  uint32_t access_alignment = std::min(base_alignment, natural_alignment);
  if (GetKnownAccessByteOffset(pointer_id, accessed_type_id, &byte_offset)) {
    const uint64_t gcd = std::gcd(uint64_t(base_alignment), byte_offset);
    if (gcd == 0 || gcd > std::numeric_limits<uint32_t>::max()) return false;
    access_alignment = static_cast<uint32_t>(gcd);
  }
  alignment_operand.words[0] = access_alignment;
  return true;
}

bool HwLowerToStandardPass::GetKnownAccessByteOffset(
    uint32_t pointer_id, uint32_t accessed_type_id,
    uint64_t* byte_offset) const {
  if (!byte_offset) return false;
  *byte_offset = 0;
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer) return false;
  if (pointer->opcode() != spv::Op::OpAccessChain &&
      pointer->opcode() != spv::Op::OpInBoundsAccessChain) {
    return true;
  }
  if (pointer->NumInOperands() != 2) return true;

  Instruction* base =
      get_def_use_mgr()->GetDef(pointer->GetSingleWordInOperand(0));
  Instruction* base_pointer_type =
      base ? get_def_use_mgr()->GetDef(base->type_id()) : nullptr;
  if (!base_pointer_type ||
      base_pointer_type->opcode() != spv::Op::OpTypePointer ||
      base_pointer_type->NumInOperands() < 2) {
    return false;
  }
  Instruction* pointee =
      get_def_use_mgr()->GetDef(base_pointer_type->GetSingleWordInOperand(1));
  if (!pointee ||
      (pointee->opcode() != spv::Op::OpTypeArray &&
       pointee->opcode() != spv::Op::OpTypeRuntimeArray) ||
      pointee->NumInOperands() == 0 ||
      pointee->GetSingleWordInOperand(0) != accessed_type_id) {
    // The access chain produced the pointer passed to the original HW
    // operation.  Its Aligned promise is relative to that pointer, not to the
    // module variable at the root of the chain.
    return true;
  }

  uint32_t element_index = 0;
  if (!TryEvaluateConstantU32Expression(pointer->GetSingleWordInOperand(1),
                                        &element_index)) {
    return false;
  }
  uint32_t stride = GetArrayStride(pointee->result_id());
  if (stride == 0) stride = GetTypeNaturalAlignment(accessed_type_id);
  if (stride == 0) return false;
  *byte_offset = uint64_t(element_index) * uint64_t(stride);
  return true;
}

bool HwLowerToStandardPass::TryEvaluateConstantFloat32(uint32_t id,
                                                       float* value,
                                                       uint32_t depth) const {
  if (!value || depth > 16) return false;
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  if (!inst) return false;
  if (inst->opcode() == spv::Op::OpConstantNull) {
    *value = 0.0f;
    return true;
  }
  if (inst->opcode() == spv::Op::OpConstant) {
    Instruction* type = get_def_use_mgr()->GetDef(inst->type_id());
    if (!type || type->opcode() != spv::Op::OpTypeFloat ||
        type->GetSingleWordInOperand(0) != 32 || inst->NumInOperands() != 1) {
      return false;
    }
    const uint32_t bits = inst->GetSingleWordInOperand(0);
    static_assert(sizeof(bits) == sizeof(*value), "unexpected float size");
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }
  if (inst->opcode() == spv::Op::OpCopyObject && inst->NumInOperands() == 1) {
    return TryEvaluateConstantFloat32(inst->GetSingleWordInOperand(0), value,
                                      depth + 1);
  }
  if (inst->opcode() == spv::Op::OpCompositeExtract &&
      inst->NumInOperands() == 2) {
    Instruction* composite =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
    if (!composite || composite->opcode() != spv::Op::OpConstantComposite) {
      return false;
    }
    const uint32_t index = inst->GetSingleWordInOperand(1);
    if (index >= composite->NumInOperands()) return false;
    return TryEvaluateConstantFloat32(composite->GetSingleWordInOperand(index),
                                      value, depth + 1);
  }
  return false;
}

bool HwLowerToStandardPass::TryEvaluateConstantU32Expression(
    uint32_t id, uint32_t* value, uint32_t depth) const {
  if (!value || depth > 16) return false;
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  if (!inst) return false;
  if (inst->opcode() == spv::Op::OpConstantNull) {
    *value = 0;
    return true;
  }
  if (inst->opcode() == spv::Op::OpConstant) {
    Instruction* type = get_def_use_mgr()->GetDef(inst->type_id());
    if (!type || type->opcode() != spv::Op::OpTypeInt ||
        type->GetSingleWordInOperand(0) > 32 || inst->NumInOperands() != 1) {
      return false;
    }
    *value = inst->GetSingleWordInOperand(0);
    return true;
  }
  if ((inst->opcode() == spv::Op::OpCopyObject ||
       inst->opcode() == spv::Op::OpBitcast ||
       inst->opcode() == spv::Op::OpSConvert ||
       inst->opcode() == spv::Op::OpUConvert) &&
      inst->NumInOperands() == 1) {
    return TryEvaluateConstantU32Expression(inst->GetSingleWordInOperand(0),
                                            value, depth + 1);
  }
  if (inst->opcode() == spv::Op::OpConvertFToU && inst->NumInOperands() == 1) {
    float float_value = 0.0f;
    if (!TryEvaluateConstantFloat32(inst->GetSingleWordInOperand(0),
                                    &float_value, depth + 1) ||
        !std::isfinite(float_value) || float_value < 0.0f ||
        double(float_value) > double(std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    *value = static_cast<uint32_t>(float_value);
    return true;
  }
  if (inst->opcode() == spv::Op::OpCompositeExtract &&
      inst->NumInOperands() == 2) {
    Instruction* composite =
        get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
    if (!composite || composite->opcode() != spv::Op::OpConstantComposite) {
      return false;
    }
    const uint32_t index = inst->GetSingleWordInOperand(1);
    if (index >= composite->NumInOperands()) return false;
    return TryEvaluateConstantU32Expression(
        composite->GetSingleWordInOperand(index), value, depth + 1);
  }
  if (inst->NumInOperands() != 2) return false;
  uint32_t lhs = 0;
  uint32_t rhs = 0;
  if (!TryEvaluateConstantU32Expression(inst->GetSingleWordInOperand(0), &lhs,
                                        depth + 1) ||
      !TryEvaluateConstantU32Expression(inst->GetSingleWordInOperand(1), &rhs,
                                        depth + 1)) {
    return false;
  }
  switch (inst->opcode()) {
    case spv::Op::OpIAdd:
      *value = lhs + rhs;
      return true;
    case spv::Op::OpISub:
      *value = lhs - rhs;
      return true;
    case spv::Op::OpIMul:
      *value = lhs * rhs;
      return true;
    case spv::Op::OpUDiv:
      if (rhs == 0) return false;
      *value = lhs / rhs;
      return true;
    case spv::Op::OpUMod:
      if (rhs == 0) return false;
      *value = lhs % rhs;
      return true;
    default:
      return false;
  }
}

uint32_t HwLowerToStandardPass::GetTypeNaturalAlignment(
    uint32_t type_id) const {
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  if (!type) return 0;
  switch (type->opcode()) {
    case spv::Op::OpTypeBool:
      return 1;
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat: {
      if (type->NumInOperands() == 0) return 0;
      const uint32_t width = type->GetSingleWordInOperand(0);
      return width != 0 && width % 8 == 0 ? width / 8 : 0;
    }
    case spv::Op::OpTypeVector: {
      if (type->NumInOperands() < 2) return 0;
      const uint32_t component_alignment =
          GetTypeNaturalAlignment(type->GetSingleWordInOperand(0));
      const uint32_t count = type->GetSingleWordInOperand(1);
      if (component_alignment == 0 || count < 2 || count > 4) return 0;
      return component_alignment * (count == 2 ? 2 : 4);
    }
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeRuntimeArray:
      return type->NumInOperands() == 0
                 ? 0
                 : GetTypeNaturalAlignment(type->GetSingleWordInOperand(0));
    default:
      return 0;
  }
}

uint32_t HwLowerToStandardPass::GetArrayStride(uint32_t array_type_id) const {
  uint32_t stride = 0;
  get_decoration_mgr()->WhileEachDecoration(
      array_type_id, uint32_t(spv::Decoration::ArrayStride),
      [&stride](const Instruction& decoration) {
        if (decoration.opcode() == spv::Op::OpDecorate &&
            decoration.NumInOperands() >= 3) {
          stride = decoration.GetSingleWordInOperand(2);
        } else if (decoration.opcode() == spv::Op::OpMemberDecorate &&
                   decoration.NumInOperands() >= 4) {
          stride = decoration.GetSingleWordInOperand(3);
        }
        return false;
      });
  return stride;
}

uint32_t HwLowerToStandardPass::GetOrCreatePackedLoadChunkFunction(
    uint32_t pointer_id, uint32_t pointer_type_id, uint32_t component_type_id,
    uint32_t vec2_type_id, const std::vector<Operand>& memory_operands) {
  const std::string key =
      std::to_string(pointer_id) + "|" + std::to_string(pointer_type_id) + "|" +
      std::to_string(component_type_id) + "|" + std::to_string(vec2_type_id) +
      "|" + MemoryOperandsKey(memory_operands);
  auto cached = packed_load_chunk_functions_.find(key);
  if (cached != packed_load_chunk_functions_.end()) return cached->second;
  if (!CanCapturePointer(pointer_id)) return 0;

  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t function_type_id =
      GetOrCreateFunctionType(vec2_type_id, {uint_type_id});
  if (uint_type_id == 0 || function_type_id == 0) return 0;

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start =
      MakeUnique<Instruction>(context(), spv::Op::OpFunction, vec2_type_id,
                              function_id, std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t base_param_id = TakeNextId();
  if (base_param_id == 0) return 0;
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, base_param_id,
      std::initializer_list<Operand>{}));

  const uint32_t entry_label_id = TakeNextId();
  if (entry_label_id == 0) return 0;
  std::unique_ptr<BasicBlock> entry_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_label_id));
  if (!entry_block) return 0;

  InstructionBuilder builder(context(), entry_block.get());
  const uint32_t captured_pointer_id =
      BuildCapturedPointer(&builder, pointer_id);
  if (captured_pointer_id == 0) return 0;

  std::vector<uint32_t> lane_ids;
  lane_ids.reserve(kPackedVec2Width);
  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    const uint32_t lane_offset_id = GetOrCreateUIntConstant(lane);
    if (lane_offset_id == 0) return 0;
    uint32_t element_index_id = base_param_id;
    if (lane != 0) {
      Instruction* element_index = builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, base_param_id, lane_offset_id);
      if (!element_index) return 0;
      element_index_id = element_index->result_id();
    }
    const uint32_t elem_ptr_id = BuildElementAccessFromPointerType(
        &builder, pointer_type_id, captured_pointer_id, component_type_id,
        element_index_id);
    if (elem_ptr_id == 0) return 0;
    const uint32_t load_id =
        AddLoad(&builder, component_type_id, elem_ptr_id, memory_operands);
    if (load_id == 0) return 0;
    lane_ids.push_back(load_id);
  }

  Instruction* vec = builder.AddCompositeConstruct(vec2_type_id, lane_ids);
  if (!vec) return 0;
  if (!builder.AddUnaryOp(0, spv::Op::OpReturnValue, vec->result_id())) {
    return 0;
  }

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(entry_block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/false);

  packed_load_chunk_functions_[key] = function_id;
  return function_id;
}

uint32_t HwLowerToStandardPass::GetOrCreatePackedStoreChunkFunction(
    uint32_t pointer_id, uint32_t pointer_type_id, uint32_t component_type_id,
    uint32_t vec2_type_id, const std::vector<Operand>& memory_operands) {
  const std::string key =
      std::to_string(pointer_id) + "|" + std::to_string(pointer_type_id) + "|" +
      std::to_string(component_type_id) + "|" + std::to_string(vec2_type_id) +
      "|" + MemoryOperandsKey(memory_operands);
  auto cached = packed_store_chunk_functions_.find(key);
  if (cached != packed_store_chunk_functions_.end()) return cached->second;
  if (!CanCapturePointer(pointer_id)) return 0;

  const uint32_t void_type_id = GetOrCreateVoidType();
  const uint32_t uint_type_id = GetOrCreateUIntType();
  const uint32_t function_type_id =
      GetOrCreateFunctionType(void_type_id, {uint_type_id, vec2_type_id});
  if (void_type_id == 0 || uint_type_id == 0 || function_type_id == 0) {
    return 0;
  }

  const uint32_t function_id = TakeNextId();
  if (function_id == 0) return 0;
  std::unique_ptr<Instruction> function_start =
      MakeUnique<Instruction>(context(), spv::Op::OpFunction, void_type_id,
                              function_id, std::initializer_list<Operand>{});
  function_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  function_start->AddOperand(IdOperand(function_type_id));
  std::unique_ptr<Function> function =
      MakeUnique<Function>(std::move(function_start));

  const uint32_t base_param_id = TakeNextId();
  const uint32_t value_param_id = TakeNextId();
  if (base_param_id == 0 || value_param_id == 0) return 0;
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, uint_type_id, base_param_id,
      std::initializer_list<Operand>{}));
  function->AddParameter(MakeUnique<Instruction>(
      context(), spv::Op::OpFunctionParameter, vec2_type_id, value_param_id,
      std::initializer_list<Operand>{}));

  const uint32_t entry_label_id = TakeNextId();
  if (entry_label_id == 0) return 0;
  std::unique_ptr<BasicBlock> entry_block =
      std::unique_ptr<BasicBlock>(MakeBasicBlock(entry_label_id));
  if (!entry_block) return 0;

  InstructionBuilder builder(context(), entry_block.get());
  const uint32_t captured_pointer_id =
      BuildCapturedPointer(&builder, pointer_id);
  if (captured_pointer_id == 0) return 0;

  for (uint32_t lane = 0; lane < kPackedVec2Width; ++lane) {
    const uint32_t lane_offset_id = GetOrCreateUIntConstant(lane);
    if (lane_offset_id == 0) return 0;
    uint32_t element_index_id = base_param_id;
    if (lane != 0) {
      Instruction* element_index = builder.AddBinaryOp(
          uint_type_id, spv::Op::OpIAdd, base_param_id, lane_offset_id);
      if (!element_index) return 0;
      element_index_id = element_index->result_id();
    }
    const uint32_t elem_ptr_id = BuildElementAccessFromPointerType(
        &builder, pointer_type_id, captured_pointer_id, component_type_id,
        element_index_id);
    const uint32_t value_id = ExtractCompositeElement(
        &builder, component_type_id, value_param_id, lane);
    if (elem_ptr_id == 0 || value_id == 0) return 0;
    if (!AddStore(&builder, elem_ptr_id, value_id, memory_operands,
                  component_type_id)) {
      return 0;
    }
  }

  if (!builder.AddNullaryOp(0, spv::Op::OpReturn)) return 0;

  function->SetFunctionEnd(
      MakeUnique<Instruction>(context(), spv::Op::OpFunctionEnd, 0, 0,
                              std::initializer_list<Operand>{}));
  function->AddBasicBlock(std::move(entry_block));
  AddGeneratedFunction(std::move(function), function_id,
                       /*may_write_memory=*/true);

  packed_store_chunk_functions_[key] = function_id;
  return function_id;
}

}  // namespace opt
}  // namespace spvtools
