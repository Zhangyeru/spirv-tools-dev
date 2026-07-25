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

#include "source/opt/hw_fuse_two_layer_vector_matmul_pass.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "source/opcode.h"
#include "source/opt/basic_block.h"
#include "source/opt/decoration_manager.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/feature_manager.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/util/make_unique.h"
#include "source/util/string_utils.h"
#include "spirv/unified1/GLSL.std.450.h"

namespace spvtools {
namespace opt {
namespace {

constexpr uint32_t kInputOperand = 0;
constexpr uint32_t kMatrixOperand = 1;
constexpr uint32_t kBiasOperand = 2;
constexpr uint32_t kSplitWidth = 16;
constexpr uint32_t kMatrixLoadPointerOperand = 0;
constexpr uint32_t kMatrixLoadShapeOperand = 1;
constexpr uint32_t kMatrixLoadOffsetOperand = 2;
constexpr uint32_t kMatrixLoadLayoutOperand = 3;
constexpr uint32_t kMatrixLoadMemoryOperands = 4;
constexpr uint32_t kVectorLoadPointerOperand = 0;
constexpr uint32_t kVectorLoadOffsetOperand = 1;
constexpr uint32_t kVectorLoadMemoryOperands = 2;

Operand IdOperand(uint32_t id) { return {SPV_OPERAND_TYPE_ID, {id}}; }

}  // namespace

Pass::Status HwFuseTwoLayerVectorMatmulPass::Process() {
  if (max_unrolled_matmul_macs_ == 0) return Status::SuccessWithoutChange;

  // Collect first.  A successful rewrite kills instructions in the same basic
  // block, so mutating while Module::ForEachInst is walking the list is unsafe.
  std::vector<Instruction*> candidates;
  get_module()->ForEachInst([&candidates](Instruction* inst) {
    if (inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulHW ||
        inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddHW) {
      candidates.push_back(inst);
    }
  });

  bool changed = false;
  for (Instruction* candidate : candidates) {
    // An earlier fusion can consume an instruction that was also collected as
    // a candidate.  Re-querying its definition avoids dereferencing a killed
    // list node.
    if (!candidate || candidate->result_id() == 0 ||
        get_def_use_mgr()->GetDef(candidate->result_id()) != candidate) {
      continue;
    }

    Match match;
    if (!MatchSecondLayer(candidate, &match)) continue;
    if (!RewriteMatch(match)) return Status::Failure;
    changed = true;
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool HwFuseTwoLayerVectorMatmulPass::MatchSecondLayer(Instruction* second,
                                                      Match* match) const {
  if (!second || !match || !IsVectorMatrixMul(second)) return false;
  const uint32_t expected_operands = IsVectorMatrixMulAdd(second) ? 3u : 2u;
  if (second->NumInOperands() != expected_operands) return false;

  BasicBlock* block = context()->get_instr_block(second);
  if (!block) return false;

  Match candidate;
  candidate.second = second;
  candidate.block = block;
  if (!GetVectorType(second->type_id(), &candidate.output)) return false;

  Instruction* second_input =
      get_def_use_mgr()->GetDef(second->GetSingleWordInOperand(kInputOperand));
  Instruction* second_matrix =
      get_def_use_mgr()->GetDef(second->GetSingleWordInOperand(kMatrixOperand));
  VectorTypeInfo second_input_type;
  if (!second_input || !second_matrix ||
      !GetVectorType(second_input->type_id(), &second_input_type) ||
      !GetMatrixType(second_matrix->type_id(), &candidate.second_matrix)) {
    return false;
  }

  if (IsVectorMatrixMulAdd(second)) {
    Instruction* bias =
        get_def_use_mgr()->GetDef(second->GetSingleWordInOperand(kBiasOperand));
    VectorTypeInfo bias_type;
    if (!bias || !GetVectorType(bias->type_id(), &bias_type) ||
        bias_type.component_type_id != candidate.output.component_type_id ||
        bias_type.length != candidate.output.length) {
      return false;
    }
  }

  std::vector<Instruction*> second_transport;
  Instruction* producer =
      TraceFunctionValueSource(second_input, second, &second_transport);
  if (!producer) return false;

  std::vector<Instruction*> first_transport;
  if (IsGlslFMax(producer)) {
    candidate.relu = producer;
    if (producer->type_id() != second_input->type_id() ||
        context()->get_instr_block(producer) != block) {
      return false;
    }

    const uint32_t lhs_id = producer->GetSingleWordInOperand(2);
    const uint32_t rhs_id = producer->GetSingleWordInOperand(3);
    uint32_t value_id = 0;
    if (IsZeroConstant(lhs_id)) {
      value_id = rhs_id;
    } else if (IsZeroConstant(rhs_id)) {
      value_id = lhs_id;
    } else {
      return false;
    }

    Instruction* value = get_def_use_mgr()->GetDef(value_id);
    producer =
        TraceFunctionValueSource(value, candidate.relu, &first_transport);
    if (!producer) return false;
  }

  if (!IsVectorMatrixMul(producer) ||
      producer->NumInOperands() != (IsVectorMatrixMulAdd(producer) ? 3u : 2u) ||
      context()->get_instr_block(producer) != block) {
    return false;
  }
  candidate.first = producer;

  Instruction* first_input = get_def_use_mgr()->GetDef(
      producer->GetSingleWordInOperand(kInputOperand));
  Instruction* first_matrix = get_def_use_mgr()->GetDef(
      producer->GetSingleWordInOperand(kMatrixOperand));
  if (!first_input || !first_matrix ||
      !GetVectorType(first_input->type_id(), &candidate.input) ||
      !GetVectorType(producer->type_id(), &candidate.middle) ||
      !GetMatrixType(first_matrix->type_id(), &candidate.first_matrix)) {
    return false;
  }

  if (IsVectorMatrixMulAdd(producer)) {
    Instruction* bias = get_def_use_mgr()->GetDef(
        producer->GetSingleWordInOperand(kBiasOperand));
    VectorTypeInfo bias_type;
    if (!bias || !GetVectorType(bias->type_id(), &bias_type) ||
        bias_type.component_type_id != candidate.middle.component_type_id ||
        bias_type.length != candidate.middle.length) {
      return false;
    }
  }

  const uint32_t component_type_id = candidate.input.component_type_id;
  if (!IsFloat16Type(component_type_id) ||
      candidate.middle.component_type_id != component_type_id ||
      candidate.output.component_type_id != component_type_id ||
      candidate.first_matrix.component_type_id != component_type_id ||
      candidate.second_matrix.component_type_id != component_type_id ||
      second_input_type.component_type_id != component_type_id) {
    return false;
  }

  const uint32_t k = candidate.input.length;
  const uint32_t n = candidate.middle.length;
  const uint32_t p = candidate.output.length;
  if (k == 0 || n <= kSplitWidth || p == 0 || p > kSplitWidth ||
      candidate.first_matrix.rows != k || candidate.first_matrix.cols != n ||
      second_input_type.length != n || candidate.second_matrix.rows != n ||
      candidate.second_matrix.cols != p) {
    return false;
  }

  // Apply the configured cap before emitting any instructions.  The guarded
  // divisions also avoid overflowing K*N + N*P for hostile dimensions.
  if (uint64_t(k) > max_unrolled_matmul_macs_ / uint64_t(n)) return false;
  const uint64_t first_macs = uint64_t(k) * uint64_t(n);
  const uint64_t remaining = max_unrolled_matmul_macs_ - first_macs;
  if (uint64_t(p) > remaining / uint64_t(n)) return false;

  candidate.first_fp_fast_math_mode =
      GetFPFastMathMode(candidate.first->result_id());
  candidate.relu_fp_fast_math_mode =
      candidate.relu ? GetFPFastMathMode(candidate.relu->result_id()) : 0;
  candidate.second_fp_fast_math_mode =
      GetFPFastMathMode(candidate.second->result_id());

  candidate.kill_list.push_back(candidate.first);
  if (candidate.relu) candidate.kill_list.push_back(candidate.relu);
  candidate.kill_list.insert(candidate.kill_list.end(),
                             second_transport.begin(), second_transport.end());
  candidate.kill_list.insert(candidate.kill_list.end(), first_transport.begin(),
                             first_transport.end());

  // Preserve the pointer-aware nature of the existing direct lower path.
  // Stream each private matrix or bias operand independently from its original
  // load instead of first materializing a cooperative aggregate in Function
  // memory.  A shared or unsafe operand falls back without disabling direct
  // loads for the other operands.
  std::vector<Instruction*> first_matrix_chain;
  std::vector<Instruction*> second_matrix_chain;
  const bool has_first_direct =
      GetDirectMatrixLoad(first_matrix, second, candidate.first_matrix,
                          &candidate.first_direct_matrix, &first_matrix_chain);
  const bool has_second_direct = GetDirectMatrixLoad(
      second_matrix, second, candidate.second_matrix,
      &candidate.second_direct_matrix, &second_matrix_chain);
  std::vector<Instruction*> first_bias_chain;
  std::vector<Instruction*> second_bias_chain;
  bool has_first_direct_bias = false;
  bool has_second_direct_bias = false;
  if (IsVectorMatrixMulAdd(candidate.first)) {
    Instruction* bias = get_def_use_mgr()->GetDef(
        candidate.first->GetSingleWordInOperand(kBiasOperand));
    has_first_direct_bias =
        GetDirectVectorLoad(bias, second, candidate.middle,
                            &candidate.first_direct_bias, &first_bias_chain);
  }
  if (IsVectorMatrixMulAdd(candidate.second)) {
    Instruction* bias = get_def_use_mgr()->GetDef(
        candidate.second->GetSingleWordInOperand(kBiasOperand));
    has_second_direct_bias =
        GetDirectVectorLoad(bias, second, candidate.output,
                            &candidate.second_direct_bias, &second_bias_chain);
  }

  auto accept_direct = [this, second, &candidate](
                           bool available,
                           const std::vector<Instruction*>& chain,
                           Instruction* source_load,
                           Instruction* protected_value = nullptr) {
    if (!available || !source_load) return false;
    // The first input remains an aggregate and gains new extract users during
    // RewriteMatch.  IsClosedIntermediateChain can only see the old users, so
    // explicitly prevent a bias candidate from killing that future source.
    if (source_load == protected_value ||
        std::find(chain.begin(), chain.end(), protected_value) != chain.end()) {
      return false;
    }
    std::vector<Instruction*> trial = candidate.kill_list;
    trial.insert(trial.end(), chain.begin(), chain.end());
    trial.push_back(source_load);
    if (!IsClosedIntermediateChain(second, &trial)) return false;
    candidate.kill_list = std::move(trial);
    return true;
  };

  if (!accept_direct(has_first_direct, first_matrix_chain,
                     candidate.first_direct_matrix.source_load)) {
    candidate.first_direct_matrix = {};
  }
  if (!accept_direct(has_second_direct, second_matrix_chain,
                     candidate.second_direct_matrix.source_load)) {
    candidate.second_direct_matrix = {};
  }
  if (!accept_direct(has_first_direct_bias, first_bias_chain,
                     candidate.first_direct_bias.source_load, first_input)) {
    candidate.first_direct_bias = {};
  }
  if (!accept_direct(has_second_direct_bias, second_bias_chain,
                     candidate.second_direct_bias.source_load, first_input)) {
    candidate.second_direct_bias = {};
  }

  if (!IsClosedIntermediateChain(second, &candidate.kill_list)) return false;

  *match = std::move(candidate);
  return true;
}

bool HwFuseTwoLayerVectorMatmulPass::RewriteMatch(const Match& match) {
  if (!match.first || !match.second || !match.block) return false;

  // TypeManager does not model the private HW cooperative type opcodes.  Scan
  // and extend the module's type/value section directly while those types are
  // still present; the outer lowering pass rebuilds TypeManager afterwards.
  const uint32_t vec2_type_id =
      GetOrCreateVectorType(match.input.component_type_id, 2);
  if (vec2_type_id == 0) {
    ReportError(match.second, "failed to create f16vec2 type for HW fusion");
    return false;
  }
  const uint32_t zero_id = GetOrCreateZero(match.input.component_type_id);
  const uint32_t zero_vec2_id = GetOrCreateZero(vec2_type_id);
  const uint32_t glsl_import_id = GetOrCreateGLSLStd450Import();
  if (vec2_type_id == 0 || zero_id == 0 || zero_vec2_id == 0 ||
      glsl_import_id == 0) {
    ReportError(match.second, "failed to create standard types for HW fusion");
    return false;
  }

  InstructionBuilder builder(
      context(), match.second,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  const uint32_t first_input_id =
      match.first->GetSingleWordInOperand(kInputOperand);
  const uint32_t first_matrix_id =
      match.first->GetSingleWordInOperand(kMatrixOperand);
  const uint32_t second_matrix_id =
      match.second->GetSingleWordInOperand(kMatrixOperand);

  // Keep adjacent output columns in f16vec2 accumulators.  There are at most
  // eight of these (P <= 16), and they remain live across all N splits.
  std::vector<uint32_t> output_pair_ids;
  output_pair_ids.reserve((match.output.length + 1) / 2);
  for (uint32_t output_col = 0; output_col < match.output.length;
       output_col += 2) {
    uint32_t bias0 = zero_id;
    uint32_t bias1 = zero_id;
    if (IsVectorMatrixMulAdd(match.second)) {
      const uint32_t bias_id =
          match.second->GetSingleWordInOperand(kBiasOperand);
      bias0 =
          match.second_direct_bias.source_load
              ? BuildVectorElementLoad(&builder, match.input.component_type_id,
                                       match.second_direct_bias, output_col)
              : BuildExtract(&builder, match.input.component_type_id, bias_id,
                             {output_col});
      if (output_col + 1 < match.output.length) {
        bias1 = match.second_direct_bias.source_load
                    ? BuildVectorElementLoad(
                          &builder, match.input.component_type_id,
                          match.second_direct_bias, output_col + 1)
                    : BuildExtract(&builder, match.input.component_type_id,
                                   bias_id, {output_col + 1});
      }
      if (bias0 == 0 || bias1 == 0) return false;
    }
    const uint32_t pair = BuildVec2(&builder, vec2_type_id, bias0, bias1);
    if (pair == 0) return false;
    output_pair_ids.push_back(pair);
  }

  // Calculate one hidden pair and immediately consume it with the
  // corresponding two rows of the second matrix.  The f16vec2 lanes represent
  // adjacent columns, avoiding horizontal reductions and a 16-scalar hidden
  // live range.
  for (uint32_t split_begin = 0; split_begin < match.middle.length;
       split_begin += kSplitWidth) {
    const uint32_t split_end =
        std::min(split_begin + kSplitWidth, match.middle.length);
    for (uint32_t hidden_col = split_begin; hidden_col < split_end;
         hidden_col += 2) {
      uint32_t bias0 = zero_id;
      uint32_t bias1 = zero_id;
      if (IsVectorMatrixMulAdd(match.first)) {
        const uint32_t bias_id =
            match.first->GetSingleWordInOperand(kBiasOperand);
        bias0 = match.first_direct_bias.source_load
                    ? BuildVectorElementLoad(
                          &builder, match.input.component_type_id,
                          match.first_direct_bias, hidden_col)
                    : BuildExtract(&builder, match.input.component_type_id,
                                   bias_id, {hidden_col});
        if (hidden_col + 1 < split_end) {
          bias1 = match.first_direct_bias.source_load
                      ? BuildVectorElementLoad(
                            &builder, match.input.component_type_id,
                            match.first_direct_bias, hidden_col + 1)
                      : BuildExtract(&builder, match.input.component_type_id,
                                     bias_id, {hidden_col + 1});
        }
        if (bias0 == 0 || bias1 == 0) return false;
      }
      uint32_t hidden_pair = BuildVec2(&builder, vec2_type_id, bias0, bias1);
      if (hidden_pair == 0) return false;

      for (uint32_t row = 0; row < match.input.length; ++row) {
        const uint32_t input = BuildExtract(
            &builder, match.input.component_type_id, first_input_id, {row});
        if (input == 0) return false;
        const uint32_t input_pair =
            BuildVec2(&builder, vec2_type_id, input, input);
        if (input_pair == 0) return false;
        const uint32_t weight0 =
            match.first_direct_matrix.source_load
                ? BuildMatrixElementLoad(
                      &builder, match.input.component_type_id,
                      match.first_direct_matrix, row, hidden_col)
                : BuildExtract(&builder, match.input.component_type_id,
                               first_matrix_id, {row, hidden_col});
        if (weight0 == 0) return false;
        uint32_t weight1 = zero_id;
        if (hidden_col + 1 < split_end) {
          weight1 = match.first_direct_matrix.source_load
                        ? BuildMatrixElementLoad(
                              &builder, match.input.component_type_id,
                              match.first_direct_matrix, row, hidden_col + 1)
                        : BuildExtract(&builder, match.input.component_type_id,
                                       first_matrix_id, {row, hidden_col + 1});
          if (weight1 == 0) return false;
        }
        const uint32_t weight_pair =
            BuildVec2(&builder, vec2_type_id, weight0, weight1);
        if (weight_pair == 0) return false;
        hidden_pair =
            BuildFma(&builder, vec2_type_id, glsl_import_id, input_pair,
                     weight_pair, hidden_pair, match.first_fp_fast_math_mode);
        if (hidden_pair == 0) return false;
      }

      if (match.relu) {
        hidden_pair =
            BuildFMaxZero(&builder, vec2_type_id, glsl_import_id, hidden_pair,
                          zero_vec2_id, match.relu_fp_fast_math_mode);
        if (hidden_pair == 0) return false;
      }

      const uint32_t hidden0 = BuildExtract(
          &builder, match.input.component_type_id, hidden_pair, {0});
      if (hidden0 == 0) return false;
      const uint32_t hidden1 = BuildExtract(
          &builder, match.input.component_type_id, hidden_pair, {1});
      if (hidden1 == 0) return false;
      const uint32_t hidden0_pair =
          BuildVec2(&builder, vec2_type_id, hidden0, hidden0);
      if (hidden0_pair == 0) return false;
      const uint32_t hidden1_pair =
          BuildVec2(&builder, vec2_type_id, hidden1, hidden1);
      if (hidden1_pair == 0) return false;

      for (uint32_t output_col = 0; output_col < match.output.length;
           output_col += 2) {
        const size_t output_pair = output_col / 2;
        const uint32_t weight00 =
            match.second_direct_matrix.source_load
                ? BuildMatrixElementLoad(
                      &builder, match.input.component_type_id,
                      match.second_direct_matrix, hidden_col, output_col)
                : BuildExtract(&builder, match.input.component_type_id,
                               second_matrix_id, {hidden_col, output_col});
        if (weight00 == 0) return false;
        uint32_t weight01 = zero_id;
        if (output_col + 1 < match.output.length) {
          weight01 = match.second_direct_matrix.source_load
                         ? BuildMatrixElementLoad(&builder,
                                                  match.input.component_type_id,
                                                  match.second_direct_matrix,
                                                  hidden_col, output_col + 1)
                         : BuildExtract(&builder, match.input.component_type_id,
                                        second_matrix_id,
                                        {hidden_col, output_col + 1});
          if (weight01 == 0) return false;
        }
        const uint32_t weight0_pair =
            BuildVec2(&builder, vec2_type_id, weight00, weight01);
        if (weight0_pair == 0) return false;
        output_pair_ids[output_pair] = BuildFma(
            &builder, vec2_type_id, glsl_import_id, hidden0_pair, weight0_pair,
            output_pair_ids[output_pair], match.second_fp_fast_math_mode);
        if (output_pair_ids[output_pair] == 0) return false;

        if (hidden_col + 1 < split_end) {
          const uint32_t weight10 =
              match.second_direct_matrix.source_load
                  ? BuildMatrixElementLoad(
                        &builder, match.input.component_type_id,
                        match.second_direct_matrix, hidden_col + 1, output_col)
                  : BuildExtract(&builder, match.input.component_type_id,
                                 second_matrix_id,
                                 {hidden_col + 1, output_col});
          if (weight10 == 0) return false;
          uint32_t weight11 = zero_id;
          if (output_col + 1 < match.output.length) {
            weight11 =
                match.second_direct_matrix.source_load
                    ? BuildMatrixElementLoad(&builder,
                                             match.input.component_type_id,
                                             match.second_direct_matrix,
                                             hidden_col + 1, output_col + 1)
                    : BuildExtract(&builder, match.input.component_type_id,
                                   second_matrix_id,
                                   {hidden_col + 1, output_col + 1});
            if (weight11 == 0) return false;
          }
          const uint32_t weight1_pair =
              BuildVec2(&builder, vec2_type_id, weight10, weight11);
          if (weight1_pair == 0) return false;
          output_pair_ids[output_pair] =
              BuildFma(&builder, vec2_type_id, glsl_import_id, hidden1_pair,
                       weight1_pair, output_pair_ids[output_pair],
                       match.second_fp_fast_math_mode);
          if (output_pair_ids[output_pair] == 0) return false;
        }
      }
    }
  }

  std::vector<uint32_t> output_ids;
  output_ids.reserve(match.output.length);
  for (uint32_t output_col = 0; output_col < match.output.length;
       ++output_col) {
    const uint32_t scalar =
        BuildExtract(&builder, match.input.component_type_id,
                     output_pair_ids[output_col / 2], {output_col % 2});
    if (scalar == 0) return false;
    output_ids.push_back(scalar);
  }

  // FP arithmetic decorations are invalid on the replacement construct.  The
  // corresponding mode has already been copied to every generated arithmetic
  // instruction.  NoContraction is deliberately not carried across: this
  // internal optimization follows the relaxed HW matmul lowering contract.
  RemoveFloatingPointDecorations(match.second->result_id());
  std::vector<Operand> operands;
  operands.reserve(output_ids.size());
  for (uint32_t id : output_ids) operands.push_back(IdOperand(id));
  match.second->SetOpcode(spv::Op::OpCompositeConstruct);
  match.second->SetResultType(match.output.type_id);
  match.second->SetInOperands(std::move(operands));
  context()->UpdateDefUse(match.second);

  std::unordered_map<Instruction*, uint32_t> order;
  uint32_t index = 0;
  for (Instruction& inst : *match.block) order[&inst] = index++;
  std::vector<Instruction*> kill_list = match.kill_list;
  std::sort(kill_list.begin(), kill_list.end(),
            [&order](Instruction* lhs, Instruction* rhs) {
              return order[lhs] > order[rhs];
            });
  for (Instruction* inst : kill_list) context()->KillInst(inst);
  return true;
}

Instruction* HwFuseTwoLayerVectorMatmulPass::TraceFunctionValueSource(
    Instruction* value, Instruction* before, std::vector<Instruction*>* chain,
    uint32_t depth) const {
  if (!value || !before || !chain || depth > 16) return nullptr;
  BasicBlock* before_block = context()->get_instr_block(before);

  if (value->opcode() == spv::Op::OpCopyObject ||
      value->opcode() == spv::Op::OpBitcast) {
    if (value->NumInOperands() < 1 ||
        context()->get_instr_block(value) != before_block) {
      return nullptr;
    }
    chain->push_back(value);
    return TraceFunctionValueSource(
        get_def_use_mgr()->GetDef(value->GetSingleWordInOperand(0)), value,
        chain, depth + 1);
  }

  if (value->opcode() != spv::Op::OpLoad) return value;
  if (value->NumInOperands() != 1 ||
      context()->get_instr_block(value) != before_block) {
    return nullptr;
  }

  const uint32_t pointer_id = value->GetSingleWordInOperand(0);
  if (!IsFunctionPointer(pointer_id)) return value;
  Instruction* store = FindLastStoreToFunctionPointer(pointer_id, value);
  if (!store || store->NumInOperands() != 2) return nullptr;

  chain->push_back(value);
  chain->push_back(store);
  return TraceFunctionValueSource(
      get_def_use_mgr()->GetDef(store->GetSingleWordInOperand(1)), store, chain,
      depth + 1);
}

Instruction* HwFuseTwoLayerVectorMatmulPass::FindLastStoreToFunctionPointer(
    uint32_t pointer_id, Instruction* before) const {
  BasicBlock* block = context()->get_instr_block(before);
  if (!block) return nullptr;
  Instruction* last_store = nullptr;
  for (Instruction& inst : *block) {
    if (&inst == before) break;
    if (inst.opcode() == spv::Op::OpStore && inst.NumInOperands() >= 2 &&
        inst.GetSingleWordInOperand(0) == pointer_id) {
      last_store = &inst;
    }
  }
  return last_store;
}

bool HwFuseTwoLayerVectorMatmulPass::IsFunctionPointer(
    uint32_t pointer_id) const {
  Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
  if (!pointer || pointer->type_id() == 0) return false;
  Instruction* pointer_type = get_def_use_mgr()->GetDef(pointer->type_id());
  return pointer_type && pointer_type->opcode() == spv::Op::OpTypePointer &&
         pointer_type->NumInOperands() >= 2 &&
         pointer_type->GetSingleWordInOperand(0) ==
             uint32_t(spv::StorageClass::Function);
}

bool HwFuseTwoLayerVectorMatmulPass::IsClosedIntermediateChain(
    Instruction* second, std::vector<Instruction*>* kill_list) const {
  if (!second || !kill_list) return false;
  BasicBlock* block = context()->get_instr_block(second);
  if (!block) return false;

  std::vector<Instruction*> unique;
  std::unordered_set<Instruction*> kill_set;
  for (Instruction* inst : *kill_list) {
    if (!inst || context()->get_instr_block(inst) != block) return false;
    if (kill_set.insert(inst).second) unique.push_back(inst);
  }

  std::unordered_set<uint32_t> function_pointers;
  for (Instruction* inst : unique) {
    if (inst->result_id() != 0) {
      bool closed = get_def_use_mgr()->WhileEachUser(
          inst, [this, second, &kill_set](Instruction* user) {
            return user == second || IsIgnorableUser(user) ||
                   kill_set.count(user) != 0;
          });
      if (!closed) return false;
    }
    if ((inst->opcode() == spv::Op::OpLoad ||
         inst->opcode() == spv::Op::OpStore) &&
        inst->NumInOperands() >= 1) {
      const uint32_t pointer_id = inst->GetSingleWordInOperand(0);
      if (!IsFunctionPointer(pointer_id)) return false;
      function_pointers.insert(pointer_id);
    }
  }

  for (uint32_t pointer_id : function_pointers) {
    Instruction* pointer = get_def_use_mgr()->GetDef(pointer_id);
    if (!pointer || pointer->opcode() != spv::Op::OpVariable ||
        context()->get_instr_block(pointer) != block) {
      return false;
    }
    const bool closed = get_def_use_mgr()->WhileEachUser(
        pointer, [this, &kill_set](Instruction* user) {
          return IsIgnorableUser(user) || kill_set.count(user) != 0;
        });
    if (!closed) return false;
    if (kill_set.insert(pointer).second) unique.push_back(pointer);
  }

  *kill_list = std::move(unique);
  return true;
}

bool HwFuseTwoLayerVectorMatmulPass::GetDirectMatrixLoad(
    Instruction* value, Instruction* before, const MatrixTypeInfo& matrix,
    DirectMatrixLoadInfo* info, std::vector<Instruction*>* chain) const {
  if (!value || !before || !info || !chain) return false;
  *info = {};
  chain->clear();

  Instruction* source = TraceFunctionValueSource(value, before, chain);
  if (!source || source->opcode() != spv::Op::OpCooperativeMatrixLoadHW ||
      source->type_id() != matrix.type_id ||
      source->NumInOperands() < kMatrixLoadMemoryOperands ||
      context()->get_instr_block(source) !=
          context()->get_instr_block(before) ||
      !CanMoveDirectLoad(source, before) ||
      !MemoryOperandsAreMovable(source, kMatrixLoadMemoryOperands)) {
    chain->clear();
    return false;
  }

  uint32_t layout = 0;
  if (!GetConstantU32(source->GetSingleWordInOperand(kMatrixLoadLayoutOperand),
                      &layout) ||
      layout !=
          static_cast<uint32_t>(spv::CooperativeMatrixLayout::RowMajorKHR)) {
    chain->clear();
    return false;
  }

  DirectMatrixLoadInfo direct;
  direct.source_load = source;
  direct.pointer_id = source->GetSingleWordInOperand(kMatrixLoadPointerOperand);
  if (!GetConstantPairU32(
          source->GetSingleWordInOperand(kMatrixLoadShapeOperand),
          &direct.shape_rows, &direct.shape_cols) ||
      !GetConstantPairU32(
          source->GetSingleWordInOperand(kMatrixLoadOffsetOperand),
          &direct.offset_row, &direct.offset_col)) {
    chain->clear();
    return false;
  }

  const uint64_t last_row = uint64_t(direct.offset_row) + matrix.rows;
  const uint64_t last_col = uint64_t(direct.offset_col) + matrix.cols;
  if (direct.shape_rows == 0 || direct.shape_cols == 0 ||
      last_row > direct.shape_rows || last_col > direct.shape_cols) {
    chain->clear();
    return false;
  }
  // Scalar OpAccessChain indices are 32-bit.  Reject the direct path before
  // rewriting if even the last element would require a wider flattened index;
  // the ordinary aggregate lowering remains available as a fallback.
  const uint64_t max_flat_index =
      (last_row - 1u) * uint64_t(direct.shape_cols) + (last_col - 1u);
  if (max_flat_index > std::numeric_limits<uint32_t>::max()) {
    chain->clear();
    return false;
  }

  Instruction* pointer = get_def_use_mgr()->GetDef(direct.pointer_id);
  Instruction* pointer_type =
      pointer ? get_def_use_mgr()->GetDef(pointer->type_id()) : nullptr;
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer ||
      pointer_type->NumInOperands() < 2) {
    chain->clear();
    return false;
  }
  direct.storage_class = pointer_type->GetSingleWordInOperand(0);
  // Function stores are intentionally allowed while tracing the closed SSA
  // transport chain below.  A direct matrix backed by Function memory could
  // alias those stores, and Private memory has similarly broad module-local
  // aliasing, so keep both on the aggregate fallback path.
  if (direct.storage_class == uint32_t(spv::StorageClass::Function) ||
      direct.storage_class == uint32_t(spv::StorageClass::Private)) {
    chain->clear();
    return false;
  }
  if (direct.storage_class ==
      uint32_t(spv::StorageClass::PhysicalStorageBuffer)) {
    if (source->NumInOperands() <= kMatrixLoadMemoryOperands ||
        (source->GetSingleWordInOperand(kMatrixLoadMemoryOperands) &
         uint32_t(spv::MemoryAccessMask::Aligned)) == 0) {
      chain->clear();
      return false;
    }
  }
  Instruction* pointee =
      get_def_use_mgr()->GetDef(pointer_type->GetSingleWordInOperand(1));
  if (!pointee ||
      (pointee->opcode() != spv::Op::OpTypeArray &&
       pointee->opcode() != spv::Op::OpTypeRuntimeArray) ||
      pointee->NumInOperands() < 1 ||
      pointee->GetSingleWordInOperand(0) != matrix.component_type_id) {
    chain->clear();
    return false;
  }
  direct.element_stride_bytes = GetArrayStride(pointee->result_id());
  if (direct.element_stride_bytes == 0) {
    // This pass only accepts f16 matrices.  Undecorated arrays therefore use
    // the scalar element size for the conservative alignment calculation.
    direct.element_stride_bytes = 2;
  }

  for (uint32_t index = kMatrixLoadMemoryOperands;
       index < source->NumInOperands(); ++index) {
    direct.memory_operands.push_back(source->GetInOperand(index));
  }
  *info = std::move(direct);
  return true;
}

bool HwFuseTwoLayerVectorMatmulPass::GetDirectVectorLoad(
    Instruction* value, Instruction* before, const VectorTypeInfo& vector,
    DirectVectorLoadInfo* info, std::vector<Instruction*>* chain) const {
  if (!value || !before || !info || !chain || vector.length == 0) return false;
  *info = {};
  chain->clear();

  Instruction* source = TraceFunctionValueSource(value, before, chain);
  if (!source || source->opcode() != spv::Op::OpCooperativeVectorLoadHW ||
      source->type_id() != vector.type_id ||
      source->NumInOperands() < kVectorLoadMemoryOperands ||
      context()->get_instr_block(source) !=
          context()->get_instr_block(before) ||
      !CanMoveDirectLoad(source, before) ||
      !MemoryOperandsAreMovable(source, kVectorLoadMemoryOperands)) {
    chain->clear();
    return false;
  }

  DirectVectorLoadInfo direct;
  direct.source_load = source;
  direct.pointer_id = source->GetSingleWordInOperand(kVectorLoadPointerOperand);
  const uint32_t offset_id =
      source->GetSingleWordInOperand(kVectorLoadOffsetOperand);
  if (!GetConstantU32(offset_id, &direct.offset) ||
      uint64_t(direct.offset) + vector.length - 1u >
          std::numeric_limits<uint32_t>::max()) {
    chain->clear();
    return false;
  }
  Instruction* offset_constant = get_def_use_mgr()->GetDef(offset_id);
  Instruction* offset_type =
      offset_constant ? get_def_use_mgr()->GetDef(offset_constant->type_id())
                      : nullptr;
  if (!offset_type || offset_type->opcode() != spv::Op::OpTypeInt ||
      offset_type->NumInOperands() < 2 ||
      (offset_type->GetSingleWordInOperand(1) != 0 &&
       (direct.offset & 0x80000000u) != 0)) {
    chain->clear();
    return false;
  }

  Instruction* pointer = get_def_use_mgr()->GetDef(direct.pointer_id);
  Instruction* pointer_type =
      pointer ? get_def_use_mgr()->GetDef(pointer->type_id()) : nullptr;
  if (!pointer_type || pointer_type->opcode() != spv::Op::OpTypePointer ||
      pointer_type->NumInOperands() < 2) {
    chain->clear();
    return false;
  }
  direct.storage_class = pointer_type->GetSingleWordInOperand(0);
  if (direct.storage_class == uint32_t(spv::StorageClass::Function) ||
      direct.storage_class == uint32_t(spv::StorageClass::Private)) {
    chain->clear();
    return false;
  }
  if (direct.storage_class ==
      uint32_t(spv::StorageClass::PhysicalStorageBuffer)) {
    if (source->NumInOperands() <= kVectorLoadMemoryOperands ||
        (source->GetSingleWordInOperand(kVectorLoadMemoryOperands) &
         uint32_t(spv::MemoryAccessMask::Aligned)) == 0) {
      chain->clear();
      return false;
    }
  }

  Instruction* pointee =
      get_def_use_mgr()->GetDef(pointer_type->GetSingleWordInOperand(1));
  if (!pointee ||
      (pointee->opcode() != spv::Op::OpTypeArray &&
       pointee->opcode() != spv::Op::OpTypeRuntimeArray) ||
      pointee->NumInOperands() < 1 ||
      pointee->GetSingleWordInOperand(0) != vector.component_type_id) {
    chain->clear();
    return false;
  }
  if (pointee->opcode() == spv::Op::OpTypeArray) {
    uint32_t array_length = 0;
    if (pointee->NumInOperands() < 2 ||
        !GetConstantU32(pointee->GetSingleWordInOperand(1), &array_length) ||
        uint64_t(direct.offset) + vector.length > array_length) {
      chain->clear();
      return false;
    }
  }
  direct.element_stride_bytes = GetArrayStride(pointee->result_id());
  if (direct.element_stride_bytes == 0) direct.element_stride_bytes = 2;

  for (uint32_t index = kVectorLoadMemoryOperands;
       index < source->NumInOperands(); ++index) {
    direct.memory_operands.push_back(source->GetInOperand(index));
  }
  *info = std::move(direct);
  return true;
}

bool HwFuseTwoLayerVectorMatmulPass::CanMoveDirectLoad(
    Instruction* load, Instruction* before) const {
  if (!load || !before) return false;
  BasicBlock* block = context()->get_instr_block(load);
  if (!block || block != context()->get_instr_block(before)) return false;

  bool after_load = false;
  for (Instruction& inst : *block) {
    if (&inst == load) {
      after_load = true;
      continue;
    }
    if (&inst == before) return after_load;
    if (!after_load) continue;
    if (inst.IsAtomicOp()) return false;

    if (inst.IsDebugLineInst() || inst.IsNonSemanticInstruction()) continue;
    switch (inst.opcode()) {
      case spv::Op::OpStore:
        // Closed cooperative-value transport uses plain Function stores.  The
        // direct source itself is never Function/Private storage, so these
        // stores cannot alias it.  Optional memory operands are rejected.
        if (inst.NumInOperands() != 2 ||
            !IsFunctionPointer(inst.GetSingleWordInOperand(0))) {
          return false;
        }
        break;
      case spv::Op::OpCooperativeMatrixLoadHW:
        if (!MemoryOperandsAreMovable(&inst, kMatrixLoadMemoryOperands)) {
          return false;
        }
        break;
      case spv::Op::OpCooperativeVectorLoadHW:
        if (!MemoryOperandsAreMovable(&inst, 2)) return false;
        break;
      case spv::Op::OpCooperativeVectorMatrixMulHW:
      case spv::Op::OpCooperativeVectorMatrixMulAddHW:
        break;
      case spv::Op::OpExtInst:
        if (!IsGlslFMax(&inst)) return false;
        break;
      default:
        // Use a pure-op allowlist instead of a memory-effect denylist.  New or
        // extension instructions (including async copies, split/named
        // barriers, cooperative stores, calls and image writes) are unsafe by
        // default and force the aggregate fallback.
        if (!inst.IsOpcodeCodeMotionSafe()) return false;
        if (inst.opcode() == spv::Op::OpLoad &&
            !MemoryOperandsAreMovable(&inst, 1)) {
          return false;
        }
        break;
    }
  }
  return false;
}

bool HwFuseTwoLayerVectorMatmulPass::MemoryOperandsAreMovable(
    const Instruction* inst, uint32_t first_memory_operand) const {
  if (!inst || inst->NumInOperands() <= first_memory_operand) return true;
  const Operand& access = inst->GetInOperand(first_memory_operand);
  if (access.type != SPV_OPERAND_TYPE_MEMORY_ACCESS ||
      access.words.size() != 1) {
    return false;
  }

  const uint32_t mask = access.words[0];
  const uint32_t aligned = uint32_t(spv::MemoryAccessMask::Aligned);
  const uint32_t alias_scope =
      uint32_t(spv::MemoryAccessMask::AliasScopeINTELMask);
  const uint32_t no_alias = uint32_t(spv::MemoryAccessMask::NoAliasINTELMask);
  const uint32_t allowed = aligned |
                           uint32_t(spv::MemoryAccessMask::Nontemporal) |
                           uint32_t(spv::MemoryAccessMask::NonPrivatePointer) |
                           alias_scope | no_alias;
  if ((mask & ~allowed) != 0) return false;

  uint32_t parameter_index = first_memory_operand + 1;
  if ((mask & aligned) != 0) {
    if (parameter_index >= inst->NumInOperands()) return false;
    const uint32_t alignment = inst->GetSingleWordInOperand(parameter_index++);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
  }
  if ((mask & alias_scope) != 0) {
    if (parameter_index >= inst->NumInOperands() ||
        !IsModuleVisibleValue(
            inst->GetSingleWordInOperand(parameter_index++))) {
      return false;
    }
  }
  if ((mask & no_alias) != 0) {
    if (parameter_index >= inst->NumInOperands() ||
        !IsModuleVisibleValue(
            inst->GetSingleWordInOperand(parameter_index++))) {
      return false;
    }
  }
  return parameter_index == inst->NumInOperands();
}

uint32_t HwFuseTwoLayerVectorMatmulPass::GetArrayStride(
    uint32_t array_type_id) const {
  uint32_t stride = 0;
  context()->get_decoration_mgr()->WhileEachDecoration(
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

bool HwFuseTwoLayerVectorMatmulPass::IsModuleVisibleValue(uint32_t id) const {
  Instruction* inst = get_def_use_mgr()->GetDef(id);
  return inst && context()->get_instr_block(inst) == nullptr;
}

bool HwFuseTwoLayerVectorMatmulPass::GetVectorType(uint32_t type_id,
                                                   VectorTypeInfo* info) const {
  if (!info) return false;
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  if (!type || type->opcode() != spv::Op::OpTypeCooperativeVectorHW ||
      type->NumInOperands() != 2) {
    return false;
  }
  uint32_t length = 0;
  if (!GetConstantU32(type->GetSingleWordInOperand(1), &length)) return false;
  *info = {type_id, type->GetSingleWordInOperand(0), length};
  return true;
}

bool HwFuseTwoLayerVectorMatmulPass::GetMatrixType(uint32_t type_id,
                                                   MatrixTypeInfo* info) const {
  if (!info) return false;
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  // MatrixUse* is an optional fourth operand on the HW type.  It does not
  // affect the logical row/column shape used by this fusion.
  if (!type || type->opcode() != spv::Op::OpTypeCooperativeMatrixHW ||
      type->NumInOperands() < 3) {
    return false;
  }
  uint32_t rows = 0;
  uint32_t cols = 0;
  if (!GetConstantU32(type->GetSingleWordInOperand(1), &rows) ||
      !GetConstantU32(type->GetSingleWordInOperand(2), &cols)) {
    return false;
  }
  *info = {type_id, type->GetSingleWordInOperand(0), rows, cols};
  return true;
}

bool HwFuseTwoLayerVectorMatmulPass::GetConstantU32(uint32_t id,
                                                    uint32_t* value) const {
  if (!value) return false;
  Instruction* constant = get_def_use_mgr()->GetDef(id);
  if (!constant || constant->opcode() != spv::Op::OpConstant ||
      constant->NumInOperands() != 1) {
    return false;
  }
  Instruction* type = get_def_use_mgr()->GetDef(constant->type_id());
  if (!type || type->opcode() != spv::Op::OpTypeInt ||
      type->NumInOperands() < 2 || type->GetSingleWordInOperand(0) != 32) {
    return false;
  }
  *value = constant->GetSingleWordInOperand(0);
  return true;
}

bool HwFuseTwoLayerVectorMatmulPass::GetConstantPairU32(
    uint32_t id, uint32_t* first, uint32_t* second) const {
  if (!first || !second) return false;
  Instruction* pair = get_def_use_mgr()->GetDef(id);
  if (!pair) return false;
  if (pair->opcode() == spv::Op::OpConstantComposite &&
      pair->NumInOperands() == 2) {
    return GetConstantU32(pair->GetSingleWordInOperand(0), first) &&
           GetConstantU32(pair->GetSingleWordInOperand(1), second);
  }
  if (pair->opcode() == spv::Op::OpConstantCompositeReplicateEXT &&
      pair->NumInOperands() == 1) {
    return GetConstantU32(pair->GetSingleWordInOperand(0), first) &&
           GetConstantU32(pair->GetSingleWordInOperand(0), second);
  }
  return false;
}

bool HwFuseTwoLayerVectorMatmulPass::IsFloat16Type(uint32_t type_id) const {
  Instruction* type = get_def_use_mgr()->GetDef(type_id);
  return type && type->opcode() == spv::Op::OpTypeFloat &&
         type->NumInOperands() == 1 && type->GetSingleWordInOperand(0) == 16;
}

bool HwFuseTwoLayerVectorMatmulPass::IsZeroConstant(uint32_t id) const {
  std::unordered_set<uint32_t> visited;
  return IsZeroConstantImpl(id, &visited);
}

bool HwFuseTwoLayerVectorMatmulPass::IsZeroConstantImpl(
    uint32_t id, std::unordered_set<uint32_t>* visited) const {
  if (!visited) return false;
  // Constants form an acyclic graph in valid SPIR-V.  A constituent id may be
  // repeated many times in OpConstantComposite, which is still a zero value.
  if (!visited->insert(id).second) return true;
  Instruction* constant = get_def_use_mgr()->GetDef(id);
  if (!constant) return false;
  if (constant->opcode() == spv::Op::OpConstantNull) return true;
  if (constant->opcode() == spv::Op::OpConstant) {
    if (!IsFloat16Type(constant->type_id()) || constant->NumInOperands() == 0) {
      return false;
    }
    for (uint32_t i = 0; i < constant->NumInOperands(); ++i) {
      if (constant->GetSingleWordInOperand(i) != 0) return false;
    }
    return true;
  }
  if ((constant->opcode() != spv::Op::OpConstantComposite &&
       constant->opcode() != spv::Op::OpConstantCompositeReplicateEXT) ||
      constant->NumInOperands() == 0) {
    return false;
  }
  for (uint32_t i = 0; i < constant->NumInOperands(); ++i) {
    if (!IsZeroConstantImpl(constant->GetSingleWordInOperand(i), visited)) {
      return false;
    }
  }
  return true;
}

bool HwFuseTwoLayerVectorMatmulPass::IsVectorMatrixMul(
    const Instruction* inst) const {
  return inst && (inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulHW ||
                  inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddHW);
}

bool HwFuseTwoLayerVectorMatmulPass::IsVectorMatrixMulAdd(
    const Instruction* inst) const {
  return inst && inst->opcode() == spv::Op::OpCooperativeVectorMatrixMulAddHW;
}

bool HwFuseTwoLayerVectorMatmulPass::IsGlslFMax(const Instruction* inst) const {
  if (!inst || inst->opcode() != spv::Op::OpExtInst ||
      inst->NumInOperands() != 4 ||
      inst->GetSingleWordInOperand(1) != GLSLstd450FMax) {
    return false;
  }
  Instruction* import =
      get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  return import && import->opcode() == spv::Op::OpExtInstImport &&
         import->NumInOperands() == 1 &&
         import->GetInOperand(0).AsString() == "GLSL.std.450";
}

bool HwFuseTwoLayerVectorMatmulPass::IsIgnorableUser(
    const Instruction* inst) const {
  return inst && (spvOpcodeIsDebug(inst->opcode()) || inst->IsDecoration() ||
                  inst->IsNonSemanticInstruction() || inst->IsDebugLineInst());
}

uint32_t HwFuseTwoLayerVectorMatmulPass::GetOrCreateGLSLStd450Import() {
  uint32_t import_id =
      context()->get_feature_mgr()->GetExtInstImportId_GLSLstd450();
  if (import_id != 0) return import_id;
  import_id = get_module()->GetExtInstImportId("GLSL.std.450");
  if (import_id != 0) return import_id;

  import_id = TakeNextId();
  if (import_id == 0) return 0;
  context()->AddExtInstImport(MakeUnique<Instruction>(
      context(), spv::Op::OpExtInstImport, 0, import_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_LITERAL_STRING,
                                      utils::MakeVector("GLSL.std.450")}}));
  return import_id;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::GetOrCreateVectorType(
    uint32_t component_type_id, uint32_t component_count) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeVector && inst.NumInOperands() == 2 &&
        inst.GetSingleWordInOperand(0) == component_type_id &&
        inst.GetSingleWordInOperand(1) == component_count) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  auto type = MakeUnique<Instruction>(
      context(), spv::Op::OpTypeVector, 0, result_id,
      std::initializer_list<Operand>{
          IdOperand(component_type_id),
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {component_count}}});
  Instruction* added = type.get();
  context()->AddType(std::move(type));
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(added);
  }
  return result_id;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::GetOrCreateUIntType() {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypeInt && inst.NumInOperands() == 2 &&
        inst.GetSingleWordInOperand(0) == 32 &&
        inst.GetSingleWordInOperand(1) == 0) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  auto type = MakeUnique<Instruction>(
      context(), spv::Op::OpTypeInt, 0, result_id,
      std::initializer_list<Operand>{{SPV_OPERAND_TYPE_LITERAL_INTEGER, {32}},
                                     {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}}});
  Instruction* added = type.get();
  context()->AddType(std::move(type));
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(added);
  }
  return result_id;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::GetOrCreateUIntConstant(
    uint32_t value) {
  const uint32_t uint_type_id = GetOrCreateUIntType();
  if (uint_type_id == 0) return 0;
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpConstant &&
        inst.type_id() == uint_type_id && inst.NumInOperands() == 1 &&
        inst.GetSingleWordInOperand(0) == value) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  auto constant = MakeUnique<Instruction>(
      context(), spv::Op::OpConstant, uint_type_id, result_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, {value}}});
  Instruction* added = constant.get();
  context()->AddGlobalValue(std::move(constant));
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(added);
  }
  return result_id;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::GetOrCreatePointerType(
    uint32_t pointee_type_id, uint32_t storage_class) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.opcode() == spv::Op::OpTypePointer && inst.NumInOperands() == 2 &&
        inst.GetSingleWordInOperand(0) == storage_class &&
        inst.GetSingleWordInOperand(1) == pointee_type_id) {
      return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  auto type = MakeUnique<Instruction>(
      context(), spv::Op::OpTypePointer, 0, result_id,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_STORAGE_CLASS, {storage_class}},
          IdOperand(pointee_type_id)});
  Instruction* added = type.get();
  context()->AddType(std::move(type));
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(added);
  }
  return result_id;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::GetOrCreateZero(uint32_t type_id) {
  for (Instruction& inst : get_module()->types_values()) {
    if (inst.type_id() != type_id) continue;
    if (inst.opcode() == spv::Op::OpConstantNull) return inst.result_id();
    if (inst.opcode() == spv::Op::OpConstant && inst.NumInOperands() != 0) {
      bool zero = true;
      for (uint32_t index = 0; index < inst.NumInOperands(); ++index) {
        const Operand& operand = inst.GetInOperand(index);
        zero &= std::all_of(operand.words.begin(), operand.words.end(),
                            [](uint32_t word) { return word == 0; });
      }
      if (zero) return inst.result_id();
    }
  }

  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  auto constant =
      MakeUnique<Instruction>(context(), spv::Op::OpConstantNull, type_id,
                              result_id, std::initializer_list<Operand>{});
  Instruction* added = constant.get();
  context()->AddGlobalValue(std::move(constant));
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    get_def_use_mgr()->AnalyzeInstDefUse(added);
  }
  context()->AddCapability(spv::Capability::Float16);
  return result_id;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::GetFPFastMathMode(
    uint32_t result_id) const {
  if (result_id == 0) return 0;
  for (const Instruction* decoration :
       context()->get_decoration_mgr()->GetDecorationsFor(
           result_id, /*include_linkage=*/false)) {
    if (decoration && decoration->NumInOperands() >= 3 &&
        decoration->GetSingleWordInOperand(1) ==
            uint32_t(spv::Decoration::FPFastMathMode)) {
      return decoration->GetSingleWordInOperand(2);
    }
  }
  return 0;
}

void HwFuseTwoLayerVectorMatmulPass::ApplyFPFastMathMode(Instruction* inst,
                                                         uint32_t mode) {
  if (!inst || inst->result_id() == 0 || mode == 0) return;
  context()->get_decoration_mgr()->AddDecorationVal(
      inst->result_id(), uint32_t(spv::Decoration::FPFastMathMode), mode);
}

void HwFuseTwoLayerVectorMatmulPass::RemoveFloatingPointDecorations(
    uint32_t result_id) {
  if (result_id == 0) return;
  context()->get_decoration_mgr()->RemoveDecorationsFrom(
      result_id, [](const Instruction& decoration) {
        if (decoration.NumInOperands() < 2) return false;
        const uint32_t kind = decoration.GetSingleWordInOperand(1);
        return kind == uint32_t(spv::Decoration::FPFastMathMode) ||
               kind == uint32_t(spv::Decoration::NoContraction);
      });
}

uint32_t HwFuseTwoLayerVectorMatmulPass::BuildExtract(
    InstructionBuilder* builder, uint32_t type_id, uint32_t composite_id,
    const std::vector<uint32_t>& indices) {
  if (!builder || type_id == 0 || composite_id == 0) return 0;
  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;

  std::vector<Operand> operands;
  operands.reserve(1 + indices.size());
  operands.push_back(IdOperand(composite_id));
  for (uint32_t index : indices) {
    operands.push_back({SPV_OPERAND_TYPE_LITERAL_INTEGER, {index}});
  }
  Instruction* extract = builder->AddInstruction(MakeUnique<Instruction>(
      context(), spv::Op::OpCompositeExtract, type_id, result_id, operands));
  return extract ? extract->result_id() : 0;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::BuildMatrixElementLoad(
    InstructionBuilder* builder, uint32_t component_type_id,
    const DirectMatrixLoadInfo& direct, uint32_t row, uint32_t col) {
  if (!builder || !direct.source_load || direct.shape_cols == 0) return 0;
  const uint64_t global_row = uint64_t(direct.offset_row) + row;
  const uint64_t global_col = uint64_t(direct.offset_col) + col;
  const uint64_t index = global_row * direct.shape_cols + global_col;
  if (index > std::numeric_limits<uint32_t>::max()) return 0;

  const uint32_t index_id = GetOrCreateUIntConstant(uint32_t(index));
  const uint32_t pointer_type_id =
      GetOrCreatePointerType(component_type_id, direct.storage_class);
  if (index_id == 0 || pointer_type_id == 0) return 0;
  const uint32_t pointer_result_id = TakeNextId();
  if (pointer_result_id == 0) return 0;
  Instruction* pointer = builder->AddInstruction(MakeUnique<Instruction>(
      context(), spv::Op::OpAccessChain, pointer_type_id, pointer_result_id,
      std::initializer_list<Operand>{IdOperand(direct.pointer_id),
                                     IdOperand(index_id)}));
  if (!pointer) return 0;

  std::vector<Operand> memory_operands = direct.memory_operands;
  if (!memory_operands.empty() &&
      (memory_operands[0].words[0] &
       uint32_t(spv::MemoryAccessMask::Aligned)) != 0) {
    const uint64_t byte_offset = index * direct.element_stride_bytes;
    const uint32_t base_alignment = memory_operands[1].words[0];
    const uint64_t alignment = std::gcd(uint64_t(base_alignment), byte_offset);
    if (alignment == 0 || alignment > std::numeric_limits<uint32_t>::max()) {
      return 0;
    }
    memory_operands[1].words[0] = uint32_t(alignment);
  }

  std::vector<Operand> operands;
  operands.reserve(1 + memory_operands.size());
  operands.push_back(IdOperand(pointer->result_id()));
  operands.insert(operands.end(), memory_operands.begin(),
                  memory_operands.end());
  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  Instruction* load = builder->AddInstruction(MakeUnique<Instruction>(
      context(), spv::Op::OpLoad, component_type_id, result_id, operands));
  return load ? load->result_id() : 0;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::BuildVectorElementLoad(
    InstructionBuilder* builder, uint32_t component_type_id,
    const DirectVectorLoadInfo& direct, uint32_t logical_index) {
  if (!builder || !direct.source_load) return 0;
  const uint64_t index = uint64_t(direct.offset) + logical_index;
  if (index > std::numeric_limits<uint32_t>::max()) return 0;

  const uint32_t index_id = GetOrCreateUIntConstant(uint32_t(index));
  const uint32_t pointer_type_id =
      GetOrCreatePointerType(component_type_id, direct.storage_class);
  if (index_id == 0 || pointer_type_id == 0) return 0;
  const uint32_t pointer_result_id = TakeNextId();
  if (pointer_result_id == 0) return 0;
  Instruction* pointer = builder->AddInstruction(MakeUnique<Instruction>(
      context(), spv::Op::OpAccessChain, pointer_type_id, pointer_result_id,
      std::initializer_list<Operand>{IdOperand(direct.pointer_id),
                                     IdOperand(index_id)}));
  if (!pointer) return 0;

  std::vector<Operand> memory_operands = direct.memory_operands;
  if (!memory_operands.empty() &&
      (memory_operands[0].words[0] &
       uint32_t(spv::MemoryAccessMask::Aligned)) != 0) {
    const uint64_t byte_offset = index * direct.element_stride_bytes;
    const uint32_t base_alignment = memory_operands[1].words[0];
    const uint64_t alignment = std::gcd(uint64_t(base_alignment), byte_offset);
    if (alignment == 0 || alignment > std::numeric_limits<uint32_t>::max()) {
      return 0;
    }
    memory_operands[1].words[0] = uint32_t(alignment);
  }

  std::vector<Operand> operands;
  operands.reserve(1 + memory_operands.size());
  operands.push_back(IdOperand(pointer->result_id()));
  operands.insert(operands.end(), memory_operands.begin(),
                  memory_operands.end());
  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  Instruction* load = builder->AddInstruction(MakeUnique<Instruction>(
      context(), spv::Op::OpLoad, component_type_id, result_id, operands));
  return load ? load->result_id() : 0;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::BuildVec2(InstructionBuilder* builder,
                                                   uint32_t vec2_type_id,
                                                   uint32_t lane0,
                                                   uint32_t lane1) {
  if (!builder || vec2_type_id == 0 || lane0 == 0 || lane1 == 0) return 0;
  const uint32_t result_id = TakeNextId();
  if (result_id == 0) return 0;
  Instruction* value = builder->AddInstruction(MakeUnique<Instruction>(
      context(), spv::Op::OpCompositeConstruct, vec2_type_id, result_id,
      std::initializer_list<Operand>{IdOperand(lane0), IdOperand(lane1)}));
  return value ? value->result_id() : 0;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::BuildFma(InstructionBuilder* builder,
                                                  uint32_t vec2_type_id,
                                                  uint32_t glsl_import_id,
                                                  uint32_t lhs, uint32_t rhs,
                                                  uint32_t accumulator,
                                                  uint32_t fp_fast_math_mode) {
  if (!builder) return 0;
  Instruction* fma = builder->AddNaryExtendedInstruction(
      vec2_type_id, glsl_import_id, GLSLstd450Fma, {lhs, rhs, accumulator});
  ApplyFPFastMathMode(fma, fp_fast_math_mode);
  return fma ? fma->result_id() : 0;
}

uint32_t HwFuseTwoLayerVectorMatmulPass::BuildFMaxZero(
    InstructionBuilder* builder, uint32_t component_type_id,
    uint32_t glsl_import_id, uint32_t value_id, uint32_t zero_id,
    uint32_t fp_fast_math_mode) {
  if (!builder) return 0;
  Instruction* maximum = builder->AddNaryExtendedInstruction(
      component_type_id, glsl_import_id, GLSLstd450FMax, {value_id, zero_id});
  ApplyFPFastMathMode(maximum, fp_fast_math_mode);
  return maximum ? maximum->result_id() : 0;
}

void HwFuseTwoLayerVectorMatmulPass::ReportError(const Instruction*,
                                                 const char* message) const {
  if (!consumer()) return;
  consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, message);
}

}  // namespace opt
}  // namespace spvtools
