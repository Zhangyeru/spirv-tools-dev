// Copyright (c) 2018 Google LLC.
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

// Validates correctness of barrier SPIR-V instructions.

#include <string>

#include "source/opcode.h"
#include "source/spirv_constant.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validate_memory_semantics.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

spv_result_t ValidateBarrierId(ValidationState_t& _, const Instruction* inst) {
  const auto operand_id = inst->GetOperandAs<uint32_t>(0);
  const auto type_id = _.GetOperandTypeId(inst, 0);
  const auto opcode_name = spvOpcodeString(inst->opcode());

  if (!_.IsIntScalarType(type_id) || _.GetBitWidth(type_id) != 32) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opcode_name << " Id <id> "
           << _.getIdName(operand_id)
           << " must be a 32-bit integer scalar.";
  }

  int64_t value = 0;
  if (_.EvalConstantValInt64(operand_id, &value) && value < 0) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opcode_name << " Id must be non-negative.";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateBarrierCount(ValidationState_t& _,
                                  const Instruction* inst) {
  const auto operand_id = inst->GetOperandAs<uint32_t>(1);
  const auto* operand_inst = _.FindDef(operand_id);
  const auto type_id = _.GetOperandTypeId(inst, 1);
  const auto opcode_name = spvOpcodeString(inst->opcode());

  if (!operand_inst || !_.IsIntScalarType(type_id) || _.GetBitWidth(type_id) != 32 ||
      (operand_inst->opcode() != spv::Op::OpConstant &&
       operand_inst->opcode() != spv::Op::OpConstantNull)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opcode_name << " N <id> " << _.getIdName(operand_id)
           << " must be a 32-bit integer compile-time constant.";
  }

  int64_t value = 0;
  if (_.EvalConstantValInt64(operand_id, &value) && value < 0) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << opcode_name << " N must be non-negative.";
  }

  return SPV_SUCCESS;
}

// Validates correctness of barrier instructions.
spv_result_t BarriersPass(ValidationState_t& _, const Instruction* inst) {
  const spv::Op opcode = inst->opcode();
  const uint32_t result_type = inst->type_id();

  switch (opcode) {
    case spv::Op::OpControlBarrier: {
      if (_.version() < SPV_SPIRV_VERSION_WORD(1, 3)) {
        _.function(inst->function()->id())
            ->RegisterExecutionModelLimitation(
                [](spv::ExecutionModel model, std::string* message) {
                  if (model != spv::ExecutionModel::TessellationControl &&
                      model != spv::ExecutionModel::GLCompute &&
                      model != spv::ExecutionModel::Kernel &&
                      model != spv::ExecutionModel::TaskNV &&
                      model != spv::ExecutionModel::MeshNV) {
                    if (message) {
                      *message =
                          "OpControlBarrier requires one of the following "
                          "Execution "
                          "Models: TessellationControl, GLCompute, Kernel, "
                          "MeshNV or TaskNV";
                    }
                    return false;
                  }
                  return true;
                });
      }

      const uint32_t execution_scope = inst->word(1);
      const uint32_t memory_scope = inst->word(2);

      if (auto error = ValidateExecutionScope(_, inst, execution_scope)) {
        return error;
      }

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, 2, memory_scope)) {
        return error;
      }
      break;
    }

    case spv::Op::OpMemoryBarrier: {
      const uint32_t memory_scope = inst->word(1);

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, 1, memory_scope)) {
        return error;
      }
      break;
    }

    case spv::Op::OpNamedBarrierInitialize: {
      if (_.GetIdOpcode(result_type) != spv::Op::OpTypeNamedBarrier) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Result Type to be OpTypeNamedBarrier";
      }

      const uint32_t subgroup_count_type = _.GetOperandTypeId(inst, 2);
      if (!_.IsIntScalarType(subgroup_count_type) ||
          _.GetBitWidth(subgroup_count_type) != 32) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Subgroup Count to be a 32-bit int";
      }
      break;
    }

    case spv::Op::OpMemoryNamedBarrier: {
      const uint32_t named_barrier_type = _.GetOperandTypeId(inst, 0);
      if (_.GetIdOpcode(named_barrier_type) != spv::Op::OpTypeNamedBarrier) {
        return _.diag(SPV_ERROR_INVALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Named Barrier to be of type OpTypeNamedBarrier";
      }

      const uint32_t memory_scope = inst->word(2);

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, 2, memory_scope)) {
        return error;
      }
      break;
    }

    case spv::Op::OpBarrierArriveHW:
    case spv::Op::OpBarrierWaitHW: {
      if (auto error = ValidateBarrierId(_, inst)) {
        return error;
      }
      if (auto error = ValidateBarrierCount(_, inst)) {
        return error;
      }
      break;
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
