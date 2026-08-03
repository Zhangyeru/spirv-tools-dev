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

#ifndef TEST_OPT_HW_LOWER_TO_STANDARD_TEST_UTILS_H_
#define TEST_OPT_HW_LOWER_TO_STANDARD_TEST_UTILS_H_

#include <cstddef>
#include <string>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {

using HwLowerToStandardTest = PassTest<::testing::Test>;

size_t CountSubstring(const std::string& text, const std::string& needle);
std::string MakeAsymmetricHwElementwiseModule(const std::string& diagnostic,
                                              const std::string& instruction);
void ExpectFailureWithMissingExtInstOperand(HwLowerToStandardTest* test,
                                            const std::string& text);
void ExpectSingleElementwiseLoop(const std::string& text);
void ExpectNoHwOrCoopMatrix(const std::string& text);
void ExpectPackedVec4Math(const std::string& text,
                          const std::string& component_name);
void ExpectPackedVec4MatmulPattern(const std::string& text,
                                   const std::string& component_name);
void ExpectScalarFallbackMath(const std::string& text,
                              const std::string& component_name = "%float");
std::string RunWithInjectedAlignedMemoryAccess(HwLowerToStandardTest* test,
                                               const std::string& text,
                                               spv::Op opcode);
std::string RunWithInjectedFullMemoryAccess(HwLowerToStandardTest* test,
                                            const std::string& text,
                                            spv::Op opcode, bool is_load);
std::string RunWithInjectedAliasMemoryAccess(HwLowerToStandardTest* test,
                                             const std::string& text);

}  // namespace opt
}  // namespace spvtools

#endif  // TEST_OPT_HW_LOWER_TO_STANDARD_TEST_UTILS_H_
