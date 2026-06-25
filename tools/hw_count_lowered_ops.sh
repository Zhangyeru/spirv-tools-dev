#!/usr/bin/env bash
# Copyright (c) 2025 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <lowered.spvasm>"
  exit 1
fi

file="$1"

echo "file: $file"

printf "OpCompositeExtract: "
grep -c "OpCompositeExtract" "$file" || true

printf "OpFMul: "
grep -c "OpFMul" "$file" || true

printf "OpFAdd: "
grep -c "OpFAdd" "$file" || true

printf "OpLoad: "
grep -c "OpLoad" "$file" || true

printf "OpStore: "
grep -c "OpStore" "$file" || true

printf "HW: "
grep -c "HW" "$file" || true

printf "CooperativeMatrixKHR: "
grep -c "CooperativeMatrixKHR" "$file" || true

printf "OpCooperativeMatrix: "
grep -c "OpCooperativeMatrix" "$file" || true

printf "OpTypeCooperativeMatrixKHR: "
grep -c "OpTypeCooperativeMatrixKHR" "$file" || true
