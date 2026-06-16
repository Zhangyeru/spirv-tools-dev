#!/usr/bin/env bash
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
