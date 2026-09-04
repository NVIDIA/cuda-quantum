#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Validate a cudaq-devel wheel: contents check + optional out-of-tree build of a
# minimal Python extension, located at `examples/plugins/mlir_extension`.
#
# Usage:
#   bash scripts/validate_devel_wheel.sh -i dist -r dist
#
# Options:
#   -i <dir>: Directory containing cudaq_devel*.whl (required)
#   -r <dir>: Directory containing cuda-quantum-cu*.whl for co-install (optional;
#             provides libcudaqMLIR that the extension links and loads). If
#             omitted, pip will attempt to resolve the cudaq dependency from the index.
#   -q: Skip the out-of-tree Python-extension smoke build (contents check only)

set -euo pipefail

devel_dir=""
runtime_dir=""
quick=false

while getopts ":i:r:q" opt; do
  case $opt in
  i) devel_dir="$OPTARG" ;;
  r) runtime_dir="$OPTARG" ;;
  q) quick=true ;;
  *) echo "Usage: $0 -i <devel_wheel_dir> [-r <runtime_wheel_dir>] [-q]" >&2; exit 1 ;;
  esac
done

if [ -z "$devel_dir" ]; then
  echo "Error: -i <devel_wheel_dir> is required" >&2
  exit 1
fi

devel_wheel=$(ls "$devel_dir"/cudaq_devel*.whl 2>/dev/null | head -1)
if [ -z "$devel_wheel" ]; then
  echo "Error: no cudaq_devel*.whl found in $devel_dir" >&2
  exit 1
fi
echo "Validating devel wheel: $devel_wheel"

devel_listing=$(unzip -l "$devel_wheel")

# Contents: expect dev artifacts, not runtime-only paths stripped from runtime wheel.
for pattern in 'include/cudaq/' 'lib/cmake/cudaq/CUDAQConfig.cmake' 'bin/mlir-tblgen'; do
  if ! grep -q "$pattern" <<< "$devel_listing"; then
    echo "Error: expected path matching '$pattern' in devel wheel" >&2
    exit 1
  fi
  echo "  OK: found $pattern"
done

if grep -q 'lib/libcudaq\.so' <<< "$devel_listing"; then
  echo "Error: devel wheel should not ship libcudaq.so (provided by cudaq runtime)" >&2
  exit 1
fi
echo "  OK: libcudaq.so not bundled in devel wheel"

# Reject build-machine absolute paths that break out-of-tree consumers.
scratch=$(mktemp -d)
venv_dir=""
smoke_dir=""
trap 'rm -rf "$venv_dir" "$smoke_dir" "$scratch"' EXIT
unzip -q -d "$scratch" "$devel_wheel" \
  'lib/cmake/*Targets*.cmake' 'lib/cmake/*/*Targets*.cmake' \
  'bin/clang.cfg' 'bin/clang++.cfg' \
  'lib/cmake/llvm/LLVMConfig.cmake' 2>/dev/null || true
if [ -z "$(find "$scratch" -type f -print -quit)" ]; then
  echo "Error: no files extracted from devel wheel" >&2
  exit 1
fi
bad_paths=$(grep -RInE '/Users/|/home/|/usr/|\.local/bin/|/opt/homebrew/' "$scratch" 2>/dev/null || true)
if [ -n "$bad_paths" ]; then
  echo "Error: devel wheel contains absolute builder paths:" >&2
  echo "$bad_paths" | head -20 >&2
  exit 1
fi
echo "  OK: no absolute builder paths found in CMake exports / clang.cfg"

if $quick; then
  echo "Skipping smoke build (-q)"
  exit 0
fi

python="${PYTHON:-python3}"
venv_dir=$(mktemp -d)
smoke_dir=$(mktemp -d)
"$python" -m venv "$venv_dir"
# shellcheck source=/dev/null
source "$venv_dir/bin/activate"
pip install -q pip wheel 'nanobind>=2.12.0,<3'

runtime_wheel=""
if [ -n "$runtime_dir" ]; then
  runtime_wheel=$(ls "$runtime_dir"/cuda_quantum_cu*.whl 2>/dev/null | head -1)
  if [ -n "$runtime_wheel" ]; then
    echo "Installing runtime wheel: $runtime_wheel"
    pip install -q "$runtime_wheel"
  fi
fi

if [ -n "$runtime_wheel" ]; then
  echo "Installing devel wheel (no-deps; runtime co-installed above)"
  pip install -q --no-deps "$devel_wheel"
else
  echo "Warning: no runtime wheel provided (-r); letting pip resolve the cudaq" >&2
  echo "         runtime dependency (which provides libcudaqMLIR) from the index." >&2
  echo "Installing devel wheel (resolving dependencies)"
  pip install -q "$devel_wheel"
fi

site_packages=$("$venv_dir/bin/python" -c 'import site; print(site.getsitepackages()[0])')
echo "Installing into site-packages: $site_packages"

# The out-of-tree example project that gets built against the installed wheels.
# Package / module names and the install component match its CMake targets.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$script_dir")"
example_src="$repo_root/examples/plugins/mlir_extension"
example_component="TrivialMLIRPythonModules"
example_pkg="cudaq_mlir_extension"
example_mod="_mlirExtension"

if [ ! -f "$example_src/CMakeLists.txt" ]; then
  echo "Error: example project not found at $example_src" >&2
  exit 1
fi

echo "=== CMake configure ($example_src) ==="
if ! cmake -S "$example_src" -B "$smoke_dir/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja 2>&1; then
  echo ""
  echo "FAIL: CMake configure failed." >&2
  exit 1
fi

echo ""
echo "=== CMake build ==="
if ! cmake --build "$smoke_dir/build" 2>&1; then
  echo ""
  echo "FAIL: Build failed." >&2
  exit 1
fi

echo ""
echo "=== Install extension into site-packages ==="
if ! cmake --install "$smoke_dir/build" \
        --prefix "$site_packages" \
        --component "$example_component" 2>&1; then
  echo ""
  echo "FAIL: Install failed." >&2
  exit 1
fi

echo ""
echo "=== Load extension ==="
if ! python - "$site_packages" "$example_pkg" "$example_mod" << 'PY'
import glob
import importlib.util
import os
import sys

site_packages, pkg, mod = sys.argv[1:4]
libdir = os.path.join(site_packages, pkg, "mlir", "_mlir_libs")
matches = glob.glob(os.path.join(libdir, mod + "*.so"))
if not matches:
    print("FAIL: built extension %s* not found in %s" % (mod, libdir), file=sys.stderr)
    sys.exit(1)

# Add the extension dir so sibling shared libs (nanobind runtime, common CAPI)
# resolve via RPATH, then import the freshly installed module.
sys.path.insert(0, libdir)
spec = importlib.util.spec_from_file_location(mod, matches[0])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if not module.run_trivial_pass():
    print("FAIL: trivial dialect/pass did not run", file=sys.stderr)
    sys.exit(1)

print("  OK: imported %s and ran the trivial pass via libcudaqMLIR" % mod)
PY
then
  echo ""
  echo "FAIL: extension did not install/load correctly." >&2
  exit 1
fi

echo ""
echo "PASS: devel wheel validation succeeded (raw extension load)."

echo ""
echo "=== pip install example package ==="
if ! pip install -q "$example_src" 2>&1; then
  echo ""
  echo "FAIL: pip install of example package failed." >&2
  exit 1
fi

echo ""
echo "=== Verify downstream dialect auto-registration ==="
if ! python - << 'PY'
import cudaq
from cudaq.mlir.ir import Context

with Context() as ctx:
    _ = ctx.dialects["trivial"]
print("  OK: trivial dialect auto-registered via cudaq.mlir_dialects entry point")
PY
then
  echo ""
  echo "FAIL: downstream dialect was not auto-registered in cudaq.mlir.ir.Context." >&2
  exit 1
fi

echo ""
echo "PASS: devel wheel validation succeeded."
