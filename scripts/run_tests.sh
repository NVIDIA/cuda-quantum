#!/bin/bash
# Run CUDA-Q test suite (ctest + llvm-lit)
# Used by both Linux and macOS CI, and for local development.
#
# Usage: bash scripts/run_tests.sh [-v] [-B build_dir]
#
# Note: GPU tests will fail gracefully on macOS (no CUDA available)

this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$this_file_dir/set_env_defaults.sh"

build_dir="build"
verbose=""

while getopts ":vB:" opt; do
  case $opt in
    v) verbose="-v" ;;
    B) build_dir="$OPTARG" ;;
  esac
done

status_sum=0

# Set PYTHONPATH to find the built cudaq module
export PYTHONPATH="$build_dir/python:${PYTHONPATH:-}"

# 1. CTest
echo "=== Running ctest ==="
ctest --output-on-failure --test-dir "$build_dir" --timeout 300 \
  -E "ctest-nvqpp|ctest-targettests"
status_sum=$((status_sum + $?))

# 2. Main lit tests
echo "=== Running llvm-lit (build/test) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests \
  --param nvqpp_site_config="$build_dir/test/lit.site.cfg.py" \
  "$build_dir/test"
status_sum=$((status_sum + $?))

# 3. Target tests
echo "=== Running llvm-lit (build/targettests) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests \
  --param nvqpp_site_config="$build_dir/targettests/lit.site.cfg.py" \
  "$build_dir/targettests"
status_sum=$((status_sum + $?))

# 4. Python MLIR tests
echo "=== Running llvm-lit (python/tests/mlir) ==="
"$LLVM_INSTALL_PREFIX/bin/llvm-lit" $verbose --time-tests \
  --param nvqpp_site_config="$build_dir/python/tests/mlir/lit.site.cfg.py" \
  "$build_dir/python/tests/mlir"
status_sum=$((status_sum + $?))

# 5. Python interop tests
echo "=== Running pytest (interop tests) ==="
python3 -m pytest $verbose --durations=0 "$build_dir/python/tests/interop/"
status_sum=$((status_sum + $?))

exit $((status_sum > 0 ? 1 : 0))
