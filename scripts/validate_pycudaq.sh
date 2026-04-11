#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Run this script to validate CUDA-Q Python wheel installation.
# If run from the repo root without -f, test files are auto-detected.
#
# Options:
#   -f <folder>: Root folder containing README.md, tests/, examples/, snippets/
#                If not provided, auto-detects from repo structure
#   -v <version>: CUDA-Q version to install (required)
#   -c <cuda_version>: Full CUDA version for conda, e.g., "12.8.0" (Linux only)
#   -i <packages_dir>: Directory containing wheel files (--find-links)
#   -p <python_version>: Python version (default: 3.11)
#   -q: Quick test mode (only run core tests)
#   -F: Force fresh venv (macOS only; delete and recreate instead of reusing)
#
# Examples:
#   # From repo root with auto-detection:
#   bash scripts/validate_pycudaq.sh -v 0.9.0 -i dist -c 12.6.0
#
#   # With custom test folder (CI container):
#   bash scripts/validate_pycudaq.sh -v 0.9.0 -f /tmp -c 12.6.0
#
# For CI containers, copy test files to a single folder:
#   COPY docs/sphinx/examples/python /tmp/examples/
#   COPY docs/sphinx/snippets/python /tmp/snippets/
#   COPY python/tests /tmp/tests/
#   COPY python/README.md.in /tmp/README.md
#
# Note: To run target tests, set the necessary API keys (IONQ_API_KEY, etc.)
#
# TODO: Unify wheel validation around this script. Currently:
#   - ci_macos.yml uses this script (auto-detects from repo, runs everything)
#   - publishing.yml uses this script (selective dir copy to /tmp, no targets/)
#   - python_wheels.yml does NOT use this script; it runs pytest, snippets,
#     and examples directly with explicit `find` exclusions in Docker containers.
# The goal is to have all wheel validation run through this script, replacing the
# inline find/pytest commands in python_wheels.yml. Use -q for quick
# (core pytest only, suitable for per-PR CI) and full mode for publishing.
# This avoids duplicating skip logic between the script and workflow files.

__optind__=$OPTIND
OPTIND=1
python_version=3.11
quick_test=false
fresh_venv=false
while getopts ":c:f:Fi:p:qv:" opt; do
    case $opt in
    c)
        cuda_version_conda="$OPTARG"
        ;;
    f)
        root_folder="$OPTARG"
        ;;
    F)
        fresh_venv=true
        ;;
    p)
        python_version="$OPTARG"
        ;;
    q)
        quick_test=true
        ;;
    i)
        extra_packages="$OPTARG"
        ;;
    v)
        cudaq_version="$OPTARG"
        ;;
    \?)
        echo "Invalid command line option -$OPTARG" >&2
        (return 0 2>/dev/null) && return 100 || exit 100
        ;;
    esac
done
OPTIND=$__optind__

# Sanitize environment: unset variables that could leak build-tree or
# system-installed CUDA-Q libraries into the validation environment.
# Without this, DYLD_LIBRARY_PATH from a prior build step can cause the
# wheel to load libraries from _skbuild/ instead of its own bundled copies,
# masking packaging bugs.
SANITIZE_VARS="
DYLD_LIBRARY_PATH
DYLD_FALLBACK_LIBRARY_PATH
LD_LIBRARY_PATH
CUDAQ_INSTALL_PREFIX
CUDA_QUANTUM_PATH
PYTHONPATH
"

for var in $SANITIZE_VARS; do
    unset "$var"
done

echo "Environment sanitized (unset: $SANITIZE_VARS)"

# Limit OpenMP threads: pytest-xdist handles process-level parallelism,
# so each worker only needs 1 OMP thread for small test simulations.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Parallel job count for snippet/example execution (xargs -P)
if [ "$(uname)" = "Darwin" ]; then
    parallel_jobs=$(sysctl -n hw.ncpu)
else
    parallel_jobs=$(nproc)
fi

# Auto-detect repo structure if -f not provided
if [ -z "$root_folder" ]; then
    # Try to find repo root
    this_file_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root=$(cd "$this_file_dir" && git rev-parse --show-toplevel 2>/dev/null)

    # Look for README.md or README.md.in (template)
    readme_src="$repo_root/python/README.md"
    if [ ! -f "$readme_src" ]; then
        readme_src="$repo_root/python/README.md.in"
    fi

    if [ -n "$repo_root" ] && [ -f "$readme_src" ]; then
        echo "Auto-detecting test files from repo: $repo_root"

        # Use staging location in repo (gitignored via /*build*/)
        staging_dir="$repo_root/build/validation"
        echo "Setting up staging directory: $staging_dir"
        rm -rf "${staging_dir:?}"
        mkdir -p "$staging_dir"

        # Copy test files to staging (mirrors CI copy structure).
        # Use cp -r instead of symlinks for robustness: find(1) may not
        # follow initial symlinks in all environments, and CI runners may
        # have different filesystem semantics.
        cp -f "$readme_src" "$staging_dir/README.md"
        cp -r "$repo_root/python/tests" "$staging_dir/tests"
        cp -r "$repo_root/docs/sphinx/examples/python" "$staging_dir/examples"
        cp -r "$repo_root/docs/sphinx/snippets/python" "$staging_dir/snippets"
        if [ -d "$repo_root/docs/sphinx/targets/python" ]; then
            cp -r "$repo_root/docs/sphinx/targets/python" "$staging_dir/targets"
        fi

        root_folder="$staging_dir"
    fi
fi

readme_file="$root_folder/README.md"
if [ ! -d "$root_folder" ] || [ ! -f "$readme_file" ]; then
    ls "$root_folder" 2>/dev/null
    echo -e "\e[01;31mDid not find Python root folder. Please pass the folder containing the README and tests with -f, or run from repo root.\e[0m" >&2
    (return 0 2>/dev/null) && return 100 || exit 100
fi

echo "Using test root folder: $root_folder"

# Detect platform and GPU availability
is_macos=false
has_cuda=false
if [ "$(uname)" = "Darwin" ]; then
    is_macos=true
    echo "macOS detected: running CPU-only validation"
else
    if [ -x "$(command -v nvidia-smi)" ] && nvidia-smi &>/dev/null; then
        has_cuda=true
    fi
fi
if ! $has_cuda; then
    echo "No CUDA GPU detected: GPU-dependent tests will be skipped"
fi

# Check if a Python file requires a GPU target that is not available.
# Returns 0 (true) if the file should be skipped, 1 (false) otherwise.
requires_unavailable_gpu_target() {
    local file="$1"
    if $has_cuda; then
        return 1
    fi
    local targets
    targets=$(awk -F'"' '/cudaq\.set_target/ {print $2}' "$file")
    for t in $targets; do
        case "$t" in
            nvidia|nvidia-fp64|nvidia-mgpu|dynamics|tensornet|remote-mqpu)
                echo "Skipping $file (requires GPU target '$t')" >&2
                return 0
                ;;
        esac
    done
    return 1
}

# Check that the `cuda_version_conda` is a full version string like "12.8.0" (Linux only)
if ! $is_macos && ! [[ $cuda_version_conda =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "\e[01;31mThe cuda_version_conda (-c) must be a full version string like '12.8.0'. Provided: '${cuda_version_conda}'.\e[0m" >&2
    (return 0 2>/dev/null) && return 100 || exit 100
fi

# Install Miniconda (Linux only - macOS uses venv)
if ! $is_macos && [ ! -x "$(command -v conda)" ]; then
    mkdir -p ~/.miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ~/.miniconda3/miniconda.sh
    bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
    rm -rf ~/.miniconda3/miniconda.sh
    eval "$(~/.miniconda3/bin/conda shell.bash hook)"
fi

# Execute instructions from the README file
if [ -n "${extra_packages}" ]; then
    pip_extra_arg="--find-links ${extra_packages}"
fi

if $is_macos; then
    # macOS: extract install commands from README markers, matching the Linux
    # pattern.
    venv_dir="$HOME/.venv/cudaq-validation"

    if $fresh_venv && [ -d "$venv_dir" ]; then
        echo "Removing existing venv at $venv_dir"
        rm -rf "$venv_dir"
    fi

    macos_script="$(awk '/(Begin macos install)/{flag=1;next}/(End macos install)/{flag=0}flag' "$readme_file" | grep . | sed '/^```/d')"
    if [ -z "$macos_script" ]; then
        echo -e "\e[01;31mNo macOS install instructions found in $readme_file (missing Begin/End macos install markers).\e[0m" >&2
        (return 0 2>/dev/null) && return 100 || exit 100
    fi

    # Build the pip install replacement for the README's "pip install cudaq".
    # When a local packages dir is provided (-i), install the wheel file
    # directly since the wheel's distribution name (cuda_quantum) differs
    # from the metapackage name (cudaq).
    if [ -n "${extra_packages}" ]; then
        metapackage=$(ls "${extra_packages}"/cudaq-*.tar.gz 2>/dev/null | head -1)
        if [ -n "$metapackage" ]; then
            pip_install_replacement="pip install --force-reinstall cudaq==${cudaq_version} --find-links ${extra_packages}"
        else
            wheel_file=$(ls "${extra_packages}"/cuda_quantum*.whl 2>/dev/null | head -1)
            if [ -n "$wheel_file" ]; then
                pip_install_replacement="pip install --force-reinstall $wheel_file"
            else
                echo -e "\e[01;31mNo wheel or metapackage found in ${extra_packages}.\e[0m" >&2
                (return 0 2>/dev/null) && return 100 || exit 100
            fi
        fi
    else
        pip_install_replacement="pip install cudaq==${cudaq_version}"
    fi

    while IFS= read -r line; do
        # Redirect venv creation into the CI-managed venv_dir
        line=$(echo "$line" | sed -E "s|python3 -m venv [^ ]+|python3 -m venv $venv_dir|g")
        line=$(echo "$line" | sed -E "s|source [^ ]+/bin/activate|source $venv_dir/bin/activate|g")
        # Replace 'pip install cudaq' with versioned install + local wheel path
        line=$(echo "$line" | sed -E 's/\$\{\{\s*[^}]+\s*\}\}/cudaq/g')
        line=$(echo "$line" | sed -E "s|pip install cudaq|${pip_install_replacement}|g")
        if [ -n "$(echo $line | tr -d '[:space:]')" ]; then
            echo "+ $line"
            eval "$line"
        fi
    done <<<"$macos_script"

    # Install test/dev dependencies (pytest, etc.)
    echo "Installing dev/test dependencies..."
    pip install -r "$this_file_dir/../requirements-dev.txt"
else
    # Linux: full conda setup with CUDA and MPI
    conda_script="$(awk '/(Begin conda install)/{flag=1;next}/(End conda install)/{flag=0}flag' "$readme_file" | grep . | sed '/^```/d')"
    if [ -z "$conda_script" ]; then
        echo -e "\e[01;31mNo conda install instructions found in $readme_file (missing Begin/End conda install markers).\e[0m" >&2
        (return 0 2>/dev/null) && return 100 || exit 100
    fi

    while IFS= read -r line; do
        line=$(echo $line | sed -E "s/cuda_version=(.\{\{)?\s?\S+\s?(\}\})?/cuda_version=${cuda_version_conda} /g")
        line=$(echo $line | sed -E "s/python(=)?3.[0-9]{1,}/python\1${python_version}/g")
        # Replace template variables like ${{ package_name }} or package names with actual install
        line=$(echo "$line" | sed -E 's/\$\{\{\s*[^}]+\s*\}\}/cudaq/g')
        line=$(echo "$line" | sed -E "s|pip install cudaq|pip install cudaq==${cudaq_version} -v ${pip_extra_arg}|g")
        if [ -n "$(echo $line | grep "conda activate")" ]; then
            conda_env=$(echo "$line" | sed "s#conda activate##" | tr -d '[:space:]')
            source $(conda info --base)/bin/activate $conda_env
        elif [ -n "$(echo $line | grep "conda create")" ]; then
            # Skip conda create if environment already exists
            env_name=$(echo "$line" | grep -oE '\-n\s+[^\s]+' | sed 's/-n //')
            if ! conda env list | grep -q "$env_name"; then
                # Use conda-forge to avoid Anaconda ToS requirements
                line=$(echo "$line" | sed 's/conda create/conda create -c conda-forge/')
                eval "$line"
            fi
        elif [ -n "$(echo $line | tr -d '[:space:]')" ]; then
            eval "$line"
        fi
    done <<<"$conda_script"
    pip install pytest pytest-xdist
fi

# Run OpenMPI setup (Linux only)
if ! $is_macos; then
    ompi_script="$(awk '/(Begin ompi setup)/{flag=1;next}/(End ompi setup)/{flag=0}flag' "$readme_file" | grep . | sed '/^```/d')"
    while IFS= read -r line; do
        if [ -n "$(echo $line | tr -d '[:space:]')" ]; then
            eval "$line"
        fi
    done <<<"$ompi_script"
fi
status_sum=0

# Run all tests from a temp directory so the repo tree (cwd, _skbuild/,
# etc.) cannot accidentally be found via sys.path or dyld search paths
# to ensure the wheel is installed in isolation.
test_workdir=$(mktemp -d)
echo "Running tests from isolated directory: $test_workdir"
cd "$test_workdir"

# Verify that the necessary GPU targets are installed and usable (Linux only)
if $is_macos; then
    echo "Skipping GPU target verification on macOS (CPU-only)"
else
    for tgt in nvidia nvidia-fp64 nvidia-mgpu tensornet; do
        python3 -c "import cudaq; cudaq.set_target('${tgt}')"
        if [ $? -ne 0 ]; then
            echo -e "\e[01;31mPython trivial test for target ${tgt} failed.\e[0m" >&2
            status_sum=$((status_sum + 1))
        fi
    done
fi

# Run core tests
echo "Running core tests."
core_test_args=(
    -v
    -n
    auto
    "$root_folder/tests"
    --ignore
    "$root_folder/tests/backends"
    --ignore
    "$root_folder/tests/dynamics/integrators"
    --ignore
    "$root_folder/tests/parallel"
    --ignore
    "$root_folder/tests/domains"
)

macos_serial_core_tests=()
if $is_macos; then
    ghz_noise_test="test_simple_run_ghz_with_noise"
    ghz_noise_nodeid="$root_folder/tests/kernel/test_run_kernel.py::${ghz_noise_test}"
    core_test_args+=(
        -k
        "not ${ghz_noise_test}"
    )
    macos_serial_core_tests+=("$ghz_noise_nodeid")
    echo "macOS xdist workaround: excluding ${ghz_noise_test} from parallel core run; it will run serially."
fi

python3 -m pytest "${core_test_args[@]}"
if [ ! $? -eq 0 ]; then
    echo -e "\e[01;31mPython tests failed.\e[0m" >&2
    status_sum=$((status_sum + 1))
fi

for serial_test in "${macos_serial_core_tests[@]}"; do
    echo "Running serial core test: $serial_test"
    python3 -m pytest -v --rootdir "$root_folder/tests" "$serial_test"
    if [ ! $? -eq 0 ]; then
        echo -e "\e[01;31mPython serial core test failed: $serial_test\e[0m" >&2
        status_sum=$((status_sum + 1))
    fi
done

# If this is a quick test, we return here.
if $quick_test; then
    if [ ! $status_sum -eq 0 ]; then
        echo -e "\e[01;31mValidation produced errors.\e[0m" >&2
    fi
    (return 0 2>/dev/null) && return $status_sum || exit $status_sum
fi

# Run backend tests (single invocation with xdist; --rootdir matches upstream import layout)
echo "Running backend tests."
python3 -m pytest -v -n auto --rootdir "$root_folder/tests" "$root_folder/tests/backends"
status=$?
# Exit code 5 indicates that no tests were collected.
if [ ! $status -eq 0 ] && [ ! $status -eq 5 ]; then
    echo -e "\e[01;31mPython backend tests failed with code $status.\e[0m" >&2
    status_sum=$((status_sum + 1))
fi

# Run platform tests (Linux only - requires MPI)
if $is_macos; then
    echo "Skipping parallel/platform tests on macOS (requires MPI)"
else
    echo "Running platform tests."
    for parallelTest in "$root_folder/tests/parallel"/*.py; do
        python3 -m pytest -v --rootdir "$root_folder/tests" $parallelTest
        if [ ! $? -eq 0 ]; then
            echo -e "\e[01;31mPython platform test $parallelTest failed.\e[0m" >&2
            status_sum=$((status_sum + 1))
        fi
    done
fi

# Run torch integrator tests.
# This is an optional integrator, which requires torch and torchdiffeq.
if $is_macos; then
    echo "Skipping torch GPU integrator tests on macOS (CPU-only)"
else
    # Install torch separately to match the cuda version.
    # Torch if installed as part of torchdiffeq's dependencies, may default to the latest cuda version.
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu$(echo $cuda_version | cut -d '.' -f-2 | tr -d .)
    python3 -m pip install torchdiffeq
    python3 -m pytest -v "$root_folder/tests/dynamics/integrators"
    if [ ! $? -eq 0 ]; then
        echo -e "\e[01;31mPython tests failed.\e[0m" >&2
        status_sum=$((status_sum + 1))
    fi
fi

# Run snippets in docs
snippet_list=$(mktemp)
for ex in $(find "$root_folder/snippets" -name '*.py'); do
    if ! requires_unavailable_gpu_target "$ex"; then
        printf '%s\0' "$ex"
    fi
done > "$snippet_list"
if [ -s "$snippet_list" ]; then
    xargs -0 -P "$parallel_jobs" -n 1 bash -c \
        'echo "Executing $1"; python3 "$1" || { echo -e "\e[01;31mFailed to execute $1.\e[0m" >&2; exit 1; }' _ \
        < "$snippet_list"
    if [ $? -ne 0 ]; then
        status_sum=$((status_sum + 1))
    fi
fi
[ -f "$snippet_list" ] && rm -f "$snippet_list"

# Run examples (pre-filter sequentially, execute in parallel)
example_list=$(mktemp)
for ex in $(find "$root_folder/examples" -name '*.py'); do
    if requires_unavailable_gpu_target "$ex"; then continue; fi
    skip_example=false
    explicit_targets=$(awk -F'"' '/cudaq\.set_target/ {print $2}' "$ex")
    for t in $explicit_targets; do
        if [ "$t" == "quera" ] || [ "$t" == "braket" ]; then
            echo -e "\e[01;31mWarning: Explicitly set target braket or quera in $ex; skipping validation due to paid submission.\e[0m" >&2
            skip_example=true
        elif [ "$t" == "pasqal" ] && [ -z "${PASQAL_PASSWORD}" ]; then
            echo -e "\e[01;31mWarning: Explicitly set target pasqal in $ex; skipping validation due to missing token.\e[0m" >&2
            skip_example=true
        fi
    done
    if ! $skip_example; then
        printf '%s\0' "$ex"  # don't split on spaces
    fi
done > "$example_list"
if [ -s "$example_list" ]; then
    xargs -0 -P "$parallel_jobs" -n 1 bash -c \
        'echo "Executing $1"; python3 "$1" || { echo -e "\e[01;31mFailed to execute $1.\e[0m" >&2; exit 1; }' _ \
        < "$example_list"
    if [ $? -ne 0 ]; then
        status_sum=$((status_sum + 1))
    fi
fi
[ -f "$example_list" ] && rm -f "$example_list"

snippet_count=$(find "$root_folder/snippets" -name '*.py' 2>/dev/null | wc -l)
example_count=$(find "$root_folder/examples" -name '*.py' 2>/dev/null | wc -l)
if [ "$snippet_count" -eq 0 ] && [ "$example_count" -eq 0 ]; then
    echo -e "\e[01;31mNo snippets or examples found in $root_folder. Check staging setup.\e[0m" >&2
    status_sum=$((status_sum + 1))
fi

# Run target tests if target folder exists (pre-filter, execute in parallel).
if [ -d "$root_folder/targets" ]; then
    target_list=$(mktemp)
    for ex in $(find "$root_folder/targets" -name '*.py'); do
        if requires_unavailable_gpu_target "$ex"; then continue; fi
        skip_example=false
        explicit_targets=$(awk -F'"' '/cudaq\.set_target/ {print $2}' "$ex")
        for t in $explicit_targets; do
            if [ "$t" == "quera" ] || [ "$t" == "braket" ]; then
                echo -e "\e[01;31mWarning: Explicitly set target braket or quera in $ex; skipping validation due to paid submission.\e[0m" >&2
                skip_example=true
            elif [ "$t" == "fermioniq" ] && [ -z "${FERMIONIQ_ACCESS_TOKEN_ID}" ]; then
                echo -e "\e[01;31mWarning: Explicitly set target fermioniq in $ex; skipping validation due to missing API key.\e[0m" >&2
                skip_example=true
            elif [ "$t" == "qci" ] && [ -z "${QCI_AUTH_TOKEN}" ]; then
                echo -e "\e[01;31mWarning: Explicitly set target qci in $ex; skipping validation due to missing API key.\e[0m" >&2
                skip_example=true
            elif [ "$t" == "oqc" ] && [ -z "${OQC_URL}" ]; then
                echo -e "\e[01;31mWarning: Explicitly set target oqc in $ex; skipping validation due to missing URL.\e[0m" >&2
                skip_example=true
            elif [ "$t" == "pasqal" ] && [ -z "${PASQAL_PASSWORD}" ]; then
                echo -e "\e[01;31mWarning: Explicitly set target pasqal in $ex; skipping validation due to missing token.\e[0m" >&2
                skip_example=true
            elif [ "$t" == "ionq" ] && [ -z "${IONQ_API_KEY}" ]; then
                echo -e "\e[01;31mWarning: Explicitly set target ionq in $ex; skipping validation due to missing API key.\e[0m" >&2
                skip_example=true
            elif [ "$t" == "tii" ] || [ "$t" == "scaleway" ] || [ "$t" == "quantum_machines" ] || \
                 [ "$t" == "quantinuum" ] || [ "$t" == "orca" ] || [ "$t" == "orca-photonics" ] || \
                 [ "$t" == "iqm" ] || [ "$t" == "infleqtion" ] || [ "$t" == "anyon" ]; then
                echo "Skipping $ex (remote target '$t' not available)" >&2
                skip_example=true
            fi
        done
        if ! $skip_example; then
            printf '%s\0' "$ex"  # don't split on spaces
        fi
    done > "$target_list"
    if [ -s "$target_list" ]; then
        xargs -0 -P "$parallel_jobs" -n 1 bash -c \
            'echo "Executing $1"; python3 "$1" || { echo -e "\e[01;31mFailed to execute $1.\e[0m" >&2; exit 1; }' _ \
            < "$target_list"
        if [ $? -ne 0 ]; then
            status_sum=$((status_sum + 1))
        fi
    fi
    [ -f "$target_list" ] && rm -f "$target_list"
fi

# Run remote-mqpu platform test (Linux only - requires GPU and MPI)
if $is_macos; then
    echo "Skipping remote-mqpu platform test on macOS (requires GPU and MPI)"
else
    # Use cudaq-qpud.py wrapper script to automatically find dependencies for the Python wheel configuration.
    # Note that a derivative of this code is in
    # docs/sphinx/using/backends/platform.rst, so if you update it here, you need to
    # check if any docs updates are needed.
    cudaq_package=$(python3 -m pip list | grep -oE 'cudaq')
    cudaq_location=$(python3 -m pip show ${cudaq_package} | grep -e 'Location: .*$')
    qpud_py="${cudaq_location#Location: }/bin/cudaq-qpud.py"
    if [ -x "$(command -v nvidia-smi)" ]; then
        nr_gpus=$(nvidia-smi --list-gpus | wc -l)
    else
        nr_gpus=0
    fi
    server1_devices=$(echo $(seq $((nr_gpus >> 1)) $((nr_gpus - 1))) | tr ' ' ,)
    server2_devices=$(echo $(seq 0 $((($nr_gpus >> 1) - 1))) | tr ' ' ,)
    echo "Launching server 1..."
    servers="localhost:12001"
    CUDA_VISIBLE_DEVICES=$server1_devices mpiexec --allow-run-as-root -np 2 python3 "$qpud_py" --port 12001 &
    if [ -n "$server2_devices" ]; then
        echo "Launching server 2..."
        servers+=",localhost:12002"
        CUDA_VISIBLE_DEVICES=$server2_devices mpiexec --allow-run-as-root -np 2 python3 "$qpud_py" --port 12002 &
    fi

    sleep 20 # wait for servers to launch
    python3 "$root_folder/snippets/using/cudaq/platform/sample_async_remote.py" \
        --backend nvidia-mgpu --servers "$servers"
    if [ ! $? -eq 0 ]; then
        echo -e "\e[01;31mRemote platform test failed.\e[0m" >&2
        status_sum=$((status_sum + 1))
    fi
    kill %1 && wait %1 2>/dev/null
    if [ -n "$server2_devices" ]; then
        kill %2 && wait %2 2>/dev/null
    fi
fi

if [ ! $status_sum -eq 0 ]; then
    echo -e "\e[01;31mValidation produced errors.\e[0m" >&2
    (return 0 2>/dev/null) && return $status_sum || exit $status_sum
fi
