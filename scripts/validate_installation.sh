#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
#
# From repo root (auto-sets examples):
#   bash scripts/validate_installation.sh              # Full validation
#   bash scripts/validate_installation.sh default      # Test specific backend(s)
#
# In Docker container:
#   Launch the NVIDIA CUDA-Q Docker container,
#   and run this script from the home directory.
#   Check the logged output.

# Auto-setup if running from repo root (detected by presence of docs/sphinx/examples)
# This allows running validation directly from the repo without manual setup.
if [ -d "docs/sphinx/examples" ]; then
    echo "Setting up examples from repo structure..."
    repo_root="$(pwd)"
    staging_dir="$repo_root/build/validation_staging"
    rm -rf "${staging_dir:?}"
    mkdir -p "$staging_dir"

    for d in examples applications targets snippets; do
        cp -r "docs/sphinx/$d" "$staging_dir/$d" || exit 1
    done
    # Remove Python subdirs from examples/applications/targets (use snippets for Python)
    rm -rf "${staging_dir:?}/examples/python" "${staging_dir:?}/applications/python" "${staging_dir:?}/targets/python"
    
    cd "$staging_dir"
    echo "Running validation from: $(pwd)"
fi

passed=0
failed=0
skipped=0
samples=0

if [ -x "$(command -v nvidia-smi)" ] && [ "$(nvidia-smi | egrep -o "CUDA Version: ([0-9]{1,}\.)+[0-9]{1,}")" != "" ]; 
then gpu_available=true
else gpu_available=false
fi

if $gpu_available; 
then echo "GPU detected." && nvidia-smi
else echo "No GPU detected."
fi 

if [ -x "$(command -v ssh -V)" ]; 
then ssh_available=true
else ssh_available=false
fi

if $ssh_available; 
then echo "SSH Client detected." 
else echo "No SSH Client detected."
fi 

if [ -x "$(command -v mpiexec --version)" ]; 
then mpi_available=true
else mpi_available=false
fi
if $mpi_available; 
then echo "MPI detected."
else echo "No MPI detected."
fi 

# Check if a Python file should be skipped due to GPU/MPI requirements
# Usage: should_skip_python_example <filepath>
# Returns: 0 (true) if should skip, 1 (false) if should run
# Sets: skip_reason variable with the reason for skipping
should_skip_python_example() {
    local file="$1"
    skip_reason=""
    
    # Skip GPU-dependent examples when no GPU is available
    if ! $gpu_available; then
        if grep -q "import cupy\|from cupy" "$file"; then
            skip_reason="requires cupy (CUDA-only)"
            return 0
        fi
        # Check for nvidia/tensornet targets (direct or via remote-mqpu)
        if grep -q "set_target.*['\"]nvidia\|['\"]tensornet" "$file"; then
            skip_reason="requires GPU target"
            return 0
        fi
    fi
    
    # Skip MPI examples when MPI is not available
    if ! $mpi_available; then
        if grep -q "cudaq\.mpi\|mpi_comm_world" "$file"; then
            skip_reason="requires MPI"
            return 0
        fi
    fi
    
    return 1
}

export UCX_LOG_LEVEL=warning
requested_backends=`\
    for target in $@; \
    do echo "$target"; \
    done`

installed_backends=`\
    echo "default"
    for file in $(ls $CUDA_QUANTUM_PATH/targets/*.yml); \
    do basename $file | cut -d "." -f 1; \
    done`

# remote_rest targets are automatically filtered, 
# so is execution on the photonics backend and the stim backend
# This will test all NVIDIA-derivative targets in the legacy mode,
# i.e., nvidia-fp64, nvidia-mgpu, nvidia-mqpu, etc., are treated as standalone targets.
available_backends=`\
    echo "default"
    for file in $(ls $CUDA_QUANTUM_PATH/targets/*.yml); \
    do
        if grep -q "library-mode-execution-manager: photonics" $file ; then 
          continue
        fi 
        # Skip optimization test targets
        if [[ $file == *"opt-test.yml" ]]; then
          continue
        fi
        if grep -q "nvqir-simulation-backend: stim" $file ; then 
          continue
        fi 
        platform=$(cat $file | grep "platform-qpu:")
        qpu=${platform##* }
        requirements=$(cat $file | grep "gpu-requirements:")
        gpus=${requirements##* }
        if [ "${qpu}" != "remote_rest" ] \
        && [ "${qpu}" != "fermioniq" ] && [ "${qpu}" != "orca" ] \
        && [ "${qpu}" != "pasqal" ] && [ "${qpu}" != "quera" ] \
        && ($gpu_available || [ -z "$gpus" ] || [ "${gpus,,}" == "false" ]); then \
            basename $file | cut -d "." -f 1; \
        fi; \
    done`

missing_backend=false
if [ $# -eq 0 ]
then
    requested_backends="$available_backends"
else
    for t in $requested_backends
    do
        echo $available_backends | grep -w -q $t
        if [ ! $? -eq 0 ];
        then
            echo "No backend configuration found for $t."
            missing_backend=true
        fi
    done    
fi

# Temporary solution until we stop reading backends names from configuration file.
# This avoids duplicate testing during container validation in the publishing task.
for backend_to_remove in nvidia-fp64 nvidia-mgpu nvidia-mqpu-fp64 nvidia-mqpu-mps nvidia-mqpu
do
    requested_backends=$(echo "$requested_backends" | grep -vx "$backend_to_remove")
done

echo
echo "Installed backends:"
echo "$installed_backends"
echo
echo "Available backends:"
echo "$available_backends"
echo
echo "Testing backends:"
echo "$requested_backends"
echo

if $missing_backend || [ "$available_backends" == "" ]; 
then
    echo "Abort due to missing backend configuration."
    exit 1 
fi

# Long-running tests
tensornet_backend_skipped_tests=(\
    examples/cpp/other/builder/vqe_h2_builder.cpp \
    examples/cpp/other/builder/qaoa_maxcut_builder.cpp \
    applications/cpp/vqe_h2.cpp \
    applications/cpp/qaoa_maxcut.cpp \
    examples/cpp/other/builder/builder.cpp \
    applications/cpp/amplitude_estimation.cpp)

echo "============================="
echo "==        C++ Tests        =="
echo "============================="

# Note: piping the `find` results through `sort` guarantees repeatable ordering.
tmpFile=$(mktemp)
for ex in `find examples/ applications/ targets/ -name '*.cpp' | sort`;
do
    filename=$(basename -- "$ex")
    filename="${filename%.*}"
    echo "Testing $filename:"
    echo "Source: $ex"
    let "samples+=1"

    # Look for a --target flag to nvq++ in the 
    # comment block at the beginning of the file.
    # Note: using sed instead of grep -P for macOS compatibility
    intended_target=$(sed -e '/^$/,$d' "$ex" | sed -n 's|^//[[:space:]]*nvq++.*--target[[:space:]]\{1,\}\([^[:space:]]\{1,\}\).*|\1|p' | head -1)
    if [ -n "$intended_target" ]; then
        echo "Intended for execution on $intended_target backend."
    fi
    use_library_mode=$(sed -e '/^$/,$d' "$ex" | grep -o '^//[[:space:]]*nvq++.*-library-mode' | head -1)
    if [ -n "$use_library_mode" ]; then
        nvqpp_extra_options="--library-mode"
    fi

    for t in $requested_backends
    do
        # Skipping dynamics examples if target is not dynamics and ex is dynamics
        # or gpu is unavailable
        if { [ "$t" != "dynamics" ] && [[ "$ex" == *"dynamics"* ]]; } || { [ "$t" == "dynamics" ] && [[ "$ex" != *"dynamics"* ]]; }; then
            let "skipped+=1"
            echo "Skipping $t target for $ex.";
            echo ":white_flag: $filename: Not intended for this target. Test skipped." >> "${tmpFile}_$(echo $t | tr - _)"
            continue
        fi

        if [ "$t" == "default" ]; then target_flag=""
        else target_flag="--target $t"
        fi
    
        if [ -n "$intended_target" ] && [ "$intended_target" != "$t" ];
        then
            let "skipped+=1"
            echo "Skipping $t target.";
            echo ":white_flag: $filename: Not intended for this target. Test skipped." >> "${tmpFile}_$(echo $t | tr - _)"
            continue

        elif [ "$t" == "density-matrix-cpu" ] && [[ "$ex" != *"nois"* ]];
        then
            let "skipped+=1"
            echo "Skipping $t target."
            echo ":white_flag: $filename: Not executed for performance reasons. Test skipped." >> "${tmpFile}_$(echo $t | tr - _)"
            continue
        # Skip long-running tests, not suitable for cutensornet-based backends.
        elif [[ "$t" == "tensornet" || "$t" == "tensornet-mps" || "$t" == "nvidia-mqpu-mps" ]] && [[ " ${tensornet_backend_skipped_tests[*]} " =~ " $ex " ]]; then
            let "skipped+=1"
            echo "Skipping $t target."
            echo ":white_flag: $filename: Issue https://github.com/NVIDIA/cuda-quantum/issues/884. Test skipped." >> "${tmpFile}_$(echo $t | tr - _)"
            continue

        elif [ "$t" == "remote-mqpu" ]; then

            # Skipped long-running tests (variational optimization loops) for the "remote-mqpu" target to keep CI runtime manageable.
            # A simplified test for these use cases is included in the 'test/Remote-Sim/' test suite. 
            # Skipped tests that require passing kernel callables to entry-point kernels for the "remote-mqpu" target.
            # Also see issue: https://github.com/NVIDIA/cuda-quantum/issues/3792
            if [[ "$ex" == *"vqe_h2"* || "$ex" == *"qaoa_maxcut"* || "$ex" == *"gradients"* || "$ex" == *"grover"* || "$ex" == *"phase_estimation"* || "$ex" == *"trotter_kernel_mode"* || "$ex" == *"builder.cpp"* || "$ex" == *"iterative_qpe"* || "$ex" == *"measuring_kernels"* || "$ex" == *"sample_to_run_migration.cpp"* ]];
            then
                let "skipped+=1"
                echo "Skipping $t target.";
                echo ":white_flag: $filename: Not executed for performance reasons. Test skipped." >> "${tmpFile}_$(echo $t | tr - _)"
                continue

            # Don't run remote-mqpu if the MPI installation is incomplete (e.g., missing an ssh-client).            
            elif [[ "$mpi_available" == true && "$ssh_available" == false ]];
            then
                let "skipped+=1"
                echo "Skipping $t target due to incomplete MPI installation.";
                echo ":white_flag: $filename: Incomplete MPI installation. Test skipped." >> "${tmpFile}_$(echo $t | tr - _)"
                continue
            fi
        fi

        echo "Testing on $t target..."
        
        # Get target options to test for targets that support multiple configurations.
        # Note: Using case statement instead of associative arrays for macOS bash 3.2 compatibility
        get_target_options() {
            case "$1" in
                nvidia) echo "fp32 fp64 fp32,mqpu fp64,mqpu fp32,mgpu fp64,mgpu" ;;
                tensornet) echo "fp32 fp64" ;;
                tensornet-mps) echo "fp32 fp64" ;;
                *) echo "" ;;
            esac
        }
        target_opts=$(get_target_options "$t")
        if [ -n "$target_opts" ]; then
            for opt in $target_opts; do
                echo "  Testing $t target option: ${opt}"
                nvq++ $nvqpp_extra_options "$ex" $target_flag --target-option "${opt}"
                if [ ! $? -eq 0 ]; then
                    let "failed+=1"
                    echo "  :x: Compilation failed for $filename." >> "${tmpFile}_$(echo $t | tr - _)"
                    continue
                fi

                ./a.out &> /tmp/cudaq_validation.out
                status=$?
                echo "  Exited with code $status"
                if [ "$status" -eq "0" ]; then 
                    let "passed+=1"
                    echo "  :white_check_mark: Successfully ran $filename." >> "${tmpFile}_$(echo $t | tr - _)"
                else
                    cat /tmp/cudaq_validation.out
                    let "failed+=1"
                    echo "  :x: Failed to execute $filename." >> "${tmpFile}_$(echo $t | tr - _)"
                fi 
                rm a.out /tmp/cudaq_validation.out &> /dev/null
            done
        else
            nvq++ $nvqpp_extra_options $ex $target_flag
            if [ ! $? -eq 0 ]; then
                let "failed+=1"
                echo ":x: Compilation failed for $filename." >> "${tmpFile}_$(echo $t | tr - _)"
                continue
            fi

            ./a.out &> /tmp/cudaq_validation.out
            status=$?
            echo "Exited with code $status"
            if [ "$status" -eq "0" ]; then 
                let "passed+=1"
                echo ":white_check_mark: Successfully ran $filename." >> "${tmpFile}_$(echo $t | tr - _)"
            else
                cat /tmp/cudaq_validation.out
                let "failed+=1"
                echo ":x: Failed to execute $filename." >> "${tmpFile}_$(echo $t | tr - _)"
            fi 
            rm a.out /tmp/cudaq_validation.out &> /dev/null
        fi
    done
    echo "============================="
done

echo "============================="
echo "==      Python Tests       =="
echo "============================="

# Note: some of the tests do their own "!pip install ..." during the test, and
# for that to work correctly on the first time, the user site directory (e.g.
# ~/.local/lib/python3.10/site-packages) must already exist, so create it here.
if [ -x "$(command -v python3)" ]; then 
    mkdir -p $(python3 -m site --user-site)
fi

# Long-running dynamics examples
dynamics_backend_skipped_examples=(\
    examples/python/dynamics/transmon_resonator.py  \
    examples/python/dynamics/silicon_spin_qubit.py)

# Note divisive_clustering_src is not currently in the Published container under
# the "examples" folder, but the Publishing workflow moves all examples from
# docs/sphinx/examples, docs/sphinx/targets into the examples directory for the
# purposes of the container validation. The divisive_clustering_src Python
# files are used by the Divisive_clustering.ipynb notebook, so they are tested
# elsewhere and should be excluded from this test.
# Note: piping the `find` results through `sort` guarantees repeatable ordering.
for ex in `find examples/ targets/ -name '*.py' | sort`;
do 
    filename=$(basename -- "$ex")
    filename="${filename%.*}"
    echo "Testing $filename:"
    echo "Source: $ex"
    let "samples+=1"

    skip_example=false
    
    # Check for GPU/MPI dependencies
    if should_skip_python_example "$ex"; then
        echo "Skipping: $skip_reason."
        skip_example=true
    fi
    
    # Note: using sed instead of grep -P for macOS compatibility
    explicit_targets=$(sed -n "s/^[[:space:]]*cudaq\.set_target([[:space:]]*['\"][[:space:]]*\([^'\"]*\)[[:space:]]*['\"].*/\1/p" "$ex")
    for t in $explicit_targets; do
        if [ -z "$(echo $requested_backends | grep $t)" ]; then 
            echo "Explicitly set target $t not available."
            skip_example=true
        elif [ "$t" == "quera" ] || [ "$t" == "braket" ] ; then 
            # Skipped because GitHub does not have the necessary authentication token 
            # to submit a (paid) job to Amazon Braket (includes QuEra).
            echo "Explicitly set target braket or quera; skipping validation due to paid submission."
            skip_example=true
        elif [[ "$t" == "dynamics" ]] && [[ " ${dynamics_backend_skipped_examples[*]} " =~ " $ex " ]]; then
            echo "Skipping due to long run time."
            skip_example=true
        fi
    done

    if $skip_example;
    then
        let "skipped+=1"
        echo "Skipped.";
        echo ":white_flag: $filename: Necessary backend(s) not available. Test skipped." >> "${tmpFile}"
    else
        python3 $ex 1> /dev/null
        status=$?
        echo "Exited with code $status"
        if [ "$status" -eq "0" ]; then 
            let "passed+=1"
            echo ":white_check_mark: Successfully ran $filename." >> "${tmpFile}"
        else
            let "failed+=1"
            echo ":x: Failed to run $filename." >> "${tmpFile}"
        fi 
    fi
    echo "============================="
done

if [ -n "$(find examples/ applications/ -name '*.ipynb')" ]; then
    let "samples+=1"
    echo "============================="
    echo "== Setting up notebook venv =="
    echo "============================="
    
    # Create venv that inherits system packages (including cudaq from container)
    # Notebooks will install their own dependencies via !pip install commands
    NOTEBOOK_VENV="/tmp/cudaq_notebook_validation_venv"
    rm -rf "$NOTEBOOK_VENV"  # Clean any previous venv
    python3 -m venv --system-site-packages "$NOTEBOOK_VENV"
    source "$NOTEBOOK_VENV/bin/activate"
    
    echo "Installing Jupyter kernel infrastructure..."
    # Only install what's needed to register the kernel
    pip install --upgrade pip -q
    pip install jupyter ipykernel notebook -q
    
    # Register the venv as a Jupyter kernel
    # Notebooks will execute in this environment and can install their own packages
    JUPYTER_KERNEL_NAME="cudaq_nb_validation_container"
    python3 -m ipykernel install --user \
        --name="$JUPYTER_KERNEL_NAME" \
        --display-name="Python (CUDA-Q Container Notebook Validation)" \
        2>/dev/null
    
    echo "Jupyter kernel '${JUPYTER_KERNEL_NAME}' registered."
    echo "Notebooks will install their own dependencies during execution."
    
    echo "============================="
    echo "==  Validating notebooks   =="
    echo "============================="
    export OMP_NUM_THREADS=8
    
    # Pass the Jupyter kernel name as first argument to notebook_validation.py
    echo "$available_backends" | python3 notebook_validation.py "$JUPYTER_KERNEL_NAME"
    validation_status=$?
    
    # Cleanup - removes venv and kernel, system packages remain untouched
    echo "Cleaning up notebook validation environment..."
    jupyter kernelspec uninstall -f "$JUPYTER_KERNEL_NAME" 2>/dev/null || true
    deactivate
    rm -rf "$NOTEBOOK_VENV"
    
    if [ $validation_status -eq 0 ]; then 
        let "passed+=1"
        echo ":white_check_mark: Notebooks validation passed." >> "${tmpFile}"
    else
        let "failed+=1"
        echo ":x: Notebooks validation failed. See log for details." >> "${tmpFile}"
    fi 
else
    let "skipped+=1"
    echo "Skipped notebook validation.";
    echo ":white_flag: Notebooks validation skipped." >> "${tmpFile}"
fi

# Python snippet validation 
if [ -d "snippets/" ];
then
    # Skip multi-GPU snippets.
    for ex in `find snippets/ -name '*.py' -not -path '*/multi_gpu_workflows/*' | sort`;
    do 
        filename=$(basename -- "$ex")
        filename="${filename%.*}"
        echo "Testing $filename:"
        echo "Source: $ex"
        let "samples+=1"
        
        skip_snippet=false
        # Check for GPU/MPI dependencies
        if should_skip_python_example "$ex"; then
            echo "Skipping: $skip_reason."
            skip_snippet=true
        fi
        
        if $skip_snippet; then
            let "skipped+=1"
            echo "Skipped."
            echo ":white_flag: $filename: GPU/MPI required. Test skipped." >> "${tmpFile}"
        else
            python3 $ex 1> /dev/null
            status=$?
            echo "Exited with code $status"
            if [ "$status" -eq "0" ]; then 
                let "passed+=1"
                echo ":white_check_mark: Successfully ran $filename." >> "${tmpFile}"
            else
                let "failed+=1"
                echo ":x: Failed to run $filename." >> "${tmpFile}"
            fi
        fi
    done
fi

if [ -f "$GITHUB_STEP_SUMMARY" ]; 
then
    for t in $requested_backends
    do  
        file="${tmpFile}_$(echo $t | tr - _)"
        if [ -f "$file" ]; then
            echo "## Execution on $t target" >> $GITHUB_STEP_SUMMARY
            cat "$file" >> $GITHUB_STEP_SUMMARY
            rm -rf "$file"
        fi
    done

    if [ -f "$tmpFile" ] && [ -n "$(cat "$tmpFile")" ]; then
        echo "## Python examples and notebooks" >> $GITHUB_STEP_SUMMARY
        cat "$tmpFile" >> $GITHUB_STEP_SUMMARY
    fi
fi
rm -rf "$tmpFile"*

echo "============================="
echo "$samples examples found."
echo "Total passed: $passed"
echo "Total failed: $failed"
echo "Skipped: $skipped"
echo "============================="

if [ "$failed" -eq "0" ] && [ "$samples" != "0" ]; then exit 0; else exit 10; fi
