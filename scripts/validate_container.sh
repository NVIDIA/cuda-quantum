#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Launch the NVIDIA CUDA-Q Docker container, 
# and run this script from the home directory.
# Check the logged output.

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

export UCX_LOG_LEVEL=warning
requested_backends=`\
    for target in $@; \
    do echo "$target"; \
    done`

installed_backends=`\
    echo "default"
    for file in $(ls $CUDA_QUANTUM_PATH/targets/*.config); \
    do basename $file | cut -d "." -f 1; \
    done`

# remote_rest targets are automatically filtered, 
# so is execution on the photonics backend
available_backends=`\
    echo "default"
    for file in $(ls $CUDA_QUANTUM_PATH/targets/*.config); \
    do
        if grep -q "LIBRARY_MODE_EXECUTION_MANAGER=photonics" $file ; then 
          continue
        fi 
        platform=$(cat $file | grep "PLATFORM_QPU=")
        qpu=${platform#PLATFORM_QPU=}
        requirements=$(cat $file | grep "GPU_REQUIREMENTS=")
        gpus=${requirements#GPU_REQUIREMENTS=}
        if [ "${qpu}" != "remote_rest" ] && [ "${qpu}" != "orca" ] && [ "${qpu}" != "NvcfSimulatorQPU" ] \
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
    examples/cpp/algorithms/vqe_h2.cpp \
    examples/cpp/algorithms/qaoa_maxcut.cpp \
    examples/cpp/other/builder/builder.cpp \
    examples/cpp/algorithms/amplitude_estimation.cpp)

echo "============================="
echo "==        C++ Tests        =="
echo "============================="

tmpFile=$(mktemp)
for ex in `find examples/ -name '*.cpp'`;
do
    filename=$(basename -- "$ex")
    filename="${filename%.*}"
    echo "Testing $filename:"
    echo "Source: $ex"
    let "samples+=1"

    # Look for a --target flag to nvq++ in the 
    # comment block at the beginning of the file.
    intended_target=`sed -e '/^$/,$d' $ex | grep -oP '^//\s*nvq++.+--target\s+\K\S+'`
    if [ -n "$intended_target" ]; then
        echo "Intended for execution on $intended_target backend."
    fi

    for t in $requested_backends
    do
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

            # Skipped long-running tests (variational optimization loops) for the "remote-mqpu" target to keep CI runtime managable.
            # A simplified test for these use cases is included in the 'test/Remote-Sim/' test suite. 
            # Skipped tests that require passing kernel callables to entry-point kernels for the "remote-mqpu" target.
            if [[ "$ex" == *"vqe_h2"* || "$ex" == *"qaoa_maxcut"* || "$ex" == *"gradients"* || "$ex" == *"grover"* || "$ex" == *"multi_controlled_operations"* || "$ex" == *"phase_estimation"* || "$ex" == *"trotter_kernel"* || "$ex" == *"builder.cpp"* ]];
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

            else
                # TODO: remove this once the nvqc backend is part of the validation
                # tracked in https://github.com/NVIDIA/cuda-quantum/issues/1283
                target_flag+=" --enable-mlir"
            fi
        fi

        echo "Testing on $t target..."
        nvq++ $ex $target_flag 
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
    done
    echo "============================="
done

echo "============================="
echo "==      Python Tests       =="
echo "============================="

for ex in `find examples/ -name '*.py'`;
do 
    filename=$(basename -- "$ex")
    filename="${filename%.*}"
    echo "Testing $filename:"
    echo "Source: $ex"
    let "samples+=1"

    if [[ "$ex" == *"iqm"* ]] || [[ "$ex" == *"oqc"* ]] || [[ "$ex" == *"ionq"* ]] || [[ "$ex" == *"quantinuum"* ]] || [[ "$ex" == *"nvqc"* ]] || [[ "$ex" == *"orca"* ]];
    then
        let "skipped+=1"
        echo "Skipped.";
        echo ":white_flag: $filename: External backend. Test skipped." >> "${tmpFile}"
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

if [ -n "$(find $(pwd) -name '*.ipynb')" ]; then
    echo "Validating notebooks:"
    echo "$available_backends" | python3 notebook_validation.py
    if [ $? -eq 0 ]; then 
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
