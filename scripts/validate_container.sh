#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Launch the NVIDIA CUDA Quantum Docker container, 
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
        if [ "${qpu}" != "remote_rest" ] && [ "${qpu}" != "orca" ] \
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

# Skip some tests (multi-controlled gates) for the MPS backend;
# see https://github.com/NVIDIA/cuda-quantum/issues/884
mps_skipped_tests=(\
    examples/cpp/algorithms/grover.cpp \
    examples/cpp/basics/multi_controlled_operations.cpp \
    examples/cpp/other/builder/builder.cpp \
    examples/cpp/algorithms/amplitude_estimation.cpp)

echo "============================="
echo "==      Python Tests       =="
echo "============================="

for ex in `find examples -name '*.py'`;
do 
    filename=$(basename -- "$ex")
    filename="${filename%.*}"
    echo "Testing $filename:"
    echo "Source: $ex"
    let "samples+=1"

    if [[ "$ex" == *"iqm"* ]] || [[ "$ex" == *"ionq"* ]] || [[ "$ex" == *"quantinuum"* ]];
    then
        let "skipped+=1"
        echo "Skipped.";
    else
        python3 $ex 1> /dev/null
        status=$?
        echo "Exited with code $status"
        if [ "$status" -eq "0" ]; then 
            let "passed+=1"
        else
            let "failed+=1"
        fi 
    fi
    echo "============================="
done

echo "============================="
echo "==        C++ Tests        =="
echo "============================="

tmpFile=$(mktemp)
for ex in `find examples -name '*.cpp'`;
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
        if [ -n "$intended_target" ] && [ "$intended_target" != "$t" ];
        then
            let "skipped+=1"
            echo "Skipping $t target.";
            echo ":white_flag: Not intended for this target. Test skipped." >> "$tmpFile_$(echo $t | tr - _)"

        elif [[ "$ex" != *"nois"* ]] && [ "$t" == "density-matrix-cpu" ];
        then
            let "skipped+=1"
            echo "Skipping $t target."
            echo ":white_flag: Not executed for performance reasons. Test skipped." >> "$tmpFile_$(echo $t | tr - _)"

        elif [[ " ${mps_skipped_tests[*]} " =~ " $ex " ]] && [ "$t" == "tensornet-mps" ]; then
            let "skipped+=1"
            echo "Skipping $t target."
            echo ":white_flag: Issue: https://github.com/NVIDIA/cuda-quantum/issues/884. Test skipped." >> "$tmpFile_$(echo $t | tr - _)"

        # Skipped long-running tests (variational optimization loops) for the "remote-mqpu" target to keep CI runtime managable.
        # A simplified test for these use cases is included in the 'test/Remote-Sim/' test suite. 
        # Skipped tests that require passing kernel callables to entry-point kernels for the "remote-mqpu" target.
        elif [[ "$t" == "remote-mqpu" ]] && [[ "$ex" == *"vqe_h2"* || "$ex" == *"qaoa_maxcut"* || "$ex" == *"gradients"* || "$ex" == *"grover"* || "$ex" == *"multi_controlled_operations"* || "$ex" == *"phase_estimation"* ]];
        then
            let "skipped+=1"
            echo "Skipping $ex for $t target.";
            echo ":white_flag: Not executed for performance reasons. Test skipped." >> "$tmpFile_$(echo $t | tr - _)"

        elif [[ "$t" == "remote-mqpu" && "$mpi_available" == true && "$ssh_available" == false ]];
        then
            # Don't run remote-mqpu if the MPI installation is incomplete (e.g., missing an ssh-client).
            let "skipped+=1"
            echo "Skipping $t target due to incomplete MPI installation.";
            echo ":white_flag: Incomplete MPI installation. Test skipped." >> "$tmpFile_$(echo $t | tr - _)"

        else
            echo "Testing on $t target..."
            if [ "$t" == "default" ]; then 
                nvq++ $ex && status=$?
            else
                nvq++ $ex --target $t && status=$?
            fi
            if [ ! $status -eq 0 ]; then
                let "failed+=1"
                echo ":x: Compilation failed for $filename." >> "$tmpFile_$(echo $t | tr - _)"
                continue
            fi

            ./a.out &> /tmp/cudaq_validation.out
            status=$?
            echo "Exited with code $status"
            if [ "$status" -eq "0" ]; then 
                let "passed+=1"
                echo ":white_check_mark: Successfully ran $filename." >> "$tmpFile_$(echo $t | tr - _)"
            else
                cat /tmp/cudaq_validation.out
                let "failed+=1"
                echo ":x: Failed to execute $filename." >> "$tmpFile_$(echo $t | tr - _)"
            fi 
            rm a.out /tmp/cudaq_validation.out &> /dev/null
        fi
    done
    echo "============================="
done

if [ -f "$GITHUB_STEP_SUMMARY" ]; 
then
    for t in $requested_backends
    do
        echo "## Execution on $t target" >> $GITHUB_STEP_SUMMARY
        cat "$tmpFile_$(echo $t | tr - _)" >> $GITHUB_STEP_SUMMARY
        rm -rf "$tmpFile_$(echo $t | tr - _)"
    done
fi
rm -rf "$tmpFile"

echo "============================="
echo "$samples examples found."
echo "Total passed: $passed"
echo "Total failed: $failed"
echo "Skipped: $skipped"
echo "============================="
if [ "$failed" -eq "0" ]; then exit 0; else exit 10; fi
