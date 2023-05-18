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

requested_backends=`\
    echo "default"
    for target in $@; \
    do echo "$target"; \
    done`

available_backends=`\
    echo "default"
    for file in $(ls $CUDA_QUANTUM_PATH/targets/*.config); \
    do basename $file | cut -d "." -f 1; \
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
echo "Detected backends:"
echo "$available_backends"
echo
echo "Testing backends:"
echo "$requested_backends"
echo

if $missing_backend; 
then
    echo "Abort due to missing backend configuration."
    exit 1 
fi

echo "============================="
echo "==      Python Tests       =="
echo "============================="

for ex in `find examples -name *.py`;
do 
    filename=$(basename -- "$ex")
    filename="${filename%.*}"
    echo "Testing $filename:"
    echo "Source: $ex"
    let "samples+=1"
    python $ex 1> /dev/null
    status=$?
    echo "Exited with code $status"
    if [ "$status" -eq "0" ]; then 
        let "passed+=1"
    else
        let "failed+=1"
    fi 
    echo "============================="
done

echo "============================="
echo "==        C++ Tests        =="
echo "============================="

for ex in `find examples -name *.cpp`;
do
    filename=$(basename -- "$ex")
    filename="${filename%.*}"
    echo "Testing $filename:"
    echo "Source: $ex"
    let "samples+=1"
    for t in $requested_backends
    do
        if [[ "$ex" == *"cuquantum"* ]];
        then 
            let "skipped+=1"
            echo "Skipping $t target.";

        elif [[ "$ex" != *"nois"* ]] && [ "$t" == "density-matrix-cpu" ];
        then
            let "skipped+=1"
            echo "Skipping $t target."

        else
            echo "Testing on $t target..."
            if [ "$t" == "default" ]; then 
                if [[ "$ex" == *"mid_circuit"* ]];
                then 
                   nvq++ --enable-mlir $ex 
                else
                   nvq++ $ex
                fi
            else
                nvq++ $ex --target $t
            fi
            ./a.out &> /dev/null
            status=$?
            echo "Exited with code $status"
            if [ "$status" -eq "0" ]; then 
                let "passed+=1"
            else
                let "failed+=1"
            fi 
            rm a.out &> /dev/null
        fi
    done
    echo "============================="
done

echo "============================="
echo "$samples examples found."
echo "Total passed: $passed"
echo "Total failed: $failed"
echo "Skipped: $skipped"
echo "============================="
if [ "$failed" -eq "0" ]; then exit 0; else exit 10; fi
