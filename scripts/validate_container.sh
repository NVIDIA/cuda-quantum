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
    for t in "" "dm" "cuquantum" "cuquantum_mgpu" "tensornet";
    do
        if [[ "$ex" == *"cuquantum"* ]] && [ "$t" = "" ];
        then 
            let "skipped+=1"
            if [ "$t" = "" ]; then 
                echo "Skipping default target.";
            else 
                echo "Skipping target $t.";
            fi
        elif [[ "$ex" != *"nois"* ]] && [ "$t" = "dm" ];
        then
            let "skipped+=1"
            echo "Skipping target dm."
        else
            if [ "$t" = "" ]; then 
                echo "Testing on default target..."
            else 
                echo "Testing on target $t..."
            fi
            nvq++ $ex -qpu $t
            ./a.out 1> /dev/null
            status=$?
            echo "Exited with code $status"
            if [ "$status" -eq "0" ]; then 
                let "passed+=1"
            else
                let "failed+=1"
            fi 
            rm a.out
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
if [ "$failed" -eq "0" ]; then exit 0; else exit 1; fi
