#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Create a folder with the Python README, the tests and the python examples, 
# and runt this script passing the path to that folder as well as the path to 
# the CUDA Quantum wheel to test with -f and -w respectively.
# Check the output for any tests that were skipped.

__optind__=$OPTIND
OPTIND=1
python_version=3.11
while getopts ":f:p:w:" opt; do
  case $opt in
    f) root_folder="$OPTARG"
    ;;
    p) python_version="$OPTARG"
    ;;
    w) cudaq_wheel="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 100 || exit 100
    ;;
  esac
done
OPTIND=$__optind__

readme_file="$root_folder/README.md"
if [ ! -d "$root_folder" ] || [ ! -f "$readme_file" ] ; then
    echo -e "\e[01;31mDid not find Python root folder. Please pass the folder containing the README and test with -f.\e[0m" >&2
    (return 0 2>/dev/null) && return 100 || exit 100
elif [ ! -f "$cudaq_wheel" ]; then
    echo -e "\e[01;31mDid not find Python wheel. Please pass its absolute path with -w.\e[0m" >&2
    (return 0 2>/dev/null) && return 100 || exit 100
fi

# Install Miniconda
if [ ! -x "$(command -v conda)" ]; then
    mkdir -p ~/.miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O ~/.miniconda3/miniconda.sh
    bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
    rm -rf ~/.miniconda3/miniconda.sh
    eval "$(~/.miniconda3/bin/conda shell.bash hook)"
fi

# Execute instructions from the README file
conda_script="$(awk '/(Begin conda install)/{flag=1;next}/(End conda install)/{flag=0}flag' "$readme_file" | grep . | sed '/^```/d')" 
while IFS= read -r line; do
    line=${line//3.10/$python_version}
    line=${line//pip install cuda-quantum/pip install "$cudaq_wheel"}
    if [ -n "$(echo $line | grep "conda activate")" ]; then
        command=$(echo "$line" | sed "s#conda activate#$(conda info --base)/bin/activate#")
        source $command
    elif [ -n "$(echo $line | tr -d '[:space:]')" ]; then
        eval "$line"
    fi
done <<< "$conda_script"
ompi_script="$(awk '/(Begin ompi setup)/{flag=1;next}/(End ompi setup)/{flag=0}flag' "$readme_file" | grep . | sed '/^```/d')" 
while IFS= read -r line; do
    if [ -n "$(echo $line | tr -d '[:space:]')" ]; then
        eval "$line"
    fi
done <<< "$ompi_script"

# Run core tests
echo "Running core tests."
python3 -m pip install pytest numpy
python3 -m pytest -v "$root_folder/tests" \
    --ignore "$root_folder/tests/backends" \
    --ignore "$root_folder/tests/parallel" \
    --ignore "$root_folder/tests/domains"
if [ ! $? -eq 0 ]; then
    echo -e "\e[01;31mPython tests failed.\e[0m" >&2
    status_sum=$((status_sum+1))
fi

# Run backend tests
echo "Running backend tests."
python3 -m pip install --user fastapi uvicorn llvmlite
for backendTest in "$root_folder/tests/backends"/*.py; do 
    python3 -m pytest -v $backendTest
    # Exit code 5 indicates that no tests were collected,
    # i.e. all tests in this file were skipped, which is the case
    # for the mock server tests since they are not included.
    status=$?
    if [ ! $status -eq 0 ] && [ ! $status -eq 5 ]; then
        echo -e "\e[01;31mPython backend test $backendTest failed with code $status.\e[0m" >&2
        status_sum=$((status_sum+1))
    fi
done

# Run platform tests
echo "Running platform tests."
for parallelTest in "$root_folder/tests/parallel"/*.py; do 
    python3 -m pytest -v $parallelTest
    if [ ! $? -eq 0 ]; then
        echo -e "\e[01;31mPython platform test $parallelTest failed.\e[0m" >&2
        status_sum=$((status_sum+1))
    fi
done

# Run examples
status_sum=0
for ex in `find "$root_folder/examples" -name '*.py' -not -path '*/providers/*'`; do
    python3 "$ex"
    if [ ! $? -eq 0 ]; then
        echo -e "\e[01;31mFailed to execute $ex.\e[0m" >&2
        status_sum=$((status_sum+1))
    fi
done

if [ ! $status_sum -eq 0 ]; then
    echo -e "\e[01;31mValidation produced errors.\e[0m" >&2
    (return 0 2>/dev/null) && return $status_sum || exit $status_sum
fi
