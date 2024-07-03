#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# Create a folder with the Python README, the tests, the python examples and snippets, 
# and run this script passing the path to that folder as well as the path to 
# the CUDA-Q wheel to test with -f and -w respectively.
# Check the output for any tests that were skipped.

# E.g. run the command 
#   source validate_wheel.sh -w /tmp/cuda_quantum-*.whl -f /tmp -p 3.10 
# in a container (with GPU support) defined by:
#
# ARG base_image=ubuntu:22.04
# FROM ${base_image}
# ARG cuda_quantum_wheel=cuda_quantum-0.6.0-cp310-cp310-manylinux_2_28_x86_64.whl
# COPY $cuda_quantum_wheel /tmp/$cuda_quantum_wheel
# COPY scripts/validate_wheel.sh validate_wheel.sh
# COPY docs/sphinx/examples/python /tmp/examples/
# COPY docs/sphinx/snippets/python /tmp/snippets/
# COPY python/tests /tmp/tests/
# COPY python/README.md /tmp/README.md
# RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates vim wget openssh-client

__optind__=$OPTIND
OPTIND=1
python_version=3.11
quick_test=false
while getopts ":f:p:qw:" opt; do
  case $opt in
    f) root_folder="$OPTARG"
    ;;
    p) python_version="$OPTARG"
    ;;
    q) quick_test=true
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
        conda_env=$(echo "$line" | sed "s#conda activate##" | tr -d '[:space:]')
        source $(conda info --base)/bin/activate $conda_env
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
status_sum=0

# Verify that the necessary GPU targets are installed and usable
for tgt in nvidia nvidia-fp64 nvidia-mgpu tensornet; do
    python3 -c "import cudaq; cudaq.set_target('${tgt}')"
    if [ $? -ne 0 ]; then 
        echo -e "\e[01;31mPython trivial test for target ${tgt} failed.\e[0m" >&2
        status_sum=$((status_sum+1))
    fi
done

# Run core tests
echo "Running core tests."
python3 -m pip install pytest numpy psutil
python3 -m pytest -v "$root_folder/tests" \
    --ignore "$root_folder/tests/backends" \
    --ignore "$root_folder/tests/parallel" \
    --ignore "$root_folder/tests/domains"
if [ ! $? -eq 0 ]; then
    echo -e "\e[01;31mPython tests failed.\e[0m" >&2
    status_sum=$((status_sum+1))
fi

# If this is a quick test, we return here.
if $quick_test; then
    if [ ! $status_sum -eq 0 ]; then
        echo -e "\e[01;31mValidation produced errors.\e[0m" >&2
    fi
    (return 0 2>/dev/null) && return $status_sum || exit $status_sum
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

# Run snippets in docs
for ex in `find "$root_folder/snippets" -name '*.py'`; do
    python3 "$ex"
    if [ ! $? -eq 0 ]; then
        echo -e "\e[01;31mFailed to execute $ex.\e[0m" >&2
        status_sum=$((status_sum+1))
    fi
done

# Run examples
for ex in `find "$root_folder/examples" -name '*.py' -not -path '*/providers/*'`; do
    python3 "$ex"
    if [ ! $? -eq 0 ]; then
        echo -e "\e[01;31mFailed to execute $ex.\e[0m" >&2
        status_sum=$((status_sum+1))
    fi
done

# Run remote-mqpu platform test
# Use cudaq-qpud.py wrapper script to automatically find dependencies for the Python wheel configuration.
# Note that a derivative of this code is in
# docs/sphinx/using/backends/platform.rst, so if you update it here, you need to
# check if any docs updates are needed.
cudaq_location=`python3 -m pip show cuda-quantum | grep -e 'Location: .*$'`
qpud_py="${cudaq_location#Location: }/bin/cudaq-qpud.py"
if [ -x "$(command -v nvidia-smi)" ]; 
then nr_gpus=`nvidia-smi --list-gpus | wc -l`
else nr_gpus=0
fi
server1_devices=`echo $(seq $((nr_gpus >> 1)) $((nr_gpus - 1))) | tr ' ' ,`
server2_devices=`echo $(seq 0 $((($nr_gpus >> 1) - 1))) | tr ' ' ,`
echo "Launching server 1..."
servers="localhost:12001"
CUDA_VISIBLE_DEVICES=$server1_devices mpiexec --allow-run-as-root -np 2 python3 "$qpud_py" --port 12001 &
if [ -n "$server2_devices" ]; then
    echo "Launching server 2..."
    servers+=",localhost:12002"
    CUDA_VISIBLE_DEVICES=$server2_devices mpiexec --allow-run-as-root -np 2 python3 "$qpud_py" --port 12002 &
fi

sleep 5 # wait for servers to launch
python3 "$root_folder/snippets/using/cudaq/platform/sample_async_remote.py" \
    --backend nvidia-mgpu --servers "$servers"
if [ ! $? -eq 0 ]; then
    echo -e "\e[01;31mRemote platform test failed.\e[0m" >&2
    status_sum=$((status_sum+1))
fi
kill %1 && wait %1 2> /dev/null
if [ -n "$server2_devices" ]; then
    kill %2 && wait %2 2> /dev/null
fi

if [ ! $status_sum -eq 0 ]; then
    echo -e "\e[01;31mValidation produced errors.\e[0m" >&2
    (return 0 2>/dev/null) && return $status_sum || exit $status_sum
fi
