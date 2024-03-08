#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script will build and activate a **custom** CUDA Quantum MPI interface.
# Specifically, this script builds a MPI plugin interface against your local MPI installation 
# and exports an environment variable $CUDAQ_MPI_COMM_LIB to tell CUDA Quantum to use this MPI Communicator plugin. 
#
# It requires a GNU C++ compiler (g++).
#
# Please check/set the following environment variables:
#  - $MPI_PATH = Path to your MPI library installation directory.
#                If your MPI library is installed in system
#                directories as opposed to its own (root) directory,
#                ${MPI_PATH}/include is expected to contain the mpi.h header.
#                ${MPI_PATH}/lib64 or ${MPI_PATH}/lib is expected to contain libmpi.so.
#
# Run: bash <cuda quantum install dir>/distributed_interfaces/activate_custom_mpi.sh
# 
# You could add $CUDAQ_MPI_COMM_LIB to your ~/.bashrc file to persist the environment variable.

if [ -z "${MPI_PATH}" ]
then
    echo "Environment variable MPI_PATH is not set. Please set it to point to the MPI root directory!"
    echo "Note that MPI_PATH/include is expected to contain the mpi.h header."
    (return 0 2>/dev/null) && return 1 || exit 1
fi

this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
if [ -z "${CXX}" ]; then
    if [ -x "$(command -v "${this_file_dir}/../bin/nvq++")" ]; then
        CXX="${this_file_dir}/../bin/nvq++"
    elif [ -x "$(command -v "nvq++")" ]; then
        CXX="nvq++"
    elif [ -x "$(command -v "${MPI_PATH}/bin/mpic++")" ]; then
        CXX="${MPI_PATH}/bin/mpic++"
    else
        echo "The nvq++ compiler was not found. Please make sure that its location is included in PATH or set the environment variable CXX to a suitable C++ compiler to build the MPI plugin."
        (return 0 2>/dev/null) && return 2 || exit 2
    fi
fi

echo "Using $CXX to build the MPI plugin for MPI installation in $MPI_PATH."
$CXX -shared -std=c++17 -fPIC \
    -I${MPI_PATH}/include \
    -I$this_file_dir/ \
    $this_file_dir/mpi_comm_impl.cpp \
    -L${MPI_PATH}/lib64 -L${MPI_PATH}/lib -lmpi \
    -Wl,-rpath=${MPI_PATH}/lib64 -Wl,-rpath=${MPI_PATH}/lib \
    -o $this_file_dir/libcudaq_distributed_interface_mpi.so
export CUDAQ_MPI_COMM_LIB=$this_file_dir/libcudaq_distributed_interface_mpi.so
