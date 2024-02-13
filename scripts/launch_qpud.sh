#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script allows us to launch cudaq-qpud with MPI ranks == Number of NVIDIA GPUs
numGPUS=$(nvidia-smi --list-gpus | wc -l)
echo $numGPUS
mpiexec --allow-run-as-root -np $numGPUS cudaq-qpud
