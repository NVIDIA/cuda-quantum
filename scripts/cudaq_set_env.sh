# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The content of this script is copied into locations like /etc/profile as 
# part of the CUDA Quantum installation via installer. When doing so,
# we match and replace the CUDAQ_INSTALL_PATH - hence the odd/unnecessary 
# if-statement in this file.
# This allows to easily updated the path if needed with 
#   sed '/^CUDAQ_INSTALL_PATH=.*/ s@@CUDAQ_INSTALL_PATH=/opt/nvidia/cudaq@'

CUDAQ_INSTALL_PATH=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
if [ -n "${CUDAQ_INSTALL_PATH}" ]; then
    # Prefix all paths such that running this script manually for a local installation
    # ensures that the local installation takes precedence over a system-wide installation.
    export CUDA_QUANTUM_PATH="${CUDAQ_INSTALL_PATH}"
    export PATH="${CUDA_QUANTUM_PATH}/bin${PATH:+:$PATH}"
    export LD_LIBRARY_PATH="${CUDA_QUANTUM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi
