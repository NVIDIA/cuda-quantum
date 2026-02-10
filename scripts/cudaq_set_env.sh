# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script sets the necessary environment variables to make CUDA-Q
# discoverable for all tools.

# Prefix all paths such that running this script manually for a local installation
# ensures that the local installation takes precedence over a system-wide installation.
export CUDA_QUANTUM_PATH="$(dirname "$(realpath "${BASH_SOURCE[0]:-$0}")")"

# Platform-independent paths
export PATH="${CUDA_QUANTUM_PATH}/bin${PATH:+:$PATH}"
export CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# Platform-specific library path
if [ "$(uname)" = "Darwin" ]; then
    export DYLD_LIBRARY_PATH="${CUDA_QUANTUM_PATH}/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${CUDA_QUANTUM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
