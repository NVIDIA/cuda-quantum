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

# Prepend a directory to a path variable, skipping if already present.
_cudaq_prepend() {
    # Read the current value of the variable.
    local _cur="$(printenv "$1" 2>/dev/null)" || true
    # If the directory is already in the path, nothing to do.
    case ":${_cur}:" in *":$2:"*) return ;; esac
    # Prepend the directory and export.
    export "$1=$2${_cur:+:${_cur}}"
}

# Platform-independent paths
_cudaq_prepend PATH "${CUDA_QUANTUM_PATH}/bin"
_cudaq_prepend CPLUS_INCLUDE_PATH "${CUDA_QUANTUM_PATH}/include"

# Platform-specific library path
if [ "$(uname)" = "Darwin" ]; then
    _cudaq_prepend DYLD_LIBRARY_PATH "${CUDA_QUANTUM_PATH}/lib"
else
    _cudaq_prepend LD_LIBRARY_PATH "${CUDA_QUANTUM_PATH}/lib"
fi

unset -f _cudaq_prepend
