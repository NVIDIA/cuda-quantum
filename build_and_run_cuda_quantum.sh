#!/bin/bash

# Function to print error message and exit with error code
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Usage instructions
usage() {
    echo "Usage: $0 [toolchain]"
    echo "  toolchain: (optional) The toolchain to use for building the development dependencies image. Default is 'llvm'."
}

# If the first argument is -h or --help, display usage instructions
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Step 1: Obtain the desired toolchain or use the default LLVM toolchain
if [ -z "$1" ]; then
  toolchain="llvm"
else
  toolchain="$1"
fi

echo "Using toolchain: $toolchain"

# Step 2: Build CUDA Quantum development dependencies image
echo "Building CUDA Quantum development dependencies image..."
docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:${toolchain}-latest -f docker/build/devdeps.Dockerfile --build-arg toolchain=$toolchain . || error_exit "Failed to build development dependencies image."

# Step 3: Build CUDA Quantum development image
echo "Building CUDA Quantum development image..."
docker build -t nvidia/cuda-quantum-dev:latest -f docker/build/cudaqdev.Dockerfile . || error_exit "Failed to build development image."

# Step 4: Build CUDA Quantum runtime image
echo "Building CUDA Quantum runtime image..."
docker build -t ghcr.io/nvidia/cuda-quantum:latest -f docker/release/cudaq.Dockerfile . || error_exit "Failed to build runtime image."

# Step 5: Run CUDA Quantum runtime image
echo "Running CUDA Quantum runtime image..."
docker run -it --rm ghcr.io/nvidia/cuda-quantum:latest || error_exit "Failed to run runtime image."
