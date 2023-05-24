# Start with a base Ubuntu 22.04 image
# --platform=linux/amd64
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    LLVM_VERSION=c0b45fef155fbe3f17f9a6f99074682c69545488 \
    LLVM_INSTALL_PATH=/opt/llvm \
    CUDAQ_INSTALL_PATH=$HOME/.cudaq

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ \
    git \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
    libblas-dev \
    liblapack-dev \
    libcurl4-openssl-dev \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Build LLVM / Clang / MLIR
RUN mkdir llvm-project && \
    cd llvm-project && \
    git init && \
    git remote add origin https://github.com/llvm/llvm-project && \
    git fetch origin --depth=1 ${LLVM_VERSION} && \
    git reset --hard FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake ../llvm -G Ninja \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PATH} \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=TRUE && \
    ninja install && \
    cp bin/llvm-lit ${LLVM_INSTALL_PATH}/bin/

# Install pip dependencies
RUN pip3 install lit numpy pytest

# Build CUDA Quantum
RUN git clone https://github.com/splch/cuda-quantum && \
    cd cuda-quantum && \
    git checkout ionqTemplate && \
    mkdir build && \
    cd build && \
    cmake .. -G Ninja \
    -DCMAKE_INSTALL_PREFIX=${CUDAQ_INSTALL_PATH} \
    -DLLVM_DIR=${LLVM_INSTALL_PATH}/lib/cmake/llvm \
    -DCUDAQ_ENABLE_PYTHON=TRUE && \
    ninja install && \
    ctest -E ctest-nvqpp

# Add binaries to path
ENV PATH="${PATH}:/cuda-quantum/build/bin"

# Set the working directory
WORKDIR /workspace

# Entry point
CMD ["/bin/bash"]
