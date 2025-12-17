# macOS Build Instructions

## Prerequisites

1. **Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **Homebrew** (if not already installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

## Python Virtual Environment Setup

Create and activate a virtual environment:
```bash
python3 -m venv ~/.venv/cudaq
source ~/.venv/cudaq/bin/activate
```

Install Python dependencies:
```bash
pip install --upgrade pip
pip install numpy llvmlite==0.44.0 scipy==1.11.4 pytest==8.2.0 lit==18.1.4 \
  requests==2.31.0 fastapi==0.111.0 uvicorn==0.29.0 pydantic==2.7.1 \
  openfermionpyscf==0.5 h5py==3.12.1
```

**Note:** Always activate the venv before running build scripts:
```bash
source ~/.venv/cudaq/bin/activate
```

## Install Dependencies

Run the prerequisites script (pointing to venv Python):
```bash
Python3_EXECUTABLE=$(which python3) \
ZLIB_INSTALL_PREFIX=$HOME/.local/zlib \
./scripts/install_prerequisites.sh
```

This will install (using Apple's system Clang):
- CMake and Ninja
- Zlib, OpenSSL, Curl, BLAS (from source)
- LLVM/MLIR with Python bindings
- pybind11
- AWS SDK

Default install locations on macOS:
- LLVM: `$HOME/.llvm`
- Other libraries: `$HOME/.local/`

## Environment Setup

After installation, add to your `.zshrc` or `.bashrc`:
```bash
# CUDA-Q build paths
export LLVM_INSTALL_PREFIX=$HOME/.llvm
export ZLIB_INSTALL_PREFIX=$HOME/.local/zlib
export OPENSSL_INSTALL_PREFIX=$HOME/.local/ssl
export CURL_INSTALL_PREFIX=$HOME/.local/curl
export BLAS_INSTALL_PREFIX=$HOME/.local/blas
export PYBIND11_INSTALL_PREFIX=$HOME/.local/pybind11
export AWS_INSTALL_PREFIX=$HOME/.local/aws
```

## Build CUDA-Q

```bash
source ~/.venv/cudaq/bin/activate
./scripts/build_cudaq.sh -v
```

CUDA-Q will be installed to `$HOME/.cudaq` by default (CPU-only build, no CUDA on macOS).
