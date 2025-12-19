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

## Build CUDA-Q

On macOS, `build_cudaq.sh` automatically installs prerequisites (LLVM, zlib, openssl, curl, blas, pybind11, AWS SDK) if LLVM is not found.

```bash
source ~/.venv/cudaq/bin/activate
./scripts/build_cudaq.sh
```

The first build will take a while as it builds LLVM/MLIR from source. Subsequent builds are faster.

Use `-i` for incremental builds:
```bash
./scripts/build_cudaq.sh -i
```

## Installation Locations

Default install locations on macOS:
- LLVM: `$HOME/.llvm`
- CUDA-Q: `$HOME/.cudaq`
- Other libraries: `$HOME/.local/`

## Environment Setup

After installation, add to your `.zshrc` or `.bashrc`:
```bash
export LLVM_INSTALL_PREFIX=$HOME/.llvm
export ZLIB_INSTALL_PREFIX=$HOME/.local/zlib
export OPENSSL_INSTALL_PREFIX=$HOME/.local/ssl
export CURL_INSTALL_PREFIX=$HOME/.local/curl
export BLAS_INSTALL_PREFIX=$HOME/.local/blas
export PYBIND11_INSTALL_PREFIX=$HOME/.local/pybind11
export AWS_INSTALL_PREFIX=$HOME/.local/aws
```

## Using CUDA-Q

After building:
```bash
source $HOME/.cudaq/set_env.sh
```

For Python:
```bash
source ~/.venv/cudaq/bin/activate
export PYTHONPATH=$HOME/.cudaq/lib/python:$PYTHONPATH
python3 -c "import cudaq; print(cudaq.__version__)"
```

## Notes

- CPU-only build (no CUDA support on macOS)
- Uses Apple's system Clang compiler
- LLVM is built as a shared library to avoid symbol conflicts
