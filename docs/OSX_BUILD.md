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
# macOS SDK path (required for nvq++ to find system headers)
export SDKROOT=$(xcrun --show-sdk-path)

# CUDA-Q build paths
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

## Known Limitations and Workarounds

### LLVM Built as Shared Library (dylib)

**Problem:** macOS uses a two-level namespace that causes duplicate symbol issues when multiple libraries statically link LLVM. This manifests as runtime errors from conflicting LLVM symbol definitions.

**Workaround:** LLVM is built with `-DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON` on macOS (see `scripts/build_llvm.sh`).

**Proper fix:** Would require restructuring how CUDA-Q links against LLVM to avoid duplicate static linkage, or using symbol visibility controls.

### fast-isel Disabled on macOS

**Problem:** The `-fast-isel=0` LLVM command line option is not registered when LLVM is built as a shared library. This causes "Unknown command line argument" errors at runtime.

**Workaround:** The fast-isel option is conditionally disabled on macOS using `#if !defined(__APPLE__)` in:
- `runtime/cudaq/builder/kernel_builder.cpp`
- `runtime/common/RuntimeMLIRCommonImpl.h`
- `python/runtime/cudaq/platform/py_alt_launch_kernel.cpp`
- `tools/cudaq-qpud/RestServerMain.cpp`

**Proper fix:** Would require either:
1. Fixing LLVM to register command line options when built as a shared library, or
2. Finding an alternative API to disable fast instruction selection that doesn't rely on command line parsing

### LTO Disabled for LLVM Build

**Problem:** pybind11 has a bug where it passes `-flto=` with an empty value when LTO is enabled (see https://github.com/pybind/pybind11/issues/5098).

**Workaround:** LLVM is built with `-DLLVM_ENABLE_LTO=OFF` on macOS (see `scripts/build_llvm.sh`).

**Proper fix:** Wait for pybind11 fix or patch the pybind11 cmake modules locally.
