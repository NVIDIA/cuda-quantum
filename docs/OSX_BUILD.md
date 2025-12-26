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

The build script automatically configures prerequisite paths:
- **Homebrew packages** (zlib, openssl, openblas): Found via `CMAKE_PREFIX_PATH`
- **Custom builds** (LLVM, pybind11, curl, aws): Installed to `~/.local/`

Add to your `.zshrc` or `.bashrc`:
```bash
# macOS SDK path (required for nvq++ to find system headers)
export SDKROOT=$(xcrun --show-sdk-path)

# Required for incremental builds with ninja in the build directory
export LLVM_INSTALL_PREFIX=$HOME/.llvm
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
- LLVM is built statically with Thin LTO and an idempotent registration patch

## Known Limitations and Workarounds

### macOS Two-Level Namespace and LLVM Symbols

**Problem:** macOS uses a two-level namespace that causes issues when multiple libraries statically link LLVM:

1. **Duplicate cl::opt registration:** Each dylib's static initializers run independently, causing LLVM's command-line option registration to fail with "Option already exists" or "Duplicate option categories" assertions.

2. **APFloat pointer mismatches:** Each dylib gets its own copy of LLVM static symbols like `IEEEdouble()`. When MLIR compares `&semantics == &APFloat::IEEEdouble()`, the comparison fails because pointers differ.

**Solution:** Two complementary fixes:

1. **Idempotent option registration patch** (`tpls/customizations/llvm/idempotent_option_registration.diff`): Modifies LLVM's CommandLine.cpp to silently skip duplicate option/category registrations instead of asserting. This handles the cl::opt issue.

2. **Thin LTO** (`-DLLVM_ENABLE_LTO=Thin` in `scripts/build_llvm.sh`): Enables link-time optimization which deduplicates identical symbols (like `IEEEdouble`) across compilation units. This handles the APFloat pointer mismatch issue.

### fast-isel Option Registration

**Problem:** On macOS, the linker only includes object files from static libraries if their symbols are explicitly referenced. LLVM options like `-fast-isel` are registered via static initializers in LLVMCodeGen, which may not be included.

**Solution:** Use `-Wl,-force_load` for LLVMCodeGen to ensure all static initializers run:
- `runtime/common/CMakeLists.txt` (cudaq-mlir-runtime)
- `python/extension/CMakeLists.txt` (CUDAQuantumPythonCAPI)

### pybind11 LTO Flag Bug

**Problem:** pybind11 has a bug where it passes `-flto=` with an empty value when LTO is enabled with Clang (see https://github.com/pybind/pybind11/issues/5098).

**Solution:** A local patch `tpls/customizations/pybind11/pybind11Common.cmake.diff` fixes the LTO flag generation. The patch is automatically applied during the build.

### xtensor xio.hpp Template Ambiguity

**Problem:** Including `<xtensor/xio.hpp>` triggers a clang 17-18 template ambiguity with xtl's `svector` rebind_container (see LLVM issue #91504). This causes compilation failures on macOS when using Apple Clang.

**Solution:** Avoid using `xio.hpp` for printing xtensor arrays. Instead, manually implement printing logic in `runtime/cudaq/domains/chemistry/molecule.cpp` for the `dump()` methods.

### Why This Differs from Linux

On Linux, the ELF dynamic linker coalesces duplicate global symbols at load time. On macOS, two-level namespace keeps each dylib's symbols isolated, requiring the idempotent patch and LTO workarounds described above.
