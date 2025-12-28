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
- LLVM and MLIR are built as shared libraries (`libLLVM.dylib`, `libMLIR.dylib`) to avoid symbol duplication issues

## macOS LLVM/MLIR Configuration

CUDA-Q builds LLVM and MLIR as shared libraries on macOS to ensure all components share single instances. This approach:

- Eliminates APFloat pointer mismatch issues (single `IEEEdouble()` address)
- Ensures single `cl::opt` GlobalParser instance (no duplicate option registration)
- Removes need for LTO or registration patches

### Why This Differs from Linux

On Linux, the ELF dynamic linker coalesces duplicate global symbols at load time via symbol interposition. On macOS, two-level namespace keeps each dylib's symbols isolated. Using shared `libLLVM.dylib` and `libMLIR.dylib` avoids duplication entirely by providing single LLVM/MLIR instances that all CUDA-Q libraries link against.

We use monolithic dylibs (`LLVM_BUILD_LLVM_DYLIB` + `MLIR_LINK_MLIR_DYLIB`) rather than `BUILD_SHARED_LIBS` for simpler deployment (2 dylibs vs 100+).

### Performance Characteristics

The dylib approach has the following performance trade-offs:

| Metric | Impact |
|--------|--------|
| Cold startup | +50-200ms (dylib loading overhead) |
| JIT compilation | No change |
| LLVM build time | Faster (no Thin LTO required) |
| Binary size | Smaller per-tool (LLVM not embedded) |
| Multi-process memory | Lower (shared dylib pages) |

For typical usage, the startup overhead is acceptable and offset by faster build times during development.

### fast-isel Option Registration

On macOS, the linker only includes object files from static libraries if their symbols are explicitly referenced. LLVM options like `-fast-isel` are registered via static initializers in LLVMCodeGen. Even with the dylib approach, we use `-Wl,-force_load` for LLVMCodeGen to ensure these static initializers run:
- `runtime/common/CMakeLists.txt` (cudaq-mlir-runtime)
- `python/extension/CMakeLists.txt` (CUDAQuantumPythonCAPI and _quakeDialects.dso)
- `tools/cudaq-qpud/CMakeLists.txt` (cudaq-qpud)

## Known Limitations

### Execution Manager Selection

**Problem:** The `CUDAQ_REGISTER_EXECUTION_MANAGER` macro defines `getRegisteredExecutionManager()` in multiple libraries (default, qudit, photonics). On Linux, ELF symbol interposition selects one at runtime. On macOS with two-level namespace, each library binds to its own dependency's symbol, preventing override.

Additionally, `execution_manager.cpp` must only be compiled into one library to ensure a single `static ExecutionManager*` instance. On macOS, duplicate static variables in different dylibs are separate instances. The `cudaq` library owns this file; `cudaq-em-default` links to `cudaq` rather than compiling its own copy.

**Solution:** Use `setExecutionManagerInternal()` to explicitly set the execution manager before use.

### xtensor xio.hpp Template Ambiguity

**Problem:** Including `<xtensor/xio.hpp>` triggers a clang 17-18 template ambiguity with xtl's `svector` rebind_container (see LLVM issue #91504). This causes compilation failures on macOS when using Apple Clang.

**Solution:** Avoid using `xio.hpp` for printing xtensor arrays. Instead, manually implement printing logic in `runtime/cudaq/domains/chemistry/molecule.cpp` for the `dump()` methods.
