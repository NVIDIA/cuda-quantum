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
- LLVM is built with Thin LTO and static linking
- Uses `-Wl,-flat_namespace` to share LLVM symbols across dylibs

## macOS LLVM/MLIR Configuration

CUDA-Q uses static LLVM/MLIR linking with flat_namespace on macOS. This approach:

- Uses `-Wl,-flat_namespace` to share symbols globally (like Linux ELF)
- Applies Thin LTO during LLVM build to reduce binary size
- Requires LLVM patch for idempotent option category registration (see `tpls/customizations/llvm/idempotent_option_category.diff`)

### Why This Differs from Linux

On Linux, the ELF dynamic linker coalesces duplicate global symbols at load time via symbol interposition. On macOS, the default two-level namespace keeps each dylib's symbols isolated, causing ODR violations when multiple dylibs statically link LLVM.

The `-Wl,-flat_namespace` linker flag makes macOS use a single global symbol namespace similar to Linux, allowing the first loaded definition to be used everywhere.

### fast-isel Option Registration

On macOS, the linker only includes object files from static libraries if their symbols are explicitly referenced. LLVM options like `-fast-isel` are registered via static initializers in LLVMCodeGen. We use `-Wl,-force_load` for LLVMCodeGen to ensure these static initializers run:
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

