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

### MLIR Threading Disabled on macOS

**Problem:** When LLVM is built as a shared library, MLIR's multi-threaded pass execution can crash with segmentation faults in LLVM's ThreadPool during pattern rewriting.

**Workaround:** MLIR threading is disabled by default on macOS by setting `CUDAQ_MLIR_DISABLE_THREADING=true` in:
- `python/runtime/cudaq/platform/py_alt_launch_kernel.cpp`

Users can override this by setting the environment variable `CUDAQ_MLIR_DISABLE_THREADING=false`, but this may cause crashes.

**Proper fix:** Avoid building LLVM as a shared library, or investigate the root cause of the threading crashes.

### CUDAQTargetConfigUtil Dylib Linking

**Problem:** When `CUDAQTargetConfigUtil` is built with `DISABLE_LLVM_LINK_LLVM_DYLIB`, it statically links LLVM's `Support` component. This causes `libcudaq.dylib` (which links `CUDAQTargetConfigUtil`) to contain its own copy of LLVM symbols like `IEEEdouble()`. When MLIR code compares APFloat semantics pointers, they don't match because `libcudaq.dylib` and `libLLVM.dylib` have different copies.

This manifests as assertion failures: `Unknown FP format` in `mlir::FloatAttr::get()`.

**Workaround:** When `LLVM_LINK_LLVM_DYLIB` is enabled, `DISABLE_LLVM_LINK_LLVM_DYLIB` is not used for `CUDAQTargetConfigUtil`, ensuring all libraries use the same LLVM symbols from the shared library (see `lib/Support/Config/CMakeLists.txt`).

**Proper fix:** Avoid building LLVM as a shared library, which would eliminate the need for this workaround.

### xtensor xio.hpp Template Ambiguity

**Problem:** Including `<xtensor/xio.hpp>` triggers a clang 17-18 template ambiguity with xtl's `svector` rebind_container (see LLVM issue #91504). This causes compilation failures on macOS when using Apple Clang.

**Workaround:** Avoid using `xio.hpp` for printing xtensor arrays. Instead, manually implement printing logic in `runtime/cudaq/domains/chemistry/molecule.cpp` for the `dump()` methods.

**Proper fix:** Wait for xtensor/xtl to fix the template ambiguity, or upgrade to a clang version where this is resolved.

### pybind11 LTO Flag Bug

**Problem:** pybind11 has a bug where it passes `-flto=` with an empty value (or `-flto==thin`) when LTO is enabled with Clang (see https://github.com/pybind/pybind11/issues/5098). This causes compilation failures.

**Workaround:** A local patch `tpls/customizations/pybind11/pybind11Common.cmake.diff` fixes the LTO flag generation in pybind11's CMake. The patch is automatically applied during the build.

**Proper fix:** Wait for the pybind11 fix to be merged upstream and update the submodule.

### flat_namespace and Idempotent Option Registration

**Problem:** When building LLVM statically (to avoid dylib threading issues), each CUDA-Q dylib gets its own copy of LLVM global objects. This causes several issues:

1. **APFloat pointer mismatches:** When MLIR code compares `&semantics == &APFloat::IEEEdouble()`, the comparison fails because each dylib has a different `IEEEdouble` address.

2. **Duplicate option registration:** With `flat_namespace` enabled to share symbols, each dylib's static initializers still run independently, causing LLVM's command-line option registration to fail with "Option already exists" or "Duplicate option categories" assertions.

3. **Unregistered options on macOS:** The macOS linker only includes object files from static libraries if their symbols are referenced. This means LLVM options like `-fast-isel` (defined in `TargetPassConfig.cpp`) may not be registered if no code explicitly references symbols from that object file.

**Workaround:** Three changes are required:

1. **flat_namespace linker flag:** Added `-Wl,-flat_namespace` on macOS (see `CMakeLists.txt`). This makes all dylibs share the same symbol namespace, so the first dylib's LLVM symbols become canonical (similar to Linux ELF behavior).

2. **LLVM patch:** The patch `tpls/customizations/llvm/idempotent_option_registration.diff` modifies three functions to skip registration if an option/category already exists, instead of asserting:
   - `registerCategory()` in `CommandLine.cpp`
   - `addOption()` in `CommandLine.cpp`
   - `addLiteralOption()` in `CommandLine.cpp` and `CommandLine.h`

3. **force_load for LLVMCodeGen:** Added `-Wl,-force_load` for `LLVMCodeGen` in libraries that parse LLVM options (see `runtime/common/CMakeLists.txt`, `runtime/cudaq/builder/CMakeLists.txt`, `tools/cudaq-qpud/CMakeLists.txt`, `python/extension/CMakeLists.txt`). This forces the linker to include all object files from LLVMCodeGen, ensuring static initializers run and options like `-fast-isel` are registered.

**Why this differs from Linux:** On Linux, the ELF dynamic linker coalesces duplicate global data symbols at load time. On macOS, the two-level namespace keeps each dylib's symbols isolated. Even with `flat_namespace`, each dylib's data segment is separate, so static initializers run independently for each copy of LLVM globals.

**Alternative approaches (not yet implemented):**

1. **LLVM dylib:** Build LLVM as a shared library (`LLVM_BUILD_LLVM_DYLIB=ON`). This avoids symbol duplication but causes threading crashes in MLIR's ThreadPool.

2. **Library consolidation (preferred long-term solution, no LLVM patches required):**

   The current workaround requires patching LLVM. A simpler approach: have `cudaq-mlir-runtime` be the sole owner of LLVM static libraries, with other libraries dynamically linking to it:

   **Current (problematic):**
   ```
   cudaq-mlir-runtime.dylib ─── statically links LLVM ─── has its own GlobalParser
   cudaq-builder.dylib ──────── statically links LLVM ─── has its own GlobalParser ✗
   cudaq-qpud ───────────────── statically links LLVM ─── has its own GlobalParser ✗
   ```

   **Proposed (clean):**
   ```
   cudaq-mlir-runtime.dylib ─── statically links LLVM ─── single GlobalParser
                    ▲
                    │ (dynamic link at runtime)
   cudaq-builder.dylib ──────── NO direct LLVM link ─── resolves from above ✓
   cudaq-qpud ───────────────── NO direct LLVM link ─── resolves from above ✓
   ```

   Implementation:
   - Keep `LLVMCodeGen` + `force_load` only in `cudaq-mlir-runtime`
   - Remove direct `LLVMCodeGen` links from `cudaq-builder`, `cudaq-qpud`, Python extension
   - Other libraries resolve LLVM symbols (like `cl::ParseCommandLineOptions`) from `cudaq-mlir-runtime.dylib` at runtime

   This eliminates duplicate LLVM copies entirely, requiring **no LLVM patches**. The key insight is that when libraries use LLVM through dynamic linking to `cudaq-mlir-runtime`, they share its single GlobalParser and option registry. Only one static initialization runs, so no duplicate registration occurs.

   See `~/llvm_symbol_report.md` Section 9 for detailed implementation steps.
