# Building against CUDA-Q

CUDA-Q Logical requires the native EDEM library and its Python value package.
The standard `-L` build below builds EDEM from the same checkout and installs
its Python package beside CUDA-Q. For a separate Logical build, install a
matching `cudaq-edem` wheel or use the native installation described below.

Build CUDA-Q Logical with the standard CUDA-Q build script by adding `-L`.
From the root of the CUDA-Q repository, run:

```bash
bash scripts/build_cudaq.sh -L
```

The script builds and installs CUDA-Q, then configures, builds, and installs
CUDA-Q Logical against that installation. The CUDA-Q Logical build directory
is `build/preview/logical`.

For development environment setup, prerequisites, and general build instructions,
see [Developing.md](https://github.com/NVIDIA/cuda-quantum/blob/main/Developing.md)
in the main repository. The build script's usual options also apply when using
`-L`, including `-i` for incremental builds, `-c Debug` for a debug build, and
`-j` to set the number of build jobs.

## Build EDEM and Logical separately

To work against an existing CUDA-Q installation, first build and install EDEM:

```bash
cmake -S edem -B build-edem -DCUDAQ_EDEM_BUILD_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX=/path/to/edem-install
cmake --build build-edem
cmake --install build-edem
```

Make the installed `cudaq_edem` Python package available to the selected
interpreter, then configure Logical against both installations:

```bash
cmake -S preview/logical -B build/preview/logical \
  -DCUDAQ_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" \
  -Dcudaq-edem_DIR=/path/to/edem-install/lib/cmake/cudaq-edem \
  -DQLX_VERIFY_TOOLCHAIN=ON
cmake --build build/preview/logical
```

Use the same compiler family and LLVM/MLIR installation as CUDA-Q. CMake can
also discover EDEM through its installed Python package; in that case the
explicit `cudaq-edem_DIR` option is unnecessary.
