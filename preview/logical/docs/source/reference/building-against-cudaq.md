# Building against CUDA-Q

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

## The MLIR provider libraries

CUDA-Q Logical never builds or vendors LLVM/MLIR itself. It resolves the
implementation from the shared provider libraries that ship with the CUDA-Q
core package: `libLLVM`, `libMLIR`, `libMLIRExecutionEngineShared` and
`libMLIR-C`, next to CUDA-Q's own `libcudaqMLIR` (CUDA-Q dialects and
transforms) and `libcudaqMLIRCAPI` (CUDA-Q's C API additions). `cudaq::MLIR`
in CMake resolves that whole provider set; linking it never embeds a second
copy of upstream MLIR/LLVM into a CUDA-Q Logical binary.

When `QLX_VERIFY_TOOLCHAIN=ON`, the configure-time audit inspects the real
`libLLVM` provider for the LLVM version CUDA-Q was built with (the slimmed
`libcudaqMLIR` no longer embeds upstream LLVM), and checks the C++ compiler
family recorded in `libcudaqMLIR` against the consumer's compiler. The
version probe falls back to the legacy monolithic libcudaqMLIR form only on
installs that predate the provider split; an SDK without providers fails the
check instead of silently pretending to be one.

## Wheel dependency policy

Building MLIR extensions uses the SDK: `cudaq-devel` (toolchain: headers,
TableGen, CMake configs) plus `cudaq-core` carrying the shared runtime
providers (libLLVM/libMLIR/libMLIR-C/libMLIRExecutionEngineShared) and
`libcudaqMLIR`. Install the set with dependency resolution enabled; a
`--no-deps` bypass is not a supported install and never constitutes a
passing check.

This branch retains today's published metadata (onsen #6 note_79265934 -
approved fallback): `cudaq-devel` keeps the full matched `cudaq` frontend
dependency closure, so a normal `pip install cudaq-devel` yields the
complete toolchain including the runtime. The delegated slimming to a direct
cudaq-core dependency is staged in this branch (config-side support is
implemented) and is a separate product decision.
