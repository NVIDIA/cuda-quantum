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
