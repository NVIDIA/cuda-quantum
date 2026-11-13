# cudaq-core

`cudaq-core` is the shared native foundation of CUDA-Q. It provides the
compiler bindings (`cudaq.mlir`) and the native backend interfaces, without
frontend execution or simulator dependencies.

This package contains the runtime dependency for the `cudaq-devel` wheel, which
can be downloaded as an artifact from the [CUDA-Q GitHub releases][releases]
starting with release 0.16.0. The `cudaq-devel` wheel contains all necessary
(build) dependencies to build CUDA-Q compiler extensions and integrations.

Most users should install [`cudaq`](https://pypi.org/project/cudaq/) instead.
It selects a matching `cudaq-core` and CUDA-specific frontend. The core and
frontend versions must match exactly.

More information is available in the [CUDA-Q documentation][docs].

[releases]: https://github.com/NVIDIA/cuda-quantum/releases
[docs]: https://nvidia.github.io/cuda-quantum/latest/
