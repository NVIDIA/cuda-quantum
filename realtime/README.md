# CUDA-Q Realtime Library

CUDA-Q Realtime is a library for tightly coupling GPU accelerated compute to the control system of a quantum processor.
It fulfills two primary responsibilities:
1. It provides the low-level basis of realtime coprocessing between FPGA and CPU-GPU systems.
1. It provides the low latency networking stack of the NVQLink architecture, enabling system integrators to achieve few-microsecond data round trips between FPGA and GPU.

> [!WARNING]
> This library is currently in early access / alpha stage and will continue to rapidly evolve as we build interactively with collaborators.

> [!NOTE]
> While the library is in early access, instructions to reproduce the FPGA-GPU latency round trip on third party systems can be found at [docs/nvqlink_latency_demo.md](docs/nvqlink_latency_demo.md).

## Getting Started

```bash
# Configure, need cmake 3.28+
cmake -G Ninja .. -DNVQLINK_BUILD_TESTS=ON
# Build
ninja 
# Test
ctest 
```

## Extending the library 

Check out the tests in the `unittests` folder as well as the example codes in `examples`. 

3rd parties can extend this library with new `device` types. The goal is to define 
a subclass of `device_mixin` that allows you specify device traits that your `device` exposes. 
There are a number of traits available, and they are specified in the `device.h` file. There are 
example devices in the `devices/` folder there too. 

3rd parties can also provide custom compiler implementations. Compilers take generic 
code strings and return a `compiled_kernel`. There is one compiler implemented as of 
today, and it is the CUDA-Q compiler. For simplicity, this compiler simply delegates to 
the command line CUDA-Q toolchain. Subclasses should be able to override the `cudaq-opt` 
pass flags. This would allow one to handle CUDA-Q IR operations in a target specific manner 
(e.g., custom lowering of the device_call op).


