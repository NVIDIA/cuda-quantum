# CUDA-Q Realtime Library

CUDA-Q Realtime is a library for tightly coupling GPU accelerated compute
to the control system of a quantum processor.

It fulfills two primary responsibilities:

1. It provides the low-level basis of realtime coprocessing
between FPGA and CPU-GPU systems.

2. It provides the low latency networking stack of the NVQLink architecture,
enabling system integrators to achieve few-microsecond
data round trips between FPGA and GPU.

> [!WARNING]
> This library is currently in early access / alpha stage
> and will continue to rapidly evolve as we build interactively with collaborators.

Please refer to the specifications for the [message protocol](docs/cudaq_realtime_message_protocol.md) 
and [realtime host API](docs/cudaq_realtime_host_api.md) for more information.


> [!NOTE]
> While the library is in early access, instructions to reproduce the FPGA-GPU latency
> round trip on third party systems can be found at [docs/nvqlink_latency_demo.md](docs/nvqlink_latency_demo.md).

## Getting Started

```bash
# Configure, need cmake 3.28+
cmake -G Ninja .. -DCUDAQ_REALTIME_BUILD_TESTS=ON
# Build
ninja 
# Test
ctest 
```

Check out the tests in the `unittests` folder for examples.
