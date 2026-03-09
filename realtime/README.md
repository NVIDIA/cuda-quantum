# CUDA-Q Realtime Library

CUDA-Q Realtime is a library for tightly coupling GPU accelerated compute
to the control system of a quantum processor.

It fulfills two primary responsibilities:

1. It provides the low-level basis of realtime coprocessing
between FPGA and CPU-GPU systems.

2. It provides the low latency networking stack of the NVQLink architecture,
enabling system integrators to achieve few-microsecond
data round trips between FPGA and GPU.

## Getting Started

To learn more about how to work with CUDA-Q Realtime,
please take a look at the [user guide](docs/user_guide.md).

If you would like to install the latest iteration under development in this
repository and/or add your own modifications, take a look at [these
instructions](docs/building.md) about building CUDA-Q Realtime from source.

## Specifications

The following specifications for the CUDA-Q Realtime message protocol
and host API are available:

- [Message protocol](docs/cudaq_realtime_message_protocol.md)

- [Realtime host API](docs/cudaq_realtime_host_api.md)

## Feedback

Please let us know your feedback and ideas for the CUDA-Q platform in the
[Discussions][cuda_quantum_discussions] tab of this repository, or file an
[issue][cuda_quantum_issues].

[cuda_quantum_discussions]: https://github.com/NVIDIA/cuda-quantum/discussions
[cuda_quantum_issues]: https://github.com/NVIDIA/cuda-quantum/issues
