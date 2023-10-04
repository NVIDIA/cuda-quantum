# Welcome to the CUDA Quantum Python API

CUDA Quantum is a comprehensive framework for quantum programming. It features:

- A programming model which extends C++ and Python with quantum kernels,
  enabling high-level programming in familiar languages
- A high-performance quantum compiler, NVQ++, based on the industry standard
  low-level virtual machine (LLVM) toolchain
- Interoperability with all of the leading models and tools for accelerated
computing, including CUDA, ISO standard parallelism, OpenMP, and OpenACC
- The ability to utilize and seamlessly switch between different quantum
  technologies, including state-of-the-art simulator backends with NVIDIA
  cuQuantum and a number of different physical quantum processors (QPUs)

The CUDA Quantum Python wheels contain the Python API and core components of
CUDA Quantum. More information about available packages as well as a link to the
documentation and examples for each version can be found in the [release
notes][cudaq_docs_releases]. System requirements and compatibility are listed in
the Getting Started section of the linked documentation.

[cudaq_docs_releases]:
    https://nvidia.github.io/cuda-quantum/latest/releases.html

## Installing CUDA Quantum

To install the latest stable version of CUDA Quantum, run

```console
python3 -m pip install cuda-quantum
```

CUDA Quantum can be used to compile and run quantum programs on a CPU-only
system, but a GPU is highly recommended and necessary to use the some of the
simulators. The GPU-based simulators included in the CUDA Quantum Python wheels
require an existing CUDA installation. Additionally, multi-GPU simulators
require an existing MPI installation.

In most cases, the CUDA and MPI dependencies can be installed via package
manager. On Ubuntu 22.04, for example, the following commands install all
optional CUDA dependencies:

```console
  arch=x86_64 # set this to sbsa for ARM processors
  sudo apt-get update && sudo apt-get install -y wget
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$arch/cuda-keyring_1.0-1_all.deb
  sudo dpkg -i cuda-keyring_1.0-1_all.deb
  sudo apt-get update && sudo apt-get install -y cuda-toolkit-11.8
```

Detailed instructions for how to install the complete CUDA toolkit on different
operating systems can be found in the [CUDA
documentation](https://docs.nvidia.com/cuda/).

If you have several GPUs available but no MPI installation yet, we recommend
taking a look at the [OpenMPI documentation](https://docs.open-mpi.org/). On
Ubuntu 22.04, for example, the following commands install the necessary MPI
libraries:

```console
  sudo apt-get update && sudo apt-get install -y openmpi-common openmpi-bin libopenmpi-dev
```

## Running CUDA Quantum

You should now be able to import CUDA Quantum and start building quantum
programs in Python!

```console
import cudaq

kernel = cudaq.make_kernel()
qubit = kernel.qalloc()
kernel.x(qubit)
kernel.mz(qubit)

result = cudaq.sample(kernel)
```

Additional examples and documentation are linked in the [release
notes][cudaq_docs_releases].

## Contributing

There are many ways in which you can get involved with CUDA Quantum. If you are
interested in developing quantum applications with CUDA Quantum, our [GitHub
repository][github_link] is a great place to get started! For more information
about contributing to the CUDA Quantum platform, please take a look at
[Contributing.md](../Contributing.md).

## License

CUDA Quantum is an open source project. The source code is available on
[GitHub][github_link] and licensed under [Apache License 2.0](../LICENSE). CUDA
Quantum makes use of the NVIDIA cuQuantum SDK to enable high-performance
simulation, which is held to its own respective license.

[github_link]: https://github.com/NVIDIA/cuda-quantum/

## Feedback

Please let us know your feedback and ideas for the CUDA Quantum platform in the
[Discussions][discussions] tab of our [GitHub repository][github_repo], or [file
an issue][cuda_quantum_issues]. To report security concerns please reach out to
[cuda-quantum@nvidia.com](mailto:cuda-quantum@nvidia.com).

[discussions]: https://github.com/NVIDIA/cuda-quantum/discussions
[cuda_quantum_issues]: https://github.com/NVIDIA/cuda-quantum/issues
[github_repo]: https://github.com/NVIDIA/cuda-quantum
