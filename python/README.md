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
CUDA Quantum. For more information about available packages and documentation,
see our [release
notes](https://nvidia.github.io/cuda-quantum/latest/releases.html).

## Installing CUDA Quantum

CUDA Quantum Python wheels are available on
[PyPI.org](https://pypi.org/project/cuda-quantum). To install the latest
release, simply run

```console
pip install cuda-quantum
```

At this time, wheels are distributed for Linux operating systems only. To build
the CUDA Quantum Python API from source using pip:

```console
git clone https://github.com/NVIDIA/cuda-quantum.git
cd cuda-quantum && ./scripts/install_prerequisites.sh
pip install .
```

For more information about building the entire C++ and Python API's, please see
the CUDA Quantum [documentation][official_install].

[official_install]: https://nvidia.github.io/cuda-quantum/latest/install.html

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

## Documentation

To see more examples, go to [python examples][python_examples], or check out the
[Python API reference][python_api_reference].

[python_examples]:
    https://nvidia.github.io/cuda-quantum/latest/using/python.html
[python_api_reference]:
    https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html

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
