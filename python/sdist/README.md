# Welcome to the CUDA-Q Python API

CUDA-Q is a comprehensive framework for quantum programming. It features:

- A programming model which extends C++ and Python with quantum kernels,
  enabling high-level programming in familiar languages
- A high-performance quantum compiler, `nvq++`, based on the industry standard
  LLVM toolchain
- Interoperability with all of the leading models and tools for accelerated
  computing, including CUDA, ISO standard parallelism, OpenMP, and OpenACC
- The ability to utilize and seamlessly switch between different quantum
  technologies, including state-of-the-art simulator backends with NVIDIA
  cuQuantum and a number of different physical quantum processors (QPUs)

The CUDA-Q Python wheels contain the Python API and core components of
CUDA-Q. More information about available packages as well as a link to the
documentation and examples for each version can be found in the [release
notes][cudaq_docs_releases]. System and compatibility requirements
are listed in the Installation Guide of the linked documentation.

**Note**: CUDA-Q is currently only supported on Linux operating systems using
`x86_64` or `aarch64`/`arm64` processors.

[cudaq_docs_releases]:
    https://nvidia.github.io/cuda-quantum/latest/releases.html
