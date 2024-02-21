# Welcome to the CUDA Quantum Python API

CUDA Quantum is a comprehensive framework for quantum programming. It features:

- A programming model which extends C++ and Python with quantum kernels,
  enabling high-level programming in familiar languages
- A high-performance quantum compiler, `nvq++`, based on the industry standard
  LLVM toolchain
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

## Configuring CUDA Quantum for GPU Simulation

CUDA Quantum can be used to compile and run quantum programs on a CPU-only
system, but a GPU is highly recommended and necessary to use the some of the
simulators. The GPU-based simulators included in the CUDA Quantum Python wheels
require an existing CUDA installation. Additionally, multi-GPU simulators
require an existing CUDA-aware MPI installation.

To install the necessary dependencies, we recommend using
[Conda](https://docs.conda.io/en/latest/). If you are not already using Conda,
you can install a minimal version following the instructions
[here](https://docs.conda.io/projects/miniconda/en/latest/index.html). The
following commands will create and activate a complete environment for CUDA
Quantum with all its dependencies:

[//]: # (Begin conda install)

```console
    conda create -y -n cuda-quantum python==3.10 pip
    conda install -y -n cuda-quantum -c "nvidia/label/cuda-11.8.0" cuda
    conda install -y -n cuda-quantum -c conda-forge mpi4py openmpi cxx-compiler cuquantum
    conda env config vars set -n cuda-quantum LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/envs/cuda-quantum/lib"
    conda env config vars set -n cuda-quantum MPI_PATH=$CONDA_PREFIX/envs/cuda-quantum
    conda run -n cuda-quantum pip install cuda-quantum
    conda activate cuda-quantum
    source $CONDA_PREFIX/lib/python3.10/site-packages/distributed_interfaces/activate_custom_mpi.sh
```

[//]: # (End conda install)

You must configure MPI by setting the following environment variables:

[//]: # (Begin ompi setup)

```console
  export OMPI_MCA_opal_cuda_support=true OMPI_MCA_btl='^openib'
```

[//]: # (End ompi setup)

*If you do not set these variables you may encounter a segmentation fault.*

**Important**: It is *not* sufficient to set these variable within the Conda
environment, like the commands above do for `LD_LIBRARY_PATH`. To avoid having
to set them every time you launch a new shell, we recommend adding them to
`~/.profile` (create the file if it does not exist), and to `~/.bash_profile` or
`~/.bash_login` if such a file exists.

MPI uses [SSH](https://en.wikipedia.org/wiki/Secure_Shell) or
[RSH](https://en.wikipedia.org/wiki/Remote_Shell) to communicate with each node
unless another resource manager, such as
[SLURM](https://slurm.schedmd.com/overview.html), is used. If you are
encountering an error "The value of the MCA parameter `plm_rsh_agent` was set to
a path that could not be found", please make sure you have an SSH Client
installed.

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
[Contributing.md](https://github.com/NVIDIA/cuda-quantum/blob/main/Contributing.md).

## License

CUDA Quantum is an open source project. The source code is available on
[GitHub][github_link] and licensed under [Apache License
2.0](https://github.com/NVIDIA/cuda-quantum/blob/main/LICENSE). CUDA Quantum
makes use of the NVIDIA cuQuantum SDK to enable high-performance simulation,
which is held to its own respective license.

[github_link]: https://github.com/NVIDIA/cuda-quantum/

## Feedback

Please let us know your feedback and ideas for the CUDA Quantum platform in the
[Discussions][discussions] tab of our [GitHub repository][github_repo], or [file
an issue][cuda_quantum_issues]. To report security concerns please reach out to
[cuda-quantum@nvidia.com](mailto:cuda-quantum@nvidia.com).

[discussions]: https://github.com/NVIDIA/cuda-quantum/discussions
[cuda_quantum_issues]: https://github.com/NVIDIA/cuda-quantum/issues
[github_repo]: https://github.com/NVIDIA/cuda-quantum
