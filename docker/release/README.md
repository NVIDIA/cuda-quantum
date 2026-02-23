# Welcome to CUDA-Q

The [CUDA-Q](https://developer.nvidia.com/cuda-quantum) toolset for hybrid
quantum-classical computers enables integration and programming of quantum
processing units (QPUs), GPUs, and CPUs in one system.

This Docker image contains contains all necessary tool for application
development using CUDA-Q in C++ or Python. This includes an installation
of the `nvq++` compiler, the CUDA-Q runtime, as well as a selection of
integrated CPU and GPU backends for rapid application development and testing.
To install additional software, run the installation command as administrator
if necessary by using `sudo <command>` (no password is required).

The image includes a folder with examples in C++ and Python in the home
directory. You can find more information about CUDA-Q including a link to
the documentation on our [GitHub
repository](https://github.com/NVIDIA/cuda-quantum).

The CUDA-Q installation in this image is licensed under [Apache License
2.0](https://www.apache.org/licenses/LICENSE-2.0). More information about the
license and third party libraries can be found in the LICENSE and NOTICE files
of the CUDA-Q installation folder defined by `CUDA_QUANTUM_PATH`.
