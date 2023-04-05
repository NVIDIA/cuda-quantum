# Welcome to CUDA Quantum

The [CUDA Quantum](https://developer.nvidia.com/cuda-quantum) toolset for hybrid
quantum-classical computers enables integration and programming of quantum
processing units (QPUs), GPUs, and CPUs in one system.

This Docker image contains contains all necessary tool for application
development using CUDA Quantum in C++ or Python. This includes an installation
of the NVQ++ compiler, the CUDA Quantum runtime, as well as a selection of
integrated CPU and GPU backends for rapid application development and testing.
Additional software can be installed into the container using the password
`cuda-quantum`.

The image includes a folder with examples in C++ and Python in the home
directory. You can find more information about CUDA Quantum including a link to
the documentation on our [GitHub
repository](https://github.com/NVIDIA/cuda-quantum).

The CUDA Quantum installation in this image is licensed under [Apache License
2.0](https://www.apache.org/licenses/LICENSE-2.0). More information about the
license and third party libraries can be found in the LICENSE and NOTICE files
of the CUDA Quantum installation folder defined by `CUDA_QUANTUM_PATH`.
