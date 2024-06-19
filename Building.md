# Building CUDA-Q from Source

To learn more about how to use CUDA-Q, please take a look at the [CUDA-Q
Documentation][cuda_quantum_docs]. The page also contains [installation
instructions][official_install] for released packages and nightly builds.

[cuda_quantum_docs]: https://nvidia.github.io/cuda-quantum/latest
[official_install]: https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html#install-cuda-q
[data_center_install]:
    https://nvidia.github.io/cuda-quantum/latest/using/install/data_center_install.html

This document is intended for anyone who wants to develop their own
modifications of, or contributions to, this code base. If you want to build
CUDA-Q from source for the purpose of installing it on a system that is not
officially supported by our distributed packages, please take a look at our
[installation guide][data_center_install] instead.

To build the CUDA-Q source code locally, fork this repository and follow the
[instructions for setting up your environment](./Dev_Setup.md). Once you have
done that, you should be able to run the [build
script](./scripts/build_cudaq.sh) to build and install CUDA-Q in a local folder.
The path where CUDA-Q will be installed can be configured by setting the
environment variable `CUDAQ_INSTALL_PREFIX`. If you customize this path or do
not work in our development container, you either need to invoke the
built/installed binaries with the full path or modify your path variables as
follows:

```bash
export PATH="${CUDAQ_INSTALL_PREFIX}/bin:${PATH}"
export PYTHONPATH="${CUDAQ_INSTALL_PREFIX}:${PYTHONPATH}"
```

Our development container comes with a pre-built version of LLVM which is used
to build CUDA-Q. If you are working outside our container, please see the
instructions for [building CUDA-Q with a custom LLVM
version](#building-cuda-q-with-a-custom-llvm-version).

Once you have built CUDA-Q, you should be able to run the tests and confirm that
they pass by navigating to the `build` folder and entering the command `ctest`.
If all tests pass, you are all set to [start developing](./Developing.md).

## GPU accelerated backends

CUDA-Q comes with a range of simulator backends for testing and debugging
application code. In addition to simulators that run on CPUs, there are also
multiple backends leveraging [cuQuantum][cuquantum] and [cuTensor][cutensor] for
GPU accelerated simulation. You can take a look at the
[install_prerequisites.sh](scripts/install_prerequisites.sh) script to see how
these dependencies are installed.

Developing code in this repository does not require you to have a GPU; if the
CUDA compiler is not found during the build, the corresponding components will
simply be omitted from the build. However, to use or develop and test GPU-based
simulators, you will need to make sure you have a suitable NVIDIA GPU with the
[latest driver][nvidia_driver] installed. You can confirm the GPU is properly
detected running the command `nvidia-smi` in your development environment.

[cuquantum]: https://developer.nvidia.com/cuquantum-sdk
[cutensor]: https://developer.nvidia.com/cutensor
[nvidia_driver]: https://www.nvidia.com/download/index.aspx

## Building CUDA-Q with a custom LLVM version

CUDA-Q is intended to be built using the LLVM commit that the submodule is set
to (currently LLVM 16.0.6). If you need to update the LLVM commit, or if you are
not using the pre-built version of that commit in our development container,
LLVM will need to be built from source. ***Building with LLVM >= 17.x is not
currently supported*** due to the QIR specification's requirement to use typed
pointers. LLVM 17.x removed the ability to use typed pointers and instead now
requires [opaque
pointers](https://llvm.org/docs/OpaquePointers.html#version-support). The QIR
specification will be updated to support opaque pointers in the near future.

To configure our build scripts with a custom LLVM, please set the LLVM submodule
to the desired commit, and set the following environment variable, replacing
`<installation_path>` with the path where the freshly built LLVM tools and
libraries should be installed:

```bash
export LLVM_INSTALL_PREFIX=<installation_path>
```

The CUDA-Q [build script](./scripts/build_cudaq.sh) checks if `llvm-config` is
available in the bin subfolder of that directory, and will automatically invoke
the [LLVM build script](./scripts/build_llvm.sh) if it is not.

Please note that building LLVM may take a while and you should have about 64GB
of memory available for the build. Please configure your Docker container
settings accordingly. If you are working on Windows, the amount of resources
that are available within a container is determined by the WSL settings. Please
create or modify the [WSL configuration file][wsl_config] if necessary.

[wsl_config]: https://learn.microsoft.com/en-us/windows/wsl/wsl-config
