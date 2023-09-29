# Building CUDA Quantum from Source

To learn more about how to use CUDA Quantum, please take a look at the [CUDA
Quantum Documentation][cuda_quantum_docs]. The page also contains [installation
instructions][official_install] for released packages.

[cuda_quantum_docs]: https://nvidia.github.io/cuda-quantum/latest
[official_install]: https://nvidia.github.io/cuda-quantum/latest/install.html

This document contains instructions for how to build CUDA Quantum from source.
This is only needed if you would like to try out the latest (unreleased) version
that is currently under development in this repository and/or if you would like
to add your own modifications to this code base.

To build the CUDA Quantum source code locally, fork this repository and follow
the [instructions for setting up your environment](./Dev_Setup.md). Once you
have done that, you should be able to run the [build
script](./scripts/build_cudaq.sh) to build and install CUDA Quantum in a local
folder. The path where CUDA Quantum will be installed can be configured by
setting the environment variable `CUDAQ_INSTALL_PREFIX`. If you customize this
path or do not work in our development container, you either need to invoke the
built/installed binaries with the full path or modify your path variables as
follows:

```bash
export PATH="${CUDAQ_INSTALL_PREFIX}/bin:${PATH}"
export PYTHONPATH="${CUDAQ_INSTALL_PREFIX}:${PYTHONPATH}"
```

Our development container comes with a pre-built version of LLVM which is used to
build CUDA Quantum. If you are working outside our container, please see the
instructions for [building CUDA Quantum with a custom LLVM
version](#building-cuda-quantum-with-a-custom-llvm-version).

Once you have built CUDA Quantum, you should be able to run the tests and
confirm that they pass by navigating to the `build` folder and entering the
command `ctest`. If all tests pass, you are all set to [start
developing](./Developing.md).

## GPU accelerated backends

CUDA Quantum comes with a range of simulator backends for testing and debugging
application code. In addition to simulators that run on CPUs, there are also
multiple backends leveraging [cuQuantum][cuquantum] for GPU accelerated
simulation. This repository does not require you to have a GPU; if no GPU is
detected, the corresponding backends will simply be omitted from the build.

To use or develop GPU-based simulators, you will need to make sure you have a
suitable NVIDIA GPU with the [latest driver][nvidia_driver] installed. You can
confirm the GPU is properly detected running the command `nvidia-smi`.

If you are working in our development container, please add a `--gpus=all` flag
to the list of `runArgs` in the
[.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) file and
relaunch the container. It is unfortunately not possible to define a development
container that enables GPU support depending on the system configuration (see
[this issue](https://github.com/airo-ugent/airo-ros/issues/17)).

If you are not working in our development container, you may need to specify the
correct path to your cuQuantum installation by setting the environment variable
`CUQUANTUM_INSTALL_PREFIX`.

[cuquantum]: https://developer.nvidia.com/cuquantum-sdk
[nvidia_driver]: https://www.nvidia.com/download/index.aspx

## Building CUDA Quantum with a custom LLVM version

CUDA Quantum is intended to be built using the LLVM commit that the submodule
is set to (currently LLVM 16.0.6). If you need to update the LLVM commit, or if
you are not using the pre-built version of that commit in our development
container, LLVM will need to be built from source. ***Building with LLVM >=
17.x is not currently supported*** due to the QIR specification's requirement
to use typed pointers. LLVM 17.x removed the ability to use typed pointers and
instead now requires [opaque pointers](https://llvm.org/docs/OpaquePointers.html#version-support).
The QIR specification will be updated to support opaque pointers in the near
future.

To configure our build scripts with a custom LLVM, please set the LLVM
submodule to the desired commit, and set the following environment variable,
replacing `<installation_path>` with the path where the freshly built LLVM tools
and libraries should be installed:

```bash
export LLVM_INSTALL_PREFIX=<installation_path>
```

The CUDA Quantum [build script](./scripts/build_cudaq.sh) checks if
`llvm-config` is available in the bin subfolder of that directory, and will
automatically invoke the [LLVM build script](./scripts/build_llvm.sh) if it is
not.

Please note that building LLVM may take a while and you should have about 64GB
of memory available for the build. Please configure your Docker container
settings accordingly. If you are working on Windows, the amount of resources
that are available within a container is determined by the WSL settings. Please
create or modify the [WSL configuration file][wsl_config] if necessary.

[wsl_config]: https://learn.microsoft.com/en-us/windows/wsl/wsl-config
