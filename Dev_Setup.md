# Setting up your Environment

This document contains a guide for how to set up your development environment if
you want to contribute to the code in this repository.

## Working inside the provided development container (recommended)

This repository defines a development container to facilitate working in a
controlled environment that does not depend on, or interfere with, other
software that is installed on your system. We recommend you work within it. To
do so, please follow the instructions for enabling Docker on your respective
operating system, and then proceed to set up your IDE following the
recommendations for [developing with VS Code and
Docker](#developing-with-vs-code-and-docker).

### Enabling Docker on Mac and Linux

[Docker Desktop][docker_desktop_install] is a free one-click-install application
for Mac, Linux, or Windows that contains the Docker Engine needed to use our
development container, as well as a GUI to manage your containers, images and
volumes. It is not necessary to log into an account to use docker if you don't
want to create one.

On Linux, it is possible and sufficient to install only the engine. However, we
recommend installing Docker Desktop regardless of your operating system for a
consistent experience across all platforms.

### Enabling Docker on Windows

If you are working with a Windows machine, please enable the [Windows Subsystem
for Linux][wsl] by running the command `wsl --install`. The subsystem gives you
an easy way to work with Docker containers, as well as to develop and test code
locally for a variety of Linux distributions.

Please install [Docker Desktop][docker_desktop_install] on Windows to manage
your containers. It is not necessary to log into an account if you don't want to
create one. Make sure to select "Use WSL 2 instead of Hyper-V" during
installation.

[wsl]: https://learn.microsoft.com/en-us/windows/wsl/install
[docker_desktop_install]: https://docs.docker.com/get-docker

### Developing with VS Code and Docker

[VS Code][vs_code] is a robust cross-platform IDE that integrates well with
Docker through the use of an extension. Extensions let you add languages and
others tools to support specialized development workflows. To work with our
development container please install the [Development Containers
extension][dev_container_extension].

Create a local clone of your fork of the CUDA-Q repository, navigate to that
folder, and open VS Code by executing the command `code .`. You should now get a
pop up offering to open the folder in the container. Please select this option
to work in a pre-configured environment that contains all necessary dependencies
to build CUDA-Q:

- On Mac/Linux: Select the option `Open Folder in Container`
- On Windows: Select the option `Clone Repository in Container Volume` for the
  best performance. You can also select `Open Folder in Container`, which will
  ensure that changes you made inside the container environment are visible also
  outside the container volume. For more information, see also [this
  guide][clone_in_container].

After you have selected the above option to launch the container, the VS Code
window will reload and you should see a green box in the lower left corner that
states `Development Container: cudaq-dev`.

The container also defines which extensions will be [loaded
automatically](.devcontainer/devcontainer.json) when launching it. This does not
impact your VS Code configuration outside the container environment.

You should now be all set to build CUDA-Q and run tests. Please open a terminal
in VS Code and follow the instructions [here](./Building.md) to confirm that
everything works as expected.

[vs_code]: https://code.visualstudio.com/download
[dev_container_extension]:
    https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers

[clone_in_container]:
    https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-a-git-repository-or-github-pr-in-an-isolated-container-volume

### Language server support for LLVM and MLIR dialects in VS Code

The development container in this repository is configured with extensions to
facilitate working with MLIR and LLVM files. The [MLIR
extension][mlir_extension] requires a language server to work properly. The
`cudaq-lsp-server` tool in this repository extends the [MLIR Language
Server](https://mlir.llvm.org/docs/Tools/MLIRLSP/) to add support for CUDA-Q
specific dialects. It recognizes files with extensions `.mlir` and `.qke`.

The `cudaq-lsp-server` is built when running the [CUDA-Q build
script](./scripts/build_cudaq.sh) as described in [Building CUDA-Q from
Source](./Building.md). If you customize the installation location, you will
need to either add that location to your path, or edit the value of
`mlir.server_path` in your workspace settings for the extension to work
properly.

[mlir_extension]:
    https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir

## Working in your own environment

If you have followed the instructions for working inside the provided
development container (recommended) you are all set and can proceed to [Getting
Started with Developing CUDA-Q](./Building.md).

If you do not leverage the container definition provided in this repository, you
will need to manually install all prerequisites for building the code on this
repository, and configure your preferred IDE for working with Python, C++, MLIR,
and LLVM components. To install all necessary build and runtime dependencies,
please follow the instructions in our [official guide][data_center_install] for
building and installing CUDA-Q from source. The guide details how to get a fully
featured CUDA-Q installation that can be deployed on a system other than the one
used for the build. For development purposes, components/certain prerequisites
can be omitted as needed, and additional tools and flags for debugging purposes
may be beneficial.

[data_center_install]:
    https://nvidia.github.io/cuda-quantum/latest/using/install/data_center_install.html
