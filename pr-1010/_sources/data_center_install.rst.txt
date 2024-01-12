Installation from Source
*******************************************

In most cases, you should not need to build CUDA Quantum from source. For the
best experience, we recommend using a container runtime to avoid conflicts with
other software tools installed on the system. Note that `Singularity
<https://docs.sylabs.io/guides/2.6/user-guide/faq.html#what-is-so-special-about-singularity>`__
or `Docker rootless mode <https://docs.docker.com/engine/security/rootless/>`__
address common issue or concerns that are often the motivation for
avoiding the use of containers. Singularity, for example, can be installed 
in a user folder and its installation does not require admin permissions; see
:ref:`this section <install-singularity-image>` for more detailed instructions 
on how to do that. Our installation guide also contains instructions for how to 
:ref:`connect an IDE <local-development-with-vscode>` to a running container.

If you do not want use a container runtime, we also provide pre-built binaries
for using CUDA Quantum with C++. These binaries are built following the instructions 
in this guide and should work for you as long as your system meets the compatibility 
requirements listed under :ref:`Prerequisites <compatibility-prebuilt-binaries>`. 
To install them, please follow the instructions :ref:`here <install-prebuilt-binaries>`.

If you still want to build and install CUDA Quantum from source, you will need
to ensure that all dependencies installed in the build and host system are
compatible with your CUDA Quantum installation. The rest of this guide outlines
specific compatibility requirements during the build and after installation, and
walks through the installation steps.

.. note::
  
  The build described in this guide does not include the support
  for building the Python support for CUDA Quantum. For more information about 
  using CUDA Quantum from Python, please take a look at 
  :ref:`this page <install-python-wheels>`.

CUDA Quantum contains some components that are only included as
pre-built binaries and not part of our open source repository. We are working on
either open-sourcing these components or making them available as separate downloads
in the future. Even without these components, almost all features of CUDA
Quantum will be enabled in a source build, though some pieces may be less
performant. 
At this time, the :ref:`multi-GPU state vector simulator <nvidia-mgpu-backend>` 
backend will not be included if you build CUDA Quantum from source.

.. _compatibility-prebuilt-binaries:

Prerequisites
------------------------------------

The following pre-requisites need to be satisfied both on the build system and
on the host system, that is the system where the built CUDA Quantum binaries
will be installed and used.

- Linux operating system. The instructions in this guide have been validated
  with the `AlmaLinux 8 image <https://hub.docker.com/u/almalinux>`__ that
  serves as the base image for the `manylinux_2_28 image
  <https://github.com/pypa/manylinux>`__, and should work for the operating
  systems CentOS 8, Debian 11 and 12, Fedora 38, OpenSUSE/SELD/SLES 15.5, RHEL 8
  and 9, Rocky 8 and 9, and Ubuntu 22.04. Other operating systems may work, but
  have not been tested.
- `Bash <https://www.gnu.org/software/bash/>`__ shell. The CUDA Quantum 
  build, install and run scripts expect to use `/bin/bash`.
- `GNU C library <https://www.gnu.org/software/libc/>`__. 
  Make sure that the version on the host system is the same one
  or newer than the version on the build system. Our own builds
  use version 2.28.
- CPU with either x86-64 (x86-64-v3 architecture and newer) or ARM64
  architecture. Other architectures may work but are not tested and may require
  adjustments to the build instructions.
- Needed only on the host system: NVIDIA GPU with Volta, Turing, Ampere, Ada, or
  Hopper architecture and `Compute Capability
  <https://developer.nvidia.com/cuda-gpus>`__ 7+. Make sure you have the latest
  `drivers <https://www.nvidia.com/download/index.aspx>`__ installed for your
  GPU, and double check that the driver version listed by the `nvidia-smi`
  command is 470.57.02 or newer.

We strongly recommend using a virtual environment for the build that includes
*only* the tools and dependencies listed in this guide. If you have additional
software installed, you will need to make sure that the build is linking against
the correct libraries and versions.

Build Dependencies
------------------------------------

In addition to the prerequisites listed above, you will need to install the
following prerequisites in your build environment:

- Standard C library: We currently statically link *all* dependencies, including 
  the standard libraries. We may revise that in the future. 
  To use the current build configuration, please make sure you have the 
  static version of the `GNU C Library <https://www.gnu.org/software/libc/>`__, 
  including the POSIX Threads library, installed on your system. 
  The necessary package(s) can usually be obtained via package manager 
  for your distribution.
- Python version 3.8 or newer: The Python interpreter is required
  (only) for some of the LLVM build scripts and the Python version
  used for the build does not have to match the version on the host
  system.
- Common tools: wget, git, unzip. The commands in the rest of this guide assume
  that these tools are present on the build system, but they can be replaced by
  other alternatives (such as, for example, manually going to a web page and
  downloading a file/folder).

The above prerequisites are no longer needed once CUDA Quantum is built and 
do not need to be present on the host system.

.. note::

  The CUDA Quantum build scripts and the commands listed in the rest of this 
  document assume you are using `bash` as the shell for your build.

In addition to installing the needed build dependencies listed above, make sure
to set the following environment variables prior to proceeding:

.. literalinclude:: ../../scripts/configure_build.sh
    :language: bash
    :start-after: [>InstallLocations]
    :end-before: [<InstallLocations]

These environment variables *must* be set during the build. Their value can be
chosen freely, but the paths specified during the build are also where the
corresponding libraries will be installed on the host system. We are working on
making this more flexible in the future.

.. note::

  If you deviate from the instructions below for installing one of the
  dependencies and instead install it for example via package manager, you will
  need to make sure that the installation path matches the path you set for the
  corresponding environment variable(s).

  Please do **not** set `LLVM_INSTALL_PREFIX` to an existing directory;
  To avoid compatibility issues, it is important to use the same compiler
  to build the LLVM/MLIR dependencies from source as is later used to 
  build CUDA Quantum itself.

CUDA
+++++++++++++++++++++++++++++++

Building CUDA Quantum requires a full installation of the CUDA toolkit.
The instructions are tested using version 11.8, but any CUDA 11 or 12 version
should work, as long as the installed driver on both the build and the host 
system supports that CUDA version. We recommend using the latest CUDA version
that is supported by your driver.

Download a suitable `CUDA version <https://developer.nvidia.com/cuda-toolkit-archive>`__
following the installation guide for your platform in the online documentation
linked on that page.

Within the tested AlmaLinux 8 environment, for example, the following commands
install CUDA 11.8:

.. literalinclude:: ../../scripts/configure_build.sh
    :language: bash
    :start-after: [>CUDAInstall]
    :end-before: [<CUDAInstall]

cuQuantum
+++++++++++++++++++++++++++++++

Each version of CUDA Quantum is compatible only with a specific cuQuantum version.
As of CUDA Quantum 0.6, this is version 23.10. Newer versions of cuQuantum (if they exist)
might be compatible but have not been tested.

Make sure the environment variable `CUDA_ARCH_FOLDER` is set to either `x86_64`
or `sbsa` (for ARM64) depending on your processor architecture, and `CUDA_VERSION`
is set to the installed CUDA version.
Install cuQuantum version 23.10 using the following commands:

.. literalinclude:: ../../scripts/configure_build.sh
    :language: bash
    :start-after: [>cuQuantumInstall]
    :end-before: [<cuQuantumInstall]

cuTensor
+++++++++++++++++++++++++++++++

Depending on how you installed CUDA, the cuTensor library is usually not included 
in the installation. This library is used by some of the simulator backends. 
Please check the cuQuantum documentation to ensure you choose a version that is 
compatible with the used cuQuantum version, such as version 1.7.

Make sure the environment variable `CUDA_ARCH_FOLDER` is set to either `x86_64`
or `sbsa` (for ARM64) depending on your processor architecture, and `CUDA_VERSION`
is set to the installed CUDA version.
Install cuTensor version 1.7 using the following commands:

.. literalinclude:: ../../scripts/configure_build.sh
    :language: bash
    :start-after: [>cuTensorInstall]
    :end-before: [<cuTensorInstall]

Toolchain
+++++++++++++++++++++++++++++++

The compiler toolchain used for the build needs to support C++20 and must be a supported 
`CUDA host compiler <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-host-compilers>`__
for the installed CUDA version.
The following instructions have been tested with
`GCC-11 <https://gcc.gnu.org/index.html>`__ as your toolchain for building CUDA Quantum.
If you use a different compiler, we recommend using an OpenMP-enabled compiler. At this
time, we actively test building with GCC 11 and 12, as well as with Clang 16. 
Other toolchains may be supported but have not been tested.

Within the tested AlmaLinux 8 environment, for example, the following commands
install GCC 11:

.. literalinclude:: ../../scripts/configure_build.sh
    :language: bash
    :start-after: [>gccInstall]
    :end-before: [<gccInstall]

Independent on which compiler toolchain you installed, set the following
environment variables to point to the respective compilers on your build system:

.. literalinclude:: ../../scripts/configure_build.sh
    :language: bash
    :start-after: [>ToolchainConfiguration]
    :end-before: [<ToolchainConfiguration]

- The variables CC and CXX *must* be set for the CUDA Quantum build.
- A Fortran compiler is needed (only) to build the OpenSSL dependency; 
  if you have an existing OpenSSL installation that you set the
  `OPENSSL_INSTALL_PREFIX` variable to, you can omit setting the FC 
  environment variable.
- To use GPU-acceleration in CUDA Quantum, make sure to set CUDACXX to 
  your CUDA compiler. If the CUDA compiler is not found when building
  CUDA Quantum, some components and backends will be omitted automatically 
  during the build.

Building CUDA Quantum
------------------------------------

This installation guide has been written for a specific version/commit of CUDA Quantum. 
Make sure to obtain the source code for that version. 
Clone the CUDA Quantum `GitHub repository <https://github.com/NVIDIA/cuda-quantum>`__ and
checkout the appropriate branch, tag, or commit. 
Note that the build scripts assume that they are run from within a git repository, 
and merely downloading the source code as ZIP archive hence will not work.

From within the folder where you cloned the CUDA Quantum repository, run the following
command to build CUDA Quantum:

.. literalinclude:: ../../docker/build/assets.Dockerfile
    :language: bash
    :start-after: [>CUDAQuantumBuild]
    :end-before: [<CUDAQuantumBuild]

The CUDA Quantum build will compile or omit optional components automatically depending
on whether the necessary pre-requisites are found in the build environment.
Please check the build log to confirm that all desired components have been built. If
you see a message that a component has been skipped, make sure you followed the 
instructions for installing the necessary prerequisites and build dependencies, 
and have set the necessary environment variables as described in this document.

Preparing the Installation
------------------------------------

To easily migrate the built binaries to the host system, we recommend creating a
`self-extracting archive <https://makeself.io/>`__. To do so, download the 
`makeself script(s) <https://github.com/megastep/makeself>`__ and move the necessary 
files to install into a separate folder using the command

.. literalinclude:: ../../docker/release/installer.Dockerfile
    :language: bash
    :start-after: [>CUDAQuantumAssets]
    :end-before: [<CUDAQuantumAssets]

You can then create a self-extracting archive with the command

.. code-block:: bash

    ./makeself.sh --gzip --license cuda_quantum_assets/cudaq/LICENSE \
        cuda_quantum_assets install_cuda_quantum.$(uname -m) \
        "CUDA Quantum toolkit for heterogeneous quantum-classical workflows" \
        bash cudaq/migrate_assets.sh -t /opt/nvidia/cudaq

Installation on the Host
------------------------------------

Make sure your host system satisfies the `Prerequisites`_ listed above,
and that the same C++ standard library that was used during the
CUDA Quantum build is present and discoverable on the host system. 
While not strictly necessary, we recommend that you install the toolchain 
that was used for the CUDA Quantum build on the host.

Copy the `install_cuda_quantum` file that you created following the instructions
in the `Preparing the Installation`_ section onto the host system, and then
run the commands

.. code-block:: bash

    sudo -E bash install_cuda_quantum.* --accept
    . /opt/nvidia/cudaq/set_env.sh

This will extract the built assets and move them to the correct locations.
The `set_env.sh` script in `/opt/nvidia/cudaq` defines the necessary environment
variables to use CUDA Quantum. To avoid having to set them manually every time a 
new shell is opened, we highly recommend adding the following lines to
the `/etc/profile` file:

.. code-block:: bash

    if [ -f /opt/nvidia/cudaq/set_env.sh ];
      . /opt/nvidia/cudaq/set_env.sh
    fi

.. note:: 

  CUDA Quantum is configured to use its own linker, meaning the 
  `LLD <https://lld.llvm.org/>`__ linker, by default. While this linker should be 
  a drop-in replacement for system linkers, in rare cases it may be necessary to use
  your own linker instead. You can configure CUDA Quantum to use an external linker 
  setting the `NVQPP_LD_PATH` environment variable to point to it; for example
  `export NVQPP_LD_PATH=ld`.

The remaining sections in this document list additional runtime dependencies 
that are not included in the migrated assets and are needed to use some of the 
CUDA Quantum features and components, install the CUDA Quantum 
runtime dependencies listed in the remaining sections on the host system.

CUDA Runtime Libraries
+++++++++++++++++++++++++++++++

To use GPU-acceleration in CUDA Quantum you will need to install the necessary 
CUDA runtime libraries. While not necessary, we recommend installing 
the complete CUDA toolkit like you did for the CUDA Quantum build.
If you prefer to only install the minimal set of runtime libraries, the following 
commands, for example, install the necessary packages for the AlmaLinux 8 environment:

.. literalinclude:: ../../scripts/configure_build.sh
    :language: bash
    :start-after: [>CUDARTInstall]
    :end-before: [<CUDARTInstall]

MPI
+++++++++++++++++++++++++++++++

To work with all CUDA Quantum backends, a CUDA-aware MPI installation is required. 
If you do not have an existing CUDA-aware MPI installation, you can build one from 
source. The following commands build a sufficient CUDA-aware OpenMPI installation.
To make best use of MPI, we recommend a more fully featured installation including
additional configurations that fit your host system.
The commands below assume you have the necessary prerequisites for the OpenMPI build
installed on the build system. Within the tested AlmaLinux 8 environment, for example,
the packages `autoconf`, `libtool`, `flex`, and `make` need to be installed.

.. literalinclude:: ../../scripts/configure_build.sh
    :language: bash
    :start-after: [>OpenMPIBuild]
    :end-before: [<OpenMPIBuild]

Different MPI implementations are supported via a plugin infrastructure in CUDA Quantum.
Once you have a CUDA-aware MPI installation on your host system, you can 
configure CUDA Quantum to use it by activating the necessary plugin.
Plugins for OpenMPI and MPICH are included in CUDA Quantum and can be activated by
setting the environment variable `MPI_PATH` to the MPI installation folder 
and then running the command

.. code-block:: bash

    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"

If you use a different MPI implementation than OpenMPI or MPICH, you will need to 
implement the necessary plugin interface yourself prior to activating the plugin 
with the command above.

.. TODO:
  For more information about building and activating a custom MPI plugin, see ...
