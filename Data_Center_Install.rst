Installation from Source
*******************************************

In most cases, you should not need to build CUDA Quantum from source. For the
best experience, we recommend using a container runtime to avoid conflicts with
other software tools installed on the system. Before proceeding with
installation from source, we recommend you take a quick look at `Singularity
<https://docs.sylabs.io/guides/2.6/user-guide/faq.html#what-is-so-special-about-singularity>`__
or `Docker rootless mode <https://docs.docker.com/engine/security/rootless/>`__,
which address common issue or concerns that are often the motivation for
avoiding the use of containers. Our installation guide also contains
instructions for how to :ref:`connect an IDE <local-development-with-vscode>` to
a running container.

If you still want to build and install CUDA Quantum from source, you will need
to ensure that all dependencies installed in the build and host system are
compatible with your CUDA Quantum installation. The rest of this guide outlines
specific compatibility requirements during the build and after installation, and
walks through the installation steps.

Also note that CUDA Quantum contains some components that are only included as
pre-built binaries and not part of our open source repository. We are working on
either open-sourcing these components or making pre-built binaries available for
them in the future. Even without these components, almost all features of CUDA
Quantum will be enabled in a source build, though some pieces may be less
performant than in our pre-built images. At this time, the :ref:`multi-GPU state
vector simulator <nvidia-mgpu-backend>` backend will not be included if you
build CUDA Quantum from source.

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
- Python version 3.8 or newer, as well as `NumPy
  <https://numpy.org/install/>`__. Older Python 3 versions may work but have not
  been tested.
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

- Python header files: The header files need to match the version of your Python
  installation, and can usually be installed via package manager for your
  distribution.
- Python packages: pytest, lit, fastapi, uvicorn, pydantic, requests, and
  llvmlite. These packages can be downloaded from `PyPI <https://pypi.org/>`__
  using `PIP <https://pip.pypa.io/en/stable/installation/>`__. They are needed
  only for building and running CUDA Quantum tests.
- `Perl <https://www.perl.org/get.html>`__ and its `core modules
  <https://www.perl.com/article/what-is-the-perl-core-/>`__. Usually, this can
  be installed via package manager for your distribution.
- `Bash <https://www.gnu.org/software/bash/>`__: The CUDA Quantum build scripts
  and the commands listed in the rest of this document assume you are using
  `bash` as the Shell for your build.
- Common tools: wget, git, unzip. The commands in the rest of this guide assume
  that these tools are present on the build system, but they can be replaced by
  other alternatives (such as, for example, manually going to a web page and
  downloading a file/folder).

The above prerequisites are no longer needed once CUDA Quantum is built and 
do not need to be present on the host system.

In addition to installing the needed build dependencies listed above, make sure
to set the following environment variables prior to proceeding:

.. code-block:: bash

    export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
    export CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
    export CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
    export LLVM_INSTALL_PREFIX=/usr/local/llvm
    export OPENSSL_INSTALL_PREFIX=/usr/local/openssl
    export BLAS_PATH=/usr/local/blas
    export MPI_PATH=/usr/local/openmpi
    export MPI_HOME="${MPI_PATH}"

These environment variables *must* be set during the build. Their value can be
chosen freely, but for now the path during the build needs to match the path
where these libraries will be installed on the host system. We are working on
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

Download a suitable [CUDA version](https://developer.nvidia.com/cuda-toolkit-archive)
following the installation guide for your platform in the online documentation
linked on that page.

Within the tested AlmaLinux 8 environment, for example, the following commands
install CUDA 11.8:

.. code-block:: bash

    CUDA_VERSION=11.8
    # Go to https://developer.download.nvidia.com/compute/cuda/repos/ and
    # set the variables below to the distribution and subfolder for your platform.
    DISTRIBUTION=rhel8 CUDA_ARCH_FOLDER=x86_64
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/${DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${DISTRIBUTION}.repo
    dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-toolkit-$(echo ${CUDA_VERSION} | tr . -)

cuQuantum
+++++++++++++++++++++++++++++++

Each version of CUDA Quantum is compatible only with a specific cuQuantum version.
At present this is version 23.10. Newer versions of cuQuantum (if they exist)
might be compatible but have not been tested.

Make sure the environment variable `CUDA_ARCH_FOLDER` is set to either `x86_64`
or `sbsa` (for ARM64) depending on your processor architecture, and `CUDA_VERSION`
is set to the installed CUDA version.
Install cuQuantum version 23.10 using the following commands:

.. code-block:: bash

    CUQUANTUM_VERSION=23.10.0.6
    cuquantum_archive=cuquantum-linux-${CUDA_ARCH_FOLDER}-${CUQUANTUM_VERSION}_cuda$(echo ${CUDA_VERSION} | cut -d . -f1)-archive.tar.xz
    wget "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-${CUDA_ARCH_FOLDER}/${cuquantum_archive}"
    mkdir -p "$CUQUANTUM_INSTALL_PREFIX" && tar xf "${cuquantum_archive}" --strip-components 1 -C "$CUQUANTUM_INSTALL_PREFIX" && rm -rf "${cuquantum_archive}"

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

.. code-block:: bash

    CUTENSOR_VERSION=1.7.0.1
    cutensor_archive=libcutensor-linux-${CUDA_ARCH_FOLDER}-${CUTENSOR_VERSION}-archive.tar.xz
    wget "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-${CUDA_ARCH_FOLDER}/${cutensor_archive}"
    mkdir -p "$CUTENSOR_INSTALL_PREFIX" && tar xf "${cutensor_archive}" --strip-components 1 -C "$CUTENSOR_INSTALL_PREFIX"
    mv "$CUTENSOR_INSTALL_PREFIX"/lib/$(echo ${CUDA_VERSION} | cut -d . -f1)/* $CUTENSOR_INSTALL_PREFIX/lib/
    ls -d $CUTENSOR_INSTALL_PREFIX/lib/*/ | xargs rm -rf && rm -rf "${cutensor_archive}"

Toolchain
+++++++++++++++++++++++++++++++

The compiler toolchain used for the build needs to support C++20 and must be a supported 
[CUDA host compiler](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-host-compilers)
for the installed CUDA version.
The following instructions have been tested with
[GCC-11](https://gcc.gnu.org/index.html) as your toolchain for building CUDA Quantum.
If you use a different compiler, we recommend using an OpenMP-enabled compiler. At this
time, we actively test building with GCC 11 and 12, as well as with Clang 16. 
Other toolchains may be supported but have not been tested.

Within the tested AlmaLinux 8 environment, for example, the following commands
install GCC 11:

.. code-block:: bash

    GCC_VERSION=11
    dnf install -y --nobest --setopt=install_weak_deps=False \
        gcc-toolset-${GCC_VERSION}

Independent on which compiler toolchain you installed, set the following
environment variables to point to the appropriate values to point to the 
respective compilers on your build system:

.. code-block:: bash

    export CXX=/opt/rh/gcc-toolset-11/root/usr/bin/g++
    export CC=/opt/rh/gcc-toolset-11/root/usr/bin/gcc
    export FC=/opt/rh/gcc-toolset-11/root/usr/bin/gfortran
    export CUDACXX=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc

- The variables CC and CXX *must* be set for the CUDA Quantum build.
- A Fortran compiler is needed (only) to build the OpenSSL dependency; 
  if you have an existing OpenSSL installation that you set the
  `OPENSSL_INSTALL_PREFIX` variable to, you can omit setting the FC 
  environment variable.
- To use GPU-acceleration in CUDA Quantum, make sure to set CUDACXX to 
  your CUDA compiler. If the CUDA compiler is not found when building CUDA Quantum, some components and backends will be omitted automatically during
  the build.

MPI
+++++++++++++++++++++++++++++++

To work with all CUDA Quantum backends, a CUDA-aware MPI installation
is required. Different MPI implementations are supported via a plugin infrastructure
in CUDA Quantum. CUDA Quantum includes the necessary plugin for OpenMPI and MPICH.
Any other MPI implementation requires implementing the plugin yourself, and activating
it in a final step after installing CUDA Quantum on the host system.

The following commands build a sufficient CUDA-aware OpenMPI installation from source.
To make best use of MPI, we recommend a more fully featured installation including
additional configurations that fit your host system.
The commands below assume you have the necessary prerequisites for the OpenMPI build
installed on the build system. Within the tested AlmaLinux 8 environment, for example,
the packages `autoconf`, `libtool`, `flex`, and `make` need to be installed.

.. code-block:: bash

    OPENMPI_VERSION=4.1.4
    mkdir ~/.openmpi-project && cd ~/.openmpi-project
    git init && git remote add origin https://github.com/open-mpi/ompi
    git fetch origin --depth=1 v${OPENMPI_VERSION} && git reset --hard FETCH_HEAD

    # Make sure CUDA_PATH is set to the correct CUDA installation root.
    CUDA_PATH=/usr/local/cuda-${CUDA_VERSION}
    ./autogen.pl
    PATH="$(dirname $CC):$PATH" LDFLAGS=-Wl,--as-needed \
    ./configure \
        --prefix="${MPI_PATH}" \
        --disable-getpwuid --disable-static \
        --disable-debug --disable-mem-debug --disable-mem-profile --disable-memchecker \
        --without-verbs \
        --with-cuda="${CUDA_PATH}"
    make -j$(nproc) && make -j$(nproc) install
    cd - && rm -rf ~/.openmpi-project

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

.. code-block:: bash

    FORCE_COMPILE_GPU_COMPONENTS=true CUDAQ_WERROR=false \
    bash scripts/build_cudaq.sh -u -v
    # && $CUQUANTUM_INSTALL_PREFIX/distributed_interfaces/ && bash activate_mpi.sh

The CUDA Quantum build will compile or omit optional components automatically depending
on whether the necessary pre-requisites are found in the build environment.
Please check the build log to confirm that all desired components have been built. If
you see a message that a component has been skipped, make sure you have all necessary 
prerequisites, have followed the instructions for installing the build dependencies, 
and the necessary environment variables as described in this document are set.

Installation on the Host
------------------------------------

To be populated...