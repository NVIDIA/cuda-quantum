Installation from Source
=========================

In most cases, you should not need to build CUDA-Q from source. For the
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
for using CUDA-Q with C++, and Python wheels for using CUDA-Q with Python. 
These binaries and wheels are built following the instructions 
in this guide and should work for you as long as your system meets the compatibility 
requirements listed under :ref:`Prerequisites <compatibility-prebuilt-binaries>`. 
To install the pre-built binaries, please follow the instructions 
:ref:`here <install-prebuilt-binaries>`. To install the Python wheels, please 
follow the instructions :ref:`here <install-python-wheels>`.

If your system is not listed as supported by our official packages, e.g. because you would 
like to use CUDA-Q on an operating system that uses an older C standard library, 
please follow this guide carefully without skipping any steps to build and install
CUDA-Q from source. The rest of this guide details system requirements 
during the build and after installation, and walks through the installation steps.

.. note::

  CUDA-Q contains some components that are only included as
  pre-built binaries and not part of our open source repository. We are working on
  either open-sourcing these components or making them available as separate downloads
  in the future. Even without these components, almost all features of CUDA-Q
  will be enabled in a source build, though some pieces may be less performant. 
  At this time, the :ref:`multi-GPU state vector simulator <nvidia-mgpu-backend>` 
  backend will not be included if you build CUDA-Q from source.


.. _compatibility-prebuilt-binaries:

Prerequisites
++++++++++++++++++++++++++++++++++++

The following pre-requisites need to be satisfied both on the build system and
on the host system, that is the system where the built CUDA-Q binaries
will be installed and used.

- Linux operating system. The instructions in this guide have been validated
  with the `AlmaLinux 8 image <https://hub.docker.com/u/almalinux>`__ that
  serves as the base image for the `manylinux_2_28 image
  <https://github.com/pypa/manylinux>`__, and should work for the operating
  systems CentOS 8, Debian 11 and 12, Fedora 38, OpenSUSE/SLED/SLES 15.5, RHEL 8
  and 9, Rocky 8 and 9, and Ubuntu 22.04. Other operating systems may work, but
  have not been tested.
- `Bash <https://www.gnu.org/software/bash/>`__ shell. The CUDA-Q 
  build, install and run scripts expect to use `/bin/bash`.
- `GNU C library <https://www.gnu.org/software/libc/>`__. 
  Make sure that the version on the host system is the same one
  or newer than the version on the build system. Our own builds
  use version 2.28.
- CPU with either x86-64 (x86-64-v3 architecture and newer) or ARM64
  (ARM v8-A architecture and newer). Other architectures may work but are not tested and may require
  adjustments to the build instructions.
- Needed **only on the host** system: NVIDIA GPU with Volta, Turing, Ampere, Ada, or
  Hopper architecture and `Compute Capability
  <https://developer.nvidia.com/cuda-gpus>`__ 7+. Make sure you have the latest
  `drivers <https://www.nvidia.com/download/index.aspx>`__ installed for your
  GPU, and double check that the driver version listed by the `nvidia-smi`
  command is 470.57.02 or newer. You do *not* need to have a GPU available on the
  build system; the CUDA compiler needed for the build can be installed and used
  without a GPU.

We strongly recommend using a virtual environment for the build that includes
*only* the tools and dependencies listed in this guide. If you have additional
software installed, you will need to make sure that the build is linking against
the correct libraries and versions.


Build Dependencies
++++++++++++++++++++++++++++++++++++

In addition to the prerequisites listed above, you will need to install the
following prerequisites in your build environment prior to proceeding with 
the build as described in the subsequent sections:

- Python version 3.8 or newer: If you intend to build CUDA-Q with Python
  support, make sure the Python version on the build system matches the version
  on the host system. If you intend to only build the C++ support for
  CUDA-Q, the Python interpreter is required only for some of the 
  LLVM build scripts and the Python version used for the build does not have 
  to match the version on the host system.
- Common tools: `wget`, `git`, `unzip`. The commands in the rest of this guide assume
  that these tools are present on the build system, but they can be replaced by
  other alternatives (such as, for example, manually going to a web page and
  downloading a file/folder).

The above prerequisites are no longer needed once CUDA-Q is built and 
do not need to be present on the host system.

.. note::

  The CUDA-Q build scripts and the commands listed in the rest of this 
  document assume you are using `bash` as the shell for your build.

In addition to installing the needed build dependencies listed above, make sure
to set the following environment variables prior to proceeding:

.. literalinclude:: ../../../../scripts/configure_build.sh
    :language: bash
    :dedent:
    :start-after: [>InstallLocations]
    :end-before: [<InstallLocations]

These environment variables *must* be set during the build. We strongly
recommend that their value is set to a path that does *not* already exist; 
this will ensure that these components are built/installed as needed when
building CUDA-Q.
The configured paths can be chosen freely, but the paths specified during the
build are also where the corresponding libraries will be installed on the 
host system. We are working on making this more flexible in the future.

.. note::

  Please do **not** set `LLVM_INSTALL_PREFIX` to an existing directory;
  To avoid compatibility issues, it is important to use the same compiler
  to build the LLVM/MLIR dependencies from source as is later used to 
  build CUDA-Q itself.

.. note::

  If you are setting the `CURL_INSTALL_PREFIX` variable to an existing 
  CURL installation (not recommended), please make sure the command 
  `curl --version` lists HTTP and HTTPS as supported protocols. If these 
  protocols are not listed, please instead set the `CURL_INSTALL_PREFIX` 
  variable to a path that does *not* exist. In that case, a suitable 
  library will be automatically built from source as part of
  building CUDA-Q.

If you deviate from the instructions below for installing one of the
dependencies and instead install it, for example, via package manager, you will
need to make sure that the installation path matches the path you set for the
corresponding environment variable(s).

CUDA
~~~~~~~~~~

Building CUDA-Q requires a full installation of the CUDA toolkit.
**You can install the CUDA toolkit and use the CUDA compiler without having a GPU.**
The instructions are tested using version 11.8, but any CUDA 11 or 12 version
should work, as long as the CUDA runtime version on the host system matches the 
CUDA version used for the build, and the installed driver on the host 
system supports that CUDA version. We recommend using the latest CUDA version
that is supported by the driver on the host system.

Download a suitable `CUDA version <https://developer.nvidia.com/cuda-toolkit-archive>`__
following the installation guide for your platform in the online documentation
linked on that page.

Within the tested AlmaLinux 8 environment, for example, the following commands
install CUDA 11.8:

.. literalinclude:: ../../../../scripts/configure_build.sh
    :language: bash
    :dedent:
    :start-after: [>CUDAInstall]
    :end-before: [<CUDAInstall]

Toolchain
~~~~~~~~~~

The compiler toolchain used for the build must be a supported 
`CUDA host compiler <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#supported-host-compilers>`__
for the installed CUDA version.
The following instructions have been tested with `GCC-11 <https://gcc.gnu.org/index.html>`__.
Other toolchains may be supported but have not been tested.

Within the tested AlmaLinux 8 environment, for example, the following commands
install GCC 11:

.. literalinclude:: ../../../../scripts/configure_build.sh
    :language: bash
    :dedent:
    :start-after: [>gccInstall]
    :end-before: [<gccInstall]

Independent on which compiler toolchain you installed, set the following
environment variables to point to the respective compilers on your build system:

.. literalinclude:: ../../../../scripts/configure_build.sh
    :language: bash
    :dedent:
    :start-after: [>ToolchainConfiguration]
    :end-before: [<ToolchainConfiguration]

- The variables `CC` and `CXX` *must* be set for the CUDA-Q build.
- To use GPU-acceleration in CUDA-Q, make sure to set `CUDACXX` to 
  your CUDA compiler, and `CUDAHOSTCXX` to the CUDA compatible host 
  compiler you are using. If the CUDA compiler is not found when building
  CUDA-Q, some components and backends will be omitted automatically 
  during the build.


Building CUDA-Q
++++++++++++++++++++++++++++++++++++

This installation guide has been written for a specific version/commit of CUDA-Q. 
Make sure to obtain the source code for that version. 
Clone the CUDA-Q `GitHub repository <https://github.com/NVIDIA/cuda-quantum>`__ and
checkout the appropriate branch, tag, or commit. 
Note that the build scripts assume that they are run from within a git repository, 
and merely downloading the source code as ZIP archive hence will not work.

Please follow the instructions in the respective subsection(s) to build the necessary 
components for using CUDA-Q from C++ and/or Python.
After the build, check that the GPU-accelerated components have been built by confirming
that the file `nvidia.config` exists in the `$CUDAQ_INSTALL_PREFIX/targets` folder.
We also recommend checking the build log printed to the console to confirm that all desired 
components have been built. 

.. note::

  The CUDA-Q build will compile or omit optional components automatically depending
  on whether the necessary pre-requisites are found in the build environment.
  If you see a message that a component has been skipped, and/or the CUDA compiler
  is not properly detected, make sure you followed the 
  instructions for installing the necessary prerequisites and build dependencies, 
  and have set the necessary environment variables as described in this document.

.. _cudaq-python-from-source:

Python Support
+++++++++++++++++++++++++++++++

The most convenient way to enable Python support within CUDA-Q is to build 
a `wheel <https://pythonwheels.com/>`__ that can then easily be installed
using `pip`. To ensure the wheel can be installed on the host system, make sure to
use the same Python version for the build as the one that is installed on the host system.
To build a CUDA-Q Python wheel, you will need to install the following additional 
Python-specific tools:

- Python development headers: The development headers for your Python version are installed
  in the way as you installed Python itself. If you installed Python via the package manager
  for your system, you may need to install an additional package to get the development headers.
  The package name is usually your python version followed by either a `-dev` or `-devel` suffix.
  If you are using a `Conda environment <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python>`__,
  the necessary headers should already be installed.
- Pip package manager: Make sure the `pip` module is enable for your Python version.
  We refer to the Python `documentation <https://pip.pypa.io/en/stable/installation/>`__ for
  more information about installing/enabling `pip`.
- Python modules: Install the additional modules `numpy`, `build`, `auditwheel`, and `patchelf` for your
  Python version, e.g. `python3 -m pip install numpy build auditwheel patchelf`.

From within the folder where you cloned the CUDA-Q repository, run the following
command to build the CUDA-Q Python wheel:

.. literalinclude:: ../../../../docker/build/assets.Dockerfile
    :language: bash
    :dedent:
    :start-after: [>CUDAQuantumPythonBuild]
    :end-before: [<CUDAQuantumPythonBuild]

.. note::

  A version identifier will be automatically assigned to the wheel based on the commit
  history. You can manually override this detection to give a more descriptive identifier
  by setting the environment variable `SETUPTOOLS_SCM_PRETEND_VERSION` to the desired
  value before building the wheel.

After the initial build, `auditwheel <https://github.com/pypa/auditwheel>`__ is used to 
include dependencies in the wheel, if necessary, and correctly label the wheel.
We recommend not including the CUDA runtime libraries and instead install them separately
on the host system following the instructions in the next section. The following 
command builds the final wheel, not including CUDA dependencies: 

.. literalinclude:: ../../../../docker/build/assets.Dockerfile
    :language: bash
    :dedent:
    :start-after: [>CUDAQuantumWheel]
    :end-before: [<CUDAQuantumWheel]

The command above will create a new wheel in the `wheelhouse` folder. This wheel can be
installed on any `compatible platform <https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/>`__.

.. note::
  You can confirm that the wheel is indeed compatible with your host platform by 
  checking that the wheel tag (i.e. the file name ending of the `.whl` file) is listed under
  "Compatible Tags" when running the command `python3 -m pip debug --verbose` on the host.

.. _cudaq-cpp-from-source:

C++ Support
+++++++++++++++++++++++++++++++

From within the folder where you cloned the CUDA-Q repository, run the following
command to build CUDA-Q:

.. literalinclude:: ../../../../docker/build/assets.Dockerfile
    :language: bash
    :dedent:
    :start-after: [>CUDAQuantumCppBuild]
    :end-before: [<CUDAQuantumCppBuild]

Note that `lld` is primarily needed when the build or host system does not already
have an existing default linker on its path; CUDA-Q supports the same linkers as
`clang` does.

To easily migrate the built binaries to the host system, we recommend creating a
`self-extracting archive <https://makeself.io/>`__. To do so, download the 
`makeself script(s) <https://github.com/megastep/makeself>`__ and move the necessary 
files to install into a separate folder using the command

.. literalinclude:: ../../../../docker/release/installer.Dockerfile
    :language: bash
    :dedent:
    :start-after: [>CUDAQuantumAssets]
    :end-before: [<CUDAQuantumAssets]

You can then create a self-extracting archive with the command

.. code-block:: bash

    ./makeself.sh --gzip --sha256 --license cuda_quantum_assets/cudaq/LICENSE \
        cuda_quantum_assets install_cuda_quantum.$(uname -m) \
        "CUDA-Q toolkit for heterogeneous quantum-classical workflows" \
        bash cudaq/migrate_assets.sh -t /opt/nvidia/cudaq

Installation on the Host
++++++++++++++++++++++++++++++++++++

Make sure your host system satisfies the `Prerequisites`_ listed above.

- To use CUDA-Q with Python, you should have a working
  Python installation on the host system, including the `pip` package manager.
- To use CUDA-Q with C++, you should make sure that you have the necessary development
  headers of the C standard library installed. You can check this by searching for
  `features.h`, commonly found in `/usr/include/`. You can install the necessary headers
  via package manager (usually the package name is called something like `glibc-devel`
  or `libc6-devel`). These headers are also included with any installation of GCC.

To use CUDA-Q with Python, you should have a working
Python installation on the host system, including the `pip` package manager.

If you followed the instructions for building the 
:ref:`CUDA-Q Python wheel <cudaq-python-from-source>`,
copy the built `.whl` file to the host system, and install it using `pip`; e.g.

.. code-block:: bash

    pip install cuda_quantum*.whl

To install the necessary CUDA and MPI dependencies for some of the components, 
you can either follow the instructions on `PyPI.org <https://pypi.org/project/cuda-quantum/>`__, 
replacing `pip install cuda-quantum` with the command above, or you can follow the
instructions in the remaining sections of this document to customize and better
optimize them for your host system.

If you followed the instructions for building the 
:ref:`CUDA-Q C++ tools <cudaq-cpp-from-source>`,
copy the `install_cuda_quantum` file that you created to the host system, 
and install it by running the commands

.. code-block:: bash

    sudo bash install_cuda_quantum.$(uname -m) --accept
    . /opt/nvidia/cudaq/set_env.sh

This will extract the built assets and move them to the correct locations.
The `set_env.sh` script in `/opt/nvidia/cudaq` defines the necessary environment
variables to use CUDA-Q. To avoid having to set them manually every time a 
new shell is opened, we highly recommend adding the following lines to
the `/etc/profile` file:

.. code-block:: bash

    if [ -f /opt/nvidia/cudaq/set_env.sh ];
      . /opt/nvidia/cudaq/set_env.sh
    fi

.. note::
  CUDA-Q as built following the instructions above includes and uses the LLVM
  C++ standard library. This will not interfere with any other C++ standard library
  you may have on your system. Pre-built external libraries, you may want to use with 
  CUDA-Q, such as specific optimizers for example, have a C API to ensure compatibility
  across different versions of the C++ standard library and will work with CUDA-Q without 
  issues. The same is true for all distributed CUDA libraries. To build you own CUDA 
  libraries that can be used with CUDA-Q, please take a look at :doc:`../integration/cuda_gpu`.

The remaining sections in this document list additional runtime dependencies 
that are not included in the migrated assets and are needed to use some of the 
CUDA-Q features and components.

CUDA Runtime Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use GPU-acceleration in CUDA-Q you will need to install the necessary 
CUDA runtime libraries. Their version (at least the version major) needs to match the version
used for the build. While not necessary, we recommend installing 
the complete CUDA toolkit like you did for the CUDA-Q build.
If you prefer to only install the minimal set of runtime libraries, the following 
commands, for example, install the necessary packages for the AlmaLinux 8 environment:

.. literalinclude:: ../../../../scripts/configure_build.sh
    :language: bash
    :dedent:
    :start-after: [>CUDARTInstall]
    :end-before: [<CUDARTInstall]

MPI
~~~~~~

To work with all CUDA-Q backends, a CUDA-aware MPI installation is required. 
If you do not have an existing CUDA-aware MPI installation, you can build one from 
source. To do so, in addition to the CUDA runtime libraries listed above 
you will need to install the CUDA runtime development package 
(`cuda-cudart-devel-${version_suffix}` or `cuda-cudart-dev-${version_suffix}`, 
depending on your distribution).

The following commands build a sufficient CUDA-aware OpenMPI installation.
To make best use of MPI, we recommend a more fully featured installation including
additional configurations that fit your host system.
The commands below assume you have the necessary prerequisites for the OpenMPI build
installed on the build system. Within the tested AlmaLinux 8 environment, for example,
the packages `autoconf`, `libtool`, `flex`, and `make` need to be installed.

.. literalinclude:: ../../../../scripts/configure_build.sh
    :language: bash
    :dedent:
    :start-after: [>OpenMPIBuild]
    :end-before: [<OpenMPIBuild]

Confirm that you have a suitable MPI implementation installed. For OpenMPI and MPICH,
for example, this can be done by compiling and running the following program:

.. literalinclude:: ../../../../docker/test/installer/mpi_cuda_check.cpp
    :language: cpp

.. note::

  If you are encountering an error similar to "The value of the MCA parameter `plm_rsh_agent` 
  was set to a path that could not be found", please make sure you have an SSH Client installed
  or update the MCA parameter to another suitable agent.
  MPI uses `SSH <https://en.wikipedia.org/wiki/Secure_Shell>`__ or
  `RSH <https://en.wikipedia.org/wiki/Remote_Shell>`__ to communicate with each node
  unless another resource manager, such as
  `SLURM <https://slurm.schedmd.com/overview.html>`__, is used. 

Different MPI implementations are supported via a plugin infrastructure in CUDA-Q.
Once you have a CUDA-aware MPI installation on your host system, you can 
configure CUDA-Q to use it by activating the necessary plugin.
Plugins for OpenMPI and MPICH are included in CUDA-Q and can be activated by
setting the environment variable `MPI_PATH` to the MPI installation folder 
and then running the command

.. code-block:: bash

    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"

.. note::

  To activate the MPI plugin for the Python support, replace `${CUDA_QUANTUM_PATH}`
  with the path that is listed under "Location" when you run the command
  `pip show cuda-quantum`.

If you use a different MPI implementation than OpenMPI or MPICH, you will need to 
implement the necessary plugin interface yourself prior to activating the plugin 
with the command above.

.. TODO:
  For more information about building and activating a custom MPI plugin, see ...
