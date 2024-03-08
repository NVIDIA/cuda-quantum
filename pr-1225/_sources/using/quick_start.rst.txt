Quick Start
*******************************************

**NVIDIA CUDA Quantum**

CUDA Quantum streamlines hybrid application development and promotes productivity and scalability
in quantum algorithm research. It offers a unified programming model designed for a hybrid
setting |---| that is, CPUs, GPUs, and QPUs working together. CUDA Quantum contains support for 
programming in Python and in C++.

.. 
    Fix Me: Learn more about CUDA Quantum’s key benefits here [Link to CUDA Quantum Marketing page].
    Issue 1351.

This Quick Start page guides you through getting set up with CUDA Quantum and running your first program.
If you have already installed and configured CUDA Quantum, we encourage you to move directly to our
:doc:`Basics Section <basics/basics>`.


.. tab:: Python

   If you want to develop CUDA Quantum applications using Python, install the
   latest stable release of the CUDA Quantum Python API:  

   .. code-block:: bash

      pip install cuda-quantum

   For further information, see the `CUDA Quantum project <https://pypi.org/project/cuda-quantum/>`_ on PyPI.

   We also offer a fully featured CUDA Quantum installation, including all Python and C++ tools, via our
   `Docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-quantum>`_. For further
   installation methods and resources, see our :ref:`Installation Guide <install-cuda-quantum>`.

   You should now be able to import CUDA Quantum and start building quantum programs in Python!

   To test your installation, create a file titled `first_program.py`, containing the following code:

   .. literalinclude:: /snippets/python/quick_start.py
      :language: python
      :start-after: [Begin Documentation]

   You may now execute this file as you do any other Python program. For example, from the command line:

   .. code-block:: bash

      python3 first_program.py

   The Hadamard gate places the qubit in a superposition state, giving a roughly 50/50 mixture
   of measurements in the `|0>` and `|1>` states.

   If you have a local GPU, the following dependencies must be installed before using any GPU accelerated simulation target.

   .. code-block:: bash
      
      conda create -y -n cuda-quantum python==3.10 pip
      conda install -y -n cuda-quantum -c "nvidia/label/cuda-11.8.0" cuda
      conda install -y -n cuda-quantum -c conda-forge mpi4py openmpi cxx-compiler cuquantum
      conda env config vars set -n cuda-quantum LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/envs/cuda-quantum/lib"
      conda env config vars set -n cuda-quantum MPI_PATH=$CONDA_PREFIX/envs/cuda-quantum
      conda run -n cuda-quantum pip install cuda-quantum
      conda activate cuda-quantum
      source $CONDA_PREFIX/lib/python3.10/site-packages/distributed_interfaces/activate_custom_mpi.sh

   When a GPU is detected, the simulation target defaults to `nvidia`. Otherwise, the target will default to the
   `qpp-cpu` simulator. To confirm the detection of your GPU, you may once again run the example

   .. code-block:: bash

      python3 first_program.py

   and confirm that `Simulation Target = nvidia` is printed to the console.

   For further information on available targets, see :doc:`Backends <backends/backends>`.

   Now that you have successfully run your first program, you are ready to move on to our :doc:`Basics Section <basics/basics>`.


.. tab:: C++

   ..
     Fix Me: Condense this section better.

   To install CUDA Quantum for C++, we provide a set of pre-built binaries. 
   Support for using CUDA Quantum with Python can be installed side-by-side with
   the pre-built binaries for C++ by following the instructions on 
   `PyPI.org <https://pypi.org/project/cuda-quantum>`__. The pre-built binaries
   work across a range of Linux operating systems listed  under :ref:`dependencies-and-compatibility`. 

   Before installing our pre-built binaries, please make sure that your 
   operating system is using the `GNU C library <https://www.gnu.org/software/libc/>`__ 
   version 2.28 or newer. You can confirm this by checking the output of the command 
   `ldd --version`. If this command does not exist, or shows an older version than 2.28, 
   please double check that your operating system is listed as 
   :ref:`supported <dependencies-and-compatibility>`. If you use an operating system
   with an older GNU C library version, you will need to build the installer from 
   source following the instructions in :ref:`data-center-install`.

   You can download the `install_cuda_quantum` file for your processor architecture from
   the assets of the respective `GitHub release <https://github.com/NVIDIA/cuda-quantum/releases>`__. 
   The installer is a `self-extracting archive <https://makeself.io/>`__ that contains the 
   pre-built binaries as well as a script to move them to the correct locations. You will need
   `bash`, `tar`, and `gzip` (usually already installed on most Linux distributions) to run 
   the installer.
   The installation location of CUDA Quantum is not currently configurable and using the installer
   hence requires admin privileges on the system. We may revise that in the future; please see and
   upvote the corresponding `GitHub issue <https://github.com/NVIDIA/cuda-quantum/issues/1075>`__.

   To install CUDA Quantum, execute the command

   .. literalinclude:: ../../../docker/test/installer/linux.Dockerfile
      :language: bash
      :dedent:
      :start-after: [>CUDAQuantumInstall]
      :end-before: [<CUDAQuantumInstall]

   .. note:: 

      To use GPU-accelerated backends, you will need to install the necessary CUDA runtime libraries. 
      For more information see the corresponding section on :ref:`Additional CUDA Tools <cuda-dependencies-prebuilt-binaries>`.

   The installation ensures that the necessary environment variables for
   using the CUDA Quantum toolchain are set upon login for all POSIX shells.
   Confirm that the `nvq++` command is found. If it is not, please make sure 
   to set the environment variables defined by the `set_env.sh` script in the 
   CUDA Quantum installation folder (usually `/opt/nvidia/cudaq`).

   If an MPI installation is available in the directory defined by `MPI_PATH`, 
   the installer automatically enables MPI support in CUDA Quantum.
   If you do not have MPI installed on your system, you can simply
   leave that path empty, and CUDA Quantum will be installed without MPI support.
   If you install MPI at a later point in time, you can activate the MPI support in CUDA 
   Quantum by setting the `MPI_PATH` variable to its installation location and 
   executing the commands

   .. code-block:: console

      MPI_PATH=/usr/local/openmpi # update this path as needed
      bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"

   .. _local-development-with-vscode:

   To develop C++ code, you most likely also want to install the
   `C++ standard library <https://en.cppreference.com/w/cpp/standard_library>`__.
   CUDA Quantum supports the GNU C++ standard library (`libstdc++`), 
   version 11 or newer. Other libraries may work but can cause issues in certain cases.
   The C++ standard library, including development headers, is almost certainly 
   available via the package manager for your system. To ensure the libraries and headers
   are discoverable, the easiest option is usually to install the complete GCC toolchain.
   Note that for certain distributions, you may need to manually enable that version 
   after installation by running a script called `enable`. You can search for such a 
   script with the command `find / -path '*gcc*' -name enable`.

   We also offer a fully featured CUDA Quantum installation, including all C++ and Python tools, via our
   `Docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-quantum>`_. For further
   installation methods and resources, see our :doc:`Installation Guide <install/install>`.

   You should now have the CUDA Quantum compiler toolchain available via the command line and are ready to run your first quantum program!

   To test your installation, create a file titled `first_program.cpp`, containing the following code:

   .. literalinclude:: /snippets/cpp/quick_start.cpp
      :language: cpp
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

   You may now compile and execute this file from the command line using the `nvq++` toolchain:

   .. code-block:: bash

      nvq++ first_program.cpp && ./a.out

   The Hadamard gate places the qubit in a superposition state, giving a roughly 50/50 mixture
   of measurements in the `|0>` and `|1>` states.

   If you have a local GPU, you may check that it has been properly configured by running your program with
   the target set to `nvidia`:

   .. code-block:: bash

      nvq++ --target=nvidia first_program.cpp && ./a.out

   If no GPU is detected, an error will be returned to the terminal: `Could not find a valid CUDA or libnvqir-custatevec-fp32
   installation. Make sure you have a compatible GPU and the necessary driver installed`. 

   For further examples of simulation with NVIDIA cuQuantum, see :doc:`Examples <examples/cuquantum>`.

   For further information on available targets, see :doc:`Backends <backends/backends>`.

   Now that you have successfully run your first program, you are ready to move on to our :doc:`Basics Section <basics/basics>`.

.. |---|   unicode:: U+2014 .. EM DASH
   :trim: