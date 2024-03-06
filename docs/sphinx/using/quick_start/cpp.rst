C++
----

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

.. literalinclude:: ../../../../docker/test/installer/linux.Dockerfile
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
installation methods and resources, see our :ref:`Installation Guide <install-cuda-quantum>`.

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

For further examples of simulation with NVIDIA cuQuantum, see :ref:`Examples <cuquantum-cpp-example-page>`.

For further information on available targets, see :ref:`Backends <backends-landing-page>`.

Now that you have successfully run your first program, you are ready to move on to our :ref:`Basics Section <cudaq-basics-landing-page>`.