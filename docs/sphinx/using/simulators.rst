CUDA Quantum Simulation Backends
*********************************

The simulation backends that are currently available in CUDA Quantum are as follows.

State Vector Simulators
==================================

.. _cuQuantum single-GPU:

Single-GPU 
++++++++++++++++++++++++++++++++++

The :code:`nvidia` target provides a state vector simulator accelerated with 
the :code:`cuStateVec` library. 

To execute a program on the :code:`nvidia` target, use the following commands:

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target nvidia

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('nvidia')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target nvidia program.cpp [...] -o program.x
        ./program.x

By default, this will leverage :code:`FP32` floating point types for the simulation. To 
switch to :code:`FP64`, specify the :code:`nvidia-fp64` target instead. 

.. note:: 

  This backend requires an NVIDIA GPU and CUDA runtime libraries. If you do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

.. _nvidia-mgpu-backend:

Multi-node multi-GPU
++++++++++++++++++++++++++++++++++

The :code:`nvidia-mgpu` target provides a state vector simulator accelerated with 
the :code:`cuStateVec` library but with support for Multi-Node, Multi-GPU distribution of the 
state vector. 

The multi-node multi-GPU simulator expects to run within an MPI context.
To execute a program on the :code:`nvidia-mgpu` target, use the following commands (adjust the value of the :code:`-np` flag as needed to reflect available GPU resources on your system):

.. tab:: Python

    .. code:: bash 

        mpiexec -np 2 python3 program.py [...] --target nvidia-mgpu

    .. note::

      If you installed CUDA Quantum via :code:`pip`, you will need to install the necessary MPI dependencies separately;
      please follow the instructions for installing dependencies in the `Project Description <https://pypi.org/project/cuda-quantum/#description>`__.

    In addition to using MPI in the simulator, you can use it in your application code by installing `mpi4py <https://mpi4py.readthedocs.io/>`__, and 
    invoking the program with the command

    .. code:: bash 

        mpiexec -np 2 python3 -m mpi4py program.py [...] --target nvidia-mgpu

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('nvidia-mgpu')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target nvidia-mgpu program.cpp [...] -o program.x
        mpiexec -np 2 ./program.x

.. note:: 

  This backend requires an NVIDIA GPU, CUDA runtime libraries, as well as an MPI installation. If you do not have these dependencies installed, you may encounter either an error stating `invalid simulator requested` (missing CUDA libraries), or an error along the lines of `failed to launch kernel` (missing MPI installation). See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

.. _OpenMP CPU-only:

OpenMP CPU-only
++++++++++++++++++++++++++++++++++

This target provides a state vector simulator based on the CPU-only, OpenMP threaded `Q++ <https://github.com/softwareqinc/qpp>`_ library.
This is the default target when running on CPU-only systems.

To execute a program on the :code:`qpp-cpu` target even if a GPU-accelerated backend is available, 
use the following commands:

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target qpp-cpu

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('qpp-cpu')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target qpp-cpu program.cpp [...] -o program.x
        ./program.x


Tensor Network Simulators
==================================

CUDA Quantum provides a couple of tensor-network simulator targets accelerated with 
the :code:`cuTensorNet` library. 
These backends are available for use from both C++ and Python.

Tensor network-based simulators are suitable for large-scale simulation of certain classes of quantum circuits involving many qubits beyond the memory limit of state vector based simulators. For example, computing the expectation value of a Hamiltonian via :code:`cudaq::observe` can be performed efficiently, thanks to :code:`cuTensorNet` contraction optimization capability. On the other hand, conditional circuits, i.e., those with mid-circuit measurements or reset, despite being supported by both backends, may result in poor performance. 

Multi-node multi-GPU
+++++++++++++++++++++++++++++++++++

The :code:`tensornet` backend represents quantum states and circuits as tensor networks in an exact form (no approximation). 
Measurement samples and expectation values are computed via tensor network contractions. 
This backend supports multi-node, multi-GPU distribution of tensor operations required to evaluate and simulate the circuit.

To execute a program on the :code:`tensornet` target using a *single GPU*, use the following commands:

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target tensornet

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('tensornet')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target tensornet program.cpp [...] -o program.x
        ./program.x

If you have *multiple GPUs* available on your system, you can use MPI to automatically distribute parallelization across the visible GPUs. 

.. note::

  If you installed the CUDA Quantum Python wheels, distribution across multiple GPUs is currently not supported for this backend.
  We will add support for it in future releases. For more information, see this `GitHub issue <https://github.com/NVIDIA/cuda-quantum/issues/920>`__.

Use the following commands to enable distribution across multiple GPUs (adjust the value of the :code:`-np` flag as needed to reflect available GPU resources on your system):

.. tab:: Python

    .. code:: bash 

        mpiexec -np 2 python3 program.py [...] --target tensornet

    In addition to using MPI in the simulator, you can use it in your application code by installing `mpi4py <https://mpi4py.readthedocs.io/>`__, and 
    invoking the program with the command

    .. code:: bash 

        mpiexec -np 2 python3 -m mpi4py program.py [...] --target tensornet

.. tab:: C++

    .. code:: bash 

        nvq++ --target tensornet program.cpp [...] -o program.x
        mpiexec -np 2 ./program.x

.. note::

  If the `CUTENSORNET_COMM_LIB` environment variable is not set, MPI parallelization on the :code:`tensornet` backend may fail.
  If you are using a CUDA Quantum container, this variable is pre-configured and no additional setup is needed. If you are customizing your installation or have built CUDA Quantum from source, please follow the instructions for `activating the distributed interface <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/getting_started.html#install-cutensornet-from-nvidia-devzone>`__ for the `cuTensorNet` library. This requires 
  :ref:`installing CUDA development dependencies <additional-cuda-tools>`, and setting the `CUTENSORNET_COMM_LIB`
  environment variable to the newly built `libcutensornet_distributed_interface_mpi.so` library.

Specific aspects of the simulation can be configured by setting the following of environment variables:

* **`CUDA_VISIBLE_DEVICES=X`**: Makes the process only see GPU X on multi-GPU nodes. Each MPI process must only see its own dedicated GPU. For example, if you run 8 MPI processes on a DGX system with 8 GPUs, each MPI process should be assigned its own dedicated GPU via `CUDA_VISIBLE_DEVICES` when invoking `mpiexec` (or `mpirun`) commands. 
* **`OMP_PLACES=cores`**: Set this environment variable to improve CPU parallelization.
* **`OMP_NUM_THREADS=X`**: To enable CPU parallelization, set X to `NUMBER_OF_CORES_PER_NODE/NUMBER_OF_GPUS_PER_NODE`.

.. note:: 

  This backend requires an NVIDIA GPU and CUDA runtime libraries. 
  If you do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. 
  See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

.. note::

  Setting random seed, via :code:`cudaq::set_random_seed`, is not supported for this backend due to a limitation of the :code:`cuTensorNet` library. This will be fixed in future release once this feature becomes available.


Matrix product state 
+++++++++++++++++++++++++++++++++++

The :code:`tensornet-mps` backend is based on the matrix product state (MPS) representation of the state vector/wave function, exploiting the sparsity in the tensor network via tensor decomposition techniques such as QR and SVD. As such, this backend is an approximate simulator, whereby the number of singular values may be truncated to keep the MPS size tractable. 
The :code:`tensornet-mps` backend only supports single-GPU simulation. Its approximate nature allows the :code:`tensornet-mps` backend to handle a large number of qubits for certain classes of quantum circuits on a relatively small memory footprint.

.. warning:: 

  The :code:`tensornet-mps` cannot handle quantum gates acting on more than two qubit operands. It will throw an error when this constraint is not satisfied.

To execute a program on the :code:`tensornet-mps` target, use the following commands:

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target tensornet-mps

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('tensornet-mps')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    .. code:: bash 

        nvq++ --target tensornet-mps program.cpp [...] -o program.x
        ./program.x

Specific aspects of the simulation can be configured by defining the following environment variables:

* **`CUDAQ_MPS_MAX_BOND=X`**: The maximum number of singular values to keep (fixed extent truncation). Default: 64.
* **`CUDAQ_MPS_ABS_CUTOFF=X`**: The cutoff for the largest singular value during truncation. Eigenvalues that are smaller will be trimmed out. Default: 1e-5.
* **`CUDAQ_MPS_RELATIVE_CUTOFF=X`**: The cutoff for the maximal singular value relative to the largest eigenvalue. Eigenvalues that are smaller than this fraction of the largest singular value will be trimmed out. Default: 1e-5

.. note:: 

  This backend requires an NVIDIA GPU and CUDA runtime libraries. 
  If you do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. 
  See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

.. note::

  Setting random seed, via :code:`cudaq::set_random_seed`, is not supported for this backend due to a limitation of the :code:`cuTensorNet` library. This will be fixed in future release once this feature becomes available.


.. _default-simulator:

Default Simulator
==================================
If no explicit target is set, i.e. if the code is compiled without any :code:`--target` flags, then CUDA Quantum makes a default choice for the simulator.

If an NVIDIA GPU and CUDA runtime libraries are available, the default target is set to `nvidia`. This will utilize the :ref:`cuQuantum single-GPU state vector simulator <cuQuantum single-GPU>`.  
On CPU-only systems, the default target is set to `qpp-cpu` which uses the :ref:`OpenMP CPU-only simulator <OpenMP CPU-only>`.

The default simulator can be overridden by the environment variable `CUDAQ_DEFAULT_SIMULATOR`. If no target is explicitly specified and the environment variable has a valid value, then it will take effect.
This environment variable can be set to any non-hardware backend. Any invalid value is ignored.

For CUDA Quantum Python API, the environment variable at the time when `cudaq` module is imported is relevant, not the value of the environment variable at the time when the simulator is invoked.

For example,

.. tab:: Python

    .. code:: bash 

        CUDAQ_DEFAULT_SIMULATOR=density-matrix-cpu python3 program.py [...]
        
.. tab:: C++

    .. code:: bash 

        CUDAQ_DEFAULT_SIMULATOR=density-matrix-cpu nvq++ program.cpp [...] -o program.x
        ./program.x

This will use the density matrix simulator target.


.. note:: 

    To use targets that require an NVIDIA GPU and CUDA runtime libraries, the dependencies must be installed, else you may encounter an error stating `Invalid simulator requested`. See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.
