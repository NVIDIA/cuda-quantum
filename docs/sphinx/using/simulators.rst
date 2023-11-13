CUDA Quantum Simulation Backends
*********************************

The simulation backends that are currently available in CUDA Quantum are as follows.

State Vector Simulators
==================================

cuQuantum single-GPU 
++++++++++++++++++++++++++++++++++

The :code:`nvidia` target provides a state vector simulator accelerated with 
the :code:`cuStateVec` library. 

To specify the use of the :code:`nvidia` target, pass the following command line 
options to :code:`nvq++`

.. code:: bash 

    nvq++ --target nvidia src.cpp ...

In python, this can be specified with 

.. code:: python 

    cudaq.set_target('nvidia')

By default, this will leverage :code:`FP32` floating point types for the simulation. To 
switch to :code:`FP64`, specify the :code:`nvidia-fp64` target instead. 

.. note:: 

    This backend requires an NVIDIA GPU and CUDA runtime libraries. If you are do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

cuQuantum multi-node multi-GPU
++++++++++++++++++++++++++++++++++

The :code:`nvidia-mgpu` target provides a state vector simulator accelerated with 
the :code:`cuStateVec` library but with support for Multi-Node, Multi-GPU distribution of the 
state vector. 

To specify the use of the :code:`nvidia-mgpu` target, pass the following command line 
options to :code:`nvq++`

.. code:: bash 

    nvq++ --target nvidia-mgpu -o program.out src.cpp ...

In Python, this can be specified with 

.. code:: python 

    cudaq.set_target('nvidia-mgpu')

The multi-node multi-GPU simulator expects to run within an MPI context. A program compiled with :code:`nvq++`, for example, is invoked with

.. code:: bash 

    mpirun -np 2 ./program.out

To use the multi-node multi-GPU backend from Python, follow the instructions for installing dependencies in the `Project Description <https://pypi.org/project/cuda-quantum/#description>`__. 
Using `mpi4py <https://mpi4py.readthedocs.io/>`__, for example, a `program.py` can be invoked from the command line with

.. code:: bash 

    mpiexec -np 2 python3.10 -m mpi4py program.py

.. note:: 

    This backend requires an NVIDIA GPU, CUDA runtime libraries, as well as an MPI installation. If you are do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

OpenMP CPU-only
++++++++++++++++++++++++++++++++++

The :code:`default` target provides a state vector simulator based on the CPU-only, OpenMP
threaded `Q++ <https://github.com/softwareqinc/qpp>`_ library. This is the default 
target, so if the code is compiled without any :code:`--target` flags, this is the 
simulator that will be used. 

Tensor Network Simulators
==================================

CUDA Quantum provides a couple of tensor-network simulator targets accelerated with 
the :code:`cuTensorNet` library. 
These backends are available for use from both C++ and Python.

`cuTensorNet` Multi-Node Multi-GPU
+++++++++++++++++++++++++++++++++++

The :code:`tensornet` backend represents quantum states and circuits as tensor networks in an exact form (no approximation). 
Measurement samples and expectation values are computed via tensor network contractions. 
This backend supports Multi-Node, Multi-GPU distribution of tensor operations required to evaluate and simulate the circuit.

.. note:: 
    To enable automatic distributed parallelization across multiple/many GPUs for the :code:`tensornet` backend, `cuTensorNet`'s distributed interface needs to be activated
    as described in the Getting Started section of the `cuTensorNet` library documentation (`Installation and Compilation <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/getting_started.html#install-cutensornet-from-nvidia-devzone>`_). 
    This typically involves executing the `distributed_interfaces/activate_mpi.sh` script in your `cuquantum` install directory on your local system. The activation script will build a shared `cuTensorNet`-MPI wrapper library (`libcutensornet_distributed_interface_mpi.so`) and set the environment variable `$CUTENSORNET_COMM_LIB` to point to that wrapper library. If the `$CUTENSORNET_COMM_LIB` environment variable becomes unset, MPI parallelization on the :code:`tensornet` backend might fail. 

This backend exposes a set of environment variables to configure specific aspects of the simulation:

* **`CUDA_VISIBLE_DEVICES=X`**: Makes the process only see GPU X on multi-GPU nodes. Each MPI process must only see its own dedicated GPU. For example, if you run 8 MPI processes on a DGX system with 8 GPUs, each MPI process should be assigned its own dedicated GPU via `CUDA_VISIBLE_DEVICES` when invoking `mpirun` (or `mpiexec`) commands. This can be done via invoking a bash script instead of the binary directly, and then using MPI library specific environment variables inside that script (e.g., `OMPI_COMM_WORLD_LOCAL_RANK`).
* **`OMP_PLACES=cores`**: Set this environment variable to improve CPU parallelization.
* **`OMP_NUM_THREADS=X`**: To enable CPU parallelization, set X to `NUMBER_OF_CORES_PER_NODE/NUMBER_OF_GPUS_PER_NODE`.

.. note:: 

    The **CUDA_VISIBLE_DEVICES** environment variable should **always** be set before using the :code:`tensornet` 
    backend if you have multiple GPUs available. With OpenMPI, you can run a multi-GPU quantum circuit simulation like this:

    .. code:: bash 
    
        mpiexec -n 8 sh -c 'CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK} binary.x > tensornet.${OMPI_COMM_WORLD_RANK}.log'

    This command will assign a unique GPU to each MPI process within the node with 8 GPUs and produce a separate output for each MPI process.

`cuTensorNet` matrix product state 
+++++++++++++++++++++++++++++++++++

The :code:`tensornet-mps` backend is based on the matrix product state (MPS) representation of the state vector/wave function, exploiting the sparsity in the tensor network via tensor decomposition techniques such as QR and SVD. As such, this backend is an approximate simulator, whereby the number of singular values may be truncated to keep the MPS size tractable. 

This backend exposes a set of environment variables to configure specific aspects of the simulation:

* **`CUDAQ_MPS_MAX_BOND=X`**: The maximum number of singular values to keep (fixed extent truncation). Default: 64.
* **`CUDAQ_MPS_ABS_CUTOFF=X`**: The cutoff for the largest singular value during truncation. Eigenvalues that are smaller will be trimmed out. Default: 1e-5.
* **`CUDAQ_MPS_RELATIVE_CUTOFF=X`**: The cutoff for the maximal singular value relative to the largest eigenvalue. Eigenvalues that are smaller than this fraction of the largest singular value will be trimmed out. Default: 1e-5

The :code:`tensornet-mps` only supports single-GPU simulation. Its approximate nature allows the :code:`tensornet-mps` backend to handle a large number of qubits for certain classes of quantum circuits on a relatively small memory footprint.

.. warning:: 

    The :code:`tensornet-mps` cannot handle quantum gates acting on more than two qubit operands. It will throw an error when this constraint is not satisfied.

Usage
++++++

To specify the use of the :code:`tensornet` or :code:`tensornet-mps` target, pass the :code:`--target` command line 
options to :code:`nvq++` or Python as follows.

.. tab:: C++

    .. code:: bash 

        nvq++ --target tensornet src.cpp ...

    or 

    .. code:: bash 

        nvq++ --target tensornet-mps src.cpp ...

.. tab:: Python
    
    .. code:: bash 

        python3 src.py --target tensornet

    or 

    .. code:: bash 

        python3 src.py --target tensornet-mps



    This is equivalent to calling :code:`cudaq.set_target("tensornet")` or :code:`cudaq.set_target("tensornet-mps")` from within the Python script.

.. note:: 

    These tensor-network backends require an NVIDIA GPU and CUDA runtime libraries. 
    If you are do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. 
    See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

.. note:: 
    Setting random seed, via :code:`cudaq::set_random_seed`, is not supported by the :code:`tensornet` and :code:`tensornet-mps` backends due to a limitation of the :code:`cuTensorNet` library. This will be fixed in future release once this feature becomes available.

.. note:: 
    Tensor network-based simulators, such as the :code:`tensornet` and :code:`tensornet-mps` backends, are suitable for large-scale simulation of certain classes of quantum circuits involving many qubits beyond the memory limit of state vector based simulators. For example, computing the expectation value of a Hamiltonian (:code:`cudaq::spin_op`) via :code:`cudaq::observe` can be performed efficiently, thanks to :code:`cuTensorNet` contraction optimization capability. On the other hand, conditional circuits, i.e., those with mid-circuit measurements or reset, despite being supported by both backends, may result in poor performance. 
