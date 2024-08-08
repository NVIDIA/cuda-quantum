CUDA-Q Simulation Backends
*********************************

.. _nvidia-backend:

The simulation backends that are currently available in CUDA-Q are as follows.

State Vector Simulators
==================================

The :code:`nvidia` target provides a state vector simulator accelerated with 
the :code:`cuStateVec` library. 

The :code:`nvidia` target supports multiple configurable options.

Features 
+++++++++

* Floating-point precision configuration 

The floating point precision of the state vector data can be configured to either 
double (`fp64`) or single (`fp32`) precision. This option can be chosen for the optimal performance and accuracy.


* Distributed simulation

The :code:`nvidia` target supports distributing state vector simulations to multiple GPUs and multiple nodes (`mgpu` :ref:`distribution <nvidia-mgpu-backend>`)
and multi-QPU (`mqpu` :ref:`platform <mqpu-platform>`) distribution whereby each QPU is simulated via a single-GPU simulator instance.


* Host CPU memory utilization 

Host CPU memory can be leveraged in addition to GPU memory to accommodate the state vector 
(i.e., maximizing the number of qubits to be simulated).

.. _cuQuantum single-GPU:


Single-GPU 
++++++++++++++++++++++++++++++++++

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

.. _nvidia-fp64-backend:

By default, this will leverage :code:`FP32` floating point types for the simulation. To 
switch to :code:`FP64`, specify the :code:`--target-option fp64` `nvq++` command line option for `C++` and `Python` or 
use `cudaq.set_target('nvidia', option='fp64')` for Python in-source target modification instead. 

.. tab:: Python

    .. code:: bash 

        python3 program.py [...] --target nvidia --target-option fp64

    The precision of the :code:`nvidia` target can also be modified in the application code by calling

    .. code:: python 

        cudaq.set_target('nvidia', option='fp64')

.. tab:: C++

    .. code:: bash 

        nvq++ --target nvidia --target-option fp64 program.cpp [...] -o program.x
        ./program.x

.. note:: 

  This backend requires an NVIDIA GPU and CUDA runtime libraries. If you do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

In the single-GPU mode, the :code:`nvidia` target provides the following environment variable options.

.. list-table:: **Environment variable options supported in single-GPU mode**
  :widths: 20 30 50

  * - Option
    - Value
    - Description
  * - ``CUDAQ_FUSION_MAX_QUBITS``
    - positive integer
    - The max number of qubits used for gate fusion. The default value is `4`.
  * - ``CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS``
    - integer greater than or equal to -1
    - The max number of qubits used for diagonal gate fusion. The default value is set to `-1` and the fusion size will be automatically adjusted for the better performance. If 0, the gate fusion for diagonal gates is disabled.
  * - ``CUDAQ_FUSION_NUM_HOST_THREADS``
    - positive integer
    - Number of CPU threads used for circuit processing. The default value is `8`.
  * - ``CUDAQ_MAX_CPU_MEMORY_GB``
    - non-negative integer, or `NONE`
    - CPU memory size (in GB) allowed for state-vector migration. `NONE` means unlimited (up to physical memory constraints). Default is 0 (disabled). 
  * - ``CUDAQ_MAX_GPU_MEMORY_GB``
    - positive integer, or `NONE`
    - GPU memory (in GB) allowed for on-device state-vector allocation. As the state-vector size exceeds this limit, host memory will be utilized for migration. `NONE` means unlimited (up to physical memory constraints). This is the default. 

.. deprecated:: 0.8
    The :code:`nvidia-fp64` targets, which is equivalent setting the `fp64` option on the :code:`nvidia` target, 
    is deprecated and will be removed in a future release.

.. _nvidia-mgpu-backend:

Multi-node multi-GPU
++++++++++++++++++++++++++++++++++

The NVIDIA target also provides a state vector simulator accelerated with 
the :code:`cuStateVec` library with support for Multi-Node, Multi-GPU distribution of the 
state vector, in addition to a single GPU.

The multi-node multi-GPU simulator expects to run within an MPI context.
To execute a program on the multi-node multi-GPU NVIDIA target, use the following commands 
(adjust the value of the :code:`-np` flag as needed to reflect available GPU resources on your system):

.. tab:: Python

    Double precision simulation:

    .. code:: bash 

        mpiexec -np 2 python3 program.py [...] --target nvidia --target-option fp64,mgpu

    Single precision simulation:
    
    .. code:: bash 

        mpiexec -np 2 python3 program.py [...] --target nvidia --target-option fp32,mgpu

    .. note::

      If you installed CUDA-Q via :code:`pip`, you will need to install the necessary MPI dependencies separately;
      please follow the instructions for installing dependencies in the `Project Description <https://pypi.org/project/cuda-quantum/#description>`__.

    In addition to using MPI in the simulator, you can use it in your application code by installing `mpi4py <https://mpi4py.readthedocs.io/>`__, and 
    invoking the program with the command

    .. code:: bash 

        mpiexec -np 2 python3 -m mpi4py program.py [...] --target nvidia --target-option fp64,mgpu

    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('nvidia', option='mgpu,fp64')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

    .. note::
        (1) The order of the option settings are interchangeable.
        For example, `cudaq.set_target('nvidia', option='mgpu,fp64')` is equivalent to `cudaq.set_target('nvidia', option='fp64.mgpu')`.

        (2) The `nvidia` target has single-precision as the default setting. Thus, using `option='mgpu'` implies that `option='mgpu,fp32'`.  

.. tab:: C++

    Double precision simulation:

    .. code:: bash 

        nvq++ --target nvidia  --target-option mgpu,fp64 program.cpp [...] -o program.x
        mpiexec -np 2 ./program.x

    Single precision simulation:

    .. code:: bash 

        nvq++ --target nvidia  --target-option mgpu,fp32 program.cpp [...] -o program.x
        mpiexec -np 2 ./program.x

.. note:: 

  This backend requires an NVIDIA GPU, CUDA runtime libraries, as well as an MPI installation. If you do not have these dependencies installed, you may encounter either an error stating `invalid simulator requested` (missing CUDA libraries), or an error along the lines of `failed to launch kernel` (missing MPI installation). See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.
  
  The number of processes and nodes should be always power-of-2. 

  Host-device state vector migration is also supported in the multi-node multi-GPU configuration. 


In addition to those environment variable options supported in the single-GPU mode,
the :code:`nvidia` target provides the following environment variable options particularly for 
the multi-node multi-GPU configuration.

.. list-table:: **Additional environment variable options for multi-node multi-GPU mode**
  :widths: 20 30 50

  * - Option
    - Value
    - Description
  * - ``CUDAQ_MGPU_LIB_MPI``
    - string
    - The shared library name for inter-process communication. The default value is `libmpi.so`.
  * - ``CUDAQ_MGPU_COMM_PLUGIN_TYPE``
    - `AUTO`, `EXTERNAL`, `OpenMPI`, or `MPICH` 
    - Selecting :code:`cuStateVec` `CommPlugin` for inter-process communication. The default is `AUTO`. If `EXTERNAL` is selected, `CUDAQ_MGPU_LIB_MPI` should point to an implementation of :code:`cuStateVec` `CommPlugin` interface.
  * - ``CUDAQ_MGPU_NQUBITS_THRESH``
    - positive integer
    - The qubit count threshold where state vector distribution is activated. Below this threshold, simulation is performed as independent (non-distributed) tasks across all MPI processes for optimal performance. Default is 25. 
  * - ``CUDAQ_MGPU_FUSE``
    - positive integer
    - The max number of qubits used for gate fusion. The default value is `6` if there are more than one MPI processes or `4` otherwise.
  * - ``CUDAQ_MGPU_P2P_DEVICE_BITS``
    - positive integer
    - Specify the number of GPUs that can communicate by using GPUDirect P2P. Default value is 0 (P2P communication is disabled).
  * - ``CUDAQ_GPU_FABRIC``
    - `MNNVL`, `NVL`, or `NONE`
    - Automatically set the number of P2P device bits based on the total number of processes when multi-node NVLink (`MNNVL`) is selected; or the number of processes per node when NVLink (`NVL`) is selected; or disable P2P (with `NONE`). 
  * - ``CUDAQ_GLOBAL_INDEX_BITS``
    - comma-separated list of positive integers
    - Specify the inter-node network structure (faster to slower). For example, assuming a 8 nodes, 4 GPUs/node simulation whereby network communication is faster, this `CUDAQ_GLOBAL_INDEX_BITS` environment variable can be set to `3,2`. The first `3` represents **8** nodes with fast communication and the second `2` represents **4** 8-node groups in those total 32 nodes. Default is an empty list (no customization based on network structure of the cluster).
  * - ``CUDAQ_HOST_DEVICE_MIGRATION_LEVEL``
    - positive integer
    - Specify host-device memory migration w.r.t. the network structure. 

.. deprecated:: 0.8
    The :code:`nvidia-mgpu` target, which is equivalent to the multi-node multi-GPU double-precision option (`mgpu,fp64`) of the :code:`nvidia`
    is deprecated and will be removed in a future release.

The above configuration options of the :code:`nvidia` backend 
can be tuned to reduce your simulation runtimes. One of the
performance improvements is to fuse multiple gates together during runtime. For
example, :code:`x(qubit0)` and :code:`x(qubit1)` can be fused together into a
single 4x4 matrix operation on the state vector rather than 2 separate 2x2
matrix operations on the state vector. This fusion reduces memory bandwidth on
the GPU because the state vector is transferred into and out of memory fewer
times. By default, up to 4 gates are fused together for single-GPU simulations,
and up to 6 gates are fused together for multi-GPU simulations. The number of
gates fused can **significantly** affect performance of some circuits, so users
can override the default fusion level by setting the setting `CUDAQ_MGPU_FUSE`
environment variable to another integer value as shown below.

.. tab:: Python

    .. code:: bash 

        CUDAQ_MGPU_FUSE=5 mpiexec -np 2 python3 program.py [...] --target nvidia --target-option mgpu,fp64

.. tab:: C++

    .. code:: bash 

        nvq++ --target nvidia --target-option mgpu,fp64 program.cpp [...] -o program.x
        CUDAQ_MGPU_FUSE=5 mpiexec -np 2 ./program.x

.. _OpenMP CPU-only:

OpenMP CPU-only
++++++++++++++++++++++++++++++++++

.. _qpp-cpu-backend:

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

.. _tensor-backends:

CUDA-Q provides a couple of tensor-network simulator targets accelerated with 
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

  If you installed the CUDA-Q Python wheels, distribution across multiple GPUs is currently not supported for this backend.
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
  If you are using a CUDA-Q container, this variable is pre-configured and no additional setup is needed. If you are customizing your installation or have built CUDA-Q from source, please follow the instructions for `activating the distributed interface <https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html#from-nvidia-devzone>`__ for the `cuTensorNet` library. This requires 
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
* **`CUDAQ_MPS_SVD_ALGO=X`**: The SVD algorithm to use. Valid values are: `GESVD` (QR algorithm), `GESVDJ` (Jacobi method), `GESVDP` (`polar decomposition <https://epubs.siam.org/doi/10.1137/090774999>`__), `GESVDR` (`randomized methods <https://epubs.siam.org/doi/10.1137/090771806>`__). Default: `GESVDJ`.

.. note:: 

  This backend requires an NVIDIA GPU and CUDA runtime libraries. 
  If you do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. 
  See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

.. note::

  Setting random seed, via :code:`cudaq::set_random_seed`, is not supported for this backend due to a limitation of the :code:`cuTensorNet` library. This will be fixed in future release once this feature becomes available.

.. note::
    The parallelism of Jacobi method (the default `CUDAQ_MPS_SVD_ALGO` setting) gives GPU better performance on small and medium size matrices.
    If you expect the a large number of singular values (e.g., increasing the `CUDAQ_MPS_MAX_BOND` setting), please adjust the `CUDAQ_MPS_SVD_ALGO` setting accordingly.  

Default Simulator
==================================

.. _default-simulator:

If no explicit target is set, i.e. if the code is compiled without any :code:`--target` flags, then CUDA-Q makes a default choice for the simulator.

If an NVIDIA GPU and CUDA runtime libraries are available, the default target is set to `nvidia`. This will utilize the :ref:`cuQuantum single-GPU state vector simulator <cuQuantum single-GPU>`.  
On CPU-only systems, the default target is set to `qpp-cpu` which uses the :ref:`OpenMP CPU-only simulator <OpenMP CPU-only>`.

The default simulator can be overridden by the environment variable `CUDAQ_DEFAULT_SIMULATOR`. If no target is explicitly specified and the environment variable has a valid value, then it will take effect.
This environment variable can be set to any non-hardware backend. Any invalid value is ignored.

For CUDA-Q Python API, the environment variable at the time when `cudaq` module is imported is relevant, not the value of the environment variable at the time when the simulator is invoked.

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
