
State Vector Simulators
==================================

CPU
++++

.. _openmp cpu-only:
.. _qpp-cpu-backend:

The `qpp-cpu` backend backend provides a state vector simulator based on the CPU-only, OpenMP threaded `Q++ <https://github.com/softwareqinc/qpp>`_ library.
This backend is good for basic testing and experimentation with just a few qubits, but performs poorly for all but the smallest simulation and is the default target when running on CPU-only systems. 

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


Single-GPU 
++++++++++++++

.. _cuquantum single-gpu:
.. _default-simulator:
.. _nvidia-backend:


The :code:`nvidia` backend provides single- and multi-GPU state-vector
simulators accelerated with the `cuStateVec` library, version 1.14 or newer.
The `cuStateVec documentation <https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html>`__
provides more information about GPU-accelerated state-vector simulation.

The :code:`nvidia` target supports multiple configurable options including specification of floating point precision.

To execute a program on the :code:`nvidia` backend, use the following commands:

.. tab:: Python

    Single Precision (Default):

    .. code:: bash 

        python3 program.py [...] --target nvidia --target-option fp32

    Double Precision:

    .. code:: bash 

        python3 program.py [...] --target nvidia --target-option fp64
    
    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('nvidia', option = 'fp64')

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

     Single Precision (Default):

     .. code:: bash 

        nvq++ --target nvidia --target-option fp32 program.cpp [...] -o program.x
        ./program.x


     Double Precision (Default):

     .. code:: bash 

        nvq++ --target nvidia --target-option fp64 program.cpp [...] -o program.x
        ./program.x
     
.. note:: 
   This backend requires an NVIDIA GPU and CUDA runtime libraries. If you do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.


In the single-GPU mode, the :code:`nvidia` backend provides the following
environment variable options. Any environment variables must be set prior to setting the target or running "`import cudaq`".
It is worth drawing attention to gate fusion, a powerful tool for improving simulation performance which is discussed in greater detail `here <https://nvidia.github.io/cuda-quantum/latest/examples/python/performance_optimizations.html>`__.

.. list-table:: **Environment variable options supported in single-GPU mode**
  :widths: 20 30 50

  * - Option
    - Value
    - Description
  * - ``CUDAQ_FUSION_MAX_QUBITS``
    - integer (maximum effective value: 10)
    - The max number of qubits used for gate fusion. Values greater than 10 are clamped to 10, while a non-positive value disables gate fusion. The default value depends on `GPU Compute Capability <https://developer.nvidia.com/cuda-gpus>`__ (CC) and the floating point precision selected for the simulator as specified :ref:`here <gate-fusion-table>`.
  * - ``CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS``
    - integer greater than or equal to -1 (maximum effective value: 20)
    - The max number of qubits used for diagonal gate fusion. Values greater than 20 are clamped to 20. The default value is set to `-1` and the fusion size will be automatically adjusted for better performance. If 0, gate fusion for diagonal gates is disabled.
  * - ``CUDAQ_FUSION_NUM_HOST_THREADS``
    - positive integer (maximum effective value: 32)
    - Number of CPU threads used for circuit processing. Values greater than 32 are clamped to 32. The default value is `8`.
  * - ``CUDAQ_MAX_CPU_MEMORY_GB``
    - non-negative integer, or `NONE`
    - CPU memory size (in GB) allowed for state-vector migration. `NONE` means unlimited (up to physical memory constraints). Default is 0GB (disabled, variable is not set to any value).
  * - ``CUDAQ_MAX_GPU_MEMORY_GB``
    - positive integer, or `NONE`
    - GPU memory (in GB) allowed for on-device state-vector allocation. As the state-vector size exceeds this limit, host memory will be utilized for migration. `NONE` means unlimited (up to physical memory constraints). This is the default.
  * - ``CUDAQ_ALLOW_FP32_EMULATED``
    - `TRUE` (`1`, `ON`) or `FALSE` (`0`, `OFF`)
    - [Blackwell (compute capability 10.0+) only] Enable or disable floating point math emulation. If enabled, allows `FP32` emulation kernels using `BFloat16` (`BF16`) whenever possible. Enabled by default. 
  * - ``CUDAQ_GPU_RNG_THRESHOLD``
    - non-negative integer
    - The minimum random-number count that uses GPU generation. The default is 100,000; 0 selects GPU generation for every request.
  * - ``CUDAQ_ENABLE_MEMPOOL``
    - `TRUE` (`1`, `ON`) or `FALSE` (`0`, `OFF`)
    - Enable or disable `CUDA memory pool <https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/#memory_pools>`__ for state vector allocation/deallocation. Enabled by default. 


.. deprecated:: 0.8
    The :code:`nvidia-fp64` targets, which is equivalent setting the `fp64` option on the :code:`nvidia` target, 
    is deprecated and will be removed in a future release.

.. note:: 

    The ``CUDAQ_MATRIX_EXP_VAL_MAX_SIZE`` environment variable has been removed.
    The ``nvidia`` state-vector backend now evaluates Pauli expectations directly
    on host-migrated states without a dense-matrix fallback, so migrated Pauli
    terms no longer require a separate width limit.


Multi-GPU multi-node 
+++++++++++++++++++++++

.. _nvidia-mgpu-backend:

The :code:`nvidia` backend also provides a state vector simulator accelerated with 
the :code:`cuStateVec` library with support for Multi-GPU, Multi-node distribution of the 
state vector.

This backend is necessary to scale applications that require a state vector that cannot fit on a single GPU memory.

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
        
        * The order of the option settings are interchangeable.
          For example, `cudaq.set_target('nvidia', option='mgpu,fp64')` is equivalent to `cudaq.set_target('nvidia', option='fp64,mgpu')`.

        * The `nvidia` target has single-precision as the default setting. Thus, using `option='mgpu'` implies that `option='mgpu,fp32'`.  

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

  This backend requires an NVIDIA GPU, compatible CUDA runtime libraries,
  `cuStateVec` 1.14 or newer, and an MPI installation. Missing CUDA or
  `cuStateVec` libraries may result in an `invalid simulator requested` error. See
  :ref:`dependencies-and-compatibility` for installation instructions.
  
  The number of processes and nodes should be always power-of-2. 

  Host-device state vector migration is also supported in the multi-GPU multi-node configuration. 


In addition to those environment variable options supported in the single-GPU mode,
the :code:`nvidia` backend provides the following environment variable options particularly for 
the multi-node multi-GPU configuration. Any environment variables must be set prior to setting the target or running "`import cudaq`".


.. list-table:: **Additional environment variable options for multi-node multi-GPU mode**
  :widths: 20 30 50

  * - Option
    - Value
    - Description
  * - ``CUDAQ_MGPU_LIB_MPI``
    - string
    - The shared library name for inter-process communication. The default value is `libmpi.so`.
  * - ``CUDAQ_MGPU_COMM_PLUGIN_TYPE``
    - `AUTO`, `SELF`, `EXTERNAL`, `OpenMPI`, or `MPICH`
    - Select the communicator provider. The default `AUTO` uses activated CUDA-Q MPI when available and otherwise detects OpenMPI or MPICH from `CUDAQ_MGPU_LIB_MPI`. `SELF` requires and uses activated CUDA-Q MPI across all ranks. `OpenMPI`, `MPICH`, and `EXTERNAL` always use the corresponding ``cuStateVecEx`` provider and bypass CUDA-Q MPI. If `EXTERNAL` is selected, `CUDAQ_MGPU_LIB_MPI` must point to a ``cuStateVecEx`` communicator module. Custom communicators set with :code:`cudaq::mpi::set_communicator` are supported only by CUDA-Q-backed `AUTO` and `SELF`.
  * - ``CUDAQ_MGPU_NQUBITS_THRESH``
    - positive integer
    - The qubit count threshold where state vector distribution is activated. Below this threshold, simulation is performed as independent (non-distributed) tasks across all MPI processes for optimal performance. Default is 25. 
  * - ``CUDAQ_MGPU_FUSE``
    - integer (maximum effective value: 10)
    - The max number of qubits used for gate fusion. Values greater than 10 are clamped to 10, while a non-positive value disables gate fusion. The default value depends on `GPU Compute Capability <https://developer.nvidia.com/cuda-gpus>`__ (CC) and the floating point precision selected for the simulator as specified :ref:`here <gate-fusion-table>`.
  * - ``CUDAQ_MGPU_P2P_DEVICE_BITS``
    - non-negative integer
    - Specify the number of global device-index bits that use GPUDirect P2P communication. A value of 0 disables P2P communication.
  * - ``CUDAQ_GPU_FABRIC``
    - `MNNVL`, `NVL`, `NONE`, or NVLink domain size (power of 2 integer)
    - Automatically set the number of P2P device bits based on the total number of processes when multi-node NVLink (`MNNVL`) is selected; or the number of processes per node when NVLink (`NVL`) is selected; or disable P2P (with `NONE`); or a specific NVLink domain size.
  * - ``CUDAQ_GLOBAL_INDEX_BITS``
    - comma-separated list of positive integers
    - Specify the network structure (faster to slower). For example, assuming a 32 MPI processes simulation, whereby the network topology is divided into 4 groups of 8 processes, which have faster communication network between them. In this case, the `CUDAQ_GLOBAL_INDEX_BITS` environment variable can be set to `3,2`. The first `3` (`log2(8)`) represents **8** processes with fast communication within the group and the second `2` represents the **4** groups (8 processes each) in those total 32 processes. The sum of all elements in this list is `5`, corresponding to the total number of MPI processes (`2^5 = 32`). If none specified, the global index bits are set based on P2P device bits.
  * - ``CUDAQ_HOST_DEVICE_MIGRATION_LEVEL``
    - positive integer
    - Specify host-device memory migration w.r.t. the network structure. If provided, this setting determines the position to insert the number of migration index bits to the `CUDAQ_GLOBAL_INDEX_BITS` list. By default, if not set, the number of migration index bits (CPU-GPU data transfers) is appended to the end of the array of index bits (aka, state vector distribution scheme). This default behavior is optimized for systems with fast GPU-GPU interconnects (NVLink, InfiniBand, etc.) 
  * - ``CUDAQ_DATA_TRANSFER_BUFFER_BITS``
    - positive integer greater than or equal to 24
    - Specify the temporary buffer size (:code:`1 << CUDAQ_DATA_TRANSFER_BUFFER_BITS` bytes) for inter-node data transfer. The default is set to 26 (64 MB). The minimum allowed value is 24 (16 MB). Depending on systems, setting a larger value to `CUDAQ_DATA_TRANSFER_BUFFER_BITS` can accelerate inter-node data transfers.

.. deprecated:: 0.8
    The :code:`nvidia-mgpu` backend, which is equivalent to the multi-node multi-GPU double-precision option (`mgpu,fp64`) of the :code:`nvidia`
    is deprecated and will be removed in a future release.

.. |:spellcheck-disable:| replace:: \
.. |:spellcheck-enable:| replace:: \


.. _gate-fusion-table:

.. list-table:: **Default Gate Fusion Size**
  :widths: 20 30 50

  * - Compute Capability
    - GPU 
    - Default Gate Fusion Size
  * - 8.0
    - NVIDIA A100
    - 4 (`fp32`) or 5 (`fp64`)
  * - 9.0
    - NVIDIA H100, H200, |:spellcheck-disable:| GH200 |:spellcheck-enable:| 
    - 5 (`fp32`) or 6 (`fp64`)
  * - 10.0
    - NVIDIA GB200, B200
    - 5 (`fp32`) or 4 (`fp64`)
  * - 10.3
    - NVIDIA B300
    - 5 (`fp32`) or 1 (`fp64`)
  * - Others
    - 
    - 4 (`fp32` and `fp64`)


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


.. note:: 
  
  On multi-node systems without `MNNVL` support, the `nvidia` target in `mgpu` mode may fail to allocate memory. 
  Users can disable GPU-fabric P2P memory sharing by setting the environment variable `CUDAQ_GPU_FABRIC=NONE`.
