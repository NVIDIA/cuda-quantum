
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
    - The maximum number of qubits used for dense gate fusion. When unset, ``cuStateVecEx`` `automatically selects the fusion size <https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/api-reference/custatevecex/index.html#svupdater-api>`__. Values greater than 10 are clamped to 10, while a non-positive value disables gate fusion.
  * - ``CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS``
    - integer greater than or equal to -1 (maximum effective value: 20)
    - The maximum number of qubits used for diagonal gate fusion. When unset or set to -1, ``cuStateVecEx`` automatically selects the fusion size. Values greater than 20 are clamped to 20, while 0 disables diagonal gate fusion.
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
    - Legacy multi-GPU alias for ``CUDAQ_FUSION_MAX_QUBITS``. The generic variable takes precedence when both are set. When neither variable is set, ``cuStateVecEx`` automatically selects the dense fusion size. Values greater than 10 are clamped to 10, while a non-positive value disables gate fusion.
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

The above configuration options of the :code:`nvidia` backend 
can be tuned to reduce your simulation runtimes. One of the
performance improvements is to fuse multiple gates together during runtime. For
example, :code:`x(qubit0)` and :code:`x(qubit1)` can be fused together into a
single 4x4 matrix operation on the state vector rather than 2 separate 2x2
matrix operations on the state vector. This fusion reduces memory bandwidth on
the GPU because the state vector is transferred into and out of memory fewer
times. By default, ``cuStateVecEx`` automatically selects the dense and diagonal
fusion sizes. The number of gates fused can **significantly** affect performance
of some circuits, so users can override the dense fusion size by setting
``CUDAQ_FUSION_MAX_QUBITS`` as shown below. The legacy ``CUDAQ_MGPU_FUSE`` alias
continues to be supported for multi-GPU simulations.

.. tab:: Python

    .. code:: bash 

        CUDAQ_FUSION_MAX_QUBITS=5 mpiexec -np 2 python3 program.py [...] --target nvidia --target-option mgpu,fp64

.. tab:: C++

    .. code:: bash 

        nvq++ --target nvidia --target-option mgpu,fp64 program.cpp [...] -o program.x
        CUDAQ_FUSION_MAX_QUBITS=5 mpiexec -np 2 ./program.x


.. note::

  On multi-node systems without `MNNVL` support, the `nvidia` target in `mgpu` mode may fail to allocate memory.
  Users can disable GPU-fabric P2P memory sharing by setting the environment variable `CUDAQ_GPU_FABRIC=NONE`.


.. _mgpu-gpu-fabric:

GPU fabric and peer-to-peer memory sharing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`CUDAQ_GPU_FABRIC`, and the lower-level `CUDAQ_MGPU_P2P_DEVICE_BITS` it derives, decide how many
global index bits travel over GPUDirect P2P instead of the communicator -- equivalently, how many
ranks form one P2P domain. The `nvidia` backend maps that choice onto one of the `memory sharing
methods
<https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/ex-statevector-distribution.html#memory-sharing-methods>`__
of the `cuStateVec Ex` API, which is what lets the ranks of a domain map each other's sub state
vector memory:

.. list-table:: **P2P domain and memory sharing method**
  :widths: 25 40 35

  * - Setting
    - P2P domain
    - Memory sharing method
  * - ``MNNVL``
    - All ranks of the communicator
    - Fabric Handle, requested explicitly
  * - ``NVL``, or an explicit domain size
    - The ranks that share a physical node, or the given number of ranks
    - Auto-detect: Fabric Handle or `PidFd`, whichever the system provides
  * - ``NONE``
    - No P2P domain; every transfer goes through the communicator
    - None

`MNNVL` and `NVL` differ only in the width of that domain. `MNNVL` covers systems whose NVLink
fabric spans nodes, such as `GB200 NVL36` and `GB200 NVL72`, so the whole communicator becomes a
single P2P domain. Multi-node NVLink requires Fabric Handle (`Memory Sharing Methods
<https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/ex-statevector-distribution.html#memory-sharing-methods>`__).
`NVL` confines each domain to one physical node, which matches a node-local NVLink or `NVSwitch`
topology; CUDA-Q enforces that by requiring every host to hold the same number of ranks and each
host's ranks to occupy a contiguous block. A node-local domain works with either method, Fabric
Handle or `PidFd`, so `NVL` utilizes the `cuStateVec` auto-selection feature.

.. note::

  `MNNVL` and `NVL` size the P2P domain after the NVLink topology, so confirm that CUDA reports
  NVLink P2P between the GPUs you intend to place in one domain:

  .. code:: bash

      nvidia-smi topo -p2p n   # expect OK for every GPU pair in the domain
      nvidia-smi topo -m       # NV# entries mark the NVLink connections

  Any pair reporting `NS`, `CNS`, `TNS`, or `GNS` has no NVLink P2P path. Shrink the domain to the
  GPUs that report `OK`, or set `CUDAQ_GPU_FABRIC=NONE` and let the communicator carry those index
  bits.

Fabric Handle needs the `IMEX` channels of the NVIDIA driver. Verify they are present before
relying on it: `/proc/devices` must list `nvidia-caps-imex-channels`, and an accessible device
node must exist under `/dev/nvidia-caps-imex-channels/` (`cuMemCreate`, in `CUDA Driver API --
Virtual Memory Management
<https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html>`__):

.. code:: bash

    grep nvidia-caps-imex-channels /proc/devices
    ls /dev/nvidia-caps-imex-channels/

Without them, `CUDAQ_GPU_FABRIC=MNNVL` will abort while creating the state vector with an
`invalid configuration` error, and `CUDAQ_GPU_FABRIC=NVL` may also abort with that same message
if `PidFd` is unavailable as well. To get more information, we can raise the `cuStateVec` log
level (`Useful tips
<https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/examples.html#useful-tips>`__) to learn
which method the backend requested and why it was rejected:

.. code:: bash

    CUSTATEVEC_LOG_LEVEL=1 mpiexec -np 2 ./program.x

For example, the log may show `FABRIC_HANDLE memory sharing method is not available on this
system` for `CUDAQ_GPU_FABRIC=MNNVL`. For `CUDAQ_GPU_FABRIC=NVL` (with auto-detection), the log
may show `No memory sharing method is available on this system` when neither method is usable.

Requirements of the `PidFd` method
""""""""""""""""""""""""""""""""""

On systems without Fabric Handle, the `nvidia` target falls back to `PidFd` for per-node P2P
memory sharing (`CUDAQ_GPU_FABRIC=NVL`).
`PidFd` exports a POSIX file descriptor from each rank and imports it into the peer ranks through
the `pidfd_open` and `pidfd_getfd` Linux system calls. Miss any of the following and the `nvidia`
target aborts while creating the state vector, with `No memory sharing method is available on this
system` in the `cuStateVec` log:

* **Kernel 5.6 or newer** (`Memory Sharing Methods
  <https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/ex-statevector-distribution.html#memory-sharing-methods>`__).
  This can be checked with `uname -r`.

* **One hardware node.** File descriptor sharing crosses processes, not machines, so every rank of
  a P2P domain must run on the same host. `CUDAQ_GPU_FABRIC=NVL` already guarantees that; a
  manually set `CUDAQ_MGPU_P2P_DEVICE_BITS` is not validated against the host layout, so it needs
  to be set accordingly.

* **Permission to import the descriptor.** `pidfd_getfd` requires `PTRACE_MODE_ATTACH_REALCREDS`
  over the peer process, the same check that governs `ptrace` (`pidfd_getfd(2)
  <https://man7.org/linux/man-pages/man2/pidfd_getfd.2.html>`__). Containers and some hosts
  deny it by default, as described below.

The default `Docker` `seccomp` profile allows `pidfd_getfd` only for containers that hold the
`SYS_PTRACE` capability (`default profile
<https://github.com/moby/profiles/blob/main/seccomp/default.json>`__). Grant it at startup:

.. code:: bash

    docker run --cap-add SYS_PTRACE ...
    # or drop the profile entirely
    docker run --security-opt seccomp=unconfined ...

On bare metal, check `/proc/sys/kernel/yama/ptrace_scope`. At 1 or higher, the `Yama` module lets
a process attach only to its own descendants unless the caller holds `SYS_PTRACE` (`ptrace(2)
<https://man7.org/linux/man-pages/man2/ptrace.2.html>`__). Ranks launched by the same `mpirun` are
siblings, not descendants, so they fail that check whenever they run without the capability. Give
the ranks `SYS_PTRACE`, or relax the setting:

.. code:: bash

    cat /proc/sys/kernel/yama/ptrace_scope   # 0 adds no restriction beyond the standard checks
    sudo sysctl -w kernel.yama.ptrace_scope=0

.. note::

  When neither method is available, set `CUDAQ_GPU_FABRIC=NONE`. That drops the P2P layer, so every
  global index bit is carried by the communicator and the results stay correct; the transfers take
  the MPI path instead of direct GPU-to-GPU copies.
