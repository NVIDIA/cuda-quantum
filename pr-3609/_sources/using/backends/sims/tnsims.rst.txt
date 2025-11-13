
Tensor Network Simulators
==================================

.. _tensor-backends:

CUDA-Q provides a couple of tensor-network simulator backends accelerated with 
the :code:`cuTensorNet` library. Detailed technical information on the simulator can be found `here <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/index.html>`__. 
These backends are available for use from both C++ and Python.

Tensor network simulators are suitable for large-scale simulation of certain classes of quantum circuits involving many qubits beyond the memory limit of state vector based simulators. For example, computing the expectation value of a Hamiltonian via :code:`cudaq::observe` can be performed efficiently, thanks to :code:`cuTensorNet` contraction optimization capability. On the other hand, conditional circuits, i.e., those with mid-circuit measurements or reset, despite being supported by both backends, may result in poor performance. 

Multi-GPU multi-node 
++++++++++++++++++++++

The :code:`tensornet` backend represents quantum states and circuits as tensor networks in an exact form (no approximation). 
Measurement samples and expectation values are computed via tensor network contractions. 
This backend supports multi-GPU, multi-node distribution of tensor operations required to evaluate and simulate the circuit.

The code:`tensornet` target supports both single and double floating point precision.

To execute a program on the :code:`tensornet` target using a *single GPU*, use the following commands:

.. tab:: Python

    Double Precision (Default): 

    .. code:: bash 

        python3 program.py [...] --target tensornet

    Single Precision:
    
     .. code:: bash 

        python3 program.py [...] --target tensornet --target-option fp32
    
    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('tensornet')

    for the default double-precision setting, or
    
    .. code:: python 

        cudaq.set_target('tensornet', option='fp32')

    for the single-precision setting.   

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
  MPI parallelization on the :code:`tensornet` backend requires CUDA-Q's MPI support. 
  Please refer to the instructions on how to :ref:`enable MPI parallelization <distributed-computing-with-mpi>` within CUDA-Q.  
  CUDA-Q containers are shipped with a pre-built MPI plugin; hence no additional setup is needed.  

.. note::  
  If the `CUTENSORNET_COMM_LIB` environment variable is set following the activation procedure described in the `cuTensorNet documentation <https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html#from-nvidia-devzone>`__, the cuTensorNet MPI plugin will take precedence over the builtin support from CUDA-Q.

Specific aspects of the simulation can be configured by setting the following of environment variables:

* **`CUDA_VISIBLE_DEVICES=X`**: Makes the process only see GPU X on multi-GPU nodes. Each MPI process must only see its own dedicated GPU. For example, if you run 8 MPI processes on a DGX system with 8 GPUs, each MPI process should be assigned its own dedicated GPU via `CUDA_VISIBLE_DEVICES` when invoking `mpiexec` (or `mpirun`) commands. 
* **`CUDAQ_TIMING_TAGS=tags`**: When the environment variable includes 9 in the tag set, timing for the path-finding stage (Prepare) and contraction stage (Compute or Sample) are output for the user.
* **`CUDAQ_TENSORNET_CONTROLLED_RANK=X`**: Specify the number of controlled qubits whereby the full tensor body of the controlled gate is expanded. If the number of controlled qubits is greater than this value, the gate is applied as a controlled tensor operator to the tensor network state. Default value is 1.
* **`CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=X`**: Set this environment variable to `TRUE` (`ON`) or `FALSE` (`OFF`) to enable or disable contraction path reuse when computing expectation values. Default is `OFF`.
* **`CUDAQ_TENSORNET_NUM_HYPER_SAMPLES=X`**: Specify the number of hyper samples used in the tensor network contraction path finder. Default value is 8 if not specified. Increasing this value will increase the path-finding time, but can decrease the contraction time if a better quality path is found (and vice versa). Hyper samples are processed in parallel using multiple host threads.
* **`CUDAQ_TENSORNET_FIND_THREADS=X`**: Used to control the number of threads on the host used for path-finding. The default value is half of the available CPU hardware threads. For processors with 1 hardware thread per CPU core (no `SMT`), increasing this to equal the number of CPU cores can improve performance.
* **`CUDAQ_TENSORNET_FIND_LIMIT=X`**: Set this environment variable to `TRUE` (`ON`) or `FALSE` (`OFF`) to enable or disable a heuristic to limit the path-finding time based on the predicted contraction time. When on, increasing the number of hyper samples may have no effect beyond a certain threshold due to enforcement of the time limit. Default is `ON`.
* **`CUDAQ_TENSORNET_FIND_DETERMINISTIC=X`**: Set this environment variable to `TRUE` (`ON`) or `FALSE` (`OFF`) to enable or disable deterministic path-finding as controlled by the CUDA-Q set_random_seed() function. When on, the number of path-finding threads is limited to 1 and therefore this setting can significantly decrease performance. Default is `OFF`.

.. note::
  Setting the `CUDAQ_TENSORNET_*` environment variables will override any corresponding environment variables used by the `cuTensorNet` library.

.. note:: 

  This backend requires an NVIDIA GPU and CUDA runtime libraries. 
  If you do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. 
  See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.


Matrix product state 
+++++++++++++++++++++++

The :code:`tensornet-mps` backend is based on the matrix product state (MPS) representation of the state vector/wave function, exploiting the sparsity in the tensor network via tensor decomposition techniques such as QR and SVD. As such, this backend is an approximate simulator, whereby the number of singular values may be truncated to keep the MPS size tractable. 
The :code:`tensornet-mps` backend only supports single-GPU simulation. Its approximate nature allows the :code:`tensornet-mps` backend to handle a large number of qubits for certain classes of quantum circuits on a relatively small memory footprint.

The code:`tensornet-mps` target supports both single and double floating point precision.

To execute a program on the :code:`tensornet-mps` target, use the following commands:

.. tab:: Python

    Double Precision (Default): 
    
    .. code:: bash 

        python3 program.py [...] --target tensornet-mps

    Single Precision:

    .. code:: bash 

        python3 program.py [...] --target tensornet-mps --target-option fp32
    
    The target can also be defined in the application code by calling

    .. code:: python 

        cudaq.set_target('tensornet-mps')

    for the default double-precision setting, or
    
    .. code:: python 

        cudaq.set_target('tensornet-mps', option='fp32')

    for the single-precision setting.   

    If a target is set in the application code, this target will override the :code:`--target` command line flag given during program invocation.

.. tab:: C++

    Double Precision (Default): 

    .. code:: bash 

        nvq++ --target tensornet-mps program.cpp [...] -o program.x
        ./program.x

    Single Precision:

    .. code:: bash 

        nvq++ --target tensornet-mps --target-option fp32 program.cpp [...] -o program.x
        ./program.x

Specific aspects of the simulation can be configured by defining the following environment variables:

* **`CUDAQ_MPS_MAX_BOND=X`**: The maximum number of singular values to keep (fixed extent truncation). Default: 64.
* **`CUDAQ_MPS_ABS_CUTOFF=X`**: The cutoff for the largest singular value during truncation. Eigenvalues that are smaller will be trimmed out. Default: 1e-5.
* **`CUDAQ_MPS_RELATIVE_CUTOFF=X`**: The cutoff for the maximal singular value relative to the largest eigenvalue. Eigenvalues that are smaller than this fraction of the largest singular value will be trimmed out. Default: 1e-5
* **`CUDAQ_MPS_SVD_ALGO=X`**: The SVD algorithm to use. Valid values are: `GESVD` (QR algorithm), `GESVDJ` (Jacobi method), `GESVDP` (`polar decomposition <https://epubs.siam.org/doi/10.1137/090774999>`__), `GESVDR` (`randomized methods <https://epubs.siam.org/doi/10.1137/090771806>`__). Default: `GESVDJ`.
* **`CUDAQ_MPS_GAUGE=X`**: The optional gauge option to improve accuracy of the MPS simulation. Valid values are: `FREE` (gauge is disabled) or `SIMPLE` (simple update algorithm). By default, no gauge configuration is set, thus the default `cuquantum` MPS setting will be used (see `cuquantum` `doc <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/api/types.html#cutensornetstatempsgaugeoption-t>`__).  

.. note:: 

  This backend requires an NVIDIA GPU and CUDA runtime libraries. 
  If you do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. 
  See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

.. note::
    The parallelism of Jacobi method (the default `CUDAQ_MPS_SVD_ALGO` setting) gives GPU better performance on small and medium size matrices.
    If you expect a large number of singular values (e.g., increasing the `CUDAQ_MPS_MAX_BOND` setting), please adjust the `CUDAQ_MPS_SVD_ALGO` setting accordingly.  


.. note::

    Both `tensornet-mps` and `tensornet` backends will allocate a scratch space on GPU device memory for their operations.
    For example, the scratch space can be used to store the contracted reduced density matrix to generate measurement bit strings.
    
    By default, these backends reserve 50% of free memory for its scratch space.
    This ratio can be customized using the `CUDAQ_TENSORNET_SCRATCH_SIZE_PERCENTAGE` environment variable.
    Valid setting must be between 5% and 95%. 
    Users may encounter runtime errors, e.g., insufficient workspace or CUDA memory allocation errors,
    when setting `CUDAQ_TENSORNET_SCRATCH_SIZE_PERCENTAGE` toward its limits.


.. note::

    All floating-point data, e.g., gate matrices, noise channel Kraus operator matrices, contracted state vector, etc., are converted to
    the target's precision setting, if not already in that precision format. Hence, users would need to take into account potential precision 
    lost when using the single precision setting.


Fermioniq
++++++++++

.. _fermioniq-backend:

`Fermioniq <https://fermioniq.com/>`__ offers a cloud-based tensor-network emulation platform, `Ava <https://www.fermioniq.com/ava/>`__, 
for the approximate simulation of large-scale quantum circuits beyond the memory limit of state vector and exact tensor network based methods. 

The level of approximation can be controlled by setting the bond dimension: larger values yield more accurate simulations at the expense 
of slower computation time. For a detailed description of Ava users are referred to the `online documentation <https://docs.fermioniq.com/>`__.

Users of CUDA-Q can access a simplified version of the full Fermioniq emulator (`Ava <https://www.fermioniq.com/ava/>`__) from either
C++ or Python. This version currently supports emulation of quantum circuits without noise, and can return measurement samples and/or 
compute expectation values of observables.

.. note::
    In order to use the Fermioniq emulator, users must provide access credentials. These can be requested by contacting info@fermioniq.com 

    The credentials must be set via two environment variables:
    `FERMIONIQ_ACCESS_TOKEN_ID` and `FERMIONIQ_ACCESS_TOKEN_SECRET`.

.. tab:: Python

    The target to which quantum kernels are submitted 
    can be controlled with the ``cudaq.set_target()`` function.

    .. code:: python

        cudaq.set_target('fermioniq')

    You will have to specify a remote configuration id for the Fermioniq backend
    during compilation.

    .. code:: python

        cudaq.set_target("fermioniq",**{
            "remote_config": remote_config_id
        })

    For a comprehensive list of all remote configurations, please contact Fermioniq directly.

    When your organization requires you to define a project id, you have to specify
    the project id during compilation.

    .. code:: python

        cudaq.set_target("fermioniq",**{
            "project_id": project_id
        })

.. tab:: C++

    To target quantum kernel code for execution in the Fermioniq backends,
    pass the flag ``--target fermioniq`` to the ``nvq++`` compiler. CUDA-Q will
    authenticate via the Fermioniq REST API using the environment variables
    set earlier.

    .. code:: bash

        nvq++ --target fermioniq src.cpp ...

    You will have to specify a remote configuration id for the Fermioniq backend
    during compilation.

    .. code:: bash

        nvq++ --target fermioniq --fermioniq-remote-config <remote_config_id> src.cpp ...

    For a comprehensive list of all remote configurations, please contact Fermioniq directly.

    When your organization requires you to define a project id, you have to specify
    the project id during compilation.

    .. code:: bash

        nvq++ --target fermioniq --fermioniq-project-id <project_id> src.cpp ...

    To specify the bond dimension, you can pass the ``fermioniq-bond-dim`` parameter.

    .. code:: bash

        nvq++ --target fermioniq --fermioniq-bond-dim 10 src.cpp ...       
