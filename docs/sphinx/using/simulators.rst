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

This target provides a state vector simulator based on the CPU-only, OpenMP threaded `Q++ <https://github.com/softwareqinc/qpp>`_ library.
This is the default target when running on CPU-only systems.

Tensor Network Simulators
==================================

cuQuantum multi-node multi-GPU
++++++++++++++++++++++++++++++++++

The :code:`tensornet` target provides a tensor-network simulator accelerated with 
the :code:`cuTensorNet` library. This backend is currently available for use from C++ and supports 
Multi-Node, Multi-GPU distribution of tensor operations required to evaluate and simulate the circuit.

.. note:: 

    This backend requires an NVIDIA GPU and CUDA runtime libraries. If you are do not have these dependencies installed, you may encounter an error stating `Invalid simulator requested`. See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.

This backend exposes a set of environment variables to configure specific aspects of the simulation:

* **`CUDAQ_CUTN_HOST_RAM=8`**: Prescribes the size of the CPU Host RAM allocated by each MPI process (defaults to 4 GB). A rule of thumb is to give each MPI process the same amount of CPU Host RAM as the RAM size of the GPU assigned to it. If there is more CPU RAM available, it is fine to further increase this number.
* **`CUDAQ_CUTN_REDUCED_PRECISION=1`**: Activates reduced precision arithmetic, specifically reduces the precision from :code:`FP64` to :code:`FP32`.
* **`CUDAQ_CUTN_LOG_LEVEL=1`**: Activates logging (for debugging purposes), the larger the integer, the more detailed the logging will be.
* **`CUDA_VISIBLE_DEVICES=X`**: Makes the process only see GPU X on multi-GPU nodes. Each MPI process must only see its own dedicated GPU. For example, if you run 8 MPI processes on a DGX system with 8 GPUs, each MPI process should be assigned its own dedicated GPU via CUDA_VISIBLE_DEVICES when invoking `mpirun` (or `mpiexec`) commands. This can be done via invoking a bash script instead of the binary directly, and then using MPI library specific environment variables inside that script (e.g., `OMPI_COMM_WORLD_LOCAL_RANK`).
* **`OMP_PLACES=cores`**: Set this environment variable to improve CPU parallelization.
* **`OMP_NUM_THREADS=X`**: To enable CPU parallelization, set X to `NUMBER_OF_CORES_PER_NODE/NUMBER_OF_GPUS_PER_NODE`.

A note on **CUDA_VISIBLE_DEVICES**: This environment variable should **always** be set before using the :code:`tensornet` 
backend if you have multiple GPUs available. With OpenMPI, you can run a multi-GPU quantum circuit simulation like this:

.. code:: bash 
    
    mpiexec -n 8 sh -c 'CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK} binary.x > tensornet.${OMPI_COMM_WORLD_RANK}.log'

This command will assign a unique GPU to each MPI process within the node with 8 GPUs and produce a separate output for each MPI process.

To specify the use of the :code:`tensornet` target, pass the following command line 
options to :code:`nvq++`

.. code:: bash 

    nvq++ --target tensornet src.cpp ...


Default Simulator
==================================
If no explicit target is set, i.e. if the code is compiled without any :code:`--target` flags, then CUDA Quantum makes a default choice for the simulator.

If an NVIDIA GPU and CUDA runtime libraries are available, the default target is set to `nvidia`. This will utilize the :ref:`cuQuantum single-GPU state vector simulator <cuQuantum single-GPU>`.  
On CPU-only systems, the default target is set to `qpp-cpu` which uses the :ref:`OpenMP CPU-only simulator <OpenMP CPU-only>`.

The default simulator can be overridden by the environment variable `CUDAQ_DEFAULT_SIMULATOR`. If no target is explicitly specified and the environment variable has a valid value, then it will take effect.
This environment variable can be set to any non-hardware backend. Any invalid value is ignored.

For example,
.. code:: bash

    CUDAQ_DEFAULT_SIMULATOR=density-matrix-cpu nvq++ src.cpp

This will use the density matrix simulator target.


.. note:: 

    To use targets that require an NVIDIA GPU and CUDA runtime libraries, the dependencies must be installed, else you may encounter an error stating `Invalid simulator requested`. See the section :ref:`dependencies-and-compatibility` for more information about how to install dependencies.
