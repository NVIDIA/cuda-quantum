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

cuQuantum multi-node multi-GPU
++++++++++++++++++++++++++++++++++

The :code:`nvidia-mgpu` target provides a state vector simulator accelerated with 
the :code:`cuStateVec` library but with support for Multi-Node, Multi-GPU distribution of the 
state vector. 

To specify the use of the :code:`nvidia-mgpu` target, pass the following command line 
options to :code:`nvq++`

.. code:: bash 

    nvq++ --target nvidia-mgpu src.cpp ...

In python, this can be specified with 

.. code:: python 

    cudaq.set_target('nvidia-mgpu')

OpenMP CPU-only
++++++++++++++++++++++++++++++++++

The :code:`default` target provides a state vector simulator based on the CPU-only, OpenMP
threaded `Q++ <https://github.com/softwareqinc/qpp>`_ library. This is the default 
target, so if the code is compiled without any :code:`--target` flags, this is the 
simulator that will be used. 

Tensor Network Simulators
==================================

cuQuantum multi-node multi-GPU
++++++++++++++++++++++++++++++++++

The :code:`tensornet` target provides a tensor-network simulator accelerated with 
the :code:`cuTensorNet` library. This backend is currently available for use from C++ and supports 
Multi-Node, Multi-GPU distribution of tensor operations required to evaluate and simulate the circuit.

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
