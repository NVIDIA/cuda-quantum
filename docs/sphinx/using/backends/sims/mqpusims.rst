
Multiple QPUs
===========================

The CUDA-Q machine model elucidates the various devices considered in the 
broader quantum-classical compute node context. Programmers will have one or many 
host CPUs, zero or many NVIDIA GPUs, a classical QPU control space, and the
quantum register itself. Moreover, the :doc:`specification </specification/cudaq/platform>`
notes that the underlying platform may expose multiple QPUs. In the near-term,
this will be unlikely with physical QPU instantiations, but the availability of
GPU-based circuit simulators on NVIDIA multi-GPU architectures does give one an
opportunity to think about programming such a multi-QPU architecture in the near-term.
CUDA-Q starts by enabling one to query information about the underlying quantum
platform via the :code:`quantum_platform` abstraction. This type exposes a
:code:`num_qpus()` method that can be used to query the number of available
QPUs for asynchronous CUDA-Q kernel and :code:`cudaq::` function invocations.
Each available QPU is assigned a logical index, and programmers can launch
specific asynchronous function invocations targeting a desired QPU.

.. _mqpu-platform:

Simulate Multiple QPUs in Parallel 
+++++++++++++++++++++++++++++++++++++

In the multi-QPU mode (:code:`mqpu` option), the NVIDIA backend provides a simulated QPU for every available NVIDIA GPU on the underlying system. 
Each QPU is simulated via a `cuStateVec` simulator backend as defined by the NVIDIA backend. 
This target enables asynchronous parallel execution of quantum kernel tasks.

Here is a simple example demonstrating its usage.

.. tab:: Python

    .. literalinclude:: ../../../snippets/python/using/cudaq/platform/sample_async.py
        :language: python
        :start-after: [Begin Documentation]

.. tab:: C++

    .. literalinclude:: ../../../snippets/cpp/using/cudaq/platform/sample_async.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]


    One can specify the target multi-QPU architecture with the :code:`--target` flag:
    
    .. code-block:: console

        nvq++ sample_async.cpp --target nvidia --target-option mqpu
        ./a.out

CUDA-Q exposes asynchronous versions of the default :code:`cudaq` algorithmic
primitive functions like :code:`sample`, :code:`observe`, and :code:`get_state` 
(e.g., :code:`sample_async` function in the above code snippets).

Depending on the number of GPUs available on the system, the :code:`nvidia` multi-QPU platform will create the same number of virtual QPU instances.
For example, on a system with 4 GPUs, the above code will distribute the four sampling tasks among those :code:`GPUEmulatedQPU` instances.

The results might look like the following 4 different random samplings:

.. code-block:: console
  
    Number of QPUs: 4
    { 10011:28 01100:28 ... }
    { 10011:37 01100:25 ... }
    { 10011:29 01100:25 ... }
    { 10011:33 01100:30 ... }

.. note::

  By default, the :code:`nvidia` multi-QPU platform will utilize all available GPUs (number of QPUs instances is equal to the number of GPUs).
  To specify the number QPUs to be instantiated, one can set the :code:`CUDAQ_MQPU_NGPUS` environment variable.
  For example, use :code:`export CUDAQ_MQPU_NGPUS=2` to specify that only 2 QPUs (GPUs) are needed.

Since the underlying :code:`GPUEmulatedQPU` is a simulator backend, we can also retrieve the state vector from each
QPU via the :code:`cudaq::get_state_async` (C++) or :code:`cudaq.get_state_async` (Python) as shown in the bellow code snippets.

.. tab:: Python

    .. literalinclude:: ../../../snippets/python/using/cudaq/platform/get_state_async.py
        :language: python
        :start-after: [Begin Documentation]

.. tab:: C++

    .. literalinclude:: ../../../snippets/cpp/using/cudaq/platform/get_state_async.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]


    One can specify the target multi-QPU architecture with the :code:`--target` flag:
    
    .. code-block:: console

        nvq++ get_state_async.cpp --target nvidia --target-option mqpu
        ./a.out

See the `Hadamard Test notebook <https://nvidia.github.io/cuda-quantum/latest/applications/python/hadamard_test.html>`__ for an application that leverages the `mqpu` backend. 


.. deprecated:: 0.8
    The :code:`nvidia-mqpu` and :code:`nvidia-mqpu-fp64` targets, which are equivalent to the multi-QPU options `mqpu,fp32` and `mqpu,fp64`, respectively, of the :code:`nvidia` target, are deprecated and will be removed in a future release.

Parallel distribution mode
^^^^^^^^^^^^^^^^^^^^^^^^^^

The CUDA-Q :code:`nvidia` multi-QPU platform supports two modes of parallel distribution of expectation value computation:

* MPI: distribute the expectation value computations across available MPI ranks and GPUs for each Hamiltonian term.
* Thread: distribute the expectation value computations among available GPUs via standard C++ threads (each thread handles one GPU).

For instance, if all GPUs are available on a single node, thread-based parallel distribution 
(:code:`cudaq::parallel::thread` in C++ or :code:`cudaq.parallel.thread` in Python, as shown in the above example) is sufficient.
On the other hand, if one wants to distribute the tasks across GPUs on multiple nodes, e.g., on a compute cluster, MPI distribution mode
should be used.

An example of MPI distribution mode usage in both C++ and Python is given below:

.. tab:: Python

    .. literalinclude:: ../../../snippets/python/using/cudaq/platform/observe_mqpu_mpi.py
        :language: python
        :start-after: [Begin Documentation]

    .. code-block:: console

        mpiexec -np <N> python3 file.py

.. tab:: C++

    .. literalinclude:: ../../../snippets/cpp/using/cudaq/platform/observe_mqpu_mpi.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        nvq++ file.cpp --target nvidia --target-option mqpu
        mpiexec -np <N> a.out

In the above example, the parallel distribution mode was set to :code:`mpi` using :code:`cudaq::parallel::mpi` in C++ or :code:`cudaq.parallel.mpi` in Python.
CUDA-Q provides MPI utility functions to initialize, finalize, or query (rank, size, etc.) the MPI runtime. 
Last but not least, the compiled executable (C++) or Python script needs to be launched with an appropriate MPI command, 
e.g., :code:`mpiexec`, :code:`mpirun`, :code:`srun`, etc.
