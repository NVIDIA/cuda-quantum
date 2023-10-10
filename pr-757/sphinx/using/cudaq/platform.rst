Taking Advantage of the Underlying Quantum Platform
---------------------------------------------------
The CUDA Quantum machine model elucidates the various devices considered in the 
broader quantum-classical compute node context. One will have one or many 
host CPUs, zero or many NVIDIA GPUs, a classical QPU control space, and the
quantum register itself. Moreover, the specification notes that the underlying
platform may expose multiple QPUs. In the near-term, this will be unlikely with
physical QPU instantiations, but the availability of GPU-based circuit
simulators on NVIDIA multi-GPU architectures does give one an opportunity 
to think about programming such a multi-QPU architecture in the near-term. 
CUDA Quantum starts by enabling one to query information about the underlying quantum
platform via the :code:`quantum_platform` abstraction. This type exposes a 
:code:`num_qpus()` method that can be used to query the number of available 
QPUs for asynchronous CUDA Quantum kernel and :code:`cudaq::` function invocations. 
Each available QPU is assigned a logical index, and programmers can launch
specific asynchronous function invocations targeting a desired QPU. 

Here is a simple example demonstrating this

.. literalinclude:: ../../snippets/cpp/using/cudaq/platform/sample_async.cpp
    :language: cpp
    :start-after: [Begin Documentation]
    :end-before: [End Documentation]

CUDA Quantum exposes asynchronous versions of the default :code:`cudaq::` algorithmic
primitive functions like :code:`sample` and :code:`observe` (e.g., :code:`cudaq::sample_async` function in the above code snippet). 

One can then specify the target multi-QPU architecture (:code:`nvidia-mqpu`) with the :code:`--target` flag:
 
.. code-block:: console 

    nvq++ sample_async.cpp -target nvidia-mqpu
    ./a.out

Depending on the number of GPUs available on the system, the :code:`nvidia-mqpu` platform will create the same number of virtual QPU instances.
For example, on a system with 4 GPUs, the above code will distribute the four sampling tasks among those :code:`GPUEmulatedQPU` instances.

The results might look like the following (4 different random samplings).

.. code-block:: console 
  
    Number of QPUs: 4
    { 10011:28 01100:28 ... }
    { 10011:37 01100:25 ... }
    { 10011:29 01100:25 ... }
    { 10011:33 01100:30 ... }

.. note:: 

  By default, the :code:`nvidia-mqpu` platform will utilize all available GPUs (number of QPUs instances is equal to the number of GPUs).
  To specify the number QPUs to be instantiated, one can set the :code:`CUDAQ_MQPU_NGPUS` environment variable.
  For example, :code:`export CUDAQ_MQPU_NGPUS=2` to specify that only 2 QPUs (GPUs) are needed.


An equivalent example in Python is as follows.

.. literalinclude:: ../../snippets/python/using/cudaq/platform/sample_async.py
    :language: python
    :start-after: [Begin Documentation]

Asynchronous expectation value computations
+++++++++++++++++++++++++++++++++++++++++++

One typical use case of the :code:`nvidia-mqpu` platform is to distribute the 
expectation value computations of a multi-term Hamiltonian across multiple virtual QPUs (:code:`GPUEmulatedQPU`).

Here is an example.

.. literalinclude:: ../../snippets/cpp/using/cudaq/platform/observe_mqpu.cpp
    :language: cpp
    :start-after: [Begin Documentation]
    :end-before: [End Documentation]


One can then target the :code:`nvidia-mqpu` platform by:

.. code-block:: console 

    nvq++ observe_mqpu.cpp -target nvidia-mqpu
    ./a.out

Equivalently, in Python

.. literalinclude:: ../../snippets/python/using/cudaq/platform/observe_mqpu.py
    :language: python
    :start-after: [Begin Documentation]

In the above code snippet, since the Hamiltonian contains four non-identity terms, there are four quantum circuits that need to be executed
in order to compute the expectation value of that Hamiltonian and given the quantum state prepared by the ansatz kernel. When the :code:`nvidia-mqpu` platform
is selected, these circuits will be distributed across all available QPUs. The final expectation value result is computed from all QPU execution results.

Parallel distribution mode
++++++++++++++++++++++++++

The CUDA Quantum :code:`nvidia-mqpu` platform supports two modes of parallel distribution of expectation value computation:

* MPI: distribute the expectation value computations across available MPI ranks and GPUs for each Hamiltonian term.
* Thread: distribute the expectation value computations among available GPUs via standard C++ threads (each thread handles one GPU).

For instance, if all GPUs are available on a single node, thread-based parallel distribution 
(:code:`cudaq::parallel::thread` in C++ or :code:`cudaq.parallel.thread` in Python, as shown in the above example) is sufficient. 
On the other hand, if one wants to distribute the tasks across GPUs on multiple nodes, e.g., on a compute cluster, MPI distribution mode
should be used.

An example of MPI distribution mode usage is as follows:

C++
^^^

.. literalinclude:: ../../snippets/cpp/using/cudaq/platform/observe_mqpu_mpi.cpp
    :language: cpp
    :start-after: [Begin Documentation]
    :end-before: [End Documentation]

.. code-block:: console 

    nvq++ observe_mqpu_mpi.cpp -target nvidia-mqpu
    mpirun -np <N> a.out


Python
^^^^^^

.. literalinclude:: ../../snippets/python/using/cudaq/platform/observe_mqpu_mpi.py
    :language: python
    :start-after: [Begin Documentation]

.. code-block:: console 

    mpirun -np <N> python3 observe_mpi.py

In the above examples, the parallel distribution mode was set to :code:`mpi` using :code:`cudaq::parallel::mpi` in C++ or :code:`cudaq.parallel.mpi` in Python.
CUDA Quantum provides MPI utility functions to initialize, finalize, or query (rank, size, etc.) the MPI runtime. 
Last but not least, the compiled executable (C++) or Python script needs to be launched with an appropriate MPI command, 
e.g., :code:`mpirun`, :code:`mpiexec`, :code:`srun`, etc. 
