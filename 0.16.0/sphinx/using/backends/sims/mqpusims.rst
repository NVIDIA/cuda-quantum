
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
For example, on a system with 4 GPUs, the above code will distribute the four sampling tasks among those virtual QPU instances.

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

Since the underlying virtual QPU is a simulator backend, we can also retrieve the state vector from each
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

.. _multi-node-mqpu:

Multi-QPU with Multi-Node Multi-GPU Backends
+++++++++++++++++++++++++++++++++++++++++++++++

Some simulator backends, such as :code:`tensornet` and :code:`nvidia` with the :code:`mgpu` option,
can distribute a single simulation across multiple GPUs on multiple nodes using MPI.
This section shows how to run *multiple independent simulations in parallel* within the same MPI program,
where each simulation uses its own dedicated group of MPI ranks and GPUs.

The approach is straightforward:

1. Partition the global MPI communicator into `sub-communicators <https://www.mpich.org/static/docs/latest/www3/MPI_Comm_split.html>`__ — one per QPU group.
2. Pass each sub-communicator to CUDA-Q so the underlying simulator uses only the ranks in that group.
3. Each QPU group then calls CUDA-Q APIs independently.

.. note::

    This replaces the former :code:`remote-mqpu` target, which required standing up HTTP servers and had
    higher overhead. With direct MPI distribution, users leverage the job scheduler's resource binding
    (e.g., `SLURM's <https://slurm.schedmd.com/mc_support.html>`__ :code:`--ntasks-per-node`) and have
    full control over rank placement.

Once the communicator is set, all standard CUDA-Q execution APIs —
:code:`sample`, :code:`observe`, :code:`run`, and :code:`get_state` — work as usual within each QPU group.

In all examples below, 4 MPI ranks are divided into 2 QPU groups of 2 ranks each (2 GPUs per
simulation).

Using the CUDA-Q MPI API
^^^^^^^^^^^^^^^^^^^^^^^^^^

For users new to MPI, CUDA-Q provides its own :code:`cudaq::mpi` (C++) and :code:`cudaq.mpi`
(Python) helpers that wrap the common MPI operations needed to set up QPU groups.

.. tab:: Python

    .. literalinclude:: ../../../examples/python/mpi/sample_cudaq_mpi.py
        :language: python
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        mpirun -n 4 python3 sample_cudaq_mpi.py
        QPU 0 (40 qubits):
        { 0000000000000000000000000000000000000000:489 1111111111111111111111111111111111111111:511 }
        QPU 1 (45 qubits):
        { 000000000000000000000000000000000000000000000:495 111111111111111111111111111111111111111111111:505 }

.. tab:: C++

    .. literalinclude:: ../../../examples/cpp/mpi/sample_cudaq_mpi.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        nvq++ --target tensornet -o sample_cudaq_mpi sample_cudaq_mpi.cpp
        mpirun -n 4 ./sample_cudaq_mpi
        QPU 0 (40 qubits):
        { 0000000000000000000000000000000000000000:483 1111111111111111111111111111111111111111:517 }
        QPU 1 (45 qubits):
        { 000000000000000000000000000000000000000000000:501 111111111111111111111111111111111111111111111:499 }

Using Native MPI APIs
^^^^^^^^^^^^^^^^^^^^^^^

Users who already manage MPI in their application (e.g. with :code:`<mpi.h>` in C++ or
`mpi4py <https://mpi4py.readthedocs.io/>`__ in Python) can split the communicator using
those libraries directly and hand the result to CUDA-Q.

.. tab:: Python

    The sub-communicator handle is passed to CUDA-Q via the :code:`comm_handle` keyword argument
    of :code:`cudaq.set_target()`.

    .. literalinclude:: ../../../examples/python/mpi/sample.py
        :language: python
        :start-after: [Begin Documentation]

    .. code-block:: console

        mpirun -n 4 python3 sample.py
        QPU 0 (40 qubits):
        { 0000000000000000000000000000000000000000:489 1111111111111111111111111111111111111111:511 }
        QPU 1 (45 qubits):
        { 000000000000000000000000000000000000000000000:495 111111111111111111111111111111111111111111111:505 }

.. tab:: C++

    .. literalinclude:: ../../../examples/cpp/mpi/sample.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        nvq++ --target tensornet -o sample sample.cpp \
            -I$(mpicc -showme:incdirs) -L$(mpicc -showme:libdirs) -lmpi
        mpirun -n 4 ./sample
        QPU 0 (40 qubits):
        { 0000000000000000000000000000000000000000:483 1111111111111111111111111111111111111111:517 }
        QPU 1 (45 qubits):
        { 000000000000000000000000000000000000000000000:501 111111111111111111111111111111111111111111111:499 }

.. note::

    When using native MPI APIs, the pointer passed to :code:`cudaq::mpi::set_communicator` must
    refer to a live :code:`MPI_Comm` object. Keep it alive for the duration of all CUDA-Q calls
    in that QPU group and free it with :code:`MPI_Comm_free` once simulation is complete.

.. note::

    MPI must be initialized before calling :code:`cudaq::mpi::set_communicator`,
    :code:`cudaq.mpi.set_communicator`, or :code:`cudaq.set_target` with a :code:`comm_handle`.
    CUDA-Q will raise an error if MPI has not been initialized. Refer to
    :ref:`distributed-computing-with-mpi` for instructions on enabling MPI support in CUDA-Q.

Gradient Computation for VQE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common application of the multi-QPU pattern is computing the energy gradient
for the `Variational Quantum Eigensolver (VQE)
<https://www.nature.com/articles/ncomms5213>`__.
Most gradient rules (e.g. parameter-shift) evaluate each gradient component
independently via one or more :code:`observe` calls per parameter, which maps
naturally onto the multi-QPU model: assign one QPU group per parameter and all
gradient components are evaluated in parallel.
The pattern scales linearly — a circuit with :math:`N` variational parameters
requires :math:`N` QPU groups and :math:`N \times` :code:`ranks_per_qpu` total
MPI ranks, with no code changes beyond launching more ranks and providing more
initial parameters.

.. note::

    This MPI-based approach offers two advantages over the thread-based
    multi-QPU mode (:ref:`mqpu-platform`):

    * **Larger problems**: increasing :code:`ranks_per_qpu` assigns more GPUs
      to each virtual QPU, allowing each :code:`observe` call to simulate
      circuits that exceed the memory of a single GPU.
    * **Arbitrary cluster scale**: QPU groups span across nodes via MPI,
      so the total number of virtual QPUs is limited only by the cluster size,
      not by the number of GPUs on a single node.

The examples below use a 40-qubit placeholder ansatz and Hamiltonian to
illustrate the distribution pattern. To use this in a real VQE workflow,
substitute your application-specific ansatz and physical Hamiltonian (e.g.
generated from a chemistry package such as ``PySCF``), wrap the gradient evaluation
in an optimization loop (e.g. using :code:`cudaq.optimizers`), and feed the
gathered gradient back to the optimizer at each iteration.

.. tab:: Python (cudaq.mpi)

    .. literalinclude:: ../../../examples/python/mpi/observe_gradient_cudaq_mpi.py
        :language: python
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        mpirun -n 4 python3 observe_gradient_cudaq_mpi.py
        Gradient: [-9.588510772084065, -5.91040413322679]

.. tab:: Python (mpi4py)

    .. literalinclude:: ../../../examples/python/mpi/observe_gradient.py
        :language: python
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        mpirun -n 4 python3 observe_gradient.py
        Gradient: [-9.588510772084065, -5.91040413322679]

.. tab:: C++ (cudaq::mpi)

    .. literalinclude:: ../../../examples/cpp/mpi/observe_gradient_cudaq_mpi.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        nvq++ --target tensornet -o observe_gradient_cudaq_mpi observe_gradient_cudaq_mpi.cpp
        mpirun -n 4 ./observe_gradient_cudaq_mpi
        Gradient: [-9.588511, -5.910404]

.. tab:: C++ (native MPI)

    .. literalinclude:: ../../../examples/cpp/mpi/observe_gradient.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        nvq++ --target tensornet -o observe_gradient observe_gradient.cpp \
            -I$(mpicc -showme:incdirs) -L$(mpicc -showme:libdirs) -lmpi
        mpirun -n 4 ./observe_gradient
        Gradient: [-9.588511, -5.910404]
