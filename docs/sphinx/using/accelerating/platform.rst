Taking Advantage of the Underlying Quantum Platform
---------------------------------------------------
The CUDA Quantum machine model elucidates the various devices considered in the 
broader quantum-classical compute node context. Programmers will have one or many 
host CPUs, zero or many NVIDIA GPUs, a classical QPU control space, and the
quantum register itself. Moreover, the :doc:`specification </specification/cudaq/platform>`
notes that the underlying platform may expose multiple QPUs. In the near-term,
this will be unlikely with physical QPU instantiations, but the availability of
GPU-based circuit simulators on NVIDIA multi-GPU architectures does give one an
opportunity to think about programming such a multi-QPU architecture in the near-term.
CUDA Quantum starts by enabling one to query information about the underlying quantum
platform via the :code:`quantum_platform` abstraction. This type exposes a
:code:`num_qpus()` method that can be used to query the number of available
QPUs for asynchronous CUDA Quantum kernel and :code:`cudaq::` function invocations.
Each available QPU is assigned a logical index, and programmers can launch
specific asynchronous function invocations targeting a desired QPU.


NVIDIA `MQPU` Platform
++++++++++++++++++++++

The NVIDIA `MQPU` target (:code:`nvidia-mqpu`) provides a simulated QPU for every available NVIDIA GPU on the underlying system. 
Each QPU is simulated via a `cuStateVec` simulator backend. For more information about using multiple GPUs 
to simulate each virtual QPU, or using a different backend for virtual QPUs, please see :ref:`remote MQPU platform <remote-mqpu-platform>`.
This target enables asynchronous parallel execution of quantum kernel tasks.

Here is a simple example demonstrating its usage.

.. tab:: Python

    .. literalinclude:: ../../snippets/python/using/cudaq/platform/sample_async.py
        :language: python
        :start-after: [Begin Documentation]

.. tab:: C++

    .. literalinclude:: ../../snippets/cpp/using/cudaq/platform/sample_async.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]


    One can specify the target multi-QPU architecture (:code:`nvidia-mqpu`) with the :code:`--target` flag:
    
    .. code-block:: console

        nvq++ sample_async.cpp -target nvidia-mqpu
        ./a.out

CUDA Quantum exposes asynchronous versions of the default :code:`cudaq` algorithmic
primitive functions like :code:`sample` and :code:`observe` (e.g., :code:`sample_async` function in the above code snippets).

Depending on the number of GPUs available on the system, the :code:`nvidia-mqpu` platform will create the same number of virtual QPU instances.
For example, on a system with 4 GPUs, the above code will distribute the four sampling tasks among those :code:`GPUEmulatedQPU` instances.

The results might look like the following 4 different random samplings:

.. code-block:: console
  
    Number of QPUs: 4
    { 10011:28 01100:28 ... }
    { 10011:37 01100:25 ... }
    { 10011:29 01100:25 ... }
    { 10011:33 01100:30 ... }

.. note::

  By default, the :code:`nvidia-mqpu` platform will utilize all available GPUs (number of QPUs instances is equal to the number of GPUs).
  To specify the number QPUs to be instantiated, one can set the :code:`CUDAQ_MQPU_NGPUS` environment variable.
  For example, use :code:`export CUDAQ_MQPU_NGPUS=2` to specify that only 2 QPUs (GPUs) are needed.

Asynchronous expectation value computations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One typical use case of the :code:`nvidia-mqpu` platform is to distribute the
expectation value computations of a multi-term Hamiltonian across multiple virtual QPUs (:code:`GPUEmulatedQPU`).

Here is an example.

.. tab:: Python

    .. literalinclude:: ../../snippets/python/using/cudaq/platform/observe_mqpu.py
        :language: python
        :start-after: [Begin Documentation]

.. tab:: C++

    .. literalinclude:: ../../snippets/cpp/using/cudaq/platform/observe_mqpu.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]


    One can then target the :code:`nvidia-mqpu` platform by executing the following commands:

    .. code-block:: console

        nvq++ observe_mqpu.cpp -target nvidia-mqpu
        ./a.out

In the above code snippets, since the Hamiltonian contains four non-identity terms, there are four quantum circuits that need to be executed
in order to compute the expectation value of that Hamiltonian and given the quantum state prepared by the ansatz kernel. When the :code:`nvidia-mqpu` platform
is selected, these circuits will be distributed across all available QPUs. The final expectation value result is computed from all QPU execution results.

Parallel distribution mode
^^^^^^^^^^^^^^^^^^^^^^^^^^

The CUDA Quantum :code:`nvidia-mqpu` platform supports two modes of parallel distribution of expectation value computation:

* MPI: distribute the expectation value computations across available MPI ranks and GPUs for each Hamiltonian term.
* Thread: distribute the expectation value computations among available GPUs via standard C++ threads (each thread handles one GPU).

For instance, if all GPUs are available on a single node, thread-based parallel distribution 
(:code:`cudaq::parallel::thread` in C++ or :code:`cudaq.parallel.thread` in Python, as shown in the above example) is sufficient.
On the other hand, if one wants to distribute the tasks across GPUs on multiple nodes, e.g., on a compute cluster, MPI distribution mode
should be used.

An example of MPI distribution mode usage in both C++ and Python is given below:

.. tab:: Python

    .. literalinclude:: ../../snippets/python/using/cudaq/platform/observe_mqpu_mpi.py
        :language: python
        :start-after: [Begin Documentation]

    .. code-block:: console

        mpiexec -np <N> python3 file.py

.. tab:: C++

    .. literalinclude:: ../../snippets/cpp/using/cudaq/platform/observe_mqpu_mpi.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    .. code-block:: console

        nvq++ file.cpp -target nvidia-mqpu
        mpiexec -np <N> a.out

In the above example, the parallel distribution mode was set to :code:`mpi` using :code:`cudaq::parallel::mpi` in C++ or :code:`cudaq.parallel.mpi` in Python.
CUDA Quantum provides MPI utility functions to initialize, finalize, or query (rank, size, etc.) the MPI runtime. 
Last but not least, the compiled executable (C++) or Python script needs to be launched with an appropriate MPI command, 
e.g., :code:`mpiexec`, :code:`mpirun`, :code:`srun`, etc.

.. _remote-mqpu-platform:

Remote `MQPU` Platform
+++++++++++++++++++++++++++

As shown in the above examples, the :code:`nvidia-mqpu` platform enables
multi-QPU distribution whereby each QPU is simulated by a :ref:`single NVIDIA GPU <cuQuantum single-GPU>`.
To run multi-QPU workloads on different simulator backends, one can use the :code:`remote-mqpu` platform,
which encapsulates simulated QPUs as independent HTTP REST server instances. 
The following code illustrates how to launch asynchronous sampling tasks on multiple virtual QPUs, 
each simulated by a `tensornet` simulator backend.

.. tab:: Python

    .. literalinclude:: ../../snippets/python/using/cudaq/platform/sample_async_remote.py
        :language: python
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

.. tab:: C++

    .. literalinclude:: ../../snippets/cpp/using/cudaq/platform/sample_async_remote.cpp
        :language: cpp
        :start-after: [Begin Documentation]
        :end-before: [End Documentation]

    The code above is saved in `sample_async.cpp` and compiled with the following command, targeting the :code:`remote-mqpu` platform:

    .. code-block:: console

        nvq++ sample_async.cpp -o sample_async.x --target remote-mqpu --remote-mqpu-backend tensornet --remote-mqpu-auto-launch 2
        ./sample_async.x

In the above code snippets, the :code:`remote-mqpu` platform was used in the auto-launch mode,
whereby a specific number of server instances, i.e., virtual QPUs, are launched on the local machine
in the background. The remote QPU daemon service, :code:`cudaq-qpud`, will also be shut down automatically
at the end of the session.

.. note:: 
    By default, auto launching daemon services do not support MPI parallelism.
    Hence, using the `nvidia-mgpu` backend to simulate each virtual QPU requires 
    manually launching each server instance. How to do that is explained in the rest of this section.

.. _custom_remote_qpud_launch:

To customize how many and which GPUs are used for simulating each virtual QPU, one can launch each server manually.
For instance, on a machine with 8 NVIDIA GPUs, one may wish to partition those GPUs into
4 virtual QPU instances, each manages 2 GPUs. To do so, first launch a :code:`cudaq-qpud` server for each virtual QPU:


.. code-block:: console
    
    CUDA_VISIBLE_DEVICES=0,1 mpiexec -np 2 cudaq-qpud --port <QPU 1 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=2,3 mpiexec -np 2 cudaq-qpud --port <QPU 2 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=4,5 mpiexec -np 2 cudaq-qpud --port <QPU 3 TCP/IP port number>
    CUDA_VISIBLE_DEVICES=6,7 mpiexec -np 2 cudaq-qpud --port <QPU 4 TCP/IP port number>


In the above code snippet, four :code:`nvidia-mgpu` daemons are started in MPI context via the :code:`mpiexec` launcher.
This activates MPI runtime environment required by the :code:`nvidia-mgpu` backend. Each QPU daemon is assigned a unique 
TCP/IP port number via the :code:`--port` command-line option. The :code:`CUDA_VISIBLE_DEVICES` environment variable restricts the GPU devices 
that each QPU daemon sees so that it targets specific GPUs. 

With these invocations, each virtual QPU is locally addressable at the URL `localhost:<port>`. 

.. warning:: 

    There is no authentication required to communicate with this server app. 
    Hence, please make sure to either (1) use a non-public TCP/IP port for internal use or 
    (2) use firewalls or other security mechanisms to manage user access. 

User code can then target these QPUs for multi-QPU workloads, such as asynchronous sample or observe shown above for the :code:`nvidia-mqpu` platform.

.. tab:: Python

     .. code:: python 

        cudaq.set_target("remote-mqpu", url="localhost:<port1>,localhost:<port2>,localhost:<port3>,localhost:<port4>", backend="nvidia-mgpu")
        
.. tab:: C++

    .. code-block:: console

        nvq++ distributed.cpp --target remote-mqpu --remote-mqpu-url localhost:<port1>,localhost:<port2>,localhost:<port3>,localhost:<port4> --remote-mqpu-backend nvidia-mgpu
    

Each URL is treated as an independent QPU, hence the number of QPUs (:code:`num_qpus()`) is equal to the number of URLs provided. 
The multi-node multi-GPU simulator backend (:code:`nvidia-mgpu`) is requested via the :code:`--remote-mqpu-backend` command-line option.

.. note:: 

    The requested backend (:code:`nvidia-mgpu`) will be executed inside the context of the QPU daemon service, thus 
    inherits its GPU resource allocation (two GPUs per backend simulator instance). 

Supported Kernel Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^

The platform serializes kernel invocation to QPU daemons via REST APIs. 
Please refer to the `Open API Docs <../../openapi.html>`_  for the latest API information.
Runtime arguments are serialized into a flat memory buffer (`args` field of the request JSON). 
For more information about argument type serialization, please see :ref:`the table below <type_serialization_table>`.

When using a remote backend to simulate each virtual QPU, 
by default, we currently do not support passing complex data structures, 
such as nested vectors or class objects, or other kernels as arguments to the entry point kernels.
These type limitations only apply to the **entry-point** kernel and not when passing arguments
to other quantum kernels.

Support for the full range of argument types within CUDA Quantum can be enabled by compiling the 
code with the :code:`--enable-mlir` option. This flag forces quantum kernels to be compiled with 
the CUDA Quantum MLIR-based compiler. As a result, runtime arguments can be resolved by the CUDA 
Quantum compiler infrastructure to support wider range of argument types. However, certain
language constructs within quantum kernels may not yet be fully supported.

.. _type_serialization_table:

.. list-table:: Kernel argument serialization
   :widths: 50 50 50
   :header-rows: 1

   * - Data type
     - Example
     - Serialization
   * -  Trivial type (occupies a contiguous memory area)
     -  `int`, `std::size_t`, `double`, etc.
     - Byte data (via `memcpy`)
   * - `std::vector` of trivial type
     - `std::vector<int>`, `std::vector<double>`, etc. 
     - Total vector size in bytes as a 64-bit integer followed by serialized data of all vector elements.

