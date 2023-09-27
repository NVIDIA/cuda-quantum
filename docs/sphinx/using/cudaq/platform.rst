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

.. code-block:: cpp 

    auto kernelToBeSampled = [](int runtimeParam) __qpu__ {
      cudaq::qreg q(runtimeParam);
      h(q);
      mz(q);
    };

    // Get the quantum_platform singleton
    auto& platform = cudaq::get_platform();

    // Query the number of QPUs in the system
    auto num_qpus = platform.num_qpus();
    printf("Number of QPUs: %zu\n", num_qpus);
    // We will launch asynchronous sampling tasks
    // and will store the results immediately as a future 
    // we can query at some later point
    std::vector<cudaq::async_sample_result> countFutures;
    for (std::size_t i = 0; i < num_qpus; i++) {
      countFutures.emplace_back(cudaq::sample_async(i, kernelToBeSampled, 5 /*runtimeParam*/));
    }

    // 
    // Go do other work, asynchronous execution of sample tasks on-going
    // 

    // Get the results, note future::get() will kick off a wait
    // if the results are not yet available.
    for (auto& counts : countFutures) {
      counts.get().dump();
    }

CUDA Quantum exposes asynchronous versions of the default :code:`cudaq::` algorithmic
primitive functions like :code:`sample` and :code:`observe` (e.g., :code:`cudaq::sample_async` function in the above code snippet). 

One can then specify the target multi-QPU architecture (:code:`nvidia-mqpu`) with the :code:`--target` flag:
 
.. code-block:: console 

    nvq++ simple.cpp -target nvidia-mqpu
    ./a.out

Depending on the number of GPUs available on the system, :code:`nvidia-mqpu` platform will create the same number of virtual QPU instances.

For example, on a system that has 4 GPUs, the above code will distribute the 4 sampling tasks among those :code:`GPUEmulatedQPU` instances.

The results might look like the followings (4 different random samplings).

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


An equivalent example in Python is as follows:

.. code:: python

    import cudaq

    cudaq.set_target("nvidia-mqpu")
    target = cudaq.get_target()
    num_qpus = target.num_qpus()
    print("Number of QPUs:", num_qpus)

    kernel, runtime_param = cudaq.make_kernel(int)
    qubits = kernel.qalloc(runtime_param)
    # Place qubits in superposition state.
    kernel.h(qubits)
    # Measure.
    kernel.mz(qubits)

    count_futures = []
    for qpu in range(num_qpus):
        count_futures.append(cudaq.sample_async(kernel, 5, qpu_id=qpu))


    for counts in count_futures:
      print(counts.get())

Asynchronous expectation value computations
+++++++++++++++++++++++++++++++++++++++++++

One typical use case of the :code:`nvidia-mqpu` platform is to distribute the 
expectation value computations of a multi-term Hamiltonian across multiple virtual QPUs (:code:`GPUEmulatedQPU`).

Here is an example.

.. code-block:: cpp 
    using namespace cudaq::spin;
    cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                      .21829 * z(0) - 6.125 * z(1);

    // Get the quantum_platform singleton
    auto& platform = cudaq::get_platform();

    // Query the number of QPUs in the system
    auto num_qpus = platform.num_qpus();
    printf("Number of QPUs: %zu\n", num_qpus);

    auto ansatz = [](double theta) __qpu__ {
      cudaq::qubit q, r;
      x(q);
      ry(theta, r);
      x<cudaq::ctrl>(r, q);
    };

    double result = cudaq::observe<cudaq::parallel::thread>(ansatz, h, 0.59);
    printf("Expectation value: %lf\n", result);


One can then target the :code:`nvidia-mqpu` platform by:

.. code-block:: console 

    nvq++ observe.cpp -target nvidia-mqpu
    ./a.out

Equivalently, in Python

.. code:: python

    import cudaq
    from cudaq import spin
    cudaq.set_target("nvidia-mqpu")
    target = cudaq.get_target()
    num_qpus = target.num_qpus()
    print("Number of QPUs:", num_qpus)

    # Define spin ansatz.
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])
    # Define spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)


    exp_val = cudaq.observe(kernel, hamiltonian, 0.59, execution=cudaq.parallel.thread).expectation_z()
    print("Expectation value: ", exp_val)

In the above code snippet, since the Hamiltonian contains four non-identity terms, there are four quantum circuits that need to be executed
in order to compute the expectation value of that Hamiltonian given the quantum state prepared by the ansatz kernel. When the :code:`nvidia-mqpu` platform
is selected, these circuits will be distributed across all available QPUs. The final expectation value result is computed from all QPU execution results.

Parallel distribution mode
++++++++++++++++++++++++++

The CUDA Quantum :code:`nvidia-mqpu` platform supports two modes of parallel distribution of expectation value computation:

* MPI: distribute the expectation value computations across available MPI ranks and GPUs for each Hamiltonian term.
* Thread: distribute the expectation value computations available GPUs via standard C++ threads (each thread handles one GPU).

For instance, if all of the GPUs are available on a single node, thread-based parallel distribution 
(:code:`cudaq::parallel::thread` in C++ or :code:`cudaq.parallel.thread` in Python as shown in the above example) is sufficient. 
On the other hand, if one wants to distribute the tasks across GPUs on multiple nodes, e.g., on HPC clusters, MPI distribution mode
should be used.

An example of MPI distribution mode usage is as follows:

C++
^^^

.. code-block:: cpp 

    #include "cudaq.h"

    int main() {
      cudaq::mpi::initialize();
      using namespace cudaq::spin;
      cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                        .21829 * z(0) - 6.125 * z(1);

      auto ansatz = [](double theta) __qpu__ {
        cudaq::qubit q, r;
        x(q);
        ry(theta, r);
        x<cudaq::ctrl>(r, q);
      };

      double result = cudaq::observe<cudaq::parallel::mpi>(ansatz, h, 0.59);
      if (cudaq::mpi::rank() == 0)
        printf("Expectation value: %lf\n", result);
      cudaq::mpi::finalize();

      return 0;
    }

.. code-block:: console 

    nvq++ observe.cpp -target nvidia-mqpu
    mpirun -np <N> a.out


Python
^^^^^^

.. code:: python

    import cudaq
    from cudaq import spin

    cudaq.mpi.initialize()
    cudaq.set_target("nvidia-mqpu")

    # Define spin ansatz.
    kernel, theta = cudaq.make_kernel(float)
    qreg = kernel.qalloc(2)
    kernel.x(qreg[0])
    kernel.ry(theta, qreg[1])
    kernel.cx(qreg[1], qreg[0])
    # Define spin Hamiltonian.
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)


    exp_val = cudaq.observe(kernel, hamiltonian, 0.59, execution=cudaq.parallel.mpi).expectation_z()
    if cudaq.mpi.rank() == 0:
        print("Expectation value: ", exp_val)


    cudaq.mpi.finalize()

.. code-block:: console 

    mpirun -np <N> python3 observe_mpi.py

In the above examples, we specified the parallel distribution mode to :code:`mpi` and used CUDA Quantum MPI utility functions 
to initialize, finalize, or query (rank, size, etc.) the MPI runtime. 
Last but not least, the compiled executable (C++) or Python script needs to be launched with an appropriate MPI command, 
e.g., :code:`mpirun`, :code:`mpiexec`, :code:`srun`, etc. 
