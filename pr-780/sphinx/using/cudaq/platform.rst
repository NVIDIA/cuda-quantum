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
    for (auto& counts : countsFutures) {
      counts.get().dump();
    }

CUDA Quantum exposes asynchronous versions of the default :code:`cudaq::` algorithmic
primitive functions like :code:`sample` and :code:`observe`. 