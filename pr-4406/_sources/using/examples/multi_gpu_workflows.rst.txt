Multi-GPU Workflows
===================

There are many backends available with CUDA-Q that enable seamless
switching between GPUs, QPUs and CPUs and also allow for workflows
involving multiple architectures working in tandem. This page will walk through the simple steps to accelerate any quantum circuit simulation with a GPU and how to scale large simulations using multi-GPU multi-node capabilities.



From CPU to GPU
------------------

The code below defines a kernel that creates a GHZ state using :math:`N` qubits. 
   
.. literalinclude:: ../../snippets/python/using/examples/multi_gpu_workflows/multiple_targets.py
    :language: python
    :start-after: [Begin state]
    :end-before: [End state]

You can run a state vector simulation using your CPU with the :code:`qpp-cpu` backend. This is helpful for debugging code and testing small circuits.

.. literalinclude:: ../../snippets/python/using/examples/multi_gpu_workflows/multiple_targets.py
    :language: python
    :start-after: [Begin CPU]
    :end-before: [End CPU]

.. parsed-literal::

    { 00:475 11:525 }

As the number of qubits increases to even modest size, the CPU simulation will become impractically slow.  By switching to the :code:`nvidia` backend, you can accelerate the same code on a single GPU and achieve a speedup of up to **425x**.  If you have a GPU available, this the default backend to ensure maximum productivity.

.. literalinclude:: ../../snippets/python/using/examples/multi_gpu_workflows/multiple_targets.py
    :language: python
    :start-after: [Begin GPU]
    :end-before: [End GPU]

.. parsed-literal::

    { 0000000000000000000000000:510 1111111111111111111111111:490 }


Pooling the memory of multiple GPUs (`mgpu`)
---------------------------------------------


As :code:`N` gets larger, the size of the state vector that needs to be stored in memory increases exponentially. 
The state vector has :math:`2^N` elements, each a complex number requiring 8 bytes. This means a 30 qubit simulation 
requires roughly 8 GB. Adding a few more qubits will quickly exceed the memory of as single GPU.  The `mqpu` backend 
solved this problem by pooling the memory of multiple GPUs across multiple nodes to perform a single state vector simulation. 


.. image:: images/mgpu.png


If you have multiple GPUs, you can use the following command to run the simulation across :math:`n` GPUs. 


:code:`mpiexec -np n python3 program.py --target nvidia --target-option mgpu`

This code will execute in an MPI context and provide additional memory to simulation much larger state vectors.

You can also set :code:`cudaq.set_target('nvidia', option='mgpu')` within the file to select the target.


Parallel execution over multiple QPUs (`mqpu`)
------------------------------------------------

Batching Hamiltonian Terms 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple GPUs can also come in handy for cases where applications might benefit from multiple QPUs running in parallel.  The `mqpu` backend uses multiple GPUs to simulate QPUs so you can accelerate quantum applications with parallelization.


.. image:: images/mqpu.png

The most simple example is Hamiltonian Batching. In this case, an expectation value of a large Hamiltonian is distributed across multiple simulated QPUs, where each QPUs evaluates a subset of the Hamiltonian terms. 



.. image:: ../../applications/python/images/hsplit.png


The code below evaluates the expectation value of a random 100000 term Hamiltonian. A standard :code:`observe` call will run the program on a single GPU.  Adding the argument :code:`execution=cudaq.parallel.thread` or :code:`execution=cudaq.parallel.mpi` will automatically distribute the Hamiltonian terms across multiple GPUs on a single node or multiple GPUs on multiple nodes, respectively.  

The code is executed with :code:`mpiexec -np n python3 program.py --target nvidia --target-option mqpu` where :math:`n` is the number of GPUs available. 

   
.. literalinclude:: ../../snippets/python/using/examples/multi_gpu_workflows/hamiltonian_batching.py
    :language: python
    :start-after: [Begin Docs]
    :end-before: [End Docs]



Circuit Batching 
^^^^^^^^^^^^^^^^^

A second way to leverage the `mqpu` backend is to batch circuit evaluations across multiple simulated QPUs.   



.. image:: ../../applications/python/images/circsplit.png


One example where circuit batching is helpful might be evaluating a parameterized circuit many times with different parameters. The code below prepares a list of 10000 parameter sets for a 5 qubit circuit. 


.. literalinclude:: ../../snippets/python/using/examples/multi_gpu_workflows/circuit_batching.py
    :language: python
    :start-after: [Begin prepare]
    :end-before: [End prepare]

All of these circuits can be broadcast through a single :code"`observe` call and run by default on a single GPU.  The code below times this entire process.

.. literalinclude:: ../../snippets/python/using/examples/multi_gpu_workflows/circuit_batching.py
    :language: python
    :start-after: [Begin single]
    :end-before: [End single]

.. parsed-literal::

    3.185340642929077

This can be greatly accelerated by batching the circuits on multiple QPUs. The first step is to slice the large list of parameters unto smaller arrays. The example below divides by four, in preparation to run on four GPUs.

.. literalinclude:: ../../snippets/python/using/examples/multi_gpu_workflows/circuit_batching.py
    :language: python
    :start-after: [Begin split]
    :end-before: [End split]

.. parsed-literal::

    There are now 10000 parameter sets split into 4 batches of 2500 , 2500 , 2500 , 2500


As the results are run asynchronously, they need to be stored in a list (:code:`asyncresults`) and retrieved later with the :code:`get` command. The following loops over the parameter batches, and the sets of parameters in each batch. The parameter sets are provided as inputs to :code:`observe_async` along with specification of a :code:`qpu_id` which designates the GPU (of the four available) which will run computation. A speedup of up to 4x can be expected with results varying by problem size.


.. literalinclude:: ../../snippets/python/using/examples/multi_gpu_workflows/circuit_batching.py
    :language: python
    :start-after: [Begin multiple]
    :end-before: [End multiple]

.. parsed-literal::

    1.1754660606384277


    
Multi-QPU + Other Backends (`remote-mqpu`)
-------------------------------------------
    
    
The `mqpu` backend can be extended so that each parallel simulated QPU run backends other than :code:`nvidia`.  This provides a way to simulate larger scale circuits and execute parallel algorithms. This accomplished by launching remotes servers which each simulated a QPU.  
The code example below demonstrates this using the :code:`tensornet-mps` backend which allows sampling of a 40 qubit circuit too larger for state vector simulation. In this case, the target is specified as :code:`remote-mqpu` while an additional :code:`backend` is specified for the simulator used for each QPU.  

The default approach uses one GPU per QPU and can both launch and close each server automatically. This is accomplished by specifying :code:`auto_launch` and :code"`url` within :code:`cudaq.set_target`.  Running the script below will then sample the 40 qubit circuit using two QPUs each running :code:`tensornet-mps`.  
    
.. code:: python
  
        import cudaq

        backend = 'tensornet-mps'

        servers = '2'

        @cudaq.kernel
        def kernel(controls_count: int):
            controls = cudaq.qvector(controls_count)
            targets = cudaq.qvector(40)
            # Place controls in superposition state.
            h(controls)
            for target in range(40):
                x.ctrl(controls, targets[target])
            # Measure.
            mz(controls)
            mz(targets)

        # Set the target to execute on and query the number of QPUs in the system;
        # The number of QPUs is equal to the number of (auto-)launched server instances.
        cudaq.set_target("remote-mqpu",
                         backend=backend,
                         auto_launch=str(servers) if servers.isdigit() else "",
                         url="" if servers.isdigit() else servers)
        qpu_count = cudaq.get_target().num_qpus()
        print("Number of virtual QPUs:", qpu_count)

        # We will launch asynchronous sampling tasks,
        # and will store the results as a future we can query at some later point.
        # Each QPU (indexed by an unique Id) is associated with a remote REST server.
        count_futures = []
        for i in range(qpu_count):

            result = cudaq.sample_async(kernel, i + 1, qpu_id=i)
            count_futures.append(result)
        print("Sampling jobs launched for asynchronous processing.")

        # Go do other work, asynchronous execution of sample tasks on-going.
        # Get the results, note future::get() will kick off a wait
        # if the results are not yet available.
        for idx in range(len(count_futures)):
            counts = count_futures[idx].get()
            print(counts)

:code:`remote-mqpu` can also be used with `mqpu`, allowing each QPU to be simulated by multiple GPUs. 
This requires manual preparation of the servers and detailed instructions are in the :ref:`remote multi-QPU platform <remote-mqpu-platform>`
section of the docs.            
