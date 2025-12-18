Running your first CUDA-Q Program
----------------------------------------

Now that you have defined your first quantum kernel, let's look at different options for how to execute it.
In CUDA-Q, quantum circuits are stored as quantum kernels. For estimating the probability distribution of 
a measured quantum state in a circuit, we use the ``sample`` function call, for analyzing individual return values 
from multiple executions, we use the ``run`` function call, and for computing the expectation value of a quantum 
state with a given observable, we use the ``observe`` function call.

Sample
++++++++

Quantum states collapse upon measurement and hence need to be sampled many times to gather statistics. The CUDA-Q `sample` call enables this.

.. tab:: Python

  The :func:`cudaq.sample` method takes a kernel and its arguments as inputs, and returns a :class:`cudaq.SampleResult`.

.. tab:: C++

  The `cudaq::sample` method takes a kernel and its arguments as inputs, and returns a `cudaq::SampleResult`.

This result dictionary contains the distribution of measured states for the system.

Continuing with the GHZ kernel defined in :doc:`Building Your First CUDA-Q Program <build_kernel>`,
we will set the concrete value of our `qubit_count` to be two.

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample1]
        :end-before: [End Sample1]

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample1]
        :end-before: [End Sample1]

The code above can be run like any other program:

.. tab:: Python

  Assuming the program is saved in the file `sample.py`, we can execute it with the command

  .. code-block:: console

      python3 sample.py

.. tab:: C++

  Assuming the program is saved in the file `sample.cpp`, we can now compile this file with the `nvq++` toolchain, and then run the compiled executable.

  .. code-block:: console

      nvq++ sample.cpp
      ./a.out

By default, `sample` produces an ensemble of 1000 shots. This can be changed by specifying an integer argument
for the `shots_count`.

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample2]
        :end-before: [End Sample2]

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample2]
        :end-before: [End Sample2]

Note that there is a subtle difference between how sample is executed with the target device set to a simulator or with the target device set to a QPU. When run on a simulator, the quantum state is built once and then sampled repeatedly, where the number of samples is defined by `shots_count`. When executed on quantum hardware, the quantum state collapses upon measurement and hence needs to be rebuilt every time to collect a sample.

A variety of methods can be used to extract useful information from a `SampleResult`. For example,
to return the most probable measurement and its respective probability:

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample3]
        :end-before: [End Sample3]

  See the :doc:`API specification <../../../api/languages/python_api>` for further information.

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample3]
        :end-before: [End Sample3]

  See the :doc:`API specification <../../../api/languages/cpp_api>` for further information.

Sampling a distribution can be a time intensive task. An asynchronous version of sample exists and can be useful to parallelize your application. Asynchronous programming is a technique that enables your program to start a potentially long-running task and still be able to be responsive to other events while that task runs, rather than having to wait until that task has finished. Once that task has finished, your program is presented with the result.

Asynchronous execution allows to easily parallelize execution of multiple kernels on a multi-processor platform. Such a platform
is available, for example, by choosing the target `nvidia-mqpu`:

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin SampleAsync]
        :end-before: [End SampleAsync]

.. note::

  This kind of parallelization is most effective
  if you actually have multiple QPUs or GPUs available. Otherwise, the 
  sampling will still have to execute sequentially due to resource constraints. 

More information about parallelizing execution can be found on the :ref:`mqpu-platform`  page.

Run
+++++++++

The `run` method executes a quantum kernel multiple times and returns each individual result. Unlike `sample`, 
which collects measurement statistics as counts, `run` preserves each individual return value from each 
execution. This is useful when you need to analyze the distribution of returned values which may not be possible from just 
aggregated measurement counts. Additionally, the `run` method also supports returning various types of values 
from the quantum kernel, including scalar types (bool, int, float and their variants) and user-defined data structures.

.. tab:: Python

  The ``cudaq.run`` method takes a kernel and its arguments as inputs and returns a list containing 
  the result values from each execution. The kernel must return a non-void value.

.. tab:: C++

  The ``cudaq::run`` method takes a kernel and its arguments as inputs and returns a `std::vector` containing
  the result values from each execution. The kernel must return a non-void value.

Below is an example of a quantum kernel that creates a GHZ state, measures all qubits, and returns the total 
count of qubits in state :math:`|1\rangle`:

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_run.py
        :language: python
        :start-after: [Begin Run1]
        :end-before: [End Run1]

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_run.cpp
        :language: cpp
        :start-after: [Begin Run1]
        :end-before: [End Run1]

The code above will execute the kernel multiple times (defined by `shots_count`) and return a list of 
individual results. By default, the `shots_count` for `run` is 100.

You can process the results to get statistics or other insights:

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_run.py
        :language: python
        :start-after: [Begin Run2]
        :end-before: [End Run2]

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_run.cpp
        :language: cpp
        :start-after: [Begin Run2]
        :end-before: [End Run2]


.. note::

  Currently, `run` supports kernels returning scalar types (bool, int, float) and custom data structures.

.. note:: 

  When using custom data structures, they must be defined with `slots=True` in Python or as simple aggregates in C++.


Similar to `sample_async`, the `run` API also supports asynchronous execution through `run_async`. 
This is particularly useful for parallelizing execution of multiple kernels on a multi-processor platform:

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_run.py
        :language: python
        :start-after: [Begin RunAsync]
        :end-before: [End RunAsync]

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_run.cpp
        :language: cpp
        :start-after: [Begin RunAsync]
        :end-before: [End RunAsync]

More information about parallelizing execution can be found at the :ref:`mqpu-platform` page.


.. note:: 

  Currently, `run` and `run_async` are supported on simulator targets and select hardware platforms.


Observe
+++++++++

The observe function allows us to calculate expectation values for a defined quantum operator, that is the value of :math:`\bra{\psi}H\ket{\psi}`, where :math:`H` is the desired operator and :math:`\ket{\psi}` is the quantum state after executing a given kernel. 

.. tab:: Python

  The :func:`cudaq.observe` method takes a kernel and its arguments as inputs, along with a :class:`cudaq.operators.spin.SpinOperator`.

  Using the `cudaq.spin` module, operators may be defined as a linear combination of Pauli strings. Functions, such
  as :func:`cudaq.spin.i`, :func:`cudaq.spin.x`, :func:`cudaq.spin.y`, :func:`cudaq.spin.z` may be used to construct more
  complex spin Hamiltonians on multiple qubits.

.. tab:: C++

  The `cudaq::observe` method takes a kernel and its arguments as inputs, along with a `cudaq::spin_op`.

  Operators may be defined as a linear combination of Pauli strings. Functions, such
  as `cudaq::spin_op::i`, `cudaq::spin_op::x`, `cudaq::spin_op::y`, `cudaq::spin_op::z` may be used to construct more
  complex spin Hamiltonians on multiple qubits.

Below is an example of a spin operator object consisting of a `Z(0)` operator, or a Pauli Z-operator on the qubit zero. 
This is followed by the construction of a kernel with a single qubit in an equal superposition. 
The Hamiltonian is printed to confirm that it has been constructed properly.

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_observe.py
        :language: python
        :start-after: [Begin Observe1]
        :end-before: [End Observe1]

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_observe.cpp
        :language: cpp
        :start-after: [Begin Observe1]
        :end-before: [End Observe1]

The `observe` function takes a kernel, any kernel arguments, and a spin operator as inputs and produces an `ObserveResult` object.
The expectation value can be printed using the `expectation` method. 

.. note:: 
  
  It is important to exclude a measurement in the kernel, otherwise the expectation value will be determined from a collapsed 
  classical state. For this example, the expected result of 0.0 is produced.

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_observe.py
        :language: python
        :start-after: [Begin Observe2]
        :end-before: [End Observe2]

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_observe.cpp
        :language: cpp
        :start-after: [Begin Observe2]
        :end-before: [End Observe2]

Unlike `sample`, the default `shots_count` for `observe` is 1. This result is deterministic and equivalent to the
expectation value in the limit of infinite shots.  To produce an approximate expectation value from sampling, `shots_count` can
be specified to any integer.

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/first_observe.py
        :language: python
        :start-after: [Begin Observe3]
        :end-before: [End Observe3]

.. tab:: C++

  .. literalinclude:: ../../snippets/cpp/using/first_observe.cpp
        :language: cpp
        :start-after: [Begin Observe3]
        :end-before: [End Observe3]

Similar to `sample_async` above, observe also supports asynchronous execution. 
More information about parallelizing execution can be found at 
the :ref:`mqpu-platform` page.

Running on a GPU
++++++++++++++++++

.. tab:: Python

  Using :func:`cudaq.set_target`, different targets can be specified for kernel execution.

.. tab:: C++

  Using the `--target` argument to `nvq++`, different targets can be specified for kernel execution.

If a local GPU is detected, the target will default to `nvidia`. Otherwise, the CPU-based simulation
target, `qpp-cpu`,  will be selected.
  
We will demonstrate the benefits of using a GPU by sampling our GHZ kernel with 25 qubits and a
`shots_count` of 1 million. Using a GPU accelerates this task by more than 35x. To learn about
all of the available targets and ways to accelerate kernel execution, visit the
:doc:`Backends <../backends/backends>` page.

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/time.py
        :language: python
        :start-after: [Begin Time]
        :end-before: [End Time]

.. tab:: C++

  To compare the performance, we can create a simple timing script that isolates just the call
  to `cudaq::sample`. We are still using the same GHZ kernel as earlier, but the following
  modification is made to the main function:

  .. literalinclude:: ../../snippets/cpp/using/time.cpp
    :language: cpp
    :start-after: [Begin Time]
    :end-before: [End Time]

  First we execute on the CPU backend:

  .. code:: console

    nvq++ --target=qpp-cpu sample.cpp
    ./a.out
  
  seeing an output of the order:
  ``It took 22.8337 seconds.``

  Now we can execute on the GPU enabled backend:

  .. code:: console

    nvq++ --target=nvidia sample.cpp
    ./a.out

  seeing an output of the order:
  ``It took 3.18988 seconds.``
