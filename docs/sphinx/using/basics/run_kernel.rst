Running your first CUDA-Q Program
----------------------------------------

Now that you have defined your first quantum kernel, let's look at different options for how to execute it.
In CUDA-Q, quantum circuits are stored as quantum kernels. For estimating the probability distribution of 
a measured quantum state in a circuit, we use the ``sample`` function call, and for computing the
expectation value of a quantum state with a given observable, we use the ``observe`` function call.

Sample
++++++++

.. tab:: Python

  The :func:`cudaq.sample` method takes a kernel and its arguments as inputs, and returns a :class:`cudaq.SampleResult`.
  This result dictionary contains the distribution of measured states for the system.
  Continuing with the GHZ kernel defined in :doc:`Building Your First CUDA-Q Program <build_kernel>`,
  we will set the concrete value of our `qubit_count` to be two. The following will assume this code exists in
  a file named `sample.py`.

  .. literalinclude:: ../../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample1]
        :end-before: [End Sample1]

  By default, `sample` produces an ensemble of 1000 shots. This can be changed by specifying an integer argument
  for the `shots_count`.

  .. literalinclude:: ../../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample2]
        :end-before: [End Sample2]

  A variety of methods can be used to extract useful information from a :class:`cudaq.SampleResult`. For example,
  to return the most probable measurement and its respective probability:

  .. literalinclude:: ../../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample3]
        :end-before: [End Sample3]

  We can execute this program as we do any Python file.

  .. code-block:: console

      python3 sample.py

  See the :doc:`API specification <../../../api/languages/python_api>` for further information.

.. tab:: C++

  The :func:`cudaq.sample` method takes a kernel and its arguments as inputs, and returns a :class:`cudaq.SampleResult`.
  This result dictionary contains the distribution of measured states for the system.
  Continuing with the GHZ kernel defined in :doc:`Building Your First CUDA-Q Program <build_kernel>`,
  we will set the concrete value of our `qubit_count` to be two. The following will assume this code exists in
  a file named `sample.cpp`.

  .. literalinclude:: ../../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample1]
        :end-before: [End Sample1]

  By default, `sample` produces an ensemble of 1000 shots. This can be changed by specifying an integer argument
  for the `shots_count`.

  .. literalinclude:: ../../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample2]
        :end-before: [End Sample2]

  A variety of methods can be used to extract useful information from a :class:`cudaq.SampleResult`. For example,
  to return the most probable measurement and its respective probability:

  .. literalinclude:: ../../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample3]
        :end-before: [End Sample3]

  We can now compile this file with the `nvq++` toolchain, and run it as we do any other
  C++ executable.

  .. code-block:: console

      nvq++ sample.cpp
      ./a.out

  See the :doc:`API specification <../../../api/languages/cpp_api>` for further information.

  Asynchronous version of sample exists and more information can be found at 
  :doc:`Sample Asynchronous <../backends/platform>` page.

Observe
+++++++++

.. tab:: Python

  The :func:`cudaq.observe` method takes a kernel and its arguments as inputs, along with a :class:`cudaq.SpinOperator`.
  As opposed to :func:`cudaq.sample`, `observe` is primarily used to produce expectation values of a kernel with respect
  to a provider operator.

  Using the `cudaq.spin` module, operators may be defined as a linear combination of Pauli strings. Functions, such
  as :func:`cudaq.spin.i`, :func:`cudaq.spin.x`, :func:`cudaq.spin.y`, :func:`cudaq.spin.z` may be used to construct more
  complex spin Hamiltonians on multiple qubits.
  
  Below is an example of a spin operator object consisting of a `Z(0)` operator, or a Pauli Z-operator on the zeroth qubit. 
  This is followed by the construction of a kernel with a single qubit in an equal superposition. 
  The Hamiltonian is printed to confirm it has been constructed properly.

  .. literalinclude:: ../../snippets/python/using/first_observe.py
        :language: python
        :start-after: [Begin Observe1]
        :end-before: [End Observe1]

  :code:`cudaq::observe` takes a kernel, any kernel arguments, and a spin operator as inputs and produces an `ObserveResult` object.
  The expectation value can be printed using the `expectation` method. 
  
  .. note:: 
    It is important to exclude a measurement in the kernel, otherwise the expectation value will be determined from a collapsed 
    classical state. For this example, the expected result of 0.0 is produced.

  .. literalinclude:: ../../snippets/python/using/first_observe.py
        :language: python
        :start-after: [Begin Observe2]
        :end-before: [End Observe2]

  Unlike `sample`, the default `shots_count` for :code:`cudaq::observe` is 1. This result is deterministic and equivalent to the
  expectation value in the limit of infinite shots.  To produce an approximate expectation value from sampling, `shots_count` can
  be specified to any integer.

  .. literalinclude:: ../../snippets/python/using/first_observe.py
        :language: python
        :start-after: [Begin Observe3]
        :end-before: [End Observe3]

.. tab:: C++

  The :func:`cudaq.observe` method takes a kernel and its arguments as inputs, along with a `cudaq::spin_op`.
  As opposed to :func:`cudaq.sample`, `observe` is primarily used to produce expectation values of a kernel with respect
  to a provider operator.

  Within the `cudaq::spin` namespace, operators may be defined as a linear combination of Pauli strings. Functions, such
  as `cudaq::spin::i`, `cudaq::spin::x`, `cudaq::spin::y`, `cudaq::spin::z` may be used to construct more
  complex spin Hamiltonians on multiple qubits.
  
  Below is an example of a spin operator object consisting of a `Z(0)` operator, or a Pauli Z-operator on the zeroth qubit. 
  This is followed by the construction of a kernel with a single qubit in an equal superposition. 
  The Hamiltonian is printed to confirm it has been constructed properly.

  .. literalinclude:: ../../snippets/cpp/using/first_observe.cpp
        :language: cpp
        :start-after: [Begin Observe1]
        :end-before: [End Observe1]

  :code:`cudaq::observe` takes a kernel, any kernel arguments, and a spin operator as inputs and produces an `ObserveResult` object.
  The expectation value can be printed using the `expectation` method. 
  
  .. note:: 
    It is important to exclude a measurement in the kernel, otherwise the expectation value will be determined from a collapsed 
    classical state. For this example, the expected result of 0.0 is produced.

  .. literalinclude:: ../../snippets/cpp/using/first_observe.cpp
        :language: cpp
        :start-after: [Begin Observe2]
        :end-before: [End Observe2]

  Unlike `sample`, the default `shots_count` for :code:`cudaq::observe` is 1. This result is deterministic and equivalent to the
  expectation value in the limit of infinite shots.  To produce an approximate expectation value from sampling, `shots_count` can
  be specified to any integer.

  .. literalinclude:: ../../snippets/cpp/using/first_observe.cpp
        :language: cpp
        :start-after: [Begin Observe3]
        :end-before: [End Observe3]

  Asynchronous version of observe exists and more information can be found 
  :doc:`Observe Asynchronous <../backends/platform>` page.

Running on a GPU
++++++++++++++++++

.. tab:: Python

  Using :func:`cudaq.set_target`, different targets can be specified for kernel execution.
  
  If a local GPU is detected, the target will default to `nvidia`. Otherwise, the CPU-based simulation
  target, `qpp-cpu`,  will be selected.
  
  We will demonstrate the benefits of using a GPU by sampling our GHZ kernel with 25 qubits and a
  `shots_count` of 1 million. Using a GPU accelerates this task by more than 35x. To learn about
  all of the available targets and ways to accelerate kernel execution, visit the
  :doc:`Backends <../backends/backends>` page.

  .. literalinclude:: ../../snippets/python/using/time.py
        :language: python
        :start-after: [Begin Time]
        :end-before: [End Time]


.. tab:: C++

  Using the `--target` argument to `nvq++`, different targets can be specified for kernel execution.
  
  If a local GPU is detected, the target will default to `nvidia`. Otherwise, the CPU-based simulation
  target, `qpp-cpu`,  will be selected.
  
  We will demonstrate the benefits of using a GPU by sampling our GHZ kernel with 25 qubits and a
  `shots_count` of 1 million. Using a GPU accelerates this task by more than 35x. To learn about
  all of the available targets and ways to accelerate kernel execution, visit the 
  :doc:`Backends <../backends/backends>` page.

  To compare the performance, we can create a simple timing script that isolates just the call
  to :func:`cudaq.sample`. We are still using the same GHZ kernel as earlier, but the following
  modification made to the main function:

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