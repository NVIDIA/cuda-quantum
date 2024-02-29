CUDA Quantum Basics
*******************

.. _cudaq-basics-landing-page:

What is a CUDA Quantum kernel?
-------------------------------

Quantum kernels are defined as functions that are executed on a quantum processing unit (QPU) or
a simulator. They generalize quantum circuits and provide a new abstraction for quantum programming.
Quantum kernels can be combined with classical functions to create quantum-classical applications
that can be executed on a heterogeneous system of QPUs, GPUs, and CPUs to solve real-world problems.

**What’s the difference between a quantum kernel and a quantum circuit?**

Every quantum circuit is a kernel, but not every quantum kernel is a circuit. For instance, a quantum
kernel can be built up from other kernels, allowing us to interpret a large quantum program as a sequence
of subroutines or subcircuits.  

Moreover, since quantum kernels are functions, there is more expressibility available compared to a
standard quantum circuit. We can not only parameterize the kernel, but can also apply classical controls
(`if`, `for`, `while`, etc.). As functions, quantum kernels can return void, Boolean values, integers,
floating point numbers, and vectors of Boolean values. Conditional statements on quantum memory and qubit
measurements can be included in quantum kernels to enable dynamic circuits and fast feedback, particularly
useful for quantum error correction. 

**How do I build and run a quantum kernel?**

Once a quantum kernel has been defined in a program, it can be executed using the `sample` or the `observe` primitives.
Let’s take a closer look at how to build and execute a quantum kernel with CUDA Quantum.


Building your first CUDA Quantum Program
-----------------------------------------

.. _building-your-first-kernel:

.. tab:: Python

  .. 
    FIXME :: comment back in when updated python rolls out.
    We can define our quantum kernel as we do any other function in Python, through the use of the
   `@cudaq.kernel` decorator. 
  
  We can define our quantum kernel in Python through the use of the :func:`cudaq.make_kernel` function.
  Let's begin with a simple GHZ-state example, producing a state of
  maximal entanglement amongst an allocated set of qubits. 
  
  .. literalinclude:: ../snippets/python/using/first_kernel.py
      :language: python
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

  This kernel function can accept any number of arguments, allowing for flexibility in the construction
  of the quantum program. In this case, the `qubit_count` argument allows us to dynamically control the
  number of qubits allocated to the kernel. As we will see in further :ref:`examples <python-examples-landing-page>`,
  we could also use these arguments to control various parameters of the gates themselves, such as rotation
  angles.


.. tab:: C++

  We can define our quantum kernel as we do any other typed callable in C++, through the use of the
  `__qpu__` annotation. For the following example, we will define a kernel for a simple GHZ-state as
  a standard free function.

  .. literalinclude:: ../snippets/cpp/using/first_kernel.cpp
      :language: cpp
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

  This kernel function can accept any number of arguments, allowing for flexibility in the construction
  of the quantum program. In this case, the `qubit_count` argument allows us to dynamically control the
  number of qubits allocated to the kernel. As we will see in further :ref:`examples <cpp-examples-landing-page>`,
  we could also use these arguments to control various parameters of the gates themselves, such as rotation
  angles.



Running your first CUDA Quantum Program
----------------------------------------

Now that you have built your first quantum kernel, we will learn how to physically execute the program.

Sample
++++++++

.. tab:: Python

  The :func:`cudaq.sample` method takes a kernel and it arguments as inputs, and returns a :class:`cudaq.SampleResult`
  to the programmer. This result dictionary contains the distribution of measured states for the system.
  Continuing with the GHZ kernel defined in :ref:`Building Your First CUDA Quantum Program <building-your-first-kernel>`,
  we will set the concrete value of our `qubit_count` to be two. The following will assume this code exists in
  a file named `sample.py`.

  .. literalinclude:: ../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample1]
        :end-before: [End Sample1]

  By default, `sample` produces an ensemble of 1000 shots. This can be changed by specifying an integer argument
  for the `shots_count`.

  .. literalinclude:: ../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample2]
        :end-before: [End Sample2]

  A variety of methods can be used to extract useful information from a :class:`cudaq.SampleResult`. For example,
  to return the most probable measurement and its respective probability:

  .. literalinclude:: ../snippets/python/using/first_sample.py
        :language: python
        :start-after: [Begin Sample3]
        :end-before: [End Sample3]

  We can execute this program as we do a typical python file.

  .. code-block:: console

      python3 sample.py

  See the :ref:`API specification <python-api-landing-page>` for further information.

.. tab:: C++

  The :func:`cudaq.sample` method takes a kernel and it arguments as inputs, and returns a :class:`cudaq.SampleResult`
  to the programmer. This result dictionary contains the distribution of measured states for the system.
  Continuing with the GHZ kernel defined in :ref:`Building Your First CUDA Quantum Program <building-your-first-kernel>`,
  we will set the concrete value of our `qubit_count` to be two. The following will assume this code exists in
  a file named `sample.cpp`.

  .. literalinclude:: ../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample1]
        :end-before: [End Sample1]

  By default, `sample` produces an ensemble of 1000 shots. This can be changed by specifying an integer argument
  for the `shots_count`.

  .. literalinclude:: ../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample2]
        :end-before: [End Sample2]

  A variety of methods can be used to extract useful information from a :class:`cudaq.SampleResult`. For example,
  to return the most probable measurement and its respective probability:

  .. literalinclude:: ../snippets/cpp/using/first_sample.cpp
        :language: cpp
        :start-after: [Begin Sample3]
        :end-before: [End Sample3]

  We can now compile this file with the `nvq++` toolchain, and run it as we do any other
  C++ executable.

  .. code-block:: console

      nvq++ sample.cpp
      ./a.out

  See the :ref:`API specification <cpp-api-landing-page>` for further information.

Observe
+++++++++

:func:`cudaq.observe` is used to produce expectation values provided a quantum state and a spin operator. 

First, an operator (i.e. a linear combination of Pauli strings) must be specified. To do so, import `spin` from cudaq. Any linear combination of the I, X, Y, and Z spin operators can be constructed with using spin.i(q), spin.x(q), spin.y(q), and spin.z(q), respectively, where q is the index of the target qubit. 

Below is an example of a spin operator object consisting of a Z(0) operator, followed by construction of a kernel with a single qubit in an equal superposition. The Hamiltonian is printed to confirm it has been constructed properly.

.. literalinclude:: ../snippets/python/using/observe.py
      :language: python
      :start-after: [Begin Observe1]
      :end-before: [End Observe1]

:code:`cudaq::observe` takes a kernel, kernel arguments (if any),  and a spin operator as inputs and produces an `ObserveResult` object. The expectation value can be printed using the `expectation` method. It is important to exclude a measurement in the kernel, otherwise the expectation value will be determined from a collapsed classical state. For this example, the expected result of 0.0 is produced.

.. literalinclude:: ../snippets/python/using/observe.py
      :language: python
      :start-after: [Begin Observe2]
      :end-before: [End Observe2]

Unlike `sample`, the default `shots_count` for :code:`cudaq::observe` is 1. This result is deterministic and equivalent to the expectation value in the limit of infinite shots.  To produce an approximate expectation value from sampling, `shots_count` can be specified to any integer.

.. literalinclude:: ../snippets/python/using/observe.py
      :language: python
      :start-after: [Begin Observe3]
      :end-before: [End Observe3]

Running on a GPU
++++++++++++++++++

.. tab:: Python

  Using :func:`cudaq.set_target`, different targets can be specified for kernel execution.
  
  If a local GPU is detected, the target will default to `nvidia`. Otherwise, the CPU-based simulation
  target, `qpp-cpu`,  will be selected.
  
  We will demonstrate the benefits of using a GPU by sampling our GHZ kernel with 25 qubits and a
  `shots_count` of 1 million. Using a GPU accelerates this task by more than 35x. To learn about
  all of the available targets and ways to accelerate kernel execution, visit the
  :ref:`Backends <backends-landing-page>` page.

  .. literalinclude:: ../snippets/python/using/time.py
        :language: python
        :start-after: [Begin Time]
        :end-before: [End Time]


.. tab:: C++

  Using the `-- target` argument to `nvq++`, different targets can be specified for kernel execution.
  
  If a local GPU is detected, the target will default to `nvidia`. Otherwise, the CPU-based simulation
  target, `qpp-cpu`,  will be selected.
  
  We will demonstrate the benefits of using a GPU by sampling our GHZ kernel with 25 qubits and a
  `shots_count` of 1 million. Using a GPU accelerates this task by more than 35x. To learn about
  all of the available targets and ways to accelerate kernel execution, visit the 
  :ref:`Backends <backends-landing-page>` page.

  To compare the performance, we can create a simple timing script that isolates just the call
  to :func:`cudaq.sample`. We are still using the same GHZ kernel as earlier, but the following
  modification made to the main function:

  .. literalinclude:: ../snippets/cpp/using/time.cpp
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
..
  FIXME:: Decide on what to do with this for march release without updated python.
    Spending the rest of the week writing this for the kernel builder when it will
    just be taken out doesn't seem like a great use of time.

  Language Fundamentals
  ----------------------

  CUDA Quantum kernels support a subset of native Python syntax. We will now outline the supported syntax
  and highlight important features of the CUDA Quantum kernel API.
  .. FIXME ... better copy here

  Quantum Memory
  ++++++++++++++++++++++++++++++++++

  Todo

Troubleshooting
-----------------


Debugging and Verbose Simulation Output
+++++++++++++++++++++++++++++++++++++++++

One helpful mechanism of debugging CUDA Quantum simulation execution is
the :code:`CUDAQ_LOG_LEVEL` environment variable. For any CUDA Quantum
executable, just prepend this and turn it on:

.. tab:: Python

  .. code-block:: bash

      CUDAQ_LOG_LEVEL=info python3 file.py

.. tab:: C++

    .. code-block:: bash

      CUDAQ_LOG_LEVEL=info ./a.out

Similarly, one may write the IR to their console or to a file before remote
submission. This may be done through the :code:`CUDAQ_DUMP_JIT_IR` environment
variable. For any CUDA Quantum executable, just prepend as follows:

.. tab:: Python

  .. code-block:: bash

      CUDAQ_DUMP_JIT_IR=1 python3 file.py
      # or
      CUDAQ_DUMP_JIT_IR=<output_filename> python3 file.py

.. tab:: C++

  .. code-block:: bash

      CUDAQ_DUMP_JIT_IR=1 ./a.out
      # or
      CUDAQ_DUMP_JIT_IR=<output_filename> ./a.out