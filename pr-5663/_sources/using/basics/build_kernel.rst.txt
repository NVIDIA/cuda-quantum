Building your first CUDA-Q Program
-----------------------------------------

.. tab:: Python

  We can define our quantum kernel as a typical Python function, with the additional use of the
  `@cudaq.kernel` decorator. Let's begin with a simple GHZ-state example, producing a state of
  maximal entanglement amongst an allocated set of qubits. 
  
  .. literalinclude:: ../../snippets/python/using/first_sample.py
      :language: python
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

.. tab:: C++

  We can define our quantum kernel as we do any other typed callable in C++, through the use of the
  `__qpu__` annotation. For the following example, we will define a kernel for a simple GHZ-state as
  a standard free function.

  .. literalinclude:: ../../snippets/cpp/using/first_sample.cpp
      :language: cpp
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

This kernel function can accept any number of arguments, allowing for flexibility in the construction
of the quantum program. In this case, the `qubit_count` argument allows us to dynamically control the
number of qubits allocated to the kernel. As we will see in further :doc:`examples <../examples/examples>`,
we could also use these arguments to control various parameters of the gates themselves, such as rotation
angles.
