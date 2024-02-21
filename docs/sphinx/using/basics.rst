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

.. tab:: Python

  We can define our quantum kernel as we do any other function in Python, through the use of the
  `@cudaq.kernel` decorator. Let's begin with a simple GHZ-state example, producing a state of
  maximal entanglement amongst an allocated set of qubits. 
  
  .. literalinclude:: ../snippets/python/using/first_kernel.py
      :language: python
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]

  This kernel function can accept any number of arguments, allowing for flexibility in the construction
  of the quantum program. In this case, the `qubit_count` argument allows us to dynamically control the
  number of qubits allocated to the kernel. As we will see in further `examples <cuda-quantum-examples>`,
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
  number of qubits allocated to the kernel. As we will see in further `examples <cuda-quantum-examples>`,
  we could also use these arguments to control various parameters of the gates themselves, such as rotation
  angles.



Running your first CUDA Quantum Program
----------------------------------------

Todo

Language Fundamentals
----------------------

Todo