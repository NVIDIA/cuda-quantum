Introduction
--------------------------------

Welcome to CUDA-Q! On this page we will illustrate CUDA-Q with several examples. 

.. tab:: Python

   We're going to take a look at how to construct quantum programs through CUDA-Q's `Kernel` API.

   When you create a `Kernel` and invoke its methods, a quantum program is constructed that can then be executed by calling, for example, `cudaq::sample`. Let's take a closer look!

   .. literalinclude:: ../../examples/python/intro.py
      :language: python

.. tab:: C++

   We're going to take a look at how to construct quantum programs using CUDA-Q kernels.

   CUDA-Q kernels are any typed callable in the language that is annotated with the :code:`__qpu__` attribute. Let's take a look at a very 
   simple "Hello World" example, specifically a CUDA-Q kernel that prepares a GHZ state on a programmer-specified number of qubits. 

   .. literalinclude:: ../../examples/cpp/basics/static_kernel.cpp
      :language: cpp

   Here we see that we can define a custom :code:`struct` that is templated on a :code:`size_t` parameter. 
   Our kernel expression is free to use this template parameter in the allocation of a 
   compile-time-known register of qubits. Within the kernel, we are free to apply various quantum operations, 
   like a Hadamard on qubit 0 :code:`h(q[0])`. Controlled operations are **modifications** of single-qubit 
   operations, like the :code:`x<cudaq::ctrl>(q[0],q[1])` operation which implements a controlled-X gate. We 
   can measure single qubits or entire registers. 

   In this example we are interested in sampling the final state produced by this CUDA-Q kernel. 
   To do so, we leverage the generic :code:`cudaq::sample` function, which returns a data type 
   encoding the qubit measurement strings and the corresponding number of times that string 
   was observed (here the default number of shots is used, :code:`1000`).

   The following example illustrates how to compile and execute this code.

   .. code:: bash 

      nvq++ static_kernel.cpp -o ghz.x
      ./ghz.x