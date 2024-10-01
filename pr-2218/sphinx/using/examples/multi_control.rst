Multi-control Synthesis 
-------------------------
Now let's take a look at how CUDA-Q allows one to control a general unitary 
on an arbitrary number of control qubits. 

.. tab:: Python

   Our first option is to describe our general unitary by another pre-defined
   CUDA-Q kernel. Alternatively, one may pass multiple arguments for control qubits or vectors
   to any controlled operation.

   .. toctree::
      :hidden:

      ../../examples/python/building_kernels.ipynb

.. tab:: C++ 

   For this scenario, our general unitary can be described by another pre-defined 
   CUDA-Q kernel expression. 

   .. literalinclude:: ../../examples/cpp/basics/multi_controlled_operations.cpp
      :language: cpp
