Multi-control Synthesis 
-------------------------

Now let's take a look at how CUDA-Q allows one to control a general unitary 
on an arbitrary number of control qubits. 

.. tab:: Python

   Our first option is to describe our general unitary by another pre-defined
   CUDA-Q kernel. 

   .. literalinclude:: ../../examples/python/multi_controlled_operations.py
      :language: python
      :start-after: [Begin OptionA]
      :end-before: [End OptionA]

   Alternatively, one may pass multiple arguments for control qubits or vectors
   to any controlled operation.

   .. literalinclude:: ../../examples/python/multi_controlled_operations.py
      :language: python
      :start-after: [Begin OptionB]
      :end-before: [End OptionB]


.. tab:: C++ 

   For this scenario, our general unitary can be described by another pre-defined 
   CUDA-Q kernel expression. 

   .. literalinclude:: ../../examples/cpp/basics/multi_controlled_operations.cpp
      :language: cpp