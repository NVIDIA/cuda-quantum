Quantum Approximate Optimization Algorithm
-------------------------------------------

.. tab:: Python

   Let's now see how we can implement the Quantum Approximate Optimization Algorithm (QAOA) to compute the Max-Cut of a rectangular graph by leveraging 
   `cudaq:vqe`. For more on the QAOA algorithm and the Max-Cut problem, refer to 
   `this paper <https://arxiv.org/abs/1411.4028>`__.

   .. literalinclude:: ../../examples/python/qaoa_maxcut.py
      :language: python

.. tab:: C++

   Let's now see how we can implement the Quantum Approximate Optimization Algorithm (QAOA) to compute the Max-Cut of a rectangular graph
   For more on the QAOA algorithm and the Max-Cut problem, refer to 
   `this paper <https://arxiv.org/abs/1411.4028>`__.

   .. literalinclude:: ../../examples/cpp/algorithms/qaoa_maxcut.cpp
      :language: cpp
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]