Quantum Approximate Optimization Algorithm
-------------------------------------------

.. tab:: Python

   `Farhi et al.`<https://arxiv.org/abs/1411.4028>` introduced the quantum approximation optimization algorithm (QAOA) to solve optimization problems like the Max Cut problem. In short, QAOA is a variational algortihm with a particular ansatz. QAOA is made up of a variational quantum circuit (i.e., a kernel that depends on a set of parameter values) and a classical optimizer. The aim of QAOA is to use the classical optimizer to identify parameter values that generate a quantum circuit whose expectation value for a given cost Hamilitonian is minimized. 
   What distinguishes QAOA from other variational algorithms is the structure of the quantum circuit. For each vertex in the graph, there is an associated qubit in the circuit. The circuit is initialized in a superposition state. The remainder of the QAOA circuit is made up of blocks (referred to as layers). Each layer contains a problem kernel and a mixer kernel. The problem kernel encodes our graph. The mixer kernel is composed of parameterized rotation gates applied to each qubit. The more layers there are, the better the approximation the algorithm achieves. 
   Let's now see how we can implement the Quantum Approximate Optimization Algorithm (QAOA) to compute the Max-Cut of a rectangular graph.

   .. literalinclude:: ../../examples/python/qaoa_maxcut.py
      :language: python

.. tab:: C++

   Let's now see how we can implement the Quantum Approximate Optimization Algorithm (QAOA) to compute the Max-Cut of a rectangular graph.
   For more on the QAOA algorithm and the Max-Cut problem, refer to 
   `this paper <https://arxiv.org/abs/1411.4028>`__.

   .. literalinclude:: ../../examples/cpp/algorithms/qaoa_maxcut.cpp
      :language: cpp
      :start-after: [Begin Documentation]
      :end-before: [End Documentation]
