Quantum Operators
*****************

:code:`cudaq::spin_op`
----------------------
**[1]** CUDA-Q provides a native :code:`spin_op` data type in the :code:`cudaq` namespace for the
expression of quantum mechanical spin operators. 

**[2]** The :code:`spin_op` provides an abstraction for a general tensor product of Pauli
spin operators, and sums thereof:

.. math:: 

    H = \sum_{i=0}^M P_i, P_i = \prod_{j=0}^N \sigma_j^a

for :math:`a = {x,y,z}`, :math:`j` the qubit index, and :math:`N` the number of qubits.

**[3]** The :code:`spin_op` exposes common C++ operator overloads for algebraic expressions. 

**[4]** CUDA-Q defines static functions to create
the primitive X, Y, and Z Pauli operators on specified qubit indices
which can subsequently be used in algebraic expressions to build up
more complicated Pauli tensor products and their sums.

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/operators/spin_op_creation.cpp
     :language: cpp
     :start-after: [Begin SpinOp Creation C++]
     :end-before: [End SpinOp Creation C++]
     :lines: 2-4

.. tab:: Python

  .. literalinclude:: /../snippets/python/operators/spin_op_creation.py
     :language: python
     :start-after: [Begin SpinOp Creation Python]
     :end-before: [End SpinOp Creation Python]
     :lines: 2-4
     
**[5]** The :code:`spin_op` also provides a mechanism for the expression of circuit
synthesis tasks within quantum kernel code. Specifically, operations
that encode :math:`N`\ :sup:`th`\ order trotterization of exponentiated :code:`spin_op`
rotations, e.g. :math:`U = \exp(-i H t)`, where :math:`H` is the provided :code:`spin_op`.
Currently, H is limited to a single product term.

**[6]** The :code:`spin_op` can be created within classical host code and quantum kernel
code, and can also be passed by value to quantum kernel code from host code. 

