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

    .. code-block:: cpp

        auto h = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) - \
                 2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) + \
                 .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

.. tab:: Python

    .. code-block:: python 

        from cudaq import spin 
        h = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(1) + \
                 .21829 * spin.z(0) - 6.125 * spin.z(1)


**[5]** The :code:`spin_op` also provides a mechanism for the expression of circuit
synthesis tasks within quantum kernel code. Specifically, operations
that encode :math:`N`\ :sup:`th`\ order trotterization of exponentiated :code:`spin_op`
rotations, e.g. :math:`U = \exp(-i H t)`, where :math:`H` is the provided :code:`spin_op`.
Currently, H is limited to a single product term.

**[6]** The :code:`spin_op` can be created within classical host code and quantum kernel
code, and can also be passed by value to quantum kernel code from host code. 

