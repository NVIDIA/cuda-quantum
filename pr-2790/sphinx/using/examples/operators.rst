Operators
=========

Operators are important constructs for many quantum applications. This section covers how to define and use spin operators as well as additional tools for defining more sophisticated operators.

Constructing Spin Operators
---------------------------

The `spin_op` type provides an abstraction for a general tensor product of Pauli spin operators, and their sums.

Spin operators are constructed using the `spin.z()`, `spin.y()`, `spin.x()`, and `spin.i()` functions, corresponding to the :math:`Z`, :math:`Y`, :math:`X`, and :math:`I` Pauli operators. For example, `spin.z(0)` corresponds to a Pauli Z 
operation acting on qubit 0. The example below demonstrates how to construct the following operator
2 :math:`XYX` - 3 :math:`ZZY`.

.. tab:: Python

    .. literalinclude:: ../../examples/python/operators.py
        :language: python
        :start-after: [Begin Spin]
        :end-before: [End Spin]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/operators.cpp
        :language: cpp
        :start-after: [Begin Spin]
        :end-before: [End Spin]

There are a number of convenient methods for combining, comparing, iterating through, and extracting information from spin operators and can be referenced `here <https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.SpinOperator>`_ in the API.

Pauli Words and Exponentiating Pauli Words
------------------------------------------

The `pauli_word` type specifies a string of Pauli operations (e.g. `XYXZ`) and is convenient for applying operations based on exponentiated Pauli words. The code below demonstrates how a list of Pauli words, along with their coefficients, are provided as kernel inputs and converted into operators by the `exp_pauli` function.

The code below applies the following operation: :math:`e^{i(0.432XYZ)} + e^{i(0.324IXX)}`

.. tab:: Python

    .. literalinclude:: ../../examples/python/operators.py
        :language: python
        :start-after: [Begin Pauli]
        :end-before: [End Pauli]

.. tab:: C++

    .. literalinclude:: ../../examples/cpp/operators.cpp
        :language: cpp
        :start-after: [Begin Pauli]
        :end-before: [End Pauli]