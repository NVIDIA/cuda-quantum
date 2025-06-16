Quantum Operations
******************************

CUDA-Q provides a default set of quantum operations on qubits. 
These operations can be used to define custom kernels and libraries.
Since the set of quantum intrinsic operations natively supported on a specific target 
depends on the backends architecture, the :code:`nvq++` compiler automatically
decomposes the default operations into the appropriate set of intrinsic operations 
for that target.

The sections `Unitary Operations on Qubits`_ and `Measurements on Qubits`_ list the default set of quantum operations on qubits.

Operations that implement unitary transformations of the quantum state are templated.
The template argument allows to invoke the adjoint and controlled version of the quantum transformation, see the section on `Adjoint and Controlled Operations`_.

CUDA-Q additionally provides overloads to support broadcasting of
single-qubit operations across a vector of qubits.  For example,
:code:`x(cudaq::qvector<>&)` flips the state of each qubit in the provided
:code:`cudaq::qvector`. 


Unitary Operations on Qubits
=============================

:code:`x`
---------------------

This operation implements the transformation defined by the Pauli-X matrix. It is also known as the quantum version of a `NOT`-gate.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/x_op.py
       :language: python
       :start-after: [Begin X Op]
       :end-before: [End X Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/x_op.cpp
       :language: cpp
       :start-after: [Begin X Op]
       :end-before: [End X Op]

:code:`y`
---------------------

This operation implements the transformation defined by the Pauli-Y matrix.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/y_op.py
       :language: python
       :start-after: [Begin Y Op]
       :end-before: [End Y Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/y_op.cpp
       :language: cpp
       :start-after: [Begin Y Op]
       :end-before: [End Y Op]

:code:`z`
---------------------

This operation implements the transformation defined by the Pauli-Z matrix.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/z_op.py
       :language: python
       :start-after: [Begin Z Op]
       :end-before: [End Z Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/z_op.cpp
       :language: cpp
       :start-after: [Begin Z Op]
       :end-before: [End Z Op]

:code:`h`
---------------------

This operation is a rotation by π about the X+Z axis, and enables one to create a superposition of computational basis states.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/h_op.py
       :language: python
       :start-after: [Begin H Op]
       :end-before: [End H Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/h_op.cpp
       :language: cpp
       :start-after: [Begin H Op]
       :end-before: [End H Op]
       
:code:`r1`
---------------------

This operation is an arbitrary rotation about the :code:`|1>` state.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/r1_op.py
       :language: python
       :start-after: [Begin R1 Op]
       :end-before: [End R1 Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/r1_op.cpp
       :language: cpp
       :start-after: [Begin R1 Op]
       :end-before: [End R1 Op]

:code:`rx`
---------------------

This operation is an arbitrary rotation about the X axis.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/rx_op.py
       :language: python
       :start-after: [Begin Rx Op]
       :end-before: [End Rx Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/rx_op.cpp
       :language: cpp
       :start-after: [Begin Rx Op]
       :end-before: [End Rx Op]

:code:`ry`
---------------------

This operation is an arbitrary rotation about the Y axis.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/ry_op.py
       :language: python
       :start-after: [Begin Ry Op]
       :end-before: [End Ry Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/ry_op.cpp
       :language: cpp
       :start-after: [Begin Ry Op]
       :end-before: [End Ry Op]
:code:`rz`
---------------------

This operation is an arbitrary rotation about the Z axis.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/rz_op.py
       :language: python
       :start-after: [Begin Rz Op]
       :end-before: [End Rz Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/rz_op.cpp
       :language: cpp
       :start-after: [Begin Rz Op]
       :end-before: [End Rz Op]

:code:`s`
---------------------

This operation applies to its target a rotation by π/2 about the Z axis.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/s_op.py
       :language: python
       :start-after: [Begin S Op]
       :end-before: [End S Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/s_op.cpp
       :language: cpp
       :start-after: [Begin S Op]
       :end-before: [End S Op]

:code:`t`
---------------------

This operation applies to its target a π/4 rotation about the Z axis.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/t_op.py
       :language: python
       :start-after: [Begin T Op]
       :end-before: [End T Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/t_op.cpp
       :language: cpp
       :start-after: [Begin T Op]
       :end-before: [End T Op]

:code:`swap`
---------------------

This operation swaps the states of two qubits.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/swap_op.py
       :language: python
       :start-after: [Begin Swap Op]
       :end-before: [End Swap Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/swap_op.cpp
       :language: cpp
       :start-after: [Begin Swap Op]
       :end-before: [End Swap Op]

:code:`u3`
---------------------

This operation applies the universal three-parameters operator to target qubit. The three parameters are Euler angles - theta (θ), phi (φ), and lambda (λ).

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/u3_op.py
       :language: python
       :start-after: [Begin U3 Op]
       :end-before: [End U3 Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/u3_op.cpp
       :language: cpp
       :start-after: [Begin U3 Op]
       :end-before: [End U3 Op]

Adjoint and Controlled Operations
==================================

.. tab:: Python

    The :code:`adj` method of any gate can be used to invoke the 
    `adjoint <https://en.wikipedia.org/wiki/Conjugate_transpose>`__ transformation:

    .. literalinclude:: /../sphinx/snippets/python/default_ops/adjoint_op.py
       :language: python
       :start-after: [Begin Adjoint Op]
       :end-before: [End Adjoint Op]

.. tab:: C++

    The template argument :code:`cudaq::adj` can be used to invoke the 
    `adjoint <https://en.wikipedia.org/wiki/Conjugate_transpose>`__ transformation:

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/adjoint_op.cpp
       :language: cpp
       :start-after: [Begin Adjoint Op]
       :end-before: [End Adjoint Op]
.. tab:: Python

    The :code:`ctrl` method of any gate can be used to apply the transformation
    conditional on the state of one or more control qubits, see also this 
    `Wikipedia entry <https://en.wikipedia.org/wiki/Quantum_logic_gate#Controlled_gates>`__.

    .. literalinclude:: /../sphinx/snippets/python/default_ops/controlled_op.py
       :language: python
       :start-after: [Begin Controlled Op]
       :end-before: [End Controlled Op]

.. tab:: C++

    The template argument :code:`cudaq::ctrl` can be used to apply the transformation
    conditional on the state of one or more control qubits, see also this 
    `Wikipedia entry <https://en.wikipedia.org/wiki/Quantum_logic_gate#Controlled_gates>`__.

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/controlled_op.cpp
       :language: cpp
       :start-after: [Begin Controlled Op]
       :end-before: [End Controlled Op]

Following common convention, by default the transformation is applied to the target qubit(s)
if all control qubits are in a :code:`|1>` state. 
However, that behavior can be changed to instead apply the transformation when a control qubit is in 
a :code:`|0>` state by negating the polarity of the control qubit.
The syntax for negating the polarity is the not-operator preceding the
control qubit: 

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/negated_control.cpp
       :language: cpp
       :start-after: [Begin Negated Control]
       :end-before: [End Negated Control]

This notation is only supported in the context of applying a controlled operation and is only valid for control qubits. For example, negating either of the target qubits in the
:code:`swap` operation is not allowed.
Negating the polarity of control qubits is similarly supported when using :code:`cudaq::control` to conditionally apply a custom quantum kernel.

Measurements on Qubits
=============================

:code:`mz`
---------------------

This operation measures a qubit with respect to the computational basis, 
i.e., it projects the state of that qubit onto the eigenvectors of the Pauli-Z matrix.
This is a non-linear transformation, and no template overloads are available.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/mz_op.py
       :language: python
       :start-after: [Begin MZ Op]
       :end-before: [End MZ Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/mz_op.cpp
       :language: cpp
       :start-after: [Begin MZ Op]
       :end-before: [End MZ Op]

:code:`mx`
---------------------

This operation measures a qubit with respect to the Pauli-X basis, 
i.e., it projects the state of that qubit onto the eigenvectors of the Pauli-X matrix.
This is a non-linear transformation, and no template overloads are available.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/mx_op.py
       :language: python
       :start-after: [Begin MX Op]
       :end-before: [End MX Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/mx_op.cpp
       :language: cpp
       :start-after: [Begin MX Op]
       :end-before: [End MX Op]

:code:`my`
---------------------

This operation measures a qubit with respect to the Pauli-Y basis, 
i.e., it projects the state of that qubit onto the eigenvectors of the Pauli-Y matrix.
This is a non-linear transformation, and no template overloads are available.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/my_op.py
       :language: python
       :start-after: [Begin MY Op]
       :end-before: [End MY Op]
        
.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/my_op.cpp
       :language: cpp
       :start-after: [Begin MY Op]
       :end-before: [End MY Op]

User-Defined Custom Operations
==============================

Users can define a custom quantum operation by its unitary matrix. First use 
the API to register a custom operation, outside of a CUDA-Q kernel. Then the 
operation can be used within a CUDA-Q kernel like any of the built-in operations
defined above.
Custom operations are supported on qubits only (`qudit` with `level = 2`).

.. tab:: Python

    The :code:`cudaq.register_operation` API accepts an identifier string for 
    the custom operation and its unitary matrix. The matrix can be a `list` or
    `numpy` array of complex numbers. A 1D matrix is interpreted as row-major.
    

    .. literalinclude:: /../sphinx/snippets/python/default_ops/custom_op.py
       :language: python
       :start-after: [Begin Custom Op]
       :end-before: [End Custom Op]
        
.. tab:: C++

    The macro :code:`CUDAQ_REGISTER_OPERATION` accepts a unique name for the 
    operation, the number of target qubits, the number of rotation parameters 
    (can be 0), and the unitary matrix as a 1D row-major `std::vector<complex>` 
    representation.
    
    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/custom_op.cpp
       :language: cpp
       :start-after: [Begin Custom Op]
       :end-before: [End Custom Op]

For multi-qubit operations, the matrix is interpreted with MSB qubit ordering,
i.e. big-endian convention. The following example shows two different custom
operations, each operating on 2 qubits.

.. tab:: Python

    .. literalinclude:: ../snippets/python/using/examples/two_qubit_custom_op.py
      :language: python
      :start-after: [Begin Docs]
      :end-before: [End Docs]

.. tab:: C++

    .. literalinclude:: ../snippets/cpp/using/two_qubit_custom_op.cpp
      :language: cpp
      :start-after: [Begin Docs]
      :end-before: [End Docs]

.. note:: 

  When a custom operation is used on hardware backends, it is synthesized to a
  set of native quantum operations. Currently, only 1-qubit and 2-qubit custom 
  operations are supported on hardware backends.

Photonic Operations on Qudits
=============================

These operations are valid only on the `orca-photonics` target which does not support
the quantum operations above.

:code:`create`
---------------------

This operation increments the number of photons in a qumode up to a maximum value
defined by the qudit level that represents the qumode. If it is applied to a qumode
where the number of photons is already at the maximum value, the operation has no
effect.

:math:`C|0\rangle → |1\rangle, C|1\rangle → |2\rangle, C|2\rangle → |3\rangle, \cdots, C|d\rangle → |d\rangle`
where :math:`d` is the qudit level.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/create_op.py
       :language: python
       :start-after: [Begin Create Op]
       :end-before: [End Create Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/create_op.cpp
       :language: cpp
       :start-after: [Begin Create Op]
       :end-before: [End Create Op]

:code:`annihilate`
---------------------

This operation reduces the number of photons in a qumode up to a minimum value of
0 representing the vacuum state. If it is applied to a qumode where the number of
photons is already at the minimum value 0, the operation has no effect.

:math:`A|0\rangle → |0\rangle, A|1\rangle → |0\rangle, A|2\rangle → |1\rangle, \cdots, A|d\rangle → |d-1\rangle`
where :math:`d` is the qudit level.

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/annihilate_op.py
       :language: python
       :start-after: [Begin Annihilate Op]
       :end-before: [End Annihilate Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/annihilate_op.cpp
       :language: cpp
       :start-after: [Begin Annihilate Op]
       :end-before: [End Annihilate Op]

:code:`phase_shift`
---------------------

A phase shifter adds a phase :math:`\phi` on a qumode. For the annihilation (:math:`a_1`)
and creation operators (:math:`a_1^\dagger`) of a qumode, the phase shift operator
is defined  by

.. math::
    P(\phi) = \exp\left(i \phi a_1^\dagger a_1  \right)

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/phase_shift_op.py
       :language: python
       :start-after: [Begin Phase Shift Op]
       :end-before: [End Phase Shift Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/phase_shift_op.cpp
       :language: cpp
       :start-after: [Begin Phase Shift Op]
       :end-before: [End Phase Shift Op]

:code:`beam_splitter`
---------------------

Beam splitters act on two qumodes together and it is parameterized by a single angle 
:math:`\theta`, relating to reflectivity.
For the annihilation (:math:`a_1` and :math:`a_2`) and creation operators (:math:`a_1^\dagger`
and :math:`a_2^\dagger`) of two qumodes, the beam splitter operator is defined by

.. math::
    B(\theta) = \exp\left[i \theta (a_1^\dagger a_2 + a_1 a_2^\dagger) \right]

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/beam_splitter_op.py
       :language: python
       :start-after: [Begin Beam Splitter Op]
       :end-before: [End Beam Splitter Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/beam_splitter_op.cpp
       :language: cpp
       :start-after: [Begin Beam Splitter Op]
       :end-before: [End Beam Splitter Op]

:code:`mz`
---------------------

This operation returns the measurement results of the input qumode(s).

.. tab:: Python

    .. literalinclude:: /../sphinx/snippets/python/default_ops/mz_qumode_op.py
       :language: python
       :start-after: [Begin MZ Op]
       :end-before: [End MZ Op]

.. tab:: C++

    .. literalinclude:: /../sphinx/snippets/cpp/default_ops/mz_qumode_op.cpp
       :language: cpp
       :start-after: [Begin MZ Op]
       :end-before: [End MZ Op]