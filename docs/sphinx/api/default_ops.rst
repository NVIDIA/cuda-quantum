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

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # X = | 0  1 |
        #     | 1  0 |
        x(qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // X = | 0  1 |
        //     | 1  0 |
        x(qubit);

:code:`y`
---------------------

This operation implements the transformation defined by the Pauli-Y matrix.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # Y = | 0  -i |
        #     | i   0 |
        y(qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // Y = | 0  -i |
        //     | i   0 |
        y(qubit);

:code:`z`
---------------------

This operation implements the transformation defined by the Pauli-Z matrix.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # Z = | 1   0 |
        #     | 0  -1 |
        z(qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // Z = | 1   0 |
        //     | 0  -1 |
        z(qubit);

:code:`h`
---------------------

This operation is a rotation by π about the X+Z axis, and 
enables one to create a superposition of computational basis states.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # H = (1 / sqrt(2)) * | 1   1 |
        #                     | 1  -1 |
        h(qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // H = (1 / sqrt(2)) * | 1   1 |
        //                     | 1  -1 |
        h(qubit);

:code:`r1`
---------------------

This operation is an arbitrary rotation about the :code:`|1>` state.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # R1(λ) = | 1     0    |
        #         | 0  exp(iλ) |
        r1(math.pi, qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // R1(λ) = | 1     0    |
        //         | 0  exp(iλ) |
        r1(std::numbers::pi, qubit);

:code:`rx`
---------------------

This operation is an arbitrary rotation about the X axis.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
        #         | -isin(θ/2)  cos(θ/2)  |
        rx(math.pi, qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
        //         | -isin(θ/2)  cos(θ/2)  |
        rx(std::numbers::pi, qubit);

:code:`ry`
---------------------

This operation is an arbitrary rotation about the Y axis.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # Ry(θ) = | cos(θ/2)  -sin(θ/2) |
        #         | sin(θ/2)   cos(θ/2) |
        ry(math.pi, qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // Ry(θ) = | cos(θ/2)  -sin(θ/2) |
        //         | sin(θ/2)   cos(θ/2) |
        ry(std::numbers::pi, qubit);

:code:`rz`
---------------------

This operation is an arbitrary rotation about the Z axis.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # Rz(λ) = | exp(-iλ/2)      0     |
        #         |     0       exp(iλ/2) |
        rz(math.pi, qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // Rz(λ) = | exp(-iλ/2)      0     |
        //         |     0       exp(iλ/2) |
        rz(std::numbers::pi, qubit);

:code:`s`
---------------------

This operation applies to its target a rotation by π/2 about the Z axis.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # S = | 1   0 |
        #     | 0   i |
        s(qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // S = | 1   0 |
        //     | 0   i |
        s(qubit);

:code:`t`
---------------------

This operation applies to its target a π/4 rotation about the Z axis.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # T = | 1      0     |
        #     | 0  exp(iπ/4) |
        t(qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // T = | 1      0     |
        //     | 0  exp(iπ/4) |
        t(qubit);

:code:`swap`
---------------------

This operation swaps the states of two qubits.

.. tab:: Python

    .. code-block:: python

        qubit_1, qubit_2 = cudaq.qubit(), cudaq.qubit()

        # Apply the unitary transformation
        # Swap = | 1 0 0 0 |
        #        | 0 0 1 0 |
        #        | 0 1 0 0 |
        #        | 0 0 0 1 |
        swap(qubit_1, qubit_2)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit_1, qubit_2;

        // Apply the unitary transformation
        // Swap = | 1 0 0 0 |
        //        | 0 0 1 0 |
        //        | 0 1 0 0 |
        //        | 0 0 0 1 |
        swap(qubit_1, qubit_2);

:code:`u3`
---------------------

This operation applies the universal three-parameters operator to target qubit. The three parameters are Euler angles - theta (θ), phi (φ), and lambda (λ).

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()

        # Apply the unitary transformation
        # U3(θ,φ,λ) = | cos(θ/2)            -exp(iλ) * sin(θ/2)       |
        #             | exp(iφ) * sin(θ/2)   exp(i(λ + φ)) * cos(θ/2) |
        u3(np.pi, np.pi, np.pi / 2, q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;

        // Apply the unitary transformation
        // U3(θ,φ,λ) = | cos(θ/2)            -exp(iλ) * sin(θ/2)       |
        //             | exp(iφ) * sin(θ/2)   exp(i(λ + φ)) * cos(θ/2) |
        u3(M_PI, M_PI, M_PI_2, q);


Adjoint and Controlled Operations
==================================

.. tab:: Python

    The :code:`adj` method of any gate can be used to invoke the 
    `adjoint <https://en.wikipedia.org/wiki/Conjugate_transpose>`__ transformation:

    .. code-block:: python

        # Create a kernel and allocate a qubit in a |0> state.
        qubit = cudaq.qubit()

        # Apply the unitary transformation defined by the matrix
        # T = | 1      0     |
        #     | 0  exp(iπ/4) |
        # to the state of the qubit `q`:
        t(qubit)

        # Apply its adjoint transformation defined by the matrix
        # T† = | 1      0     |
        #      | 0  exp(-iπ/4) |
        t.adj(qubit)
        # `qubit` is now again in the initial state |0>.

    The :code:`ctrl` method of any gate can be used to apply the transformation
    conditional on the state of one or more control qubits, see also this 
    `Wikipedia entry <https://en.wikipedia.org/wiki/Quantum_logic_gate#Controlled_gates>`__.

    .. code-block:: python

        # Create a kernel and allocate qubits in a |0> state.
        ctrl_1, ctrl_2, target = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
        # Create a superposition.
        h(ctrl_1)
        # `ctrl_1` is now in a state (|0> + |1>) / √2.

        # Apply the unitary transformation
        # | 1  0  0  0 |
        # | 0  1  0  0 |
        # | 0  0  0  1 |
        # | 0  0  1  0 |
        x.ctrl(ctrl_1, ctrl_2)
        # `ctrl_1` and `ctrl_2` are in a state (|00> + |11>) / √2.

        # Set the state of `target` to |1>:
        x(target)
        # Apply the transformation T only if both 
        # control qubits are in a |1> state:
        t.ctrl([ctrl_1, ctrl_2], target)
        # The qubits ctrl_1, ctrl_2, and target are now in a state
        # (|000> + exp(iπ/4)|111>) / √2.

.. tab:: C++

    The template argument :code:`cudaq::adj` can be used to invoke the 
    `adjoint <https://en.wikipedia.org/wiki/Conjugate_transpose>`__ transformation:

    .. code-block:: cpp

        // Allocate a qubit in a |0> state.
        cudaq::qubit qubit;

        // Apply the unitary transformation defined by the matrix
        // T = | 1      0     |
        //     | 0  exp(iπ/4) |
        // to the state of the qubit `q`:
        t(qubit);

        // Apply its adjoint transformation defined by the matrix
        // T† = | 1      0     |
        //      | 0  exp(-iπ/4) |
        t<cudaq::adj>(qubit);
        // Qubit `q` is now again in the initial state |0>.

    The template argument :code:`cudaq::ctrl` can be used to apply the transformation
    conditional on the state of one or more control qubits, see also this 
    `Wikipedia entry <https://en.wikipedia.org/wiki/Quantum_logic_gate#Controlled_gates>`__.

    .. code-block:: cpp

        // Allocate qubits in a |0> state.
        cudaq::qubit ctrl_1, ctrl_2, target;
        // Create a superposition.
        h(ctrl_1);
        // Qubit ctrl_1 is now in a state (|0> + |1>) / √2.

        // Apply the unitary transformation
        // | 1  0  0  0 |
        // | 0  1  0  0 |
        // | 0  0  0  1 |
        // | 0  0  1  0 |
        x<cudaq::ctrl>(ctrl_1, ctrl_2);
        // The qubits ctrl_1 and ctrl_2 are in a state (|00> + |11>) / √2.

        // Set the state of `target` to |1>:
        x(target);
        // Apply the transformation T only if both 
        // control qubits are in a |1> state:
        t<cudaq::ctrl>(ctrl_1, ctrl_2, target);
        // The qubits ctrl_1, ctrl_2, and target are now in a state
        // (|000> + exp(iπ/4)|111>) / √2.


Following common convention, by default the transformation is applied to the target qubit(s)
if all control qubits are in a :code:`|1>` state. 
However, that behavior can be changed to instead apply the transformation when a control qubit is in 
a :code:`|0>` state by negating the polarity of the control qubit.
The syntax for negating the polarity is the not-operator preceding the
control qubit: 

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit c, q;
        h(c);
        x<cudaq::ctrl>(!c, q);
        // The qubits c and q are in a state (|01> + |10>) / √2.

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

    .. code-block:: python

        qubit = cudaq.qubit()
        mz(qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;
        mz(qubit);

:code:`mx`
---------------------

This operation measures a qubit with respect to the Pauli-X basis, 
i.e., it projects the state of that qubit onto the eigenvectors of the Pauli-X matrix.
This is a non-linear transformation, and no template overloads are available.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()
        mx(qubit)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;
        mx(qubit);

:code:`my`
---------------------

This operation measures a qubit with respect to the Pauli-Y basis, 
i.e., it projects the state of that qubit onto the eigenvectors of the Pauli-Y matrix.
This is a non-linear transformation, and no template overloads are available.

.. tab:: Python

    .. code-block:: python

        qubit = cudaq.qubit()
        kernel.my(qubit)
        
.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit qubit;
        my(qubit);


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
    

    .. code-block:: python

        import cudaq
        import numpy as np

        cudaq.register_operation("custom_h", 1. / np.sqrt(2.) * np.array([1, 1, 1, -1]))

        cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

        @cudaq.kernel
        def bell():
            qubits = cudaq.qvector(2)
            custom_h(qubits[0])
            custom_x.ctrl(qubits[0], qubits[1])

        cudaq.sample(bell).dump()

        
.. tab:: C++

    The macro :code:`CUDAQ_REGISTER_OPERATION` accepts a unique name for the 
    operation, the number of target qubits, the number of rotation parameters 
    (can be 0), and the unitary matrix as a 1D row-major `std::vector<complex>` 
    representation.
    
    .. code-block:: cpp

        #include <cudaq.h>

        CUDAQ_REGISTER_OPERATION(custom_h, 1, 0,
                                {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2})

        CUDAQ_REGISTER_OPERATION(custom_x, 1, 0, {0, 1, 1, 0})

        __qpu__ void bell_pair() {
            cudaq::qubit q, r;
            custom_h(q);
            custom_x<cudaq::ctrl>(q, r);
        }

        int main() {
            auto counts = cudaq::sample(bell_pair);
            for (auto &[bits, count] : counts) {
                printf("%s\n", bits.data());
            }
        }


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

  Custom operations are currently supported only on :doc:`../using/backends/simulators`.
  Attempt to use with a hardware backend will result in runtime error.
