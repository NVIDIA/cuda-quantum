Quantum Operations
******************************

CUDA Quantum provides a default set of quantum operations on qubits. 
These operations can be used to define custom kernels and libraries.
Since the set of quantum intrinsic operations natively supported on a specific target 
depends on the backends architecture, the :code:`nvq++` compiler automatically
decomposes the default operations into the appropriate set of intrinsic operations 
for that target.

The sections `Unitary Operations on Qubits`_ and `Measurements on Qubits`_ list the default set of quantum operations on qubits.

Operations that implement unitary transformations of the quantum state are templated.
The template argument allows to invoke the adjoint and controlled version of the quantum transformation, see the section on `Adjoint and Controlled Operations`_.

CUDA Quantum additionally provides overloads to support broadcasting of
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

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # X = | 0  1 |
        #     | 1  0 |
        kernel.x(q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // X = | 0  1 |
        //     | 1  0 |
        x(q);

:code:`y`
---------------------

This operation implements the transformation defined by the Pauli-Y matrix.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # Y = | 0  -i |
        #     | i   0 |
        kernel.y(q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // Y = | 0  -i |
        //     | i   0 |
        y(q);

:code:`z`
---------------------

This operation implements the transformation defined by the Pauli-Z matrix.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # Z = | 1   0 |
        #     | 0  -1 |
        kernel.z(q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // Z = | 1   0 |
        //     | 0  -1 |
        z(q);

:code:`h`
---------------------

This operation is a rotation by π about the X+Z axis, and 
enables one to create a superposition of computational basis states.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # H = (1 / sqrt(2)) * | 1   1 |
        #                     | 1  -1 |
        kernel.h(q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // H = (1 / sqrt(2)) * | 1   1 |
        //                     | 1  -1 |
        h(q);

:code:`r1`
---------------------

This operation is an arbitrary rotation about the :code:`|1>` state.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # R1(λ) = | 1     0    |
        #         | 0  exp(iλ) |
        kernel.r1(math.pi, q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // R1(λ) = | 1     0    |
        //         | 0  exp(iλ) |
        r1(std::numbers::pi, q);

:code:`rx`
---------------------

This operation is an arbitrary rotation about the X axis.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
        #         | -isin(θ/2)  cos(θ/2)  |
        kernel.rx(math.pi, q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
        //         | -isin(θ/2)  cos(θ/2)  |
        rx(std::numbers::pi, q);

:code:`ry`
---------------------

This operation is an arbitrary rotation about the Y axis.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # Ry(θ) = | cos(θ/2)  -sin(θ/2) |
        #         | sin(θ/2)   cos(θ/2) |
        kernel.ry(math.pi, q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // Ry(θ) = | cos(θ/2)  -sin(θ/2) |
        //         | sin(θ/2)   cos(θ/2) |
        ry(std::numbers::pi, q);

:code:`rz`
---------------------

This operation is an arbitrary rotation about the Z axis.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # Rz(λ) = | exp(-iλ/2)      0     |
        #         |     0       exp(iλ/2) |
        kernel.rz(math.pi, q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // Rz(λ) = | exp(-iλ/2)      0     |
        //         |     0       exp(iλ/2) |
        rz(std::numbers::pi, q);

:code:`s`
---------------------

This operation applies to its target a rotation by π/2 about the Z axis.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # S = | 1   0 |
        #     | 0   i |
        kernel.s(q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // S = | 1   0 |
        //     | 0   i |
        s(q);

:code:`t`
---------------------

This operation applies to its target a π/4 rotation about the Z axis.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation
        # T = | 1      0     |
        #     | 0  exp(iπ/4) |
        kernel.t(q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;

        // Apply the unitary transformation
        // T = | 1      0     |
        //     | 0  exp(iπ/4) |
        t(q);

:code:`swap`
---------------------

This operation swaps the states of two qubits.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        qs = kernel.qalloc(2)

        # Apply the unitary transformation
        # Swap = | 1 0 0 0 |
        #        | 0 0 1 0 |
        #        | 0 1 0 0 |
        #        | 0 0 0 1 |
        kernel.swap(qs[0], qs[1])

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q1, q2;

        // Apply the unitary transformation
        // Swap = | 1 0 0 0 |
        //        | 0 0 1 0 |
        //        | 0 1 0 0 |
        //        | 0 0 0 1 |
        swap(q1, q2);


Adjoint and Controlled Operations
==================================

The template argument :code:`cudaq::adj` can be used to invoke the 
`adjoint <https://en.wikipedia.org/wiki/Conjugate_transpose>`__ transformation:

.. tab:: Python

    .. code-block:: python

        # Create a kernel and allocate a qubit in a |0> state.
        kernel = cudaq.make_kernel()
        q = kernel.qalloc()

        # Apply the unitary transformation defined by the matrix
        # T = | 1      0     |
        #     | 0  exp(iπ/4) |
        # to the state of the qubit `q`:
        kernel.t(q)

        # Apply its adjoint transformation defined by the matrix
        # T† = | 1      0     |
        #      | 0  exp(-iπ/4) |
        kernel.tdg(q)
        # Qubit `q` is now again in the initial state |0>.

.. tab:: C++

    .. code-block:: cpp

        // Allocate a qubit in a |0> state.
        cudaq::qubit q

        // Apply the unitary transformation defined by the matrix
        // T = | 1      0     |
        //     | 0  exp(iπ/4) |
        // to the state of the qubit `q`:
        t(q);

        // Apply its adjoint transformation defined by the matrix
        // T† = | 1      0     |
        //      | 0  exp(-iπ/4) |
        t<cudaq::adj>(q);
        // Qubit `q` is now again in the initial state |0>.

The template argument :code:`cudaq::ctrl` can be used to apply the transformation
conditional on the state of one or more control qubits, see also this 
`Wikipedia entry <https://en.wikipedia.org/wiki/Quantum_logic_gate#Controlled_gates>`__.

.. tab:: Python

    .. code-block:: python

        # Create a kernel and allocate qubits in a |0> state.
        kernel = cudaq.make_kernel()
        qs = kernel.qalloc(3)
        c1, c2, q = qs[0], qs[1], qs[2]
        # Create a superposition.
        kernel.h(c1)
        # Qubit c1 is now in a state (|0> + |1>) / √2.

        # Apply the unitary transformation
        # | 1  0  0  0 |
        # | 0  1  0  0 |
        # | 0  0  0  1 |
        # | 0  0  1  0 |
        kernel.cx(c1, c2)
        # The qubits c1 and c2 are in a state (|00> + |11>) / √2.

        # Set the state of qubit q to |1>:
        kernel.x(q)
        # Apply the transformation T only if both 
        # control qubits are in a |1> state:
        kernel.ct([c1, c2], q)
        # The qubits c1, c2, and q are now in a state
        # (|000> + exp(iπ/4)|111>) / √2.

.. tab:: C++

    .. code-block:: cpp

        // Allocate qubits in a |0> state.
        cudaq::qubit c1, c2, q;
        // Create a superposition.
        h(c1);
        // Qubit c1 is now in a state (|0> + |1>) / √2.

        // Apply the unitary transformation
        // | 1  0  0  0 |
        // | 0  1  0  0 |
        // | 0  0  0  1 |
        // | 0  0  1  0 |
        x<cudaq::ctrl>(c1, c2);
        // The qubits c1 and c2 are in a state (|00> + |11>) / √2.

        // Set the state of qubit q to |1>:
        x(q);
        // Apply the transformation T only if both 
        // control qubits are in a |1> state:
        t<cudaq::ctrl>(c1, c2, q);
        // The qubits c1, c2, and q are now in a state
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

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()
        kernel.mz(q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;
        mz(q);

:code:`mx`
---------------------

This operation measures a qubit with respect to the Pauli-X basis, 
i.e., it projects the state of that qubit onto the eigenvectors of the Pauli-X matrix.
This is a non-linear transformation, and no template overloads are available.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()
        kernel.mx(q)

.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;
        mx(q);

:code:`my`
---------------------

This operation measures a qubit with respect to the Pauli-Y basis, 
i.e., it projects the state of that qubit onto the eigenvectors of the Pauli-Y matrix.
This is a non-linear transformation, and no template overloads are available.

.. tab:: Python

    .. code-block:: python

        kernel = cudaq.make_kernel()
        q = kernel.qalloc()
        kernel.my(q)
        
.. tab:: C++

    .. code-block:: cpp

        cudaq::qubit q;
        my(q);
