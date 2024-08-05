Quantum Computing 101
======================

The fundamental unit of classical information storage, processing and
transmission is the bit. Analogously, we define its quantum counterpart,
a quantum bit or simply the qubit.

Classical bits are transistor elements whose states can be altered to
perform computations. Similarly qubits too have physical realizations
within superconducting materials, ion-traps and photonic systems. We
shall not concern ourselves with specific qubit architectures but rather
think of them as systems which obey the laws of quantum mechanics and
the mathematical language physicists have developed to describe the
theory: linear algebra.

Quantum States
-----------------------------

Information storage scales linearly if bits have a single state. Access
to multiple states, namely a 0 and a 1 allows for information encoding
to scale logarithmically. Similarly we define a qubit to have the states
:math:`\ket{0}` and :math:`\ket{1}` in Dirac notation where:

.. math:: \ket{0} = \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix}

.. math:: \ket{1} = \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}

Rather than just the two states each classical bit can be in,
quantum theory allows one to explore linear combinations of states,
also called superpositions:

.. math::   \ket{\psi} = \alpha\ket{0} + \beta\ket{1} 

where :math:`\alpha` and :math:`\beta` :math:`\in \mathbb{C}`. It is
important to note that this is still the state of one qubit; 
although we have two kets, they both represent a
superposition state of one qubit.

If we have two classical bits, the possible states we could encode
information in would be 00, 01, 10 and 11. Correspondingly, multiple
qubits can be combined and the possible combinations of their states
used to process information.

A two qubit system has four computational basis states:
:math:`\ket{00}, \ket{01}, \ket{10}, \ket{11}`.

More generally, the quantum state of a :math:`n` qubit system is written
as a sum of :math:`2^n` possible basis states where the coefficients
track the probability of the system collapsing into that state if a
measurement is applied.

For :math:`n = 500`, :math:`2^n \approx 10^{150}` which is greater than
the number of atoms in the universe. Storing the complex numbers
associated with :math:`2^{500}` amplitudes would not be feasible using
bits and classical computations but nature seems to only require 500
qubits to do so. The art of quantum computation is thus to build quantum
systems that we can manipulate with fine precision such that evolving a
large statevector can be offloaded onto a quantum computer.


Quantum Gates
-----------------------------

We can manipulate the state of a qubit via quantum gates. 
For example, the Pauli X gate allows us to flip the state of the qubit:

.. math::  X \ket{0} = \ket{1} 

.. math::  \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix} 

.. literalinclude:: ../../snippets/python/using/examples/pauli_x_gate.py
    :language: python
    :start-after: [Begin Docs]
    :end-before: [End Docs]

.. parsed-literal::

    { 1:1000 }

The Hadamard gate allows us to put the qubit in an equal superposition
state:

.. math::  H \ket{0} =  \tfrac{1}{\sqrt{2}} \ket{0} + \tfrac{1}{\sqrt{2}} \ket{1}  \equiv \ket{+}

.. math:: \tfrac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} = \tfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} + \tfrac{1}{\sqrt{2}} \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}. 

The probability of finding the qubit in the 0 / 1 state is hence
:math:`\lvert \tfrac{1}{\sqrt{2}} \rvert ^2 = \tfrac{1}{2}`. Lets verify
this with some code:

.. literalinclude:: ../../snippets/python/using/examples/hadamard_gate.py
    :language: python
    :start-after: [Begin Docs]
    :end-before: [End Docs]

.. parsed-literal::

    { 0:502 1:498 }

For a qubit in a superposition state, quantum gates
act linearly:

.. math::    X (\alpha\ket{0} + \beta\ket{1}) = \alpha\ket{1} + \beta\ket{0} 

As we evolve quantum states via quantum gates, the normalization
condition requires that the sum of modulus squared of amplitudes must
equal 1 at all times:

.. math::   \ket{\psi} = \alpha\ket{0} + \beta\ket{1},          |\alpha|^2 + |\beta|^2 = 1. 

This is to adhere to the conservation of probabilities which translates
to a constraint on types of quantum gates we can define.
For a general quantum state :math:`\ket{\psi}`, upholding the
normalization condition requires quantum gates to be unitary, that is
:math:`U^{\dagger}U = U^{*^{T}}U = \mathbb{I}`.

Just like the single-qubit gates above, we can define 
multi-qubit gates to act on multiple-qubits.
The controlled-NOT or CNOT gate, for example, acts on 2 qubits: the control qubit and
the target qubit. Its effect is to flip the target if the control is in
the excited :math:`\ket{1}` state.

.. literalinclude:: ../../snippets/python/using/examples/cnot_gate.py
    :language: python
    :start-after: [Begin Docs]
    :end-before: [End Docs]

.. parsed-literal::

    { 11:1000 }
    

The CNOT gate in matrix notation is represented as

.. math::  CNOT \equiv \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix} 

and one can check that :math:`CNOT^\dagger CNOT = \mathbb{I}`.
Its effect on the computational basis states is:

.. math::  CNOT\ket{00} = \ket{00} 

.. math::  CNOT\ket{01} = \ket{01} 

.. math::  CNOT\ket{10} = \ket{11} 

.. math::  CNOT\ket{11} = \ket{10} 

For a full list of gates supported in CUDA-Q see :doc:`../../api/default_ops`. 

Measurements
-----------------------------

Quantum theory is probabilistic and hence requires statistical inference
to derive observations. Prior to measurement, the state of a qubit is
all possible combinations of :math:`\alpha` and :math:`\beta` and upon
measurement, wavefunction collapse yields either a classical 0 or 1.

The mathematical theory devised to explain quantum phenomena tells us
that the probability of observing the qubit in the state
:math:`\ket{0}` / :math:`\ket{1}` yielding a classical 0 / 1 is
:math:`\lvert \alpha \rvert ^2` / :math:`\lvert \beta \rvert ^2`. 

As we see in the example of the Hadamard gate above,
the result 0 or 1 each is yielded roughly 50% of the times as predicted 
by the postulate stated above thus proving the theory.

Classically, we cannot encode information within states such as 00 + 11
but quantum mechanics allows us to write linear superpositions

.. math::   \ket{\psi} = \alpha_{00}\ket{00} + \alpha_{01}\ket{01} + \alpha_{10}\ket{10} + \alpha_{11}\ket{11}

where the probability of measuring :math:`x = 00, 01, 10, 11` occurs
with probability :math:`\lvert \alpha_{x} \rvert ^2` with the
normalization condition that
:math:`\sum_{x \in \{ 0,1 \}^2} \lvert \alpha_{x} \rvert ^2 = 1`.
