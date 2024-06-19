Quantum Operations
==================

Qubit
-----

The fundamental unit of classical information storage, processing and
transmission is the bit. Analogously, we define its quantum counterpart,
a quantum bit or simply the qubit. Below we define a qubit in CUDA-Q.

.. literalinclude:: ../../snippets/python/using/examples/build_kernel.py
    :language: python

Classical bits are transistor elements whose states can be altered to
perform computations. Similarly qubits too have physical realizations
within superconducting materials, ion-traps and photonic systems. We
shall not concern ourselves with specific qubit architectures but rather
think of them as systems which obey the laws of quantum mechanics and
the mathematical language physicists have developed to describe the
theory: linear algebra.

Information storage scales linearly if bits have a single state. Access
to multiple states, namely a 0 and a 1 allows for information encoding
to scale logarithmically. Similarly we define a qubit to have the states
:math:`\ket{0}` and :math:`\ket{1}` in Dirac notation where:

.. math:: \ket{0} = \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix}

.. math:: \ket{1} = \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}

Pauli X Gate
------------

We can manipulate the state of the qubit via quantum gates. The Pauli X
gate allows us to flip the state of the qubit:

.. math::  X \ket{0} = \ket{1} 

.. math::  \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix} 

.. literalinclude:: ../../snippets/python/using/examples/pauli_x_gate.py
    :language: python

.. parsed-literal::

    { 1:1000 }
    


Superpositions & Measurements
-----------------------------

We have explored the 2 states accessible to us via a qubit. In fact,
quantum theory allows one to explore linear combinations of states
namely superpositions:

.. math::   \ket{\psi} = \alpha\ket{0} + \beta\ket{1} 

where :math:`\alpha` and :math:`\beta` :math:`\in \mathbb{C}`. It is
important to note that this is still the state of one qubit even though
:math:`\ket{\psi}` has 2 kets.

Quantum theory is probabilistic and hence requires statistical inference
to derive observations. Prior to measurement, the state of a qubit is
all possible combinations of :math:`\alpha` and :math:`\beta` and upon
measurement, wavefunction collapse yields either a classical 0 or 1.

The mathematical theory devised to explain quantum phenomena tells us
that the probability of observing the qubit in the state
:math:`\ket{0}` / :math:`\ket{1}` yielding a classical 0 / 1 is
:math:`\lvert \alpha \rvert ^2` / :math:`\lvert \beta \rvert ^2`. The
theory has been verified experimentally countless times and we shall
verify it once more below.

The Hadamard gate allows us to put the qubit in an equal superposition
state:

.. math::  H \ket{0} =  \tfrac{1}{\sqrt{2}} \ket{0} + \tfrac{1}{\sqrt{2}} \ket{1}  \equiv \ket{+}

.. math:: \tfrac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} = \tfrac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 0 \\ \end{bmatrix} + \tfrac{1}{\sqrt{2}} \begin{bmatrix} 0 \\ 1 \\ \end{bmatrix}. 

The probability of finding the qubit in the 0 / 1 state is hence
:math:`\lvert \tfrac{1}{\sqrt{2}} \rvert ^2 = \tfrac{1}{2}`. Lets verify
this with some code:

.. literalinclude:: ../../snippets/python/using/examples/hadamard_gate.py
    :language: python

.. parsed-literal::

    { 0:502 1:498 }
    


Quantum theory is statistical and statistical accuracy increases with
sampling. Above we see how with a 1000 shots, the result 0 / 1 is
yielded roughly 50% of the times as predicted by the postulate stated
above thus proving the theory.

For completeness:

.. math::  H \ket{1} =  \tfrac{1}{\sqrt{2}} \ket{0} - \tfrac{1}{\sqrt{2}} \ket{1}  \equiv \ket{-}

Qubit Visualizations
--------------------

What are the possible states our qubit can be in and how can we build up
a visual cue to help us make sense of quantum states and their
evolution?

We know our qubit can have 2 distinct states: :math:`\ket{0}` and
:math:`\ket{1}`. Maybe we need a 1 dimensional line whose vertices can
represent each of the aforementioned states.

We also know that qubits can be in an equal superposition states:
:math:`\ket{+}` and :math:`\ket{-}`. This now forces us to extend our
1-D line to a 2-D Cartesian coordinate system.

Later, we will learn the existence of states that can be represented
with :math:`\ket{+i}` and :math:`\ket{-i}`, this calls for a 3-D
extension.

It turns out that a sphere is able to depict all the possible states of
a single qubit. This is called a Bloch sphere and as shown in figure below:

Gate Linearity
--------------

Lets manipulate a single qubit:

1. 

   .. math::   X  \ket{0} = \ket{1}  

2. 

   .. math::   X  \ket{1} = \ket{0}  

And more generally, for a qubit in a superposition state, quantum gates
act linearly:

.. math::    X (\alpha\ket{0} + \beta\ket{1}) = \alpha\ket{1} + \beta\ket{0} 

It is important to note that states such as
:math:`\alpha\ket{0} + \beta\ket{1}` reference a single qubit in a
superposition state. Although we have two kets, they both represent a
superposition state of one qubit. We shall explore multiple qubits and
their notation in the next chapter.

Gate Unitarity
--------------

As we evolve quantum states via quantum gates, the normalization
condition requires that the sum of modulus squared of amplitudes must
equal 1 at all times:

.. math::   \ket{\psi} = \alpha\ket{0} + \beta\ket{1},          |\alpha|^2 + |\beta|^2 = 1. 

This is to adhere to the conservation of probabilities which translates
to a constraint on types of quantum gates we can define.

For a general quantum state :math:`\ket{\psi}`, upholding the
normalization condition requires quantum gates to be unitary, that is
:math:`U^{\dagger}U = U^{*^{T}}U = \mathbb{I}`.

Single Qubit Gates
------------------

Below we summarize a few single qubit gates and their effects on quantum
states:

.. math::  X \equiv \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \hspace{1cm} X(\alpha\ket{0} + \beta\ket{1}) = \alpha\ket{1} + \beta\ket{0} \hspace{1cm}  

.. math::  Z \equiv \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \hspace{1cm} Z(\alpha\ket{0} + \beta\ket{1}) = \alpha\ket{0} - \beta\ket{1}  

.. math::  H \equiv \tfrac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \hspace{1cm} H(\alpha\ket{0} + \beta\ket{1}) = \alpha\tfrac{\ket{0}+\ket{1}}{\sqrt{2}} + \beta\tfrac{\ket{0}-\ket{1}}{\sqrt{2}}  

Multiple Qubits
===============

If we have 2 classical bits, the possible states we could encode
information in would be 00, 01, 10 and 11. Correspondingly, multiple
qubits can be combined and the possible combinations of their states
used to process information.

A two qubit system has 4 computational basis states:
:math:`\ket{00}, \ket{01}, \ket{10}, \ket{11}`.

Classically, we cannot encode information within states such as 00 + 11
but quantum mechanics allows us to write linear superpositions

.. math::   \ket{\psi} = \alpha_{00}\ket{00} + \alpha_{01}\ket{01} + \alpha_{10}\ket{10} + \alpha_{11}\ket{11}

where the probability of measuring :math:`x = 00, 01, 10, 11` occurs
with probability :math:`\lvert \alpha_{x} \rvert ^2` with the
normalization condition that
:math:`\sum_{x \in \{ 0,1 \}^2} \lvert \alpha_{x} \rvert ^2 = 1`

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

Some Notation Conventions
-------------------------

Qubit counting starts from 0 and the 0th qubit is represented on the
left most side in Dirac notation. For e.g. in :math:`\ket{01}` the 0th 
qubit is in state :math:`\ket{0}` and the first in state
:math:`\ket{1}`.

For brevity, we denote gate application with subscripts to reference the
qubit it acts on. For e.g. :math:`X_{0}\ket{00} = \ket{10}` refers to
:math:`X_{0}` acting on the 0th qubit flipping it to the state 1 as
shown. Below we see how this is done in CUDA-Q.

.. literalinclude:: ../../examples/python/notation.py
    :language: python

.. parsed-literal::

         ╭───╮
    q0 : ┤ x ├
         ╰───╯
    
    { 10:1000 }
    


Controlled-NOT Gate
-------------------

Analogous to classical computing, we now introduce multi-qubit gates to
quantum computing.

The controlled-NOT or CNOT gate acts on 2 qubits: the control qubit and
the target qubit. Its effect is to flip the target if the control is in
the excited :math:`\ket{1}` state.

We use the notation CNOT01\ :math:`\ket{10} = \ket{11}` to describe its
effects. The subscripts denote that the 0th qubit is the control qubit
and the 1st qubit is the target qubit.

.. literalinclude:: ../../examples/python/cnot.py
    :language: python

.. parsed-literal::

    { 11:1000 }
    


In summary, the CNOT gate in matrix notation is represented as:

.. math::  CNOT \equiv \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix} 

To conserve probability and preserve the normalization condition,
quantum gates must obey unitarity and one can check that
:math:`CNOT^\dagger CNOT = \mathbb{I}`

and its effect on the computational basis states is:

.. math::  CNOT_{01}\ket{00} = \ket{00} 

.. math::  CNOT_{01}\ket{01} = \ket{01} 

.. math::  CNOT_{01}\ket{10} = \ket{11} 

.. math::  CNOT_{01}\ket{11} = \ket{10} 
