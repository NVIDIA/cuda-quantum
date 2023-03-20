# QTX Dialect

## General Introduction

The quantum circuit model is the most widely used model of quantum
computation.  It provides a convenient tool for formulating quantum
algorithms and an architecture for the physical construction of quantum
computers.

A _quantum circuit_ represents a computation as a sequence of quantum
operators applied to quantum data.  In our case, the quantum data is a set
of quantum bits, or qubits for short.  Physically, a qubit is an object with
only two distinguishable states, i.e., it is a two-state quantum mechanical
system such as a spin-1/2 particle.

Conceptually, a _quantum operator_ is an effect that might modify the state
of a subset of qubits. Most often, this effect is unitary evolution---in
this case, we say that the operator is a _unitary_.  The number of target
qubits an operator acts upon is an intrinsic property.

A _quantum instruction_ is the embodiment of a quantum operator when applied
to a specific subset of qubits.  The number of qubits must be equal to (or
greater than) the number of target qubits intrinsic to the operator.  If
greater, the extra qubits are considered controls.

## Motivation

The main motivation behind QTX is to directly expose quantum and classical
data dependencies for optimization purposes, i.e., to represent the dataflow
in quantum computations.  In contrast to Quake, which uses memory semantics
(quantum operators act as side-effects on qubit references), QTX uses value
semantics, that is quantum operators consume and produce values.

Let's see an example to clarify such an important difference.  Take the
following Quake implementation of some toy quantum computation:

```cpp
func.func foo(%qvec : !quake.qvec<2>) {
    // Boilerplate to extract each qubit from the vector
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q0 = quake.qextract %qvec[%c0 : index] : !quake.qvec<2> -> !quake.qref
    %q1 = quake.qextract %qvec[%c1 : index] : !quake.qvec<2> -> !quake.qref

    // We apply some operators to those extracted qubits
    ... bunch of operators using %q0 and %q1 ...
    quake.h (%q0)

    // We decide to measure the vector
    %result = quake.mz (%qvec) : vector<2xi1>

    // And then apply another Hadamard to %q0
    quake.h (%q0)
    ...
}
```

Now imagine we want to optimize this code by removing pair of adjacent
adjoint operators, e.g., if we have a pair Hadamard operations next to each
other on the same qubit---visually:

```txt
    ┌───┐ ┌───┐         ┌───┐
   ─┤ H ├─┤ H ├─  =  ───┤ I ├───  =  ─────────────
    └───┘ └───┘         └───┘
```

Where `I` is the identity operator. Now note that a naive implemention of
this optimization for Quake would optimize away both `quake.h` operators
being applied to `%q0`.  Such an implementation would have missed the fact
that a measurement is being applied to the vector, `%qvec`, which contains
`%q0`.

Of course it is possible to correctly implement this optimization for Quake.
However such an implementation would be quite error-prone and require
complex analyses.  For this reason, we have QTX.

In QTX operators consume values and returns new values:

```cpp
%q0_1 = qtx.op %q0_0 : !qtx.wire
```

We can visualize the difference between Quake and QTX as:

```txt
            Quake                                    QTX

        ┌──┐ ┌──┐     ┌──┐                  ┌──┐ %q0_1 ┌──┐     ┌──┐
   %q0 ─┤  ├─┤  ├─···─┤  ├─ %q0  vs  %q0_0 ─┤  ├───────┤  ├─···─┤  ├─ %q0_Z
        └──┘ └──┘     └──┘                  └──┘       └──┘     └──┘
```

If we look at the Quake implementation again, we notice that the problem
with the naive optimization happens because the Hadamard operators are
connected by the same value `%q0`.  Since QTX consumes values and return
new values, we won't have this problem.  The following is the implementation
in QTX (omitting some types to not clutter the code too much):

```cpp
qtx.circuit foo(%array : !qtx.wire_array<2>) -> (!qtx.wire_array<2>) {
    // Boilerplate to extract each qubit from the array
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q0, %new_array = array_borrow %c0 from %array : ...
    %q1, %new_array_1 = array_borrow %c1 from %new_array_1 : ...

    ... bunch o operators ...
    %q0_X = h %q0_W : !qtx.wire

    // We give the wire back to the array
    %new_array_2 = array_yield %q0_X, %q1_W to %new_array_1 : ...

    // Measure the array
    %result, %new_array_3 = mz %new_array_2 : !qtx.wire_array<2> -> vector<2xi1>, !qtx.wire_array<2>

    // Borrow the wire for qubit 0 again:
    %q0_Y, %new_array_4 = array_borrow %c0 from %new_array_3 : ...

    %q0_Z = h %q0_Y : !qtx.wire
    ...
    return %new_array_Z : !qtx.wire_array<2>
}
```

In this code we can more straightforwardly see that the Hadamard operators
cannot cancel each other.  One way of reasoning about this is as follows:
In QTX we need to follow a chain of values to know the qubit operators are
being applied to, in this example:

```txt
Quake                        QTX
    %q0      [%q0_0, %q0_1, ..., %q0_W, %q0_X, %q0_Y, %q0_Z]

```

We know that one Hadamard is applied to `%q0_W` and generates `%q0_X`, and
the other is applied `%q0_Y` and generates `%q0_Z`.  Hence, there is no
connection between them---which means they cannot cancel each other out.

## QTX Types

In QTX, we use the `!qtx.wire` type to represent an intermediate "state" of
a single qubit in time.  One can view the values of this type as line
segments in a quantum circuit diagram.  The dialect also defines a aggregate
type for wires, `!qtx.array<size>`, that has a fix size, known at
compilation time. (For more information about those types look at their
respective descriptions.)
