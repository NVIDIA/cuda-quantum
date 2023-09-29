# Quake Dialect

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

The main motivation behind Quake's value model is to directly expose
quantum and classical data dependencies for optimization purposes,
i.e., to represent the dataflow in quantum computations.  In contrast
to Quake's memory model, which uses memory semantics (quantum
operators act as side-effects on qubit references), the value model
uses value semantics, that is quantum operators consume and produce
values. These values are not truly SSA values, however, as operations
still have side-effects on the value itself and the value cannot be
copied.

Let's see an example to clarify the distinction between the models.  Take the
following Quake implementation of some toy quantum computation:

```cpp
func.func foo(%veq : !quake.veq<2>) {
    // Boilerplate to extract each qubit from the vector
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q0 = quake.extract_ref %veq[%c0] : (!quake.veq<2>, index) -> !quake.ref
    %q1 = quake.extract_ref %veq[%c1] : (!quake.veq<2>, index) -> !quake.ref

    // We apply some operators to those extracted qubits
    // ... bunch of operators using %q0 and %q1 ...
    quake.h %q0 : (!quake.ref) -> ()

    // We decide to measure the vector
    %result = quake.mz %veq : (!quake.veq<2>) -> cc.stdvec<i1>

    // And then apply another Hadamard to %q0
    quake.h %q0 : (!quake.ref) -> ()
    // ...
}
```

Now imagine we want to optimize this code by removing pair of adjacent
adjoint operators, e.g., if we have a pair Hadamard operations next to each
other on the same qubit---visually:

```text
    ┌───┐ ┌───┐         ┌───┐
   ─┤ H ├─┤ H ├─  =  ───┤ I ├───  =  ─────────────
    └───┘ └───┘         └───┘
```

Where `I` is the identity operator. Now note that a naive implemention of
this optimization for Quake would optimize away both `quake.h` operators
being applied to `%q0`.  Such an implementation would have missed the fact
that a measurement is being applied to the vector, `%veq`, which contains
`%q0`.

Of course it is possible to correctly implement this optimization for Quake.
However such an implementation would be quite error-prone and require
complex analyses.  For this reason, Quake has overloaded gates.

In the value model operators consume values and return new values:

```text
  %q0_1 = quake.op %q0_0 : (!quake.wire) -> !quake.wire
```

We can visualize the difference between memory and value representation as:

```text
            Memory                                   Value

        ┌──┐ ┌──┐     ┌──┐                  ┌──┐ %q0_1 ┌──┐     ┌──┐
   %q0 ─┤  ├─┤  ├─···─┤  ├─ %q0  vs  %q0_0 ─┤  ├───────┤  ├─···─┤  ├─ %q0_Z
        └──┘ └──┘     └──┘                  └──┘       └──┘     └──┘
```

If we look at the implementation again, we notice that the problem
with the naive optimization happens because the Hadamard operators are
implicitly connected by the same value `%q0`. In value form, all the gates
are explicitly connected by distinct values, which eliminates the need
to do further analysis via implicit side-effects.
The following is the implementation in value form.

```text
func.func @foo(%array : !quake.qvec<2>) {
    // Boilerplate to extract each qubit
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r0 = quake.extract_ref %array[%c0] : (!quake.qvec<2>, index) -> !quake.qref
    %r1 = quake.extract_ref %array[%c1] : (!quake.qvec<2>, index) -> !quake.qref

    // Unwrap the quantum references to expose the wires.
    %q0 = quake.unwrap %r0 : (!quake.qref) -> !quake.wire
    %q1 = quake.unwrap %r1 : (!quake.qref) -> !quake.wire

    // Misc. operators applied
    %q0_M = quake.h %q0_L : (!quake.wire) -> !quake.wire

    // Re-wrap the wire to its original source
    quake.wrap %q0_M to %r0 : !quake.wire, !quake.qref
    quake.wrap %q1_X to %r1 : !quake.wire, !quake.qref

    // Measure the entire vector of quantum references
    %result = quake.mz %array : (!quake.qvec<2>) -> !cc.stdvec<i1>

    // Unwrap the wire for qubit 0 again
    %q0_P = quake.unwrap %r0 : (!quake.qref) -> !quake.wire
    ...
    %q0_Z = quake.h %q0_Y : (!quake.wire) -> !quake.wire
    // Re-wrap the wire back to the original reference
    quake.wrap %q0_Z to %r0 : !quake.wire, !quake.qref
    return
}
```

In this code we can more straightforwardly see that the Hadamard
operators cannot cancel each other.  One way of reasoning about this
is as follows: In value form we need to follow a chain of values to
know the qubit operators are being applied to, in this example:

```text
Mmeory                          Value
    %q0         [%q0_0, %q0_1 ... %q0_L, %q0_M; %q0_P ... %q0_Y, %q0_Z]

```

We know that one Hadamard is applied to `%q0_L` and generates `%q0_M`, and
the other is applied `%q0_Y` and generates `%q0_Z`.  Hence, there is no
connection between them---which means they cannot cancel each other out.

## Quake Types

In value form, we use the `!quake.wire` type to represent an
intermediate "state" of a single qubit in time.  One can view the
values of this type as line segments in a quantum circuit diagram.
