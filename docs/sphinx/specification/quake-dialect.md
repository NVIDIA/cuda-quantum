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
of a subset of qubits. Most often, this effect is unitary evolution. In this
case, we say that the operator is a _unitary_.  The number of target
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

Let's see an example to clarify the distinction between the models. Take the
following Quake implementation of some toy quantum computation:

```mlir
func.func @foo(%veq: !quake.veq<2>)
    -> !cc.stdvec<!quake.measure> {
  // Boilerplate to extract each qubit from the vector
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %q0 = quake.extract_ref %veq[%c0]
      : (!quake.veq<2>, index) -> !quake.ref
  %q1 = quake.extract_ref %veq[%c1]
      : (!quake.veq<2>, index) -> !quake.ref

  // We apply an operator to the first extracted qubit
  quake.h %q0 : (!quake.ref) -> ()

  // We decide to measure the vector
  %result = quake.mz %veq
      : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>

  // And then apply another Hadamard to %q0
  quake.h %q0 : (!quake.ref) -> ()
  return %result : !cc.stdvec<!quake.measure>
}
```

Now imagine we want to optimize this code by removing a pair of adjacent
adjoint operators. For example, consider a pair of Hadamard operations next to
each other on the same qubit:

```text
    ┌───┐ ┌───┐         ┌───┐
   ─┤ H ├─┤ H ├─  =  ───┤ I ├───  =  ─────────────
    └───┘ └───┘         └───┘
```

Here, `I` is the identity operator. Now note that a naive implementation of
this optimization for Quake would optimize away both `quake.h` operators being
applied to `%q0`. Such an implementation would have missed the fact that a
measurement is being applied to the vector, `%veq`, which contains the qubit
referenced by `%q0`.

Of course it is possible to correctly implement this optimization for Quake.
However such an implementation would be quite error-prone and require complex
analyses. For this reason, Quake has overloaded gates. Reference and value
forms can coexist within the same function body and, where supported, on a
single quantum operation.

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

If we look at the implementation again, we notice that the problem with the
naive optimization happens because the Hadamard operators are implicitly
connected by the same reference `%q0`, while the measurement reaches that
reference through `%veq`. In value form, all the gates are explicitly
connected by distinct values, which eliminates the need to do further analysis
via implicit side-effects. The following is the implementation in value form.

```mlir
func.func @foo_value(%q0: !quake.wire, %q1: !quake.wire)
    -> (!cc.stdvec<!quake.measure>, !quake.wire, !quake.wire) {
  %q0_after_first_h = quake.h %q0
      : (!quake.wire) -> !quake.wire
  %result, %after_measurement:2 = quake.mz %q0_after_first_h, %q1
      : (!quake.wire, !quake.wire)
        -> (!cc.stdvec<!quake.measure>, !quake.wire, !quake.wire)
  %q0_after_second_h = quake.h %after_measurement#0
      : (!quake.wire) -> !quake.wire
  return %result, %q0_after_second_h, %after_measurement#1
      : !cc.stdvec<!quake.measure>, !quake.wire, !quake.wire
}
```

In this code we can more straightforwardly see that the Hadamard operators
cannot cancel each other. One way of reasoning about this is as follows: In
value form we need to follow a chain of values to know which qubit the
operators are being applied to, in this example:

```text
%q0
  -> quake.h  -> %q0_after_first_h
  -> quake.mz -> %after_measurement#0
  -> quake.h  -> %q0_after_second_h
```

We know that one Hadamard is applied to `%q0` and generates
`%q0_after_first_h`, the measurement consumes that value and generates
`%after_measurement#0`, and the other Hadamard is applied to
`%after_measurement#0` and generates `%q0_after_second_h`. Hence, the
measurement lies between them, which means they cannot cancel each other out.

The example threads each wire from a function argument through every operation
that uses it and then to a function result. Region-based control-flow
operations similarly thread wires through their region arguments and return
updated wires as operation results. For example, a `cc.if` passes `%q` into
both regions as `%arg`, and each region returns the wire produced by its gate:

```mlir
func.func @conditional(%condition: i1, %q: !quake.wire) -> !quake.wire {
  %updated:1 = cc.if (%condition) ((%arg = %q)) -> (!quake.wire) {
    %then = quake.h %arg : (!quake.wire) -> !quake.wire
    cc.continue %then : !quake.wire
  } else {
    %else = quake.x %arg : (!quake.wire) -> !quake.wire
    cc.continue %else : !quake.wire
  }
  return %updated#0 : !quake.wire
}
```

Only the selected region consumes `%arg` at runtime. In a control-flow graph,
branches instead pass wires as branch operands to successor block arguments. A
conditional branch may pass the same wire to both successors because only the
selected path consumes it. Transformations must preserve this threading when
rewriting control flow.

Value semantics applies when the individual qudits can be represented
explicitly. Reference semantics remains useful for dynamically sized
collections and runtime-selected elements. A transformation must account for
the representation it accepts rather than assume that every program can be
freely converted between the two forms.

```{only} compiler_developer_docs
See {ref}`Developing compiler passes <compiler-pass-input-output-ir>` for more
detail.
```

## Calling between reference and value forms

`quake.unwrap` obtains the current wire from a `!quake.ref`, and `quake.wrap`
writes the updated wire back to that reference. A reference-form function can
use these operations when it calls a value-form function:

```mlir
func.func private @value_kernel(!quake.wire) -> !quake.wire

func.func @call_value_kernel(%q: !quake.ref) {
  %wire = quake.unwrap %q : (!quake.ref) -> !quake.wire
  %updated = call @value_kernel(%wire)
      : (!quake.wire) -> !quake.wire
  quake.wrap %updated to %q : !quake.wire, !quake.ref
  return
}
```

In the other direction, `quake.call_by_ref` lets value-form code call a
function whose quantum parameters use reference semantics. The operation
returns the updated wire after the call:

```mlir
func.func private @reference_kernel(!quake.ref)

func.func @call_reference_kernel(%q: !quake.wire) -> !quake.wire {
  %updated = quake.call_by_ref @reference_kernel(%q)
      : (!quake.wire) -> !quake.wire
  return %updated : !quake.wire
}
```

For a call with ordinary results, `quake.call_by_ref` appends each updated
wire or cable to the result list.
