# Gadgets and verification

A gadget in CUDA-Q Logical is a bounded, typed realization of a logical
objective. Three artifacts travel together through compilation: the
`implements=` clause states the ideal logical claim, the
`cudaq.logical.patch[...]` signature states the encoded boundary (code,
encoding, ownership), and the body states the executable realization. The
compiled `fabric` artifact keeps all three inspectable, and for supported
realization classes the compiler proves the claim instead of trusting it.

## What `implements=` takes

`implements=` states a *claim*; the body supplies the *realization*. The claim
is a declarative description of an ideal logical operation — it never executes.
Instructions like `cudaq.logical.h` belong in the body, and passing one as the
claim is rejected with a `TypeError`.

There are two ways to write the claim, and the choice is about where the ideal
operation already lives.

### Author the objective in CUDA-Q Logical

When the ideal operation has no existing definition, declare it with
`@cudaq.logical.objective` and implement it in a gadget:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/02_code_and_gadget.py
   :language: python
   :start-at: @cql.objective
   :end-before: # Materialize both definitions
   :caption: An authored objective and the gadget that realizes it (examples/standalone/02_code_and_gadget.py).
```

`terminal_memory` says what the operation means at P0 — a logical qubit is
discarded. `steane_memory` says how a Steane block does it: extract the
syndrome, measure the data, release the block. The claim carries no code and no
carriers; the realization carries both.

### Reuse a CUDA-Q kernel as the claim

When the ideal operation already exists as a `@cudaq.kernel`, pass it directly.
The Carbon code carries two logical qubits per block, so its gadgets implement
*paired* operations:

```{eval-rst}
.. literalinclude:: ../../../examples/04_carbon_code.py
   :language: python
   :start-at: @cudaq.kernel
   :end-before: # Implement the paired operations
   :caption: Ideal operations written as ordinary CUDA-Q kernels (examples/04_carbon_code.py).
```

```{eval-rst}
.. literalinclude:: ../../../examples/04_carbon_code.py
   :language: python
   :start-at: @cql.gadget(
   :end-before: def paired_cx_gadget
   :caption: The gadget that realizes the kernel's claim on Carbon patches.
```

A kernel's parameters are plain qubits, but a gadget's inputs are patches, so
something has to say which logical port each parameter denotes. That is
`logical_ports=`: here `left` and `right` name the two logical qubits inside
one Carbon block. Without it, a claim whose parameters do not map unambiguously
onto the patch's ports is a construction error
(`cudaq.logical.errors.AmbiguousLogicalPortMap`), never a silent guess.

Protocols take the same `implements=` field, with one extra form:
`cudaq.logical.logical.produce(kind)` claims the production of a typed resource
— the shape every distillation factory uses. See
[Magic states and protocols](magic-states-and-protocols.md).

### What the signature and body add

`cudaq.logical.patch[cudaq.logical.codes.Steane]` states the encoded boundary:
the signature derives the inout encoded port and its linear ownership. The body
states the bounded realization, where `cudaq.logical.h(block.data)` expands to
one carrier operation per data carrier — “transversal” is a property the
compiler reads off the support map, not a separate instruction.

The compiled artifact keeps claim and realization side by side (abbreviated —
the patch types spell out code, encoding, and epoch in full):

```mlir
fabric.gadget @steane_h(%arg0: !fabric.patch<@Steane, …>) -> !fabric.patch<@Steane, …> {
  %0 = fabric.h %arg0 data : !fabric.patch<@Steane, …>
  fabric.return %0 : !fabric.patch<@Steane, …>
}
```

The typed claim is machine-readable from Python: passing the standard action
`h` to `cudaq.logical.gadgets.clifford_action` returns
`CliffordAction(matrix=((0, 1), (1, 0)), phases=(0, 0), …)` — the X/Z swap that
*is* H. The same accessor applies to a compiled gadget.

## Typed records at the boundary

Syndrome-extraction results are first-class typed values, not raw bit vectors:
`cudaq.logical.types.record[Code]` names the record family of one code, and
gadget signatures may take and return records directly:

```python
import cudaq.logical as cql


@cql.objective
def memory_round(qubit: cql.types.logical_qubit) -> cql.types.logical_qubit:
    return qubit


@cql.gadget(implements=memory_round)
def extraction_round(
    block: cql.patch[cql.codes.Steane],
    previous: cql.types.record[cql.codes.Steane],
) -> tuple[cql.patch[cql.codes.Steane], cql.types.record[cql.codes.Steane]]:
    block, current = cql.extract_syndrome(block)
    return block, current
```

The compiled boundary speaks the typed `fabric.syndrome<@Steane, …>` form, and
protocols compose such gadgets by passing records along — a two-round memory
protocol is two ordinary calls, with no annotation glue. Inside a gadget,
`cudaq.logical.analysis.count` reports the authored operations of the compiled
realization (the quick start shows it on the Steane terminal-memory gadget).

## Preparation and destructive measurement

Two boundary patterns cover most library gadgets:

- **Preparation** has no encoded input seam and produces an encoded output —
  the `cudaq.logical.gadgets.prepare_zero` / `prepare_plus` factories build
  exactly this shape for any validated code.
- **Destructive measurement** consumes its encoded input and returns classical
  results. It must not fabricate a live encoded output merely to make the
  boundary look symmetric.

The shipped Steane example pairs both halves of the pattern — a terminal
objective and the gadget that realizes it by one syndrome-extraction pass
followed by data-qubit readout:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/02_code_and_gadget.py
   :language: python
   :start-at: @cql.objective
   :end-before: code = cql.materialize
   :caption: A terminal-memory objective and its gadget (examples/standalone/02_code_and_gadget.py).
```

Ownership is linear throughout: consuming a patch twice, or dropping one that is
still live, is a construction error (`UseAfterConsume`), never a silent no-op.

## Selection: retry and postselection belong to the protocol

Execution policy is not hidden inside reusable gadgets; the consuming protocol
states it. Acceptance is explicit with `cudaq.logical.postselect` — the shipped
15-to-1 distillation protocol accepts exactly when all four even-parity checks
measure +X:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/03_magic_state_distillation.py
   :language: python
   :start-at: @cql.protocol
   :end-before: # Estimate the protocol directly
   :caption: Postselection in examples/standalone/03_magic_state_distillation.py.
```

Bounded retry is the same shape. It acts on a success predicate derived from
one gadget attempt, and you spell out the policy — attempt budget, exhaustion
behavior, commit point — at the retry site:

```python
policy = cql.gadgets.RetryPolicy(
    max_attempts=8,
    exhaustion=cql.gadgets.RetryExhaustion.REPORT_FAILURE,
    commit_point=cql.gadgets.before_output(),
)
```

The compiler tracks predicate provenance: a retry predicate must derive from the
selected attempt, and the retry must carry every live patch result of that
attempt exactly once — an ambiguous or externally observable replay boundary
fails closed. Exhaustion is explicit (`RetryExhaustion.REPORT_FAILURE` / `ABORT`
/ `RETURN_LAST`), and commit points mark where an attempt becomes irreversible
(`before_output(...)`, `before_resource_output()`).

## Verification: claims are checked, not trusted

Construction and MLIR verification establish local shape, ownership, symbol, and
algebra invariants; semantic checks establish that a realization does what it
claims. Do not conflate the levels:

| Evidence              | What it establishes                                                             |
| --------------------- | ------------------------------------------------------------------------------- |
| Python construction   | typed fields, linear ownership, boundary liveness                               |
| MLIR verification     | canonical cross-object and stage invariants (`build.module.operation.verify()`) |
| code algebra checks   | symplectic rank, commutation, canonical pairing                                 |
| objective equivalence | the realization's induced action matches its `implements=` claim                |

Objective equivalence is automatic for **code-automorphism realizations** —
gadgets whose realization is a single typed permutation of the code's carriers.
The compiler derives the induced logical action from the code algebra and
compares it with the `implements=` claim, failing closed on mismatch: a
permutation that is genuinely the identity on the logical qubit verifies, and
one claiming to be a logical `H` is rejected with a `ValueError` naming the
objective it failed to implement.

For realization classes outside that set the claim is recorded rather than
derived, and the boundary is stated on the build instead of assumed.

## Design rules that keep the model crisp

1. Derive canonical facts from the code algebra instead of asking users to
   repeat them.
2. Keep the logical objective, the realization, and the execution policy in
   their owning objects — no hidden retry or acceptance inside a gadget.
3. Use typed endpoints, records, and logical ports instead of raw strings where
   typed identities exist.
4. Fail closed when a claim, a boundary, or a predicate provenance cannot be
   established.
5. Use ordinary Python modules for reusable gadget families; there is no
   registry to populate.

## Where to go next

- [Defining codes](define-a-code.md) — the validated algebra that gadgets build
  on, and the factories that consume it.
- The [quick start](../getting-started/quickstart.md) compiles a Steane gadget
  end to end and counts its authored operations.
- [Examples](examples/index.md) links the shipped gadget and protocol sources,
  including 15-to-1 distillation.
