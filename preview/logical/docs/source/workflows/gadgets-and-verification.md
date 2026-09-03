# Gadgets and verification

A gadget in CUDA-Q Logical is a bounded, typed realization of a logical
objective. Three artifacts travel together through compilation: the
`implements=` clause states the ideal logical claim, the `ql.patch[...]`
signature states the encoded boundary (code, encoding, ownership), and the body
states the executable realization. The compiled `fabric` artifact keeps all
three inspectable — and for supported realization classes the compiler proves
the claim instead of trusting it.

## Write the smallest gadget

```python
import cudaq.logical as ql


@ql.gadget(implements=ql.std.h)
def steane_h(block: ql.patch[ql.codes.Steane]) -> ql.patch[ql.codes.Steane]:
    return ql.h(block.data)


build = ql.compile(steane_h)
```

Three facts are enough:

1. `implements=` states the ideal logical claim. Standard actions and
   instruments live in `ql.std` (`h`, `cx`, `idle`, `prepare_zero`, `measure_z`,
   …); `@ql.objective` authors new ones, as the quickstart's terminal-memory
   objective shows.
2. `ql.patch[ql.codes.Steane]` states the encoded input and output types. The
   signature derives one inout encoded port and its linear ownership.
3. The body states the bounded realization. `ql.h(block.data)` expands to one
   carrier operation per data carrier — “transversal” is a property the compiler
   reads off the support map, not a separate instruction.

The compiled artifact keeps claim and realization side by side (abbreviated —
the patch types spell out code, encoding, and epoch in full):

```mlir
fabric.gadget @steane_h(%arg0: !fabric.patch<@Steane, …>) -> !fabric.patch<@Steane, …> {
  %0 = fabric.h %arg0 data : !fabric.patch<@Steane, …>
  fabric.return %0 : !fabric.patch<@Steane, …>
}
```

The typed claim is machine-readable from Python:
`ql.gadgets.clifford_action(ql.std.h)` returns the action
`CliffordAction(matrix=((0, 1), (1, 0)), phases=(0, 0), …)` — the X/Z swap that
_is_ H — and the same accessor applies to a compiled gadget.

## Typed records at the boundary

Syndrome-extraction results are first-class typed values, not raw bit vectors:
`ql.types.record[Code]` names the record family of one code, and gadget
signatures may take and return records directly. This is the idiom the shipped
test suite exercises:

```python
@ql.gadget(implements=ql.std.idle)
def extraction_round(
    block: ql.patch[ql.codes.Steane],
    previous: ql.types.record[ql.codes.Steane],
) -> tuple[ql.patch[ql.codes.Steane], ql.types.record[ql.codes.Steane]]:
    block, current = ql.extract_syndrome(block)
    return block, current
```

The compiled boundary speaks the typed `fabric.syndrome<@Steane, …>` form, and
protocols compose such gadgets by passing records along — a two-round memory
protocol is two ordinary calls, with no annotation glue. Inside a gadget,
`ql.analysis.count` reports the authored operations of the compiled realization
(the quickstart shows it on the Steane terminal-memory gadget).

## Preparation and destructive measurement

Two boundary patterns cover most library gadgets:

- **Preparation** has no encoded input seam and produces an encoded output — the
  `ql.gadgets.prepare_zero` / `prepare_plus` factories build exactly this shape
  for any validated code.
- **Destructive measurement** consumes its encoded input and returns classical
  results. It must not fabricate a live encoded output merely to make the
  boundary look symmetric.

The shipped Steane example pairs both halves of the pattern — a terminal
objective and the gadget that realizes it by one syndrome-extraction pass
followed by data-qubit readout:

```{literalinclude} ../../../examples/03_code_and_gadget.py
:language: python
:start-at: "@ql.objective"
:end-before: "code = ql.materialize"
:caption: A terminal-memory objective and its gadget (examples/03_code_and_gadget.py).
```

Ownership is linear throughout: consuming a patch twice, or dropping one that is
still live, is a construction error (`UseAfterConsume`), never a silent no-op.

## Selection: retry and postselection belong to the protocol

Execution policy is not hidden inside reusable gadgets; the consuming protocol
states it. Acceptance is explicit with `ql.postselect` — the shipped 15-to-1
distillation protocol accepts exactly when all four even-parity checks measure
+X:

```{literalinclude} ../../../examples/04_distillation.py
:language: python
:start-after: "# The positive-angle triorthogonal circuit"
:end-before: "return ql.pack_resource"
:caption: Postselection in examples/04_distillation.py.
```

Bounded retry is the same shape: `ql.ops.retry` acts on a success predicate
derived from one gadget attempt, and the policy — attempt budget, exhaustion
behavior, commit point — is spelled out at the retry site:

```python
policy = ql.gadgets.RetryPolicy(
    max_attempts=8,
    exhaustion=ql.gadgets.RetryExhaustion.REPORT_FAILURE,
    commit_point=ql.gadgets.before_output(),
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
gadgets whose realization is a single typed `ql.ops.permute`. The compiler
derives the induced logical action from the code algebra and compares it with
the claim, failing closed on mismatch:

```python
@ql.gadget(implements=ql.std.idle)
def steane_idle(block: ql.patch[ql.codes.Steane]) -> ql.patch[ql.codes.Steane]:
    return ql.ops.permute(block, tuple(range(7)))

ql.compile(steane_idle)   # verified_code_automorphism evidence recorded


@ql.gadget(implements=ql.std.h)
def wrong(block: ql.patch[ql.codes.Steane]) -> ql.patch[ql.codes.Steane]:
    return ql.ops.permute(block, tuple(range(7)))

try:
    # ValueError: code automorphism logical action does not implement the
    # declared objective under any logical-port binding
    ql.compile(wrong)
except ValueError as exc:
    assert "does not implement the declared objective" in str(exc)
```

The checks layer in order: a permutation that does not preserve the stabilizer
group is rejected before any objective comparison
(`permutation does not preserve the X-stabilizer group`), and a realization that
matches its claim under _several_ logical-port bindings raises an ambiguity
error — constrain it explicitly with `logical_ports=`. Evidence strings are
never accepted as the proof; the derivation is.

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

- [Defining codes](define-a-code.md) — the validated algebra gadgets build on,
  and the factories that consume it.
- The [quickstart](../quickstart.md) compiles a Steane gadget end to end and
  counts its authored operations.
- The [example gallery](../example-gallery/index.md) embeds the shipped gadget
  and protocol sources, including 15-to-1 distillation.
