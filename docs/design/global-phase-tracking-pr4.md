# Global phase tracking PR 4: exact `ReduceYSX`

## Stack

- Source: the applicable, manually ported PR4 work from `pt4`
- Baseline: this branch after the PR1-PR3 phase-tracking implementation
- Design stage: representative constant phase producer (`ReduceYSX`)

## Design relationship

This PR implements the design document's
[PR 4: `ReduceYSX`](../../global-phase-tracking-long.md#pr-4-reduceysx) slice
and its `Y`, `S`, `X` worked example. The source sequence has matrix
`X S Y = -S`; replacing the forward-middle form with `S` therefore requires
`quake.phase(pi)` to retain exact behavior before the late phase-lowering
boundary.

The correction uses the same ordered controls and negative-control polarities
as the matched source sequence. This matters because an unconditional
correction would phase both branches instead of only the active predicate.
The middle-adjoint identity `X S^dagger Y = S^dagger` is already exact and
therefore emits no correction.

## Code changes

- [`PhaseUtilities.h`](../../cudaq/lib/Optimizer/Transforms/PhaseUtilities.h)
  generalizes `getControlPolarities` so gate patterns can compare predicates
  with the same helper used by phase operations.
- [`QuakeSimplify.cpp`](../../cudaq/lib/Optimizer/Transforms/QuakeSimplify.cpp)
  now accepts controlled and negative-controlled `ReduceYSX` matches only when
  all three gates share the same ordered predicate. It threads replacement
  wires, preserves the middle gate's adjoint form, and emits `Phase(pi)` only
  for forward `S`.

The replacement phase is anchored to the latest target wire immediately after
the replacement `S`, following the design's deterministic local-anchor rule.

## Verification

The focused tests in
[`quake_simplify.qke`](../../cudaq/test/Transforms/quake_simplify.qke) and
[`quake_simplify_adjoint.qke`](../../cudaq/test/Transforms/quake_simplify_adjoint.qke)
cover:

- uncontrolled and directly controlled forward `YSX`;
- negative-control predicate preservation;
- exact `Y S^dagger X` reduction without a phase;
- strict circuit equivalence, including an explicit
  `H - controlled(YSX) - H` interference shape; and
- late lowering to ordinary gates with no surviving `quake.phase`.

Validation command (using an existing configured build tree):

```text
cmake --build <build-dir> --target cudaq-opt CircuitCheck
/usr/local/llvm/bin/llvm-lit -sv \
  --filter='Transforms/(quake_simplify|quake_simplify_adjoint)\.qke$' \
  --param cudaq_site_config=<build-dir>/cudaq/test/lit.site.cfg.py \
  <build-dir>/cudaq/test
```

## Commit structure

1. Generalize phase control polarity lookup.
2. Make `ReduceYSX` phase exact.
3. Add exact controlled, adjoint, lowering, and interference regressions.
4. Add this design traceability note.

## Deferred scope

This PR does not change decomposition selection or remove the broad
controlled-`quake.apply` decomposition guard. Those remain staged for later
PRs in the design sequence.

## References

- [PR4 design requirements](../../global-phase-tracking-long.md#pr-4-reduceysx)
- [Shared phase utility](../../cudaq/lib/Optimizer/Transforms/PhaseUtilities.h)
- [Exact `ReduceYSX` implementation](../../cudaq/lib/Optimizer/Transforms/QuakeSimplify.cpp)
- [IR shape regression test](../../cudaq/test/Transforms/quake_simplify.qke)
- [Strict equivalence and lowering regression test](../../cudaq/test/Transforms/quake_simplify_adjoint.qke)
- [PR3 correction-emission helper](../../cudaq/lib/Optimizer/Transforms/PhaseUtilities.h#L50)
- [Phase lifecycle pipeline](../../cudaq/lib/Optimizer/Transforms/Pipelines.cpp)
