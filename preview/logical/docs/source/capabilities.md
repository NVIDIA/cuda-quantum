# Capabilities and status

CUDA-Q Logical reports its own status the way it reports evidence: three-way,
and only against executable artifacts. A capability is **shipped** when it is
part of the installed `cudaq.logical` package, and **exercised** only when a
shipped test or example executes it — the lit/FileCheck suites, the pytest
suite (which runs every example under `preview/logical/examples/`), and the
CLI workflow test are the evidence. A capability that is neither is **out of
scope**: stated here as a deliberate boundary that fails closed, not
approximated past an implemented edge.

## What the product is

CUDA-Q Logical is a resource-estimation toolkit for fault-tolerant quantum
computing. A program is refined through three strict semantic stages — **P0**
unplaced logical, **P1** placed logical, **P2** QEC realization — and the
estimation ladder has exactly two tiers: `Tier.LOGICAL` (P0) and
`Tier.STATIC` (P2 fabric counts). Stim circuit text is the one emission
target.

## Subsystem status

Every row cites its exercising evidence in this repository.

| Subsystem | Status | Exercised by |
|---|---|---|
| Compiler foundation — immutable `Build`, typed stages/facets, serialization and clean-process replay | shipped, exercised | `examples/02_p1_placement.py`; provenance fail-closed tests in `test_cudaq_logical.py` |
| P0 authoring and logical estimation — `@program`, linear values, `Tier.LOGICAL` | shipped, exercised | `examples/01_p0_bell.py`; `examples/cli/logical_and_placement.mlir` with `qlx-estimate-logical` |
| P1 placement — `@machine`, regions/capabilities, constraints and witnesses | shipped, exercised | `examples/02_p1_placement.py`; `qlx-to-lvm` in the CLI suite |
| P2 codes and gadgets — `@code` validation, catalog (Steane, rotated surface, repetition, RM15, bare qubit), `@gadget` with `implements=`, typed records | shipped, exercised | `examples/03_code_and_gadget.py`; record-boundary tests |
| Gadget verification — code-automorphism action matching, fail-closed claims | shipped, exercised | `verified_code_automorphism` path; verifier-error lit suites |
| Protocols — `@protocol`, typed resources, postselection, bounded retry, 15-to-1 distillation | shipped, exercised | `examples/04_distillation.py`; protocol lit tests |
| Clifford+T synthesis — rotation lowering with provenance | shipped, exercised | `examples/05_clifford_t.py` |
| CUDA-Q ingress — `@cudaq.kernel` programs through CUDA-Q Logical targets | shipped, exercised | `examples/00_cudaq_logical_resource_estimate.py`, `05_clifford_t.py` |
| Static P2 estimation — gadget/operation counts with folding | shipped, exercised | `examples/00`, `04`; `fabric-count` in the CLI suite |
| Analytical projections — Gidney–Ekerå RSA-2048 and Fermi–Hubbard envelopes | shipped, exercised | `examples/06_gidney_ekera.py`, `07_fermi_hubbard.py` (analytical projections, not further compilation stages) |
| Stim text emission — `qlx-translate --fabric-to-stim`, typed `qlx.lower.emit_stim` | shipped, exercised | `examples/cli/stim_memory.mlir`; the CLI workflow lit test |

## Present but not yet exercised

These APIs exist in the package but carry no executed test or example in this
release, so the documentation does not teach them yet: dynamic codes
(`MeasurementPhase`, `EncodingEpoch`), code switching (`PatchTransform`),
concatenation (`qlx.codes.Concatenated`), metachecks (`qlx.codes.MetaChecks`),
and P2 block requests (`qlx.codes.qec_block`). Treat them as preview surface:
usable at your own risk until exercised evidence lands.

## Documented boundaries (all fail closed)

**There is no noise, detector, DEM, decoder, or sampling surface.** CUDA-Q
Logical emits Stim circuit text; it does not annotate detectors, build
detector error models, sample, or decode. Those studies belong downstream of
the emitted text, in the Stim ecosystem.

**Stages stop at P2.** There are no physical carriers, no routing or
scheduling, and no runtime or hardware submission — `qlx.stages.Stage` has
exactly `P0`, `P1`, `P2`.

**No simulator plugin surface.** Nothing in the package consumes or executes
physical simulations.

**The native P1 placer is a subset.** `qlx-to-lvm` is a deterministic
first-fit placement for explicitly machine-domained, inlined programs; the
Python placement solver (`qlx.compiler.place`) is the rich path. The native
pass fails on inputs outside the shared supported subset rather than
approximating.

**Stim emission is terminal and checked.** The translator accepts a verified
P2 entry gadget; a closed protocol is first legalized within P2
(`fabric-lower-protocols`) and only then translated. Emission never invents
an implementation that selection did not link.

**Rotation synthesis is explicit, not automatic.**
`qlx.compiler.synthesize` legalizes a logical program to a named gate set
(Clifford+T, example 05) under an operator-norm `precision=` bound;
unsupported gate sets are rejected with a `ValueError` rather than
approximated.

**Analytical projections are labeled as such.** The physical-qubit, runtime,
and retry-risk figures in examples 06 and 07 are analytical projections over
the P0/P2 artifacts, not simulated or executed results.
