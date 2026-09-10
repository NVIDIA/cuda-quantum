# Capabilities and status

CUDA-Q Logical reports its own status the way it reports evidence: three-way,
and only against executable artifacts. A capability is **shipped** when it is
part of the installed `cudaq.logical` package, and **exercised** only when a
shipped test or example executes it — the `lit`/`FileCheck` suites and the
`pytest` suite (which runs every Python example under
`preview/logical/examples/`) are the evidence. A capability that is neither is
**out of scope**: a deliberate boundary, stated here, that fails closed instead
of approximating past an implemented edge.

## What the product is

CUDA-Q Logical is a resource-estimation toolkit for fault-tolerant quantum
computing. A program is refined through four strict semantic stages — **P0**
unplaced logical, **P1** placed logical, **P2** QEC realization, and **P3**
physical event graph and schedule. The estimation ladder has four tiers:
`Tier.LOGICAL`, `Tier.STATIC`, `Tier.ANALYTICAL`, and `Tier.SCHEDULE`. Stim
circuit text is the interchange emission target from a verified P2 gadget.

## Subsystem status

Every row cites its exercising evidence in this repository.

| Subsystem                                                                                                                                               | Status             | Exercised by                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Compiler foundation — immutable `Build`, typed stages/facets, serialization and clean-process replay                                                    | shipped, exercised | `examples/standalone/01_logical_placement.py`; provenance fail-closed tests                                             |
| P0 authoring and logical estimation — `@program`, linear values, `Tier.LOGICAL`                                                                         | shipped, exercised | `examples/standalone/00_logical_program.py`, `examples/00_logical_resource_estimate.py`                                 |
| P1 placement — `@machine`, regions/capabilities, constraints, and witnesses                                                                             | shipped, exercised | `examples/standalone/01_logical_placement.py`; placement tests                                                          |
| P2 codes and gadgets — `@code` validation, catalog, `@gadget` with `implements=`, typed records                                                         | shipped, exercised | `examples/standalone/02_code_and_gadget.py`, `examples/04_carbon_code.py`; record-boundary tests                        |
| Gadget verification — code-automorphism and kernel-backed objective matching, fail-closed claims                                                       | shipped, exercised | `examples/04_carbon_code.py`; verifier-error lit suites                                                                |
| Protocols — `@protocol`, typed resources, postselection, bounded retry, 15-to-1 distillation                                                            | shipped, exercised | `examples/standalone/03_magic_state_distillation.py`, `examples/standalone/05_gidney_ekera_lookup_addition.py`; tests   |
| Clifford+T synthesis — rotation lowering with provenance                                                                                                | shipped, exercised | `examples/01_clifford_t_resource_estimate.py`, `examples/03_fermi_hubbard.py`                                           |
| CUDA-Q ingress — `@cudaq.kernel` programs through CUDA-Q Logical targets                                                                                | shipped, exercised | all six top-level numbered examples                                                                                    |
| Static P2 estimation — gadget/operation counts with folding                                                                                             | shipped, exercised | `examples/02_surface_code_resource_estimate.py`, `examples/standalone/03_magic_state_distillation.py`                   |
| P3 physical lowering, routing, native legalization, and scheduling                                                                                      | shipped, exercised | `examples/04_carbon_code.py`, `examples/standalone/04_physical_schedule.py`, `examples/standalone/05_gidney_ekera_lookup_addition.py` |
| Analytical and schedule estimation                                                                                                                      | shipped, exercised | `examples/02_surface_code_resource_estimate.py`, `examples/standalone/04_physical_schedule.py`, `examples/standalone/05_gidney_ekera_lookup_addition.py` |
| Paper-specific Gidney–Ekerå RSA-2048 projection                                                                                                         | shipped, exercised | `examples/05_gidney_ekera.py`                                                                                          |
| Stim text emission — typed `ql.lower.emit_stim` and `ql.lower.emit_stim_artifact`                                                                       | shipped, exercised | `examples/standalone/02_code_and_gadget.py`; Stim-emission tests                                                        |

## Present but not yet exercised

These APIs exist in the package but carry no executed test or example in this
release, so the documentation does not teach them yet: dynamic codes
(`MeasurementPhase`, `EncodingEpoch`), code switching (`PatchTransform`),
concatenation (`ql.codes.Concatenated`), meta-checks (`ql.codes.MetaChecks`),
and P2 block requests (`ql.codes.qec_block`). Treat them as preview surface:
use them at your own risk until exercised evidence lands.

## Documented boundaries (all fail closed)

**There is no noise, detector, DEM, decoder, or sampling surface.** CUDA-Q
Logical emits Stim circuit text; it does not annotate detectors, build detector
error models, sample, or decode. Those studies belong downstream of the emitted
text, in the Stim ecosystem.

**Stages stop at P3.** P3 represents physical carriers, routing, native events,
and schedules. It remains a compiler and estimation artifact: CUDA-Q Logical
does not submit a physical schedule to hardware or provide a runtime execution
service for it.

**No simulator plugin surface.** Nothing in the package consumes or executes
physical simulations.

**The native P1 placer is a subset.** `qlx-to-lvm` is a deterministic first-fit
placement for explicitly machine-scoped, inlined programs; the Python placement
solver (`ql.compiler.place`) is the rich path. The native pass fails on inputs
outside the shared supported subset rather than approximating.

**Stim emission is terminal and checked.** The Python emitter accepts a verified
P2 entry gadget. Emission never invents an implementation that selection did not
link.

**Rotation synthesis is explicit, not automatic.** `ql.compiler.synthesize`
legalizes a logical program to a named gate set (Clifford+T, example 01) under
an operator-norm `precision=` bound; unsupported gate sets are rejected with a
`ValueError` rather than approximated.

**Paper-specific projections are labeled as such.** Example 05's default
Gidney–Ekerå path combines compiler-counted logical resources with explicit
paper equations. Its `--physical` path instead performs P3 compilation and
scheduling. Neither path simulates or executes the workload.
