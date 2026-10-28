# Capabilities and status

CUDA-Q Logical reports its own status the way it reports evidence: three-way,
and only against executable artifacts. A capability is **shipped** when it is
part of the installed `cudaq.logical` package, and **exercised** only when a
shipped test or example executes it — the `lit`/`FileCheck` suites and the
`pytest` suite (which runs every Python example under
`preview/logical/examples/`) are the evidence. A capability that is neither is
**out of scope**: a deliberate boundary, stated here. Execution limitations and
the information omitted during lowering are listed below.

## What the product is

CUDA-Q Logical is a compiler and resource-estimation toolkit for fault-tolerant
quantum computing. A program is refined through four strict semantic stages — **P0**
unplaced logical, **P1** placed logical, **P2** QEC realization, and **P3**
physical event graph and schedule. The estimation ladder has four tiers:
`Tier.LOGICAL`, `Tier.STATIC`, `Tier.ANALYTICAL`, and `Tier.SCHEDULE`. Stim
circuit text is an interchange emission target from a verified P2 gadget.
Supported P3 physical builds also translate to Quake for execution through
ordinary `cudaq.run`.

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
| Folded PBC normalization — exact static-repeat periods, ownership closure, and immutable source Builds                                                   | shipped, exercised | `test_pbc_api.py`; `to-pbc*.mlir` and `verify-pbc*.mlir`                                                               |
| CUDA-Q ingress — `@cudaq.kernel` programs through CUDA-Q Logical targets                                                                                | shipped, exercised | all top-level numbered examples                                                                                        |
| Static P2 estimation — gadget/operation counts with folding                                                                                             | shipped, exercised | `examples/02_surface_code_resource_estimate.py`, `examples/standalone/03_magic_state_distillation.py`                   |
| P3 physical lowering, routing, native legalization, and scheduling                                                                                      | shipped, exercised | `examples/04_carbon_code.py`, `examples/standalone/04_physical_schedule.py`, `examples/standalone/05_gidney_ekera_lookup_addition.py` |
| Analytical and schedule estimation                                                                                                                      | shipped, exercised | `examples/02_surface_code_resource_estimate.py`, `examples/standalone/04_physical_schedule.py`, `examples/standalone/05_gidney_ekera_lookup_addition.py` |
| Paper-specific Gidney–Ekerå RSA-2048/3072/4096 projection                                                                                               | shipped, exercised | `examples/05_gidney_ekera.py`                                                                                          |
| Stim text emission — the `--fabric-to-stim` translation                                                                                                 | shipped, exercised | `examples/mlir/stim_memory.mlir` via `qlx-translate`; Stim-emission tests                                               |
| P3 execution — `Build.to_quake()` and `cudaq.run`                                                                                                        | shipped, exercised | `python/tests/cudaq/logical/test_run_physical.py`; standalone examples 06 and 07                                        |

## Present but not yet exercised

These APIs exist in the package but carry no executed test or example in this
release, so the documentation does not teach them yet: dynamic codes
(`MeasurementPhase`, `EncodingEpoch`), code switching (`PatchTransform`),
concatenation (`cudaq.logical.codes.Concatenated`), meta-checks
(`cudaq.logical.codes.MetaChecks`), and P2 block requests
(`cudaq.logical.codes.qec_block`). Treat them as preview surface: use them at
your own risk until exercised evidence lands.

## Execution and documented boundaries

**Execution uses ordinary CUDA-Q targets.** `cudaq.run(physical_build, ...)`
translates a supported P3 `Build` through `Build.to_quake()` and executes the
resulting circuit on the selected target. Each shot returns the entry's explicit
Boolean results: one result becomes a `bool`, and multiple results become a
`list[bool]` in physical return order. No implicit final measurements or decoding
are added. See the [Steane Bernstein–Vazirani](../use-cases/examples/steane-bernstein-vazirani.md)
and [T-factory](../use-cases/examples/t-factory-evaluation.md) execution examples.

**Execution covers a subset of P3.** Resource and factory flows, in-graph
postselection, code epoch changes, and patch-granularity resources produce
diagnostics instead of executable Quake. Support for compiling or estimating
these forms does not imply execution support.

**Target capabilities and memory limits still apply.** CUDA-Q's `stim` target
executes Clifford circuits with supported Pauli noise. Non-Clifford examples
require a compatible target, such as `nvidia` or `density-matrix-cpu`.
Successful compilation does not guarantee that a simulator has enough memory
for the physical carriers, including ancillas. The ordinary `noise_model`
argument applies supported channels to emitted gates. P3 operating-point error
and timing assumptions remain estimation inputs, not runtime noise channels.

**Execution does not model the physical schedule.** The current lowering omits
`phys.delay`, `phys.barrier`, `phys.move`, and `event.fence`. It does not execute
their timing, transport, or synchronization effects or model the associated
idle and transport noise. It does not submit P3 schedules to hardware.

**No Logical detector-error-model or decoder pipeline.** The execution path
returns authored measurement and check bits. It does not add detector or
observable annotations, generate detector error models, or decode results.
Those analyses require separate downstream tooling.

**Stages stop at P3.** P3 represents physical carriers, routing, native events,
and schedules. Quake translation connects supported physical graphs to CUDA-Q's
existing execution path and simulators without adding a Logical compilation stage.

**The native P1 placer is a subset.** `qlx-to-lvm` is a deterministic first-fit
placement for explicitly machine-scoped, inlined programs; the Python placement
solver (`cudaq.logical.compiler.place`) is the rich path. The native pass fails
on inputs outside the shared supported subset rather than approximating.

**Stim emission is terminal and checked.** The `--fabric-to-stim` translation
accepts a verified P2 entry gadget. Emission never invents an implementation
that selection did not link.

**Rotation synthesis is explicit, not automatic.**
`cudaq.logical.compiler.synthesize` legalizes a logical program to a named gate
set (Clifford+T, example 01) under an operator-norm `precision=` bound;
unsupported gate sets are rejected with a `ValueError` rather than
approximated.

**Paper-specific projections are labeled as such.** Example 05's default
Gidney–Ekerå path combines compiler-counted logical resources with explicit
paper equations. Its `--physical` path instead performs P3 compilation and
scheduling. Neither path simulates or executes the workload.
