# CUDA-Q Logical for Stim users

You know Stim: circuits as text, `REPEAT` blocks, fast stabilizer simulation.
This page maps that world onto CUDA-Q Logical, whose scope is deliberately
narrower in one direction and much wider in another.

CUDA-Q Logical compiles logical programs into selected realizations and
estimates their resources. It can export a selected P2 gadget as Stim text or
translate a supported P3 physical build to Quake for execution with `cudaq.run`.
Closed gadget-contract chains can export detector and observable annotations
with an explicit physical noise policy, or derive and compose EDEMs directly.
Sampling and decoding use downstream tools; these analysis paths are separate
from P3 execution.

## The translation table

| You do this in Stim                                            | You do this in CUDA-Q Logical                                                                                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Write a circuit qubit-by-qubit                                 | Write a _logical program_; the compiler selects code-specific realizations                                                                      |
| `stim.Circuit.generated("surface_code:rotated_memory_z", ...)` | The [profiled surface-memory example](../use-cases/stim-emission.md#export-a-surface-code-memory-with-detector-contracts), using explicit preparation, SEC, and transversal readout gadgets |
| `REPEAT 1000 { ... }`                                          | `cudaq.logical.idle(q, rounds=1000)` — stays **folded** through compilation and estimation, never unrolled                                                 |
| Hand-maintain a circuit template per code                      | One portable program; swap `code=`/`Surface[d]` and recompile — the program is untouched                                                        |
| Read the circuit to guess its cost                             | `cudaq.logical.estimate(build, tier=...)`: logical counts at P0, static and analytical counts at P2, schedule-backed counts at P3                          |
| Ship circuit text                                              | `qlx-translate <module>.mlir --fabric-to-stim` projects a selected P2 entry gadget to Stim text                                                  |
| Execute a Clifford circuit                                    | Select CUDA-Q's `stim` target and call `cudaq.run` on a supported P3 physical `Build` with explicit Boolean results                              |
| (no equivalent)                                                | Placement onto logical machines, verified gadget actions, magic-state protocols, replayable builds with evidence                             |

## Execute a physical build

`cudaq.run(physical_build, shots_count=...)` calls `Build.to_quake()` and uses
CUDA-Q's selected target. This path does not emit or load Stim text and does
not require the Python `stim` package. It returns one Boolean or Boolean list
per shot, containing only the entry's explicit results. CUDA-Q's `stim` target
supports Clifford circuits and supported Pauli channels supplied through
`noise_model`; it cannot execute T gates or arbitrary rotations.

The [Steane Bernstein–Vazirani example](../use-cases/examples/steane-bernstein-vazirani.md)
checks encoded preparation before execution. It does not run a decoder.
See [execution boundaries](capabilities.md#execution-and-documented-boundaries)
for omitted timing operations, target memory limits, and noise-model scope.

## Worked emission

Where you would reach for `stim.Circuit.generated(...)`, CUDA-Q Logical
projects a _selected_ realization — the gadget bodies that selection actually
linked, not a template:

```bash
qlx-translate preview/logical/examples/mlir/stim_steane_memory.mlir \
  --fabric-to-stim
```

That module is one round of Steane [[7,1,3]] syndrome extraction; it projects
to thirteen explicit carriers, with each `CX` line an `hx` or `hz` stabilizer
row expanded into carrier pairs. The output is standard Stim text — it loads
directly with the reference `stim` package and drops into any downstream
Stim-based analysis.

Emission is terminal and checked: it accepts a verified P2 entry gadget and
never invents an implementation that selection did not link. See
[Stim emission](../use-cases/stim-emission.md) for the full listing and the
boundaries it fails closed on.

## What has no Stim equivalent

- **Verified gadget actions.** A gadget _claims_ a logical action with
  `implements=`; for code-automorphism realizations the compiler derives the
  induced action from the code algebra and rejects mismatches.
- **Linear ownership.** A measured-out or double-consumed patch is a typed error
  at trace time, not a corrupted circuit.
- **Placement and machines.** Programs refine onto declared logical machines
  with regions, capacities, and placement witnesses — before any code is chosen.
- **Magic-state protocols.** Typed resource kinds and the concrete 15-to-1
  distillation protocol, with postselection and retry policy stated where
  execution happens.
- **Four estimation tiers.** `Tier.LOGICAL` counts a portable P0 program
  (device-independent); `Tier.STATIC` reads the realized P2 fabric with folded
  repetition counted exactly; `Tier.ANALYTICAL` adds an explicit physical
  model; `Tier.SCHEDULE` costs a scheduled P3 event graph.
- **Evidence and replay.** Every build serializes and replays in a clean
  process: `cudaq.logical.compiler.Build.replay(build.serialize())`.

## When to just use Stim

Honesty matters here: if your task is "simulate this fixed circuit fast," use
Stim directly — CUDA-Q Logical will happily _emit_ that circuit for you. CUDA-Q
Logical earns its overhead when the question is about the experiment's structure
and cost: comparing codes and realizations, composing protocols, estimating
resources at scale, or carrying one portable program through to a scheduled P3
form with an audit trail.
