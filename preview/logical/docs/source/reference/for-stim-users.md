# CUDA-Q Logical for Stim users

You know Stim: circuits as text, `REPEAT` blocks, fast stabilizer simulation.
This page maps that world onto CUDA-Q Logical, whose scope is deliberately
narrower in one direction and much wider in another.

The one-sentence version: **Stim is the assembly language of QEC experiments;
CUDA-Q Logical is the resource-estimation compiler stack above it** — and Stim
text is its emission target. CUDA-Q Logical does not simulate, sample, or
decode: no detector annotations, no detector error models, no samplers. Those
studies start from the emitted circuit and belong to the Stim ecosystem.

## The translation table

| You do this in Stim                                            | You do this in CUDA-Q Logical                                                                                                                   |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Write a circuit qubit-by-qubit                                 | Write a _logical program_; the compiler selects code-specific realizations                                                                      |
| `stim.Circuit.generated("surface_code:rotated_memory_z", ...)` | `cudaq.logical.codes.Surface[d]` + the gadget factories (`cudaq.logical.gadgets.prepare_zero`, `css_memory_round`, `logical_measure`) selected for a memory objective |
| `REPEAT 1000 { ... }`                                          | `cudaq.logical.idle(q, rounds=1000)` — stays **folded** through compilation and estimation, never unrolled                                                 |
| Hand-maintain a circuit template per code                      | One portable program; swap `code=`/`Surface[d]` and recompile — the program is untouched                                                        |
| Read the circuit to guess its cost                             | `cudaq.logical.estimate(build, tier=...)`: logical counts at P0, static and analytical counts at P2, schedule-backed counts at P3                          |
| Ship circuit text                                              | `qlx-translate <module>.mlir --fabric-to-stim` projects a selected P2 entry gadget to Stim text                                                  |
| (no equivalent)                                                | Placement onto logical machines, verified gadget objectives, magic-state protocols, replayable builds with evidence                             |

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

- **Verified gadget objectives.** A gadget _claims_ a logical action with
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
