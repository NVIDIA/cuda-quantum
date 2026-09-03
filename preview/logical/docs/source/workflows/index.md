# Workflow guides

Task-oriented guides for CUDA-Q Logical. Each guide does one job end to end
and points at the shipped example that exercises it.

| Guide | What it covers |
|---|---|
| [Defining codes](define-a-code) | `@ql.code` authoring: validated CSS algebra, distance as typed evidence, parameterized families |
| [Gadgets and verification](gadgets-and-verification) | `@ql.gadget` realizations, `implements=` claims checked against the objective, retry and commit points |
| [Devices and placement](devices-and-placement) | `@ql.machine` regions and capabilities, placement constraints and witnesses, P0 → P1 refinement |
| [Magic states and protocols](magic-states-and-protocols) | `mpp` and rotations, resource kinds, the 15-to-1 distillation protocol |
| [Logical Clifford+T synthesis](logical-synthesis) | `ql.compiler.synthesize` to a declared gate set under a projective precision bound |
| [Resource estimation](estimation) | The two estimation tiers, honesty rules, sweeps, and CUDA-Q kernel estimation |
| [Command-line estimation](command-line-estimation) | The same tiers as explicit `qlx-opt` pass pipelines over `examples/cli/` |
| [Stim emission](stim-emission) | Projecting a selected P2 program to Stim circuit text, fail-closed boundaries |

The [core concepts](../concepts.md) explain the stages, ownership, and
evidence model these guides rely on; the [architecture
page](../architecture.md) describes the dialect stack and pipeline presets
underneath them.
