# Example gallery

Every tile opens a page with the canonical executable source embedded directly
in the narrative. These are not copied snippets: the same files ship under
`examples/` and run in the test suite, so what you read is what executes.

## Authoring CUDA-Q Logical programs

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} A portable P0 Bell program
:link: example-gallery/bell-p0
:link-type: doc
Machine-independent logical intent and its first resource estimate.
^^^
`P0` · `Logical estimate`
:::

:::{grid-item-card} Placing a program on a logical machine
:link: example-gallery/p1-placement
:link-type: doc
A P0 build refined onto a declared machine, code choice still open.
^^^
`P0 → P1` · `Placement`
:::

:::{grid-item-card} Defining a code and a gadget
:link: example-gallery/code-and-gadget
:link-type: doc
The Steane [[7,1,3]] CSS code and a terminal-memory gadget with a declared objective.
^^^
`P2` · `Code` · `Gadget`
:::

:::{grid-item-card} A 15-to-1 distillation protocol
:link: example-gallery/distillation
:link-type: doc
A concrete T-state protocol with postselection and its static P2 estimate.
^^^
`P2` · `Protocol` · `Static estimate`
:::
::::

## Estimating CUDA-Q kernels

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} A first CUDA-Q kernel estimate
:link: example-gallery/cudaq-resource-estimate
:link-type: doc
A stock `cudaq.kernel` estimated on a distance-3 surface-code target.
^^^
`P2` · `Surface code` · `Fabric counts`
:::

:::{grid-item-card} Clifford+T synthesis of a rotation
:link: example-gallery/clifford-t
:link-type: doc
An arbitrary CUDA-Q rotation estimated after Clifford+T synthesis.
^^^
`Clifford+T` · `Synthesis`
:::

:::{grid-item-card} Gidney–Ekerå RSA-2048 envelope
:link: example-gallery/gidney-ekera
:link-type: doc
The folded lookup-addition workload: logical profile plus projected physical costs.
^^^
`RSA-2048` · `Projection`
:::

:::{grid-item-card} Trotterized Fermi–Hubbard evolution
:link: example-gallery/fermi-hubbard
:link-type: doc
One Trotter kernel estimated in the logical action model and after Clifford+T synthesis.
^^^
`Trotter` · `Two targets`
:::
::::

## Command line

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Command-line workflows
:link: example-gallery/cli-workflows
:link-type: doc
The same estimation stories as explicit `qlx-opt` / `qlx-translate` pass pipelines.
^^^
`qlx-opt` · `qlx-translate` · `Stim`
:::
::::
