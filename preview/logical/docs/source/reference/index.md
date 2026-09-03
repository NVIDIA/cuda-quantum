# Reference

Exact contracts for CUDA-Q Logical: how the project obtains its compiler
backbone from CUDA-Q, and how the contributor mechanics work. The
[quickstart](../quickstart.md), the [example gallery](../example-gallery/index.md),
and the [workflow guides](../workflows/index.md) remain the recommended
learning paths; this section is where precise boundaries live.

| Reference | What it answers |
|---|---|
| [Building against CUDA-Q](building-against-cudaq.md) | How CUDA-Q Logical obtains LLVM, MLIR, and CUDA-Q's shared compiler library from one CUDA-Q installation — and how builds fail when that contract is violated |
| [Developing CUDA-Q Logical](../developing.md) | Build, test, documentation, and contribution mechanics for the `preview/logical/` tree |

Related scope statements, also exact rather than aspirational:

- [Capabilities and status](../capabilities.md) — what the P0–P2 product
  does, what evidence exercises it, and what it deliberately does not do.
- [CUDA-Q Logical for Stim users](../for-stim-users.md) — the precise mapping
  between Stim concepts and CUDA-Q Logical artifacts.
