Create an out-of-tree CUDA-Q compiler pass plugin under `hxh-plugin/`.
Register the pass as `cudaq-hxh-to-z` and implement the single-qubit
optimization `H X H -> Z`. Build and test it against the provided CUDA-Q
development environment, and register the focused tests with CTest. Do not
modify CUDA-Q source or add the pass to a production pipeline.
