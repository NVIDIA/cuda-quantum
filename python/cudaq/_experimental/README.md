# Experimental features

This folder contains non-documented APIs that are experimental and may change or
get dropped at any moment without notice.

## Defining custom compile targets and runtime endpoints

CUDA-Q distinguishes between compile targets (the machine model that the user is
targeting when authoring kernels) and runtime endpoints (the final backend on which
compiled kernels should run or be simulated).

### Compile targets

Custom compile targets can be defined using the `cudaq._experimental.target.CompileTarget` class.

The target can be set for all successive launches using `cudaq.set_target(compile_target)`.`

### Runtime endpoints

Currently no Python API is provided for defining custom runtime endpoints.