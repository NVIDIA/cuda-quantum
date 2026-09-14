# Experimental features

This folder contains non-documented APIs that are experimental and may change or
get dropped at any moment without notice.

## Defining custom compile targets and runtime endpoints

CUDA-Q distinguishes between compile targets (the machine model that the user is
targeting when authoring kernels) and runtime endpoints (the final backend on
which compiled kernels should run or be simulated).

The two halves are set independently, and both are dropped again by the next
`cudaq.set_target(...)` or `cudaq.reset_target()`.

WARNING: There are currently no checks in place to ensure that the compile target
and runtime endpoint are compatible. If mis-configured, the behavior is undefined
and the user will most likely run into confusing runtime errors.

### Custom targets

A custom target pairs both halves and installs them together with
`cudaq.set_target`. Build one by subclassing or instantiating
`cudaq._experimental.CustomTarget` and pass it to `cudaq.set_target`:

```python
import cudaq
from dataclasses import dataclass, field

from cudaq._experimental import CompileTarget, CustomTarget, RuntimeEndpoint


class MyEndpoint(RuntimeEndpoint):
    def sample(self, module, args, **kwargs):
        return cudaq.SampleResult({"00": kwargs["shots_count"]})


@dataclass
class MyCustomTarget(CustomTarget):
    runtime_endpoint: RuntimeEndpoint = field(default_factory=MyEndpoint)
    compile_target: CompileTarget = field(default_factory=CompileTarget)


cudaq.set_target(MyCustomTarget())
```

Only instances of `cudaq._experimental.CustomTarget` (or a subclass) are
accepted by this overload. Duck-typed objects with the same attributes are
rejected so that using the experimental API is an explicit opt-in.

### Compile targets

A compile target owns the MLIR pass pipelines, the code generation settings and
the capabilities that kernels are compiled against. Build one with
`cudaq._experimental.CompileTarget` and install it with
`cudaq._experimental.set_compile_target`. It then takes precedence over the
compile target that the active target's QPU would provide.

### Runtime endpoints

A runtime endpoint receives an already compiled kernel and executes it. Any
Python object implementing one or more of the protocols in
`cudaq._experimental.runtime_endpoint` can serve as one; register it with
`cudaq._experimental.set_runtime_endpoint`.

Registering a Python endpoint replaces the launch half only: compilation stays
with the active compile target. If no compile target has been set explicitly,
the default local-simulator one is installed and a warning is issued.
