# Experimental features

This folder contains non-documented APIs that are experimental and may change or
get dropped at any moment without notice.

## Defining custom compile targets and runtime endpoints

CUDA-Q distinguishes between compile targets (the machine model that the user is
targeting when authoring kernels) and runtime endpoints (the final backend on
which compiled kernels should run or be simulated).

Both halves are installed together as a `CustomTarget` via `cudaq.set_target`.
The next `cudaq.set_target(...)` or `cudaq.reset_target()` replaces the
platform's QPUs and drops the custom target.

WARNING: There are currently no checks in place to ensure that the compile
target and runtime endpoint are compatible. If mis-configured, the behavior is
undefined and the user will most likely run into confusing runtime errors.

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
`cudaq._experimental.CompileTarget` and pass it together with a runtime target
name to `set_target`. `QuakeReferenceSemanticsProfile` and
`QuakeUnmappedProfile` are the supported IR profiles:

```python
from cudaq._experimental import CompileTarget, QuakeReferenceSemanticsProfile

profile = QuakeReferenceSemanticsProfile(basis_gates=["h", "t", "z(1)"])
target = CompileTarget(ir_output_profile=profile, codegen="qir-base")
cudaq.set_target("qpp-cpu", compile_target=target)
```

The target string name passed to `set_target` specifies the target that kernels
run on at runtime. Kernels are compiled against `compile_target` if it is given,
and against the named target's default compile target otherwise.

To change one setting of a named target, derive from the compile target that
`get_compile_target` returns while that target is set. For example, this
tightens the rotation tolerance of a target with a `quake-unmapped` stage:

```python
from cudaq._experimental import get_compile_target

cudaq.set_target("compiler-bench-ftqc-clifford-t")
target = get_compile_target()
stage = target.stages[0]
profile = stage.ir_output_profile
profile.approximation_tolerance = 1e-12
stage.ir_output_profile = profile
target.stages = [stage]
cudaq.set_target("compiler-bench-ftqc-clifford-t", compile_target=target)
```

Alternatively, a `cudaq._experimental.CustomTarget` can also be passed to
`set_target`. Passing `ir_output_profile=` builds a single stage whose lowering
is inferred by the registered handlers:

### Runtime endpoints

A runtime endpoint receives an already compiled kernel and executes it. Any
Python object implementing one or more of the protocols in
`cudaq._experimental.runtime_endpoint` can serve as one; pass it as the
`runtime_endpoint` field of a `CustomTarget`.
