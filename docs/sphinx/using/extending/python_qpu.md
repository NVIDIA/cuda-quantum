# Defining a Custom Python QPU

CUDA-Q allows advanced users to define their own quantum processing unit (QPU)
implementation in pure Python. A custom QPU can specify the compile target
to compile to as well as the kernel launch itself. Currently, the
supported policies are `sample` and `observe`.

## Overview

A Python QPU is a plain Python object. Whether a given launch policy is supported is
determined by the QPU protocols that the object implements:

- Implement `cudaq.qpu.SupportsSampleQPU` to support `cudaq.sample` launches
- Implement `cudaq.qpu.SupportsObserveQPU` to support `cudaq.observe` launches

Each protocol uses **policy-specific method names** so a single class can
implement both sampling and observation without name collisions:

| Protocol | Compile method | Launch method |
| --- | --- | --- |
| `SupportsSampleQPU` | `get_compile_target_sample()` | `launch_sample(module, args)` |
| `SupportsObserveQPU` | `get_compile_target_observe()` | `launch_observe(module, args)` |

## Example

The following example defines a minimal custom QPU that reuses the default
local compile settings but records launches for demonstration purposes.

```python
import cudaq
from cudaq.qpu import CompileTarget

class SampleOnesQPU:
    """Custom QPU returning always ones when sampling."""

    def __init__(self):
        self.sample_launches = 0
        self.observe_launches = 0

    # --- SupportsSampleQPU ---

    def get_compile_target_sample(self):
        # Reuse the compile pipeline for the active CUDA-Q target.
        return CompileTarget.default_sample()

    def launch_sample(self, module, args):
        self.sample_launches += 1
        print(
            f"sample launch: kernel={module.name}, args={args}"
        )
        return cudaq.SampleResult({"1": 100})


# Optional: verify the object satisfies the protocols at runtime.
qpu = SampleOnesQPU()
assert isinstance(qpu, cudaq.qpu.SupportsSampleQPU)

# Register the QPU as the active target (integration API).
cudaq.set_target(qpu)

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])

res = cudaq.sample(bell, shots_count=100)
print(res)  # { 1:100 }
```

## Method reference

`get_compile_target_sample` / `get_compile_target_observe`
: Called before the kernel is JIT-compiled. Return a `cudaq.qpu.CompileTarget`
  describing MLIR pass pipelines, codegen emission, and other compilation
  options. It is recommended to use the `CompileTarget.default_sample()` or
  `CompileTarget.default_observe()` methods to get the default compile targets
  and modify them as needed.

`launch_sample` / `launch_observe`
: Called after compilation with:

- `module` — a `cudaq.CompiledModule` handle to the compiled MLIR module and
    JIT-compiled kernel.
- `args` — a `cudaq.qpu.KernelArgs` handle to the pre-processed launch
    arguments. The arguments are packed as bytes and not stored as Python objects.
    Simple argument types such as integers and floats can be retrieved by indexing
    into the `args` object, but more complex types are treated as opaque.

  Return a `cudaq.SampleResult` or `cudaq.ObserveResult` respectively.
