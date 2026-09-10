# QPU layout preview

A CUDA-Q target that lays a kernel out on a region-based QPU and records what it
costs. It does **not** simulate quantum state -- counts come back all-zero. The
trace is the result.

The model is a *reference*: it makes its own placement, routing and scheduling
decisions while walking virtual-qubit wire-set Quake, and deliberately ignores
the `{region = @rN}` / `quake.move` annotations the region-lowering passes emit.
Those passes are an alternative path to compare against later, not an input.

## Using it

```bash
export PYTHONPATH="$PYTHONPATH:/workspaces/cuda-quantum/preview/qpu-layout/python"
```

Through a target:

```python
import cudaq
from cudaq_qpu_layout import QpuLayoutTarget

target = QpuLayoutTarget.build(num_regions=2, region_size=2)
cudaq.set_target(target)
cudaq.sample(my_kernel)

trace = target.runtime_endpoint.trace
print(trace["summary"])
```

Or on a Quake payload directly, with no target and no compilation:

```bash
python3 -m cudaq_qpu_layout payload.mlir --regions 2 --region-size 2 \
    -o trace.json --viewer trace.html
```

## Viewing a trace

`viewer.html` is self-contained and opens on its own. `write_viewer(trace, path)`
bakes a specific trace into a copy of it. To reach it from outside the container:

```bash
python3 -m cudaq_qpu_layout.serve --port 8765 --dir <trace dir>
```

then `http://localhost:8765/viewer.html?trace=NAME.json` on the host. Port 8765
is forwarded by `.devcontainer/devcontainer.json`.

## How the target is put together

CUDA-Q splits a backend into a compile half and a launch half, and
`cudaq._experimental` lets both be defined in Python:

- **`build_compile_target()`** sets `mid_level_pipeline` to
  `prepare-for-wireset{add-wireset=true}` and `codegen_translation` to `nop`, so
  the endpoint is handed wire-set Quake rather than QIR. No decomposition runs
  (the model does not care about a gate set) and `disable_qubit_mapping` is set,
  since placement is the thing being modeled.
- **`QpuLayoutEndpoint.sample()`** receives that `CompiledModule`, unrolls the
  control flow `nop` codegen leaves behind, and traces it.
- **`QpuLayoutTarget`** pairs the two so `cudaq.set_target` installs them
  together.

There is no `.yml`, no C++ `ServerHelper`, no mock HTTP server, and nothing to
build -- this preview is pure Python on top of an installed CUDA-Q.

## Layout

| Path | Role |
| --- | --- |
| `model.py` | `QpuModel`: regions, capacity, movement costs |
| `sim.py` | the walker and the reference model (placement, routing, scheduling) |
| `trace.py` | trace schema and `replay()` |
| `target.py` | the compile target, runtime endpoint and `CustomTarget` |
| `viewer.py`, `viewer.html`, `serve.py` | the trace viewer |

## Not modeled yet

Intra-region topology: a region is all-to-all, so co-located qubits interact at
no cost. When it is modeled it belongs in `Router` as a **move** -- which may or
may not lower to a SWAP -- never as a swap primitive.

Also absent: Clifford-frame merging across joined regions, heterogeneous region
sizes, magic-state factories, and classically-conditioned operations. Ports are
unbounded: they cost but never queue, so only compute wires are scarce.
