# mock_observe_qpu Plugin

`mock_observe_qpu` is a reference **external custom QPU** plugin. It shows how
to ship a Fermioniq-style server-side `observe()` backend as a pip-installable
package without modifying the CUDA-Q source tree: a custom `QPU` subclass
(`platform-qpu: mock_observe_qpu`) plus a `ServerHelper`, registered from one
shared library that CUDA-Q auto-loads as `libcudaq-<platform-qpu>-qpu.so`.

Unlike `mock_rest` (which reuses stock `remote_rest` and splits Pauli terms
client-side), this plugin disables Pauli-term splitting, attaches the full
`spin_op` on `KernelExecution::user_data["observable"]`, and returns an
expectation-shaped result.

## What It Demonstrates

- External custom QPU registration via `CUDAQ_REGISTER_TYPE(..., QPU, ...)`.
- Auto-load of `libcudaq-mock_observe_qpu-qpu.so` from the plugin `lib/` directory
  (no `plugin-libraries` YAML field required).
- Target YAML with `platform-qpu: mock_observe_qpu` (not `remote_rest`).
- Full observable attachment for `cudaq.observe` (Fermioniq-compatible JSON).
- Expectation results via `sample_result(ExecutionResult(double))`.
- The same Python packaging / `cudaq.backends` / `nvq++` install story as
  `mock_rest`.

## How It Works

- `targets/mock_observe_qpu.yml.in`: selects the custom QPU and links
  `-lcudaq-mock_observe_qpu-qpu`.
- `MockObserveQPU.cpp`: implements `MockObserveQPU` (no Pauli split; attach
  observable; sync observe through the async executor path) and
  `MockObserveServerHelper` (reads `user_data["observable"]`, returns the
  configured `value` expectation).
- Python packaging mirrors `mock_rest`.

For local testing, the helper can talk to the generic `mock_qpu` echo server
(`python/tests/utils/start_mock_qpu.py echo`). The configured `value` target
argument is used as the expectation when the echo payload is not numeric.

## Build

From the CUDA-Q repository root:

```sh
cmake -B build \
  -DCUDAQ_EXTERNAL_PROJECTS="mock-observe-qpu" \
  -DCUDAQ_EXTERNAL_MOCK_OBSERVE_QPU_SOURCE_DIR=$PWD/docs/sphinx/examples/plugins/mock_observe_qpu

ninja -C build cudaq-example-qpu-mock_observe_qpu
```

The built plugin root is `build/external/mock-observe-qpu`.

## Test

```sh
ninja -C build check-mock-observe-qpu
```

## Run

```python
import cudaq

cudaq.register_backend_path("build/external/mock-observe-qpu")
cudaq.set_target("mock_observe_qpu",
                 url="http://localhost:62454",
                 value="0.41")

@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])

H = 0.5 * cudaq.spin.z(0) + 0.3 * cudaq.spin.z(0) * cudaq.spin.z(1)
result = cudaq.observe(bell, H)
assert abs(result.expectation() - 0.41) < 1e-9
```

## Install as an End User

```sh
python3 -m pip install build/external/mock-observe-qpu
python3 -m cudaq_example_mock_observe_qpu --install-nvqpp
```
