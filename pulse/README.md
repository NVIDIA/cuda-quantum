# CUDA-Q pulse

CUDA-Q pulse is a pulse-level quantum programming research package built on
MLIR. It provides a Python kernel DSL, pulse and operator dialects, compiler
passes, and an experimental cuDensityMat execution path.

> [!WARNING]
> CUDA-Q pulse is **research-preview software**. It is not production software
> and is not a product-supported CUDA-Q feature. Its Python APIs, MLIR
> dialects, runtime interfaces, build options, numerical behavior, and file
> layout may change incompatibly or be removed without notice. No stability,
> compatibility, performance, or production-readiness guarantee is provided.
> Expect this work to evolve rapidly as the research matures.

Use this package for evaluation, experimentation, and collaboration—not for
production workloads. This preview does not publish binary wheels. Build it
from the CUDA-Q source tree or use the Docker environment below.

## Quick example

```python
import cudaq_pulse as pulse


@pulse.kernel
def rabi_oscillation(qubit):
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, gaussian(64, 0.5, 16.0), tone)


compiled_kernel = pulse.compile(
    rabi_oscillation,
    [pulse.qudit_ref()],
    qubit_freq_hz={0: 5.0e9},
)
print(compiled_kernel.mlir)
```

The compiler traces the Python kernel, builds Pulse dialect IR, applies the
selected transformations, and emits scheduled MLIR. The
[user guide](docs/index.rst) covers the kernel model, operations, compilation,
passes, and experimental GPU execution path.

## Build from Source

CUDA-Q pulse is a standalone CMake project. It does not build as part of the
CUDA-Q tree; it compiles against the CUDA-Q toolchain distributed as Python
wheels, so there is no LLVM to build and no submodule to initialize:

- **`cudaq-devel`** provides the headers, CMake packages, MLIR/LLVM archives,
  `mlir-tblgen`, `FileCheck`, and the rest of the pinned LLVM toolchain.
- **`cudaq`** (the runtime wheel) provides `libcudaqMLIR`, the single shared
  MLIR/LLVM instance that the pulse Python extension resolves its symbols from.

Both wheels must come from the same CUDA-Q revision. Beyond them, pulse needs
Python 3.10 or newer, CMake, Ninja, nanobind, `pytest`, Hypothesis, and LLVM lit.

```bash
git clone https://github.com/NVIDIA/cuda-quantum.git
cd cuda-quantum

python3 -m venv .venv-pulse
source .venv-pulse/bin/activate
python -m pip install cudaq cudaq-devel
python -m pip install "nanobind>=2.12" cmake ninja pytest hypothesis lit numpy

cmake -S pulse -B build-pulse -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-pulse --parallel
```

Note the `-S pulse`: the configure entry point is `pulse/`, not the repository
root. Nothing outside `pulse/` participates in the build.

Pulse locates the wheels through `site.getsitepackages()` of the interpreter
CMake picks up, so run CMake from the environment the wheels were installed
into, or pass `-DPython3_EXECUTABLE=/path/to/venv/bin/python`. To build against
a CUDA-Q installation that is not a wheel, point CMake at it directly with
`-DCMAKE_PREFIX_PATH=/path/to/cudaq/prefix`. nanobind is discovered the same
way and can be overridden with `-Dnanobind_DIR="$(python -m nanobind --cmake_dir)"`.

The `--target pulse` aggregate target is available for scripts that prefer a
named target.

The complete pulse package is staged in the build tree, so one `PYTHONPATH`
entry exposes both its Python sources and its native extension:

```bash
export PATH="$PWD/build-pulse/bin:$PATH"
export PYTHONPATH="$PWD/build-pulse/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import cudaq_pulse; print(cudaq_pulse.__version__)"
```

A GPU and cuDensityMat are not required to build the compiler, inspect the
generated MLIR, or run the default unit tests.

### Build the experimental cuDensityMat GPU runtime

Point `CUDENSITYMAT_ROOT` at a cuQuantum installation containing
`include/cudensitymat.h` and `lib/libcudensitymat.so`. With pulse enabled,
CMake discovers cuDensityMat and automatically adds the experimental GPU
runtime to the `pulse` target; there is no second pulse feature flag:

```bash
cmake -S pulse -B build-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAQ_BUILD_TESTS=ON \
  -DCUDENSITYMAT_ROOT=/path/to/cuquantum \
  -DPython3_EXECUTABLE="$PWD/.venv-pulse/bin/python"

cmake --build build-gpu --parallel --target pulse

export CUDAQ_PULSE_BUILD_DIR="$PWD/build-gpu"
export PATH="$PWD/build-gpu/bin:$PATH"
export PYTHONPATH="$PWD/build-gpu/python${PYTHONPATH:+:$PYTHONPATH}"
```

When `CUDENSITYMAT_ROOT` is set, configuration fails if the CUDA Toolkit or
cuDensityMat cannot be found. Without cuDensityMat, pulse remains usable in
compiler-only mode. The built runtime links the discovered
`cuDensityMat::cuDensityMat` library and records its library directory in
the runtime search path.

## Test

The aggregate target builds and runs the MLIR regression tests and the
non-GPU Python unit tests:

```bash
cmake --build build-pulse --target check-pulse
```

Both suites are also registered with `CTest`:

```bash
ctest --test-dir build-pulse -L pulse --output-on-failure
```

GPU tests remain opt-in and require a compatible NVIDIA GPU and cuDensityMat.
With the GPU runtime enabled, validate dependency linkage and GPU descriptor
creation with:

```bash
cmake --build build-gpu --target check-pulse-gpu
```

These checks validate SDK linkage and descriptor construction, then run the
numerical GPU tests for single-qubit drive evolution, T1 decay,
two-qubit XX coupling, and the public compile/JIT path. CMake enables numerical
tests only when `nvidia-smi` reports a GPU and the CUDA runtime can access at
least one device. Otherwise the target reports the tests disabled and, when
cuDensityMat is installed, retains the CPU-safe SDK linkage check. Passing this
suite is useful regression coverage, not a production numerical-accuracy
guarantee.

## Build the documentation

Install the documentation dependencies and enable the docs target at configure
time:

```bash
python3 -m pip install \
  "myst-parser>=3.0" \
  nvidia-sphinx-theme==0.0.8 \
  "sphinx>=8.1,<8.3"

cmake -S pulse -B build-pulse -G Ninja \
  -DCUDAQ_PULSE_BUILD_DOCS=ON
cmake --build build-pulse --target pulse-docs
```

The generated HTML is written to `build-pulse/docs/html` and uses NVIDIA's
Sphinx theme, matching the `QLX` documentation style.

## Docker

The package `Dockerfile` provides a turnkey compiler and Python environment. Run
the build from the CUDA-Q repository root so the full source tree is available
as context:

```bash
docker build \
  --file pulse/docker/Dockerfile \
  --tag cudaq-pulse-preview \
  .
docker run --rm -it cudaq-pulse-preview
```

Inside the container, `cudaq-pulse-opt` and `cudaq_pulse` are already on the
search paths. The `pulse-ci` Docker target additionally runs the unit tests and
generates documentation for local validation. GitHub Actions runs the same
CMake targets directly on a plain runner, installing the `cudaq` and
`cudaq-devel` wheels rather than building any part of the CUDA-Q toolchain.

CUDA-Q registers research preview roots in
`.github/research-preview-paths.txt`. Pull requests that change only registered
preview packages do not run stable CUDA-Q builds, packaging, macOS CI, `CodeQL`,
or spelling. Pulse changes instead run this package's license, test, and
documentation checks and inherit CUDA-Q's existing formatting job. A pull
request that also changes files outside the registered preview roots runs both
the preview-specific and stable suites. Trusted copy-PR branches, merge-queue
revisions, and `main` additionally run the numerical suite on CUDA-Q's GPU
runner whenever CMake detects an accessible NVIDIA GPU and CUDA runtime.

## Current scope and limitations

- The compiler, dialects, lowering experiments, and CPU unit tests are the
  primary research surface.
- GPU evolution currently models two-level transmons in per-qubit rotating
  frames. Target T1/T2 data, XX couplings, residual ZZ terms, and calibrated
  drive-amplitude scaling are supported. Transmon anharmonicity remains target
  metadata because a faithful leakage model requires three or more levels.
- GPU evolution supports the ``rk1``, ``rk2``, ``rk4``, ``magnus``, and
  ``crank_nicolson`` integrators through the pulse frontend.
  Readout/acquisition, observable evaluation, neutral-atom and multilevel
  models, unspecialized parameters, arbitrary Python waveform callbacks, and
  waveform algebra are rejected explicitly by the execution lowering.
- cuDensityMat integration is opt-in and requires a separately installed
  compatible CUDA Toolkit, cuQuantum SDK, NVIDIA driver, and GPU.
- The GPU runtime and end-to-end numerical evolution are active research.
  Regression tests exercise representative numerical paths, while results
  have no production compatibility or accuracy commitment.
- There are no binary wheels, service-level guarantees, long-term API
  guarantees, or product support commitments for this package.

## Repository contents

- `core/frontend/` — Python DSL, compiler pipeline, passes, targets, and
  runtime
- `core/mlir/` — Pulse, `QOp`, and CuDensityMat dialects and transformations
- `core/runtime/` — experimental cuDensityMat runtime shim
- `cmake/` — dependency, Python staging, testing, and documentation modules
- `test/` — lit and `FileCheck` compiler regression tests
- `tests/` — pytest unit and workload tests
- `examples/` — pulse programming examples
- `docs/` — user, API, and architecture documentation
- `benchmarks/` — compiler and simulation benchmarks

## License

CUDA-Q pulse is covered by the CUDA-Q repository's
[Apache License 2.0](../LICENSE).
