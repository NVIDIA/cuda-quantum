# MKL-Q Apple Silicon Bootstrap

MKL-Q starts as a CUDA-Q-compatible fork with Apple Silicon targets layered on
top of the existing CUDA-Q runtime contracts.

## Targets

MKL-Q targets are controlled by `CUDAQ_ENABLE_MKLQ_BACKEND`, which defaults to
ON only for Apple Silicon (`Darwin arm64/aarch64`) builds.

- `mklq-cpu`: Apple Silicon CPU baseline target. It uses MKL-Q's native fp64
  state-vector simulator with initial OpenMP loop parallelism for correctness
  and tuning work. Accelerate/vDSP dense-probability variants are benchmarked
  separately before being admitted into the default runtime path.
- `mklq-metal`: experimental Apple GPU target name. It registers an MKL-Q
  mixed-path simulator as `mklq_metal`: supported single-target and two-target
  updates, including controlled forms, can stay in a resident fp32 Metal state
  buffer across supported gate sequences. Dense full-register probability
  fills, cost-gated resident marginal probability fills, and measure/reset
  probability-and-collapse paths can read and update that resident buffer
  directly. Measurement
  probability uses a dedicated measured-qubit Metal
  reduction kernel with a small host partial-sum finish, followed by a Metal
  collapse kernel. Unsupported paths fall back to the MKL-Q fp64 CPU oracle
  after synchronizing host state.

## Usage

```python
import cudaq

cudaq.set_target("mklq-cpu")
# cudaq.set_target("mklq-metal")  # experimental mixed-path Metal target
```

```bash
nvq++ --target mklq-cpu program.cpp
nvq++ --target mklq-metal program.cpp
```

## Local Verification

Use the built Python tree when running MKL-Q target tests from this repository:

```bash
PYTHONPATH="$(pwd)/build-python/python" \
python3 -m pytest \
  python/tests/backends/test_mklq_python_api.py \
  python/tests/backends/test_mklq_nvqpp_smoke.py \
  python/tests/builder/test_mklq_targets.py \
  python/tests/backends/test_mklq_benchmark_harness.py \
  -q
```

Run the backend C++ unit tests from the build tree:

```bash
ctest --test-dir build-python -R "MKLQ(Metal|Cpu)Tester" --output-on-failure
```

Check installed and source target configs with FileCheck:

```bash
FILECHECK="${FILECHECK:-$(command -v FileCheck || true)}"
if [ -z "${FILECHECK}" ] && [ -n "${LLVM_INSTALL_PREFIX:-}" ]; then
  FILECHECK="${LLVM_INSTALL_PREFIX}/bin/FileCheck"
fi
if [ -z "${FILECHECK}" ] && [ -x "${HOME}/.local/llvm/bin/FileCheck" ]; then
  FILECHECK="${HOME}/.local/llvm/bin/FileCheck"
fi

"${FILECHECK}" \
  targettests/TargetConfig/mklq_targets.config --check-prefix=CPU \
  < runtime/nvqir/mklq/mklq-cpu.yml
"${FILECHECK}" \
  targettests/TargetConfig/mklq_targets.config --check-prefix=METAL \
  < runtime/nvqir/mklq/mklq-metal.yml
"${FILECHECK}" \
  targettests/TargetConfig/mklq_targets.config --check-prefix=CPU \
  < build-python/targets/mklq-cpu.yml
"${FILECHECK}" \
  targettests/TargetConfig/mklq_targets.config --check-prefix=METAL \
  < build-python/targets/mklq-metal.yml
```

After `cmake --build build-python --target install`, verify the installed
prefix as well:

```bash
CUDAQ_INSTALL_PREFIX="${CUDAQ_INSTALL_PREFIX:-${HOME}/.cudaq-mklq}"

PYTHONPATH="${CUDAQ_INSTALL_PREFIX}" \
python3 -m pytest \
  python/tests/backends/test_mklq_python_api.py \
  python/tests/builder/test_mklq_targets.py \
  -q

CUDAQ_NVQPP="${CUDAQ_INSTALL_PREFIX}/bin/nvq++" \
PYTHONPATH="${CUDAQ_INSTALL_PREFIX}" \
python3 -m pytest python/tests/backends/test_mklq_nvqpp_smoke.py -q
```

If you invoke `nvq++` smoke checks manually, run each target in its own
temporary working directory. The CUDA-Q toolchain emits fixed intermediate file
names based on the source basename, so parallel `nvq++` invocations in one
directory can overwrite each other's intermediates and produce misleading
target-marker output.

## Development Gates

1. Keep `qpp-cpu` as the upstream correctness reference.
2. Add CPU optimizations behind `mklq-cpu` only after circuit-level parity tests
   pass.
3. Keep `mklq-metal` experimental until Metal-native coverage no longer depends
   on CPU oracle fallback and it passes the same tests against the CPU oracle.
4. Follow the Metal sampling draw/count plan in
   `docs/superpowers/plans/2026-06-18-mklq-metal-sampling-draw-count.md`
   before moving sample count accumulation onto the GPU.
5. Benchmark on a fixed Apple Silicon host before making performance claims.

## Tooling

- Use GitHub tooling only after an `origin` fork exists; keep `upstream` pointed
  at `NVIDIA/cuda-quantum`.
- Use Linear only if the CPU, Metal, benchmark, and documentation work needs
  milestone tracking.
- Use multi-agent work only for disjoint implementation slices, for example one
  worker for CPU kernels and one worker for Metal kernels.

## Current Limitations

- `mklq-cpu` is a correctness-first fp64 state-vector baseline with initial
  OpenMP loop parallelism for hot state updates. The OpenMP path currently runs
  only for states with at least 32768 amplitudes and caps automatic OpenMP
  regions at four worker threads while this backend is being tuned.
  Single-qubit gates use an in-place 2x2 update path; built-in uncontrolled
  and controlled H/Y/Rx/Ry/Rz use dedicated structured in-place fast paths,
  built-in X/CNOT use a dedicated in-place bit-flip permutation fast path, and
  built-in controlled-Z/CZ use a dedicated in-place phase-sign fast path. Custom
  single-qubit operations still use the generic 2x2 path.
  Two-target gates, including custom two-qubit operations, use an in-place 4x4
  block update path; uncontrolled SWAP uses a dedicated in-place permutation
  fast path.
  Sampling has a fast path for full-register measurements in natural qubit
  order, including a sparse-outcome path for basis/GHZ-like states.
  Counts-only dense sampling (`includeSequentialData=false`) aggregates drawn
  outcome counts before converting bit strings, using a bounded dense counter
  buffer for small outcome spaces and a sparse map for larger ones. Standard
  non-explicit `cudaq.sample` execution now routes through this counts-only
  path, including deprecated named-register remapping, while
  explicit-measurement sampling still records per-shot data. Public
  `sample_result::sequential_data()` access remains compatible by expanding
  counts on demand when a counts-only backend result omitted stored sequential
  data. It does not use Accelerate/vDSP in the default dense probability-fill
  path yet: the current local Apple M5 microbenchmark shows the interleaved
  vDSP variant is slower than the existing OpenMP split loop at q15-q20. It
  also does not yet use BLAS or NEON micro-kernels for specialized gate paths.
- `mklq-metal` remains experimental. It no longer routes through the upstream
  `qpp` backend and now links Metal/Foundation on Apple platforms for runtime
  device discovery plus a resident fp32 Metal state buffer for supported
  single-target gates, two-target gates, and dense full-register
  probability-fill kernels. Partial-register sampling uses a cost-gated
  resident marginal probability kernel for small marginal buffers; when the
  marginal partial-sum work is no smaller than a full probability fill, it
  computes resident full-register probabilities once and folds them to
  marginal outcome probabilities on the host without first downloading the
  state vector. It still performs sample draw/count accumulation host-side; the
  current local shot-scaling gate does not justify GPU-side count accumulation
  yet.
  Resident measure/reset can compute the measured qubit probability with a
  dedicated measured-qubit Metal reduction kernel, then collapse the selected
  branch with a Metal kernel without first downloading the state. The host only
  sums per-threadgroup partial probabilities for those reductions. The host
  state remains fp64 and is synchronized at CPU fallback/readback boundaries.
  Unsupported paths still fall back to the MKL-Q CPU oracle, and the target is
  not yet a full Metal GPU backend.
- The first correctness gate covers basic state allocation, sampling, reset,
  state export/import, state index bounds, non-power-of-two state rejection,
  out-of-range qubit boundary checks, basic custom two-qubit operations,
  basic and parameterized observe paths, observe lists, shots-based observe,
  decorator-kernel `cudaq.run` mid-circuit feedback/reset paths, and
  H/X/Y/Z/Rx/Ry/Rz/CNOT/CZ/SWAP circuits. Broader noise coverage should land
  before treating MKL-Q as a complete drop-in CPU simulation target.
