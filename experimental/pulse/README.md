# cudaq-pulse

Pulse-level quantum programming on MLIR. Write pulse kernels in Python, compile
them to a Pulse MLIR dialect, transform them with Python passes, and emit MLIR.

> **Experimental.** APIs may change without notice and carry no stability guarantee.

## Hello World

```python
import cudaq_pulse as pulse

@pulse.kernel
def rabi_oscillation(qubit):
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, gaussian(64, 0.5, 16.0), tone)

compiled_kernel = pulse.compile(rabi_oscillation, [pulse.qudit_ref()],
                                qubit_freq_hz={0: 5.0e9})
print(compiled_kernel.mlir)
```

## How It Works

cudaq-pulse is a Python-first compiler pipeline:

**1. Write a kernel in Python** -- the `@pulse.kernel` DSL (`get_drive_line`,
`drive`, `gaussian`, `wait`, `sync`, ...).

**2. Compile to MLIR** -- `pulse.compile()` traces the kernel and returns a
`CompiledKernel` whose `.mlir` is the scheduled Pulse dialect.

```python
compiled = pulse.compile(rabi_oscillation, [pulse.qudit_ref()],
                         qubit_freq_hz={0: 5.0e9})
print(compiled.mlir)
```

**3. Write transform passes in Python and apply them** -- passes are plain
`Program -> Program` functions over a lightweight IR, so you can compose the
built-ins or author your own.

```python
from cudaq_pulse.passes import ProgramBuilder, run_virtual_z, run_fusion

b = ProgramBuilder("rabi", clock_ghz=2.0)
line, tone = b.get_drive_line(qubit=0, freq_hz=5.0e9)
b.drive(line, b.gaussian(64, 0.5, 16.0), tone)

program = run_fusion(run_virtual_z(b.build()))
```

**4. Emit** -- lower the transformed program back to MLIR.

```python
from cudaq_pulse.passes import program_to_pulse_mlir

print(program_to_pulse_mlir(program))
```

See [docs/user_guide/passes.rst](docs/user_guide/passes.rst) for writing custom passes.

## Install

cudaq-pulse lives in `experimental/pulse` within CUDA-Q. Build it from source:

```bash
cd experimental/pulse
mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
ninja
```

Then put the frontend and built bindings on your `PYTHONPATH`:

```bash
export PYTHONPATH=core/frontend:build/core/mlir/bindings
python -c "import cudaq_pulse as pulse; print(pulse.__version__)"
```

Preview wheels (`pip install cudaq-pulse`) may be published later.

### IDE Setup (recommended)

Add a `pyrightconfig.json` in your project root to suppress warnings for bare
DSL names inside kernels:

```json
{
    "reportUndefinedVariable": "warning"
}
```

## GPU Simulation (preview)

GPU simulation via NVIDIA cuDensityMat is under active porting and not yet
front-facing. See [docs/user_guide/gpu_execution.rst](docs/user_guide/gpu_execution.rst)
for the experimental preview.

## Learn More

- [Examples](examples/) -- 17 numbered tutorials from single-qubit Rabi to GHZ state prep
- [Documentation](docs/) -- user guide, API reference, architecture
- [Benchmarks](benchmarks/) -- compile-time and scaling measurements

## License

Apache 2.0. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
