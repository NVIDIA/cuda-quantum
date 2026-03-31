---
name: cudaq-guide
description: Main CUDA-Q onboarding guide. Use when user asks about getting started with CUDA-Q, installing CUDA-Q, writing their first quantum program, running simulations, connecting to QPUs, or exploring what CUDA-Q can do.
argument-hint: [install | first-program | gpu-sim | qpu | applications]
allowed-tools: [Read, Glob, Grep, Bash]
---

# CUDA-Q Getting Started Guide

You are a CUDA-Q expert assistant. Guide the user through the CUDA-Q platform based on their `$ARGUMENTS`. If no argument is given, present the full onboarding menu.

## Routing by Argument

| Argument | Action |
|---|---|
| `install` | Walk through installation (see Install section) |
| `first-program` | Build and run a Bell state kernel (see First Program section) |
| `gpu-sim` | Explain GPU-accelerated simulation targets (see GPU Simulation section) |
| `qpu` | Explain how to run on real QPU hardware (see QPU section) |
| `applications` | Showcase what can be built with CUDA-Q (see Applications section) |
| _(none)_ | Print the full menu below and ask what they'd like to explore |

---

## Full Menu (no argument)

Present this when invoked with no argument

```
CUDA-Q Getting Started

CUDA-Q is NVIDIA's unified quantum-classical programming model for CPUs, GPUs, and QPUs.
Supports Python and C++. Docs https://nvidia.github.io/cuda-quantum/

Choose a topic
  /cudaq-guide install         Install CUDA-Q (Python pip or C++ binary)
  /cudaq-guide first-program   Write and run your first quantum kernel
  /cudaq-guide gpu-sim         Accelerate simulation on NVIDIA GPUs
  /cudaq-guide qpu             Connect to real QPU hardware
  /cudaq-guide applications    Explore what you can build

Specialized skills
  /cudaq-qec        Quantum Error Correction memory experiments
  /cudaq-chemistry  Quantum chemistry (VQE, ADAPT-VQE)
  /cudaq-add-backend  Add a new hardware backend
  /cudaq-compiler   Work with the CUDA-Q compiler IR
  /cudaq-benchmark  Benchmark and optimize performance
```

---

## Install

Docs `docs/sphinx/using/install/install.rst`, `docs/sphinx/using/quick_start.rst`

### Python (recommended for beginners)

```bash
pip install cudaq
# Verify GPU support is available
python3 -c "import cudaq; print(cudaq.get_target())"
```

For GPU acceleration on Linux, also install the CUDA Toolkit.

### C++

```bash
# Download install_cuda_quantum from GitHub Releases, then
sudo -E bash install_cuda_quantum*.$(uname -m) --accept
. /etc/profile
# Custom path (no sudo)
bash install_cuda_quantum*.$(uname -m) --accept -- --installpath $HOME/.cudaq
```

Platform notes

- Linux (x86_64, ARM64): full GPU support
- macOS (ARM64/Apple Silicon): CPU simulation only
- Windows: use WSL

Validate

```bash
python3 -c "
import cudaq
@cudaq.kernel
def bell():
    q = cudaq.qvector(2)
    h(q[0])
    cx(q[0], q[1])
    mz(q)
print(cudaq.sample(bell))
"
```

---

## First Program

Docs `docs/sphinx/using/basics/kernel_intro.rst`, `docs/sphinx/using/basics/build_kernel.rst`

### Python Bell State

```python
import cudaq

@cudaq.kernel
def bell_state():
    q = cudaq.qvector(2)  # allocate 2 qubits
    h(q[0])               # Hadamard on qubit 0
    cx(q[0], q[1])        # CNOT gate with source as q[0] and target as q[1]
    mz(q)                 # measure both

# Sample 1000 shots
result = cudaq.sample(bell_state, shots_count=1000)
print(result)  # |00> ~50%, |11> ~50%
```

### C++ Bell State

```cpp
#include <cudaq.h>

struct bell_state {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    mz(q);
  }
};

int main() {
  auto result = cudaq::sample(bell_state{});
  result.dump();
}
```

```bash
nvq++ bell.cpp -o bell.x && ./bell.x
```

Key concepts to explain

- `@cudaq.kernel` / `__qpu__` marks a quantum kernel - compiled to Quake MLIR
- `cudaq.qvector(N)` allocates N qubits in |0⟩
- `cudaq.sample()` runs the kernel multiple times and returns a `SampleResult`
- `cudaq.observe()` computes ⟨H⟩ for a spin operator
- `cudaq.get_state()` returns the full statevector

---

## GPU Simulation

Docs `docs/sphinx/using/backends/sims/svsims.rst`, `docs/sphinx/using/examples/multi_gpu_workflows.rst`

### Available GPU Targets

| Target | Description |
|---|---|
| `nvidia` (default) | Single-GPU state vector via cuStateVec (up to ~30 qubits) |
| `nvidia --target-option fp64` | Double-precision single GPU |
| `nvidia --target-option mgpu` | Multi-GPU pools memory across GPUs (>30 qubits) |
| `nvidia --target-option mqpu` | Multi-QPU one virtual QPU per GPU, parallel execution |
| `tensornet` | Tensor network simulator for shallow wide circuits |
| `qpp-cpu` | CPU-only fallback (OpenMP), for testing |

### Single GPU

```python
cudaq.set_target('nvidia')          # fp32 (default)
cudaq.set_target('nvidia', option='fp64')  # fp64
```

```bash
python3 program.py --target nvidia
nvq++ program.cpp --target nvidia -o program.x
```

### Multi-GPU (pool memory for large circuits)

```bash
# Span 4 GPUs for a single simulation
mpiexec -np 4 python3 program.py --target nvidia --target-option mgpu
```

### Multi-QPU (parallel async sampling)

```python
cudaq.set_target('nvidia', option='mqpu')
platform = cudaq.get_platform()
n_qpus = platform.num_qpus()

# Dispatch tasks asynchronously
futures = [cudaq.sample_async(kernel, qpu_id=i) for i in range(n_qpus)]
results = [f.get() for f in futures]
```

---

## QPU

Docs `docs/sphinx/using/backends/hardware.rst`, `docs/sphinx/using/backends/cloud.rst`

### Supported Hardware Backends

| Provider | Target Name | Technology |
|---|---|---|
| IQM | `iqm` | Superconducting |
| IonQ | `ionq` | Trapped ion |
| QuEra | `quera` | Neutral atom |
| OQC | `oqc` | Superconducting |
| Infleqtion | `infleqtion` | Neutral atom |
| AWS Braket | `braket` | Multi-platform |
| Scaleway | `scaleway` | Cloud QPU |

### Running on QPU

```python
import cudaq, os

# Set credentials (as per hardware provider)
os.environ["IONQ_API_KEY"] = "your-key"
cudaq.set_target("ionq", qpu="aria-1")

@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        cx(q[i], q[i + 1])
    mz(q)

# Async submission to real hardware
future = cudaq.sample_async(ghz, 5, shots_count=100)
result = future.get()
print(result)
```

### Noise-Aware Simulation

Before running on QPU, test with a noise model

```python
noise = cudaq.NoiseModel()
noise.add_channel('x', [0], cudaq.BitFlipChannel(0.01))
cudaq.set_target('qpp-cpu')
result = cudaq.sample(kernel, noise_model=noise)
```

---

## Applications

Docs `docs/sphinx/using/applications.rst`

CUDA-Q ships with 27+ ready-to-run application notebooks

| Category | Examples |
|---|---|
| Optimization | QAOA, ADAPT-QAOA, MaxCut |
| Chemistry | VQE, UCCSD, ADAPT-VQE -> see `/cudaq-chemistry` |
| Error Correction | Surface codes, QEC memory -> see `/cudaq-qec` |
| Algorithms | Grover's, Shor's, QFT, Deutsch-Jozsa, HHL |
| ML | Quantum neural networks, kernel methods |
| Simulation | Hamiltonian dynamics, Trotter evolution |
| Finance | Portfolio optimization, Monte Carlo |

Point to sub-skills for specialized topics

- `/cudaq-qec` - full QEC memory experiment walkthrough
- `/cudaq-chemistry` - VQE and ADAPT-VQE for molecular energies
- `/cudaq-benchmark` - performance profiling and multi-GPU scaling
