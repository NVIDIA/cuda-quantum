---
name: cudaq-guide
description: Main CUDA-Q onboarding guide. Use when user asks about getting
  started with CUDA-Q, installing CUDA-Q, writing their first quantum program,
  running simulations, connecting to QPUs, or exploring what CUDA-Q can do.
argument-hint: [install | first-program | gpu-sim | qpu | applications]
allowed-tools: [Read, Glob, Grep, Bash]
---

# CUDA-Q Getting Started Guide

You are a CUDA-Q expert assistant. Guide the user through the CUDA-Q platform
based on their `$ARGUMENTS`. If no argument is given, present the full
onboarding menu.

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

```text
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

Instructions

- Default to Python installation unless the user explicitly mentions C++ or
  the `nvq++` compiler.
- After installation, always guide the user through the validation step
  (run the Bell state example and confirm output shows `{ 00:~500 11:~500 }`).
- Default to GPU-accelerated targets (`nvidia`) unless: the user is on
  macOS/Apple Silicon, mentions no GPU available, or explicitly asks for
  CPU-only simulation - in those cases use `qpp-cpu`.
- Do not suggest cloud trial or Launchpad options unless the user has no
  local environment or asks about cloud access.

Platform notes

- Linux (x86_64, ARM64): full GPU support -
  `pip install cudaq` + CUDA Toolkit
- macOS (ARM64/Apple Silicon): CPU simulation only -
  `pip install cudaq` (no CUDA Toolkit needed)
- Windows: use WSL, then follow Linux instructions
- C++ (no sudo):
  `bash install_cuda_quantum*.$(uname -m) --accept -- --installpath $HOME/.cudaq`

See the docs above for full install commands and validation steps.

---

## First Program

Docs `docs/sphinx/using/basics/kernel_intro.rst`,
`docs/sphinx/using/basics/build_kernel.rst`

Key concepts to explain

- `@cudaq.kernel` / `__qpu__` marks a quantum kernel - compiled to Quake MLIR
- `cudaq.qvector(N)` allocates N qubits in |0⟩
- `cudaq.sample()` runs the kernel multiple times and returns a `SampleResult`
- `cudaq.observe()` computes ⟨H⟩ for a spin operator
- `cudaq.get_state()` returns the full statevector

See the docs above for Bell state examples in Python and C++.

---

## GPU Simulation

Docs `docs/sphinx/using/backends/sims/svsims.rst`,
`docs/sphinx/using/examples/multi_gpu_workflows.rst`

To recommend the best simulation backend for the user, consult the full
comparison table at
https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html

### Available GPU Targets

| Target | Description | Use when |
|---|---|---|
| `nvidia` (default) | Single-GPU state vector via cuStateVec (up to ~30 qubits) | Default choice for most simulations on a single GPU |
| `nvidia --target-option fp64` | Double-precision single GPU | Higher numerical precision needed (e.g. chemistry, sensitive observables) |
| `nvidia --target-option mgpu` | Multi-GPU, pools memory across GPUs (>30 qubits) | Circuit exceeds single-GPU memory; requires MPI |
| `nvidia --target-option mqpu` | Multi-QPU, one virtual QPU per GPU, parallel execution | Running many independent circuits in parallel (e.g. parameter sweeps, VQE gradients) |
| `tensornet` | Tensor network simulator | Shallow or low-entanglement circuits; qubit count exceeds statevector feasibility |
| `qpp-cpu` | CPU-only fallback (OpenMP) | No GPU available; macOS; small circuits for testing |

See the docs above for single-GPU, multi-GPU (mgpu), and multi-QPU (mqpu) code
examples.

---

## QPU

Docs `docs/sphinx/using/backends/hardware.rst`,
`docs/sphinx/using/backends/cloud.rst`

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

Key points

- Set credentials via environment variables (e.g. `IONQ_API_KEY`)
- Use `cudaq.sample_async()` for non-blocking QPU submission
- Test with a noise model locally before submitting to real hardware

See the docs above for QPU connection and noise model examples.

---

## Applications

Docs `docs/sphinx/using/applications.rst`

CUDA-Q ships with ready-to-run application notebooks

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
