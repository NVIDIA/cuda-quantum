# CUDA-Q Onboarding Reference

Use this reference for `/cudaq-guide install`, `test-program`, `gpu-sim`,
`qpu`, `applications`, and `parallelize`.

## Documentation Map

| Section | Doc file |
| --- | --- |
| Install | `docs/sphinx/using/install/install.rst`, `docs/sphinx/using/quick_start.rst` |
| Test Program | `docs/sphinx/using/basics/kernel_intro.rst`, `docs/sphinx/using/basics/build_kernel.rst` |
| GPU Simulation | `docs/sphinx/using/backends/sims/svsims.rst`, `docs/sphinx/using/examples/multi_gpu_workflows.rst` |
| QPU | `docs/sphinx/using/backends/hardware.rst`, `docs/sphinx/using/backends/cloud.rst` |
| Applications | `docs/sphinx/using/applications.rst` |
| Parallelize | `docs/sphinx/using/examples/multi_gpu_workflows.rst` |

## Install

Instructions:

- Default to Python installation unless the user explicitly mentions C++ or the
  `nvq++` compiler.
- After installation, always guide the user through validation with the Bell
  state example and confirm output shows roughly `{ 00:~500 11:~500 }`.
- Default to GPU-accelerated targets (`nvidia`) unless the user is on
  macOS/Apple Silicon, mentions no GPU, or explicitly asks for CPU-only
  simulation. In those cases use `qpp-cpu`.
- Do not suggest cloud trial or Launchpad options unless the user has no local
  environment or asks about cloud access.

Platform notes:

- Linux x86_64 or ARM64: full GPU support with `pip install cudaq` plus CUDA
  Toolkit.
- macOS ARM64/Apple Silicon: CPU simulation only with `pip install cudaq`; no
  CUDA Toolkit needed.
- Windows: use WSL, then follow Linux instructions.
- C++ without sudo:

  ```bash
  bash install_cuda_quantum*.$(uname -m) --accept -- --installpath $HOME/.cudaq
  ```

- Brev cloud workspace: log in at the NVIDIA Application Hub, open a CUDA-Q
  workspace, then SSH in with the Brev CLI:

  ```bash
  brev open ${WORKSPACE_NAME}
  ```

  CUDA-Q and the CUDA Toolkit are pre-installed.

## Test Program

Key concepts to explain:

- `@cudaq.kernel` / `__qpu__` marks a quantum kernel compiled to Quake MLIR.
- `cudaq.qvector(N)` allocates N qubits in `|0>`.
- `cudaq.sample()` runs a measured kernel and returns a bitstring histogram
  (`SampleResult`).
- `cudaq.run()` runs a kernel with a classical return value `shots_count` times
  and returns the list of those return values.
- `cudaq.observe()` computes expectation value `<H>` for a spin operator.
- `cudaq.get_state()` returns the full statevector on simulators.

Kernel restrictions:

- Only a restricted Python subset is valid inside a kernel; it compiles to
  Quake MLIR, not normal Python.
- NumPy and SciPy cannot be used inside a kernel. Use them outside kernels for
  classical pre/post-processing.
- Kernels can call other kernels, but the callee must also be a CUDA-Q kernel.

For compiler internals (`inspect` module -> `ast_bridge.py` -> Quake MLIR ->
QIR -> JIT), route to `/cudaq-compiler` if that skill is available.

## GPU Simulation

To recommend the best simulation backend, consult the CUDA-Q simulator
documentation:
<https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html>.

| Target | Description | Use when |
|---|---|---|
| `nvidia` | Single-GPU state vector via cuStateVec, up to roughly 30 qubits | Default choice for most single-GPU simulations |
| `nvidia --target-option fp64` | Double-precision single GPU | Higher numerical precision is needed, for example chemistry or sensitive observables |
| `nvidia --target-option mgpu` | Multi-GPU statevector that pools memory across GPUs | One circuit exceeds single-GPU memory; requires MPI |
| `nvidia --target-option mqpu` | Multi-QPU mode with one virtual QPU per GPU | Many independent circuits in parallel, such as parameter sweeps or VQE gradients |
| `tensornet` | Tensor-network simulator | Shallow or low-entanglement circuits where qubit count exceeds statevector feasibility |
| `qpp-cpu` | CPU-only fallback with OpenMP | No GPU, macOS, or small local tests |

## QPU

Do not dump all providers at once. Use a two-step dialogue.

Step 1: ask which technology they want:

```text
Which QPU technology are you targeting?
  1. Ion trap         (IonQ, Quantinuum)
  2. Superconducting  (IQM, OQC, Anyon, TII, QCI)
  3. Neutral atom     (QuEra, Infleqtion, Pasqal)
  4. Cloud / multi-platform (AWS Braket, Scaleway)
```

Step 2: once they pick a technology, ask which provider, then read the
corresponding doc file and walk through it step by step.

| Technology | Provider | Doc file |
|---|---|---|
| Ion trap | IonQ | `docs/sphinx/using/backends/hardware/iontrap.rst` (IonQ section) |
| Ion trap | Quantinuum | `docs/sphinx/using/backends/hardware/iontrap.rst` (Quantinuum section) |
| Superconducting | IQM | `docs/sphinx/using/backends/hardware/superconducting.rst` (IQM section) |
| Superconducting | OQC | `docs/sphinx/using/backends/hardware/superconducting.rst` (OQC section) |
| Superconducting | Anyon | `docs/sphinx/using/backends/hardware/superconducting.rst` (Anyon section) |
| Superconducting | TII | `docs/sphinx/using/backends/hardware/superconducting.rst` (TII section) |
| Superconducting | QCI | `docs/sphinx/using/backends/hardware/superconducting.rst` (QCI section) |
| Neutral atom | Infleqtion | `docs/sphinx/using/backends/hardware/neutralatom.rst` (Infleqtion section) |
| Neutral atom | QuEra | `docs/sphinx/using/backends/hardware/neutralatom.rst` (QuEra section) |
| Neutral atom | Pasqal | `docs/sphinx/using/backends/hardware/neutralatom.rst` (Pasqal section) |
| Cloud | AWS Braket | `docs/sphinx/using/backends/cloud/braket.rst` |
| Cloud | Scaleway | `docs/sphinx/using/backends/cloud/scaleway.rst` |

After walking through the provider steps, always close with:

- Test locally first with `emulate=True` before submitting to hardware.
- Use `cudaq.sample_async()` / `cudaq.observe_async()` for non-blocking
  submission.
- Handle provider credentials securely: export them as environment variables in
  the shell session, or use a local profile or secrets manager that is not
  committed. Never paste tokens into shared files, logs, notebooks, or commits.

## Applications

CUDA-Q ships with ready-to-run application notebooks.

| Category | Examples |
|---|---|
| Optimization | QAOA, ADAPT-QAOA, MaxCut |
| Chemistry | VQE, UCCSD, ADAPT-VQE |
| Error Correction | Surface codes, QEC memory |
| Algorithms | Grover's, Shor's, QFT, Deutsch-Jozsa, HHL |
| ML | Quantum neural networks, kernel methods |
| Simulation | Hamiltonian dynamics, Trotter evolution |
| Finance | Portfolio optimization, Monte Carlo |

## Parallelize

CUDA-Q supports two distinct multi-GPU parallelization strategies. Pick based on
what the user is trying to scale.

| Goal | Strategy | Target option |
|---|---|---|
| Single circuit too large for one GPU | Pool GPU memory | `nvidia --target-option mgpu` |
| Many independent circuits at once | Run circuits in parallel | `nvidia --target-option mqpu` |
| Large Hamiltonian expectation value | Distribute terms across GPUs | `mqpu` plus `execution=cudaq.parallel.thread` |

### Circuit batching with `mqpu`

The `mqpu` option maps one virtual QPU to each GPU. Dispatch circuits
asynchronously with `qpu_id` to all GPUs simultaneously.

```python
import cudaq

cudaq.set_target("nvidia", option="mqpu")
n_qpus = cudaq.get_platform().num_qpus()

futures = [
    cudaq.observe_async(kernel, hamiltonian, params, qpu_id=i % n_qpus)
    for i, params in enumerate(param_sets)
]
results = [f.get().expectation() for f in futures]
```

### Hamiltonian batching

For a single kernel with a large Hamiltonian, add `execution=` to
`cudaq.observe`; no other code change is needed.

```python
# Single node, multiple GPUs
result = cudaq.observe(kernel, hamiltonian, *args,
                       execution=cudaq.parallel.thread)

# Multi-node via MPI
result = cudaq.observe(kernel, hamiltonian, *args,
                       execution=cudaq.parallel.mpi)
```

See the CUDA-Q docs for complete working examples of both patterns.

## Invocation Examples

- `/cudaq-guide`: print the onboarding menu and ask which topic to explore.
- `/cudaq-guide install`: walk through installation, defaulting to Python
  `pip install cudaq`, then validate with the Bell state example.
- `/cudaq-guide test-program`: build and run a Bell state kernel and confirm
  output shows roughly `{ 00:~500 11:~500 }`.
- `/cudaq-guide gpu-sim`: recommend a simulation backend, such as `nvidia` for
  one GPU or `nvidia --target-option mgpu` for circuits larger than one GPU's
  memory.
- `/cudaq-guide qpu`: start the two-step QPU dialogue and read the matching
  hardware doc.
- `/cudaq-guide parallelize`: choose between `mgpu` for pooled memory and
  `mqpu` for many independent circuits.
- `/cudaq-guide author`: route to `references/authoring.md`.

## Platform Troubleshooting

- Import error after `pip install cudaq`: ensure Python 3.10+ and a supported
  OS, Linux or macOS.
- No GPU detected: verify CUDA Toolkit and `nvidia-smi`; fall back to
  `qpp-cpu`.
- Multi-GPU `mgpu` fails: verify MPI availability and CUDA-Q target support.
- QPU submission fails: confirm credentials are set as environment variables
  or through the provider mechanism documented in local CUDA-Q docs.
