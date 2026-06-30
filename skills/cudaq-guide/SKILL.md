---
name: "cudaq-guide"
title: "Cuda Quantum"
description: "CUDA-Q onboarding and authoring guide: installation, test programs, GPU simulation, QPU hardware, quantum applications, and authoring @cudaq.kernel Python code (execution-API selection, kernel-language constraints, silent-failure pitfalls, recurring patterns, resource metrics, validation). For porting Qiskit code to CUDA-Q, route to the qiskit-to-cudaq skill."
version: "1.1.0"
author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
tags: [cuda-quantum, quantum-computing, onboarding, getting-started, authoring, kernels, nvidia]
tools: [Read, Glob, Grep]
license: "Apache-2.0"
compatibility: "Python 3.10+, C++ 20"
metadata:
    author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
    tags:
        - cuda-quantum
        - quantum-computing
        - onboarding
        - getting-started
        - nvidia
    languages:
        - python
        - c++
    domain: "quantum"
---

## CUDA-Q Getting Started Guide

You are a CUDA-Q expert assistant. Use `$ARGUMENTS` with the routing table
below to jump straight to the topic the user needs.

## Purpose

Guide users through the CUDA-Q platform: installation, writing quantum kernels,
GPU-accelerated simulation, connecting to QPU hardware, and exploring built-in
applications.

## Prerequisites

- Python 3.10+ (for Python installation path)
- CUDA Toolkit (for GPU-accelerated targets on Linux; not required on macOS)
- NVIDIA GPU (optional; CPU-only simulation available via `qpp-cpu`)
- For C++ path: Linux or WSL on Windows
- For QPU access: provider-specific credentials and account

## Instructions

- Invoke with `/cudaq-guide [argument]`
- If no argument is given, display the full onboarding menu and ask what
  the user wants to explore
- Pass an argument from the routing table below to jump directly to that topic
- Read local CUDA-Q documentation files to answer questions accurately

## References

| Section | Doc file |
| --- | --- |
| Install | `docs/sphinx/using/install/install.rst`, `docs/sphinx/using/quick_start.rst` |
| Test Program | `docs/sphinx/using/basics/kernel_intro.rst`, `docs/sphinx/using/basics/build_kernel.rst` |
| GPU Simulation | `docs/sphinx/using/backends/sims/svsims.rst`, `docs/sphinx/using/examples/multi_gpu_workflows.rst` |
| QPU | `docs/sphinx/using/backends/hardware.rst`, `docs/sphinx/using/backends/cloud.rst` |
| Applications | `docs/sphinx/using/applications.rst` |
| Parallelize | `docs/sphinx/using/examples/multi_gpu_workflows.rst` |

## Routing by Argument

| Argument | Action |
|---|---|
| `install` | Walk through installation (see Install section) |
| `test-program` | Build and run a Bell state kernel to verify CUDA-Q is working properly |
| `gpu-sim` | Explain GPU-accelerated simulation targets (see GPU Simulation section) |
| `qpu` | Explain how to run on real QPU hardware (see QPU section) |
| `applications` | Showcase what can be built with CUDA-Q (see Applications section) |
| `parallelize` | Show how to run circuits in parallel across multiple QPUs (see Parallelize section) |
| `author` | Author `@cudaq.kernel` Python code: execution API, kernel-language constraints, patterns, validation (see Authoring section) |
| _(none)_ | Print the full menu below and ask what they'd like to explore |

For porting existing Qiskit code to CUDA-Q (gate-translation table, bit-ordering vs Qiskit, framework-decoupling), route to the qiskit-to-cudaq skill instead.

---

## Full Menu (no argument)

Present this when invoked with no argument

```text
CUDA-Q Getting Started

CUDA-Q is NVIDIA's unified quantum-classical programming model for CPUs, GPUs, and QPUs.
Supports Python and C++. Docs https://nvidia.github.io/cuda-quantum/

Choose a topic
  /cudaq-guide install         Install CUDA-Q (Python pip or C++ binary)
  /cudaq-guide test-program    Write and run your quantum kernel
  /cudaq-guide gpu-sim         Accelerate simulation on NVIDIA GPUs
  /cudaq-guide qpu             Connect to real QPU hardware
  /cudaq-guide applications    Explore what you can build
  /cudaq-guide parallelize     Run circuits in parallel across multiple QPUs
  /cudaq-guide author          Author @cudaq.kernel Python code (API, patterns, pitfalls)
```

Porting Qiskit code? Use the qiskit-to-cudaq skill.

---

## Install

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
- Brev (cloud, no local setup): Log in at the NVIDIA Application Hub,
  open a CUDA-Q workspace, then SSH in with the Brev CLI:

  ```bash
  brev open ${WORKSPACE_NAME}
  ```

  CUDA-Q and the CUDA Toolkit are pre-installed.

---

## Test Program

Key concepts to explain

- `@cudaq.kernel` / `__qpu__` marks a quantum kernel - compiled to Quake MLIR
- `cudaq.qvector(N)` allocates N qubits in |0⟩
- `cudaq.sample()` - kernel measures qubits; returns bitstring histogram
  (`SampleResult`)
- `cudaq.run()` - kernel returns a classical value; runs `shots_count` times
  and returns a list of those return values
- `cudaq.observe()` - computes expectation value ⟨H⟩ for a spin operator
- `cudaq.get_state()` - returns the full statevector (simulator only)

Kernel restrictions

- Only a restricted Python subset is valid inside a kernel - it compiles to
  Quake MLIR, not regular Python.
- NumPy and SciPy cannot be used inside a kernel. Use them outside the kernel
  for classical pre/post-processing.
- Kernels can call other kernels; the callee must also be a `@cudaq.kernel`.

For compiler internals (`inspect` module -> `ast_bridge.py` -> Quake MLIR ->
QIR -> JIT), route to `/cudaq-compiler`.

---

## GPU Simulation

To recommend the best simulation backend for the user, consult the full
comparison table at
<https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html>

### Available GPU Targets

| Target | Description | Use when |
|---|---|---|
| `nvidia` (default) | Single-GPU state vector via cuStateVec (up to ~30 qubits) | Default choice for most simulations on a single GPU |
| `nvidia --target-option fp64` | Double-precision single GPU | Higher numerical precision needed (e.g. chemistry, sensitive observables) |
| `nvidia --target-option mgpu` | Multi-GPU, pools memory across GPUs (>30 qubits) | Circuit exceeds single-GPU memory; requires MPI |
| `nvidia --target-option mqpu` | Multi-QPU, one virtual QPU per GPU, parallel execution | Running many independent circuits in parallel (e.g. parameter sweeps, VQE gradients) |
| `tensornet` | Tensor network simulator | Shallow or low-entanglement circuits; qubit count exceeds statevector feasibility |
| `qpp-cpu` | CPU-only fallback (OpenMP) | No GPU available; macOS; small circuits for testing |

---

## QPU

When the user invokes this section, do not dump all providers at once.
Instead, follow this two-step dialogue:

Step 1 - ask which technology they want

```text
Which QPU technology are you targeting?
  1. Ion trap       (IonQ, Quantinuum)
  2. Superconducting (IQM, OQC, Anyon, TII, QCI)
  3. Neutral atom   (QuEra, Infleqtion, Pasqal)
  4. Cloud / multi-platform (AWS Braket, Scaleway)
```

Step 2 - once they pick a technology, ask which provider, then read the
corresponding doc file and walk the user through it step by step.

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

After walking through the provider steps, always close with

- Test locally first with `emulate=True` before submitting to real hardware.
- Use `cudaq.sample_async()` / `cudaq.observe_async()` for non-blocking submission.
- Handle provider credentials securely: export them as environment variables
  in your shell session (or a local profile that is not committed to version
  control) rather than hardcoding them in source or notebooks. Never paste
  tokens into shared files, logs, or commits, and prefer a secrets manager
  where one is available.

---

## Applications

CUDA-Q ships with ready-to-run application notebooks

| Category | Examples |
|---|---|
| Optimization | QAOA, ADAPT-QAOA, MaxCut |
| Chemistry | VQE, UCCSD, ADAPT-VQE |
| Error Correction | Surface codes, QEC memory |
| Algorithms | Grover's, Shor's, QFT, Deutsch-Jozsa, HHL |
| ML | Quantum neural networks, kernel methods |
| Simulation | Hamiltonian dynamics, Trotter evolution |
| Finance | Portfolio optimization, Monte Carlo |

---

## Parallelize

CUDA-Q supports two distinct multi-GPU parallelization strategies - pick based
on what you are trying to scale.

| Goal | Strategy | Target option |
|---|---|---|
| Single circuit too large for one GPU | Pool GPU memory | `nvidia --target-option mgpu` |
| Many independent circuits at once | Run circuits in parallel | `nvidia --target-option mqpu` |
| Large Hamiltonian expectation value | Distribute terms across GPUs | `mqpu` + `execution=cudaq.parallel.thread` |

### Circuit batching with mqpu (`sample_async` / `observe_async`)

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
`cudaq.observe` — no other code change needed.

```python
# Single node, multiple GPUs
result = cudaq.observe(kernel, hamiltonian, *args,
                       execution=cudaq.parallel.thread)

# Multi-node via MPI
result = cudaq.observe(kernel, hamiltonian, *args,
                       execution=cudaq.parallel.mpi)
```

See the docs above for complete working examples of both patterns.

---

## Authoring

Reference for authoring CUDA-Q Python code with `@cudaq.kernel`. Targets the
0.14.x Python API; concepts apply to adjacent versions but specific API
behaviors should be re-verified against the version actually installed
(`cudaq.__version__`).

For porting Qiskit code to CUDA-Q — the gate-translation table, bit-ordering
conventions vs Qiskit, floating-point precision matching, the
framework-decoupling patterns (§4.1, §4.2, §4.3), and the port-validation gate
— use the qiskit-to-cudaq skill, which builds on this section.

### Defaults and disciplines

These shape every decision in the rest of the skill.

1. Decorator mode (`@cudaq.kernel`) is the default. Do not use `cudaq.make_kernel()` builder mode without explicit user permission.
Builder mode is legacy in 0.14 with uneven feature support: no `mz(list)`, no qview slicing, no `x.ctrl`, no `cudaq.adjoint` on builder kernels, `apply_call` on subkernels has bugs, and `QuakeValue.__getitem__` rejects numpy ints. Count-key ordering also differs between modes. If a case looks like it needs builder mode, describe the advantage and trade-off and wait for confirmation.

2. Don't introduce restrictions unless the user directs.
Specific failure modes to avoid:

- Hardcoded register sizes (`cudaq.qvector(10)` literal) when `cudaq.qvector(num_qubits)` with an int parameter works fine. Register-dimension ints are lowered as runtime values; they don't bake into the compilation.
- Fixed-arity gate dispatchers (`if nc == 0 / 1 / 2`) that silently drop higher-arity multi-controlled gates. Use the runtime-arity list-comprehension pattern in §3.1 instead.
- `NotImplementedError` for flags that turn out to be no-ops in CUDA-Q (e.g., a `parameterized=True` flag — a `@cudaq.kernel` is already parameterized over its runtime float arguments, no symbolic-vs-bound distinction). Accept the flag and proceed.
- Per-shape kernel factories (`exec` + `linecache.cache`) used when a simpler int-parameter kernel works. Use the factory only when a fixed-length return-list literal is genuinely tied to a parameter (§3.6).

If a cap is genuinely required by a CUDA-Q-language constraint, raise `NotImplementedError` with a clear message describing what's needed to lift it. Never silently drop a gate or mis-compute.

1. Validate at the smallest valid configuration first.
Run at 2-4 qubits and compare raw count keys (not just aggregate fidelity) against a reference before scaling. Bit-ordering and IQFT-direction bugs produce ~50% fidelities that look plausible at a glance.

### 1. Execution API selection

Pick the API before writing the kernel — it determines the kernel's return-type signature and what conventions apply to its output.

| Need | API | Kernel signature | Notes |
|---|---|---|---|
| Aggregate counts only | `cudaq.sample(K, *args, shots_count=N)` | ends with `mz(qubits)`, no return type | Default. Fast. |
| Per-shot bits captured into Python (mid-circuit measurement, post-selection, feed-forward) | `cudaq.run(K, *args, shots_count=N)` | `-> List[bool]` (or `-> int` for encoded bits) | ~1000× slower than `sample` at 100k shots. Use only when per-shot data is genuinely required. |
| Sequential measurements of the same qubit reused across iterations | `cudaq.sample(K, *args, shots_count=N, explicit_measurements=True)` | per-shot result is a measurement-order bitstring | Bit order is measurement order, not allocation order. Reverse or remap if the target API expects classical-register display order. |
| Statevector amplitudes (exact distributions, fidelity baselines, expectation values) | `cudaq.get_state(K, *args)` | no `mz` | Index `state[i]` such that `format(i, f"0{N}b")` matches what `cudaq.sample` returns from `mz(qubits)` on the same kernel. |
| Expectation of a Pauli sum | `cudaq.observe(K, hamiltonian, *args)` | no `mz` | Specialized; not covered here. |

Rule of thumb: when the kernel doesn't need per-shot data captured into Python, end with `mz(qubits)` and omit the return type so it routes through `cudaq.sample`. The performance gap with `cudaq.run` is orders of magnitude.

`cudaq.SampleResult` (returned by `cudaq.sample`) is not a full Python dict — it supports `.items()` and indexed key access but does NOT reliably support `.keys()`, `.values()`, `.get()`, `__contains__`, or `len()` against arbitrary keys. Convert if downstream code uses dict APIs: `counts = {k: v for k, v in result.items()}`.

### 2. CUDA-Q kernel language (0.14)

#### Supported inside `@cudaq.kernel`

- Qubit allocation — `cudaq.qubit()`, `cudaq.qvector(n)`, parameter type `cudaq.qview`.
- Qview slicing `qubits[a:b]` with runtime `a, b`.
- For-loops with `range(start, stop, step)` for any sign of step.
- `if`/`elif`/`else` with runtime conditions.
- Sub-kernel calls via direct invocation: `my_subkernel(args)`.
- `cudaq.control(K, ctrl_or_list, *args)` to propagate control through every gate in K.
- `cudaq.adjoint(K, *args)` at top level (caveats below).
- Multi-control gate forms: `x.ctrl(c, t)`, `x.ctrl([c1, c2], t)`, `x.ctrl(c1, c2, t)` (variadic). Same for `y.ctrl`, `z.ctrl`, `h.ctrl`, `rx.ctrl`, `ry.ctrl`, `rz.ctrl`, `r1.ctrl`, `swap.ctrl`.
- `mz(qubit)` returning a Python-typed `bool` — usable in `if` conditions and `return` lists.
- `mz(qview)` (no return capture) for sampling via `cudaq.sample`.
- `reset(qubit)`.
- Float arithmetic with mixed int/float promotion (e.g., `2.0 * math.pi * (2 ** q)`).
- Typed parameters: `int`, `float`, `bool`, `cudaq.qubit`, `cudaq.qview`, `List[int]`, `List[float]`, `List[bool]`.
- List comprehensions over runtime-bound lengths — `[qubits[idx[k]] for k in range(nc)]` produces a fully-typed homogeneous list at kernel-decoration time. Supported even though empty-list literal and list-multiplication are not. Enables runtime-arity multi-control gate dispatch (§3.1).

#### Not supported inside `@cudaq.kernel` (CompilerError)

- Explicit casts `int(x)`, `float(x)`, `bool(x)` — use implicit promotion (multiply by a float literal: `pow_q = 2 ** q` stays int, `1.0 * pow_q` promotes).
- `reversed(range(n))` — use `range(n-1, -1, -1)`.
- Empty list literal `[]`.
- List multiplication `[False] * n` — use a literal list of the right length.
- List `.append()` mutation inside the kernel body.
- Annotated assignment without initializer (`x: int = 0`).
- `List[List[...]]` parameter or return type — flatten to parallel `List[int]` arrays.
- f-strings.
- Indexing with `np.int64` — cast to Python `int()` before passing as an index or before packing into a kernel-parameter `List[int]`.
- `np.random.*` and other runtime-dynamic library calls — generate randomness in the Python wrapper and pass values into the kernel as arguments.

#### Silent-failure and confusing-error pitfalls

`cudaq.control(K, ...)` does not compose with `cudaq.adjoint(...)` inside K.
If K's body contains `cudaq.adjoint(M, ...)`, sample-time fails with `RuntimeError: Could not successfully apply argument synth.` Fix: hand-roll M's inverse — write a separate `M_inv` kernel that walks M's gates in reverse with negated rotation angles. X/Y/Z/H/CX/CSWAP are self-inverse (angle stays); RY/RX/RZ/R1/RZZ negate angle. Then call `M_inv(...)` directly inside K. Affects QPE/QAE-style algorithms with controlled-A^†.

`cudaq.adjoint(K, *args)` fails on kernels with qview-slice access or deeply-nested loops.
Errors include `operand #1 does not dominate this use` or `'quake.extract_ref' op invalid constant index value`. Workaround: manually unroll the inverse inline (write a separate explicit-inverse kernel as above).

`cudaq.run` is much slower than `cudaq.sample`.
At 100k shots on a 10-qubit kernel, `cudaq.run` returning `List[bool]` took ~100s vs `cudaq.sample`'s ~0.1s — roughly 1000× regression. Only declare a return type when you genuinely need per-shot data.

MLIR JIT loop-unrolling failures on parameter-bounded loops.
If a `for i in range(N)` loop bound `N` is a runtime parameter, and `i` is used to index a `qvector` of fixed allocation size, and N could _theoretically_ exceed the allocation (even if classical control flow guarantees it doesn't), the JIT pass may fail with `'quake.extract_ref' op invalid constant index value`. Fix: clamp explicitly inside the kernel:

```python
safe_n = bound
if safe_n > num_qubits:
    safe_n = num_qubits
for i in range(safe_n):
    h(qubits[i])
```

Numpy ints rejected as kernel-side indices.
`qubits[np.int64(0)]` raises `RuntimeError: invalid idx passed to QuakeValue`. Cast to Python `int()` first. When packing a kernel-parameter `List[int]` from numpy/scipy output (which commonly returns `np.int64` for index-typed values), cast every element: `[int(x) for x in arr]`.

Decorator-mode `mz(qview)` count keys are positional; builder-mode count keys are by qubit allocation index.
In `@cudaq.kernel`, `mz(qview)` puts `qview[0]` leftmost in the count key — the order qubits appear in the qview. In `cudaq.make_kernel()` builder mode, the count key orders bits by smallest qubit allocation index regardless of `mz` call order. Don't mix modes within a project.

`reset(q)` on the noiseless statevector simulator is deterministic projection to |0⟩, not stochastic collapse.
Behaviorally fine for most algorithms but worth knowing if your code depends on stochastic collapse + post-selection semantics. The qubit's |1⟩-component is dropped; renormalize amplitudes as needed.

Subkernels containing `cudaq.adjoint(...)` cannot be referenced as targets of `cudaq.control(...)`. (Restated: same as the first item above; the most common manifestation is in QPE/QAE main kernels.)

### 3. Recurring patterns

> The framework-decoupling patterns §4.1 (recursive constructor -> emitter), §4.2 (pure-Python helper extraction + `sys.meta_path` verification), and §4.3 (gate-recorder class) are porting-specific and live in the qiskit-to-cudaq skill.

#### 3.1 Parallel-array gate encoding (runtime-arity multi-control)

When a kernel needs to apply a sequence of multi-controlled gates whose number of controls, control indices, control states, target, and angle all depend on runtime data (e.g., a bisection-tree state preparation, a polynomial function evaluation, a sparse-Hamiltonian oracle), encode the gate sequence as parallel arrays the kernel walks gate-by-gate.

Use the flat-with-offsets representation. Per gate `i`, the controls live in a contiguous slice of two flat arrays:

```python
# Per-gate arrays:
#   kinds[i]:     int code (0=h, 1=x, 2=ry, ...)
#   targets[i]:   qubit index in the local register
#   thetas[i]:    angle for ry; 0.0 otherwise
# Flat per-gate control data — gate i's controls are at
# ctrl_indices[ctrl_offsets[i]:ctrl_offsets[i+1]] (parallel: ctrl_states):
#   ctrl_indices: List[int]   (qubit indices)
#   ctrl_states:  List[int]   (1=normal, 0=X-wrap for ctrl_state='0')
#   ctrl_offsets: List[int]   (length = num_gates + 1; cumulative)

@cudaq.kernel
def apply_A(state: cudaq.qview,
            kinds: List[int], targets: List[int], thetas: List[float],
            ctrl_indices: List[int], ctrl_states: List[int],
            ctrl_offsets: List[int]):
    L = len(kinds)
    for i in range(L):
        kind = kinds[i]
        target = state[targets[i]]
        theta = thetas[i]
        c_start = ctrl_offsets[i]
        c_end = ctrl_offsets[i + 1]
        nc = c_end - c_start

        # X-wraps for ctrl_state=0 controls
        for k in range(c_start, c_end):
            if ctrl_states[k] == 0:
                x(state[ctrl_indices[k]])

        if nc == 0:
            if kind == 0: h(target)
            elif kind == 1: x(target)
            elif kind == 2: ry(theta, target)
        else:
            # Runtime-arity control list via list comprehension. Works for any nc.
            ctrls = [state[ctrl_indices[c_start + j]] for j in range(nc)]
            if kind == 1: x.ctrl(ctrls, target)
            elif kind == 2: ry.ctrl(theta, ctrls, target)

        # X-unwraps
        for k in range(c_start, c_end):
            if ctrl_states[k] == 0:
                x(state[ctrl_indices[k]])
```

Key construct: `[state[ctrl_indices[c_start + j]] for j in range(nc)]` — a list comprehension over a runtime-bound length, indexing into `cudaq.qview`, producing a list of qubit handles. `x.ctrl(ctrls, target)` and `ry.ctrl(theta, ctrls, target)` accept this any-arity list directly. No per-arity dispatch branches; no hardcoded max-arity cap.

Antipattern to avoid: fixed-column parallel arrays `c1[i], s1[i], c2[i], s2[i]` with hardcoded `if nc == 0 / 1 / 2` branches. That form caps the supported control count at the number of branches and silently drops anything higher. The flat-with-offsets form has no cap.

X-wrap + gate + X-unwrap is safe under outer `cudaq.control`. The wrap and unwrap pick up the same outer control and self-cancel on the control=0 branch, leaving the inner gate properly controlled.

Per-iteration concatenation with offsets: when an outer loop applies a _different_ sub-circuit per iteration, build all per-iteration gate sequences at Python time and concatenate into one flat record array, recording per-iteration offsets. The kernel walks `for k in range(N_iter): for i in range(offsets[k], offsets[k+1]): apply_op(i)`. Same `ctrl_offsets`-style indexing trick at the outer level.

Hand-roll the inverse `apply_A_inv` if the kernel will be wrapped in `cudaq.control` (see §3.7 for the trap and §2's adjoint-composition pitfall).

#### 3.2 Contiguous-swap trick for sparse controls

CUDA-Q's `.ctrl()` modifiers accept either a `qview` slice or a list of qubit handles. When the target controls are non-contiguous and the kernel needs them as a contiguous block (e.g., a `qview` slice for an optimized multi-control intrinsic), precompute swap pairs in Python and swap the needed qubits into a contiguous prefix `q[0:nc]`:

```python
# Python-side: compute swaps to gather op_controls into positions 0..nc-1
def _gather_swaps(op_controls, num_state_qubits):
    logical_at_pos = list(range(num_state_qubits))
    pos_of_logical = list(range(num_state_qubits))
    swaps = []
    for target_pos, control in enumerate(op_controls):
        source_pos = pos_of_logical[control]
        if source_pos == target_pos:
            continue
        swaps.append((target_pos, source_pos))
        a, b = logical_at_pos[target_pos], logical_at_pos[source_pos]
        logical_at_pos[target_pos], logical_at_pos[source_pos] = b, a
        pos_of_logical[a] = source_pos
        pos_of_logical[b] = target_pos
    return swaps

# Kernel-side: apply swaps, run gate on contiguous prefix, undo swaps
@cudaq.kernel
def apply_op(state: cudaq.qview, swap_a: List[int], swap_b: List[int],
             num_swaps: int, num_ctrls: int, theta: float):
    for i in range(num_swaps):
        swap(state[swap_a[i]], state[swap_b[i]])
    ry.ctrl(theta, state[0:num_ctrls], state[num_ctrls])
    for i_rev in range(num_swaps):
        i = num_swaps - 1 - i_rev
        swap(state[swap_a[i]], state[swap_b[i]])
```

The flat-list parallel-array form (§3.1) avoids this entirely by accepting any-arity, non-contiguous control sets via list comprehension. Prefer §3.1 unless a specific intrinsic forces contiguous controls.

#### 3.3 Statevector-based expected distribution via `cudaq.get_state`

When the reference uses an exact-statevector backend to compute an expected count distribution:

```python
@cudaq.kernel
def my_circuit_unmeasured(args...):
    # ...gates that prepare the state, NO mz...

state = cudaq.get_state(my_circuit_unmeasured, *args)
counts = {}
for i in range(2 ** N):
    amp = complex(state[i])
    p = amp.real**2 + amp.imag**2
    if p > 1e-12:
        counts[format(i, f"0{N}b")] = p   # raw probability
counts = {k: round(v * num_shots) for k, v in counts.items()}  # scale at lookup
```

Pair with a separate `my_circuit_measured` (same gate body + trailing `mz(qubits)`) for sampling. Both come from cudaq, so the bitstring keys line up automatically — no remapping needed.

#### 3.4 Variational loop (scipy.optimize)

```python
def expectation(thetas):
    counts = cudaq.sample(my_kernel, *args, thetas, shots_count=num_shots)
    return objective_from_counts(counts)

res = scipy.optimize.minimize(expectation, init_thetas, method='COBYLA',
                              options={'maxiter': max_iter})
```

`cudaq.sample` is synchronous — it returns counts directly, so variational loops don't need a result-stashing closure or wait-for-completion call.

Late-binding gotcha: if `expectation` closes over a loop variable (e.g., the current restart index), capture by default argument — `def expectation(thetas, _idx=i)` — to bind the value at definition time.

`parameterized=True` flags are typically no-ops in CUDA-Q. Some frameworks distinguish a "parameterized" mode that builds the circuit once with symbolic parameter placeholders and rebinds per iteration to avoid repeated transpilation. CUDA-Q has no symbolic-vs-bound distinction — `@cudaq.kernel` is already parameterized over its runtime float arguments. Accept the flag, treat it as a no-op, thread it through the run loop unchanged.

#### 3.5 Mid-circuit measurement + post-selection

```python
@cudaq.kernel
def my_kernel(args...) -> List[bool]:
    # ...gates that prepare the ancilla...
    a = mz(ancilla)        # mid-circuit, captured into a Python-typed bool
    reset(ancilla)
    # ...more gates that reuse the reset ancilla...
    return [a, mz(qa[N-1]), mz(qa[N-2]), ..., mz(qa[0])]

shots = cudaq.run(my_kernel, *args, shots_count=N)
post_selected_counts = {}
for shot in shots:
    if not shot[0]:           # post-select on ancilla = |1⟩
        continue
    key = ''.join('1' if b else '0' for b in shot[1:])
    post_selected_counts[key] = post_selected_counts.get(key, 0) + 1
```

The encoded-integer return-type alternative — kernel returns `int` instead of `List[bool]` — works when accumulating per-shot bits with bitwise OR:

```python
@cudaq.kernel
def my_kernel(args...) -> int:
    # ...
    res = 0
    if mz(ancilla):
        res = res + (1 << N)
    for q in range(N):
        if mz(qa[q]):
            res = res + (1 << q)
    return res

shots = cudaq.run(my_kernel, *args, shots_count=N)
for v in shots:
    bits = format(v, f"0{N+1}b")  # explicit width
    ...
```

The `int` return form is cleaner for fixed-width feed-forward circuits (iterative QPE) where the kernel reads its own past measurements. Inside the kernel: `if (res & (1 << j)) != 0: r1(angle, q)` enables classical-conditional gate application.

Reference: `docs/sphinx/examples/python/measuring_kernels.py` and `sample_to_run_migration.py` in the CUDA-Q 0.14 source tree.

#### 3.6 Variable-shape return lists

`@cudaq.kernel` requires the return-list literal to have a fixed length determined at decoration time. For algorithms whose return-bit count varies per configuration, there are two options.

Option A — Switch to `mz(qview)` + `cudaq.sample` (preferred when possible). If you don't need per-shot bits captured into Python (no mid-circuit measurement with conditional logic on past `mz` results), drop the return type entirely and end the kernel with `mz(qview)` over the qubits you want in the count key. No fixed-length constraint. Routes through `cudaq.sample` which is ~1000× faster than `cudaq.run` at high shots. Bit ordering in the count key follows qubit allocation order — arrange your qubit allocations accordingly.

Option B — Generate one kernel per shape via `exec` + `linecache.cache`. When the kernel genuinely needs `cudaq.run` + `List[bool]` return (e.g., mid-circuit ancilla measurement that's then reset and reused, with per-shot ancilla bit needed for post-selection), the literal length is genuinely tied to a parameter. Generate one kernel per shape on demand:

```python
import linecache

_KERNEL_CACHE = {}

def _ensure_kernel(N):
    if N in _KERNEL_CACHE:
        return _KERNEL_CACHE[N]
    return_list = ", ".join([f"mz(q[{i}])" for i in range(N - 1, -1, -1)])
    src = f"""
@cudaq.kernel
def my_kernel_N{N}(...) -> List[bool]:
    q = cudaq.qvector({N})
    # ...gate body with N spelled out as a literal...
    return [{return_list}]
"""
    fname = f"<dynamic:my_kernel_N{N}>"
    linecache.cache[fname] = (
        len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    namespace = {"cudaq": cudaq, "List": List,
                 # inject any sub-kernels the dynamic kernel calls
                 }
    exec(code, namespace)
    K = namespace[f"my_kernel_N{N}"]
    _KERNEL_CACHE[N] = K
    return K
```

`@cudaq.kernel` uses Python's `inspect.getsource` for decorator-time AST analysis, which normally fails for `exec`-produced code. Registering the source in `linecache.cache` under a synthetic filename makes `inspect.getsource` find it. Sub-kernels the dynamic kernel calls must be in the namespace passed to `exec`.

Decision rule: prefer Option A whenever the algorithm doesn't need per-shot mid-circuit data. The factory pattern is ~30 lines of boilerplate per benchmark and adds an `exec` indirection — only worth it when `cudaq.run` is genuinely required.

#### 3.7 Hand-rolled inverse (avoiding the control-of-adjoint trap)

To work around the `cudaq.control(K)` + `cudaq.adjoint(M)` composition failure:

```python
@cudaq.kernel
def A_op(state: cudaq.qview, S: int, theta: float):
    ry(theta, state[0])
    for i in range(S):
        x.ctrl(state[0], state[i + 1])

@cudaq.kernel
def A_op_inv(state: cudaq.qview, S: int, theta: float):
    # Walk A_op's gates in reverse; CXs are self-inverse, RY angle negates.
    for i in range(S):
        x.ctrl(state[0], state[i + 1])
    ry(-theta, state[0])

@cudaq.kernel
def Q_op(state: cudaq.qview, S: int, theta: float):
    # ... -S_chi ...
    A_op_inv(state, S, theta)   # NOT cudaq.adjoint(A_op, ...)
    # ... S_0, A ...

@cudaq.kernel
def main(...):
    # ...
    cudaq.control(Q_op, count_ctrl, state, S, theta)   # works
```

Inverse-emission rules:

- Walk gates in reverse order.
- Self-inverse gates (X, Y, Z, H, CX, CY, CZ, SWAP, CSWAP): angle and operands unchanged.
- Parameterized rotations (RX, RY, RZ, R1, RZZ, RXX, RYY, controlled variants): negate the angle.
- S -> S†, T -> T†.

### 4. Resource metrics

`cudaq.estimate_resources(K, *args)` returns a resources object with `count()` (total gate count) and `count_controls(gate_name, num_controls)` (number of `gate_name` invocations with exactly that many controls). Computing "total controls" for an algorithm requires iterating over arities:

```python
resources = cudaq.estimate_resources(K, *args)
total_gates = resources.count()
controlled_gates = ['x', 'y', 'z', 'r1', 'rx', 'ry', 'rz']
two_q_weighted = 0
for gate in controlled_gates:
    for arity in range(1, num_qubits):
        two_q_weighted += arity * resources.count_controls(gate, arity)
```

This sums `arity × count` so an N-controlled X contributes N to the running total — useful when comparing against a backend's per-control-wire metric.

Interpretation caveats:

- A multi-controlled gate may be counted as one operation (the high-level construct) or as many effective 2-qubit controls (after decomposition) — different tools choose differently. Define your metric before comparing across frameworks.
- The framework's resource counter may have IR-shape blind spots for specific gate forms. Verify with a small known example before trusting a count of zero.
- `CUDAQ_TIMING_TAGS=5` enables backend-level instrumentation (e.g., cuStateVec emits `Gate Count` / `Control Count` per simulator-shutdown event). These totals are aggregate across all kernel executions in a process (sums over circuits and shots) — divide by `num_circuits` (and, for `cudaq.run`-path kernels, by `num_shots`) to recover per-circuit numbers.
- Native timing/control totals from `CUDAQ_TIMING_TAGS` may be silently absent for some kernel shapes (very small circuits, kernels with certain mid-circuit-measurement patterns). Treat their absence as an instrumentation quirk, not as zero work performed.

Fidelity / output correctness is the primary validation target; resource counts are secondary diagnostics that need stated definitions.

### 5. Workflow and debugging

#### General workflow

1. Read the source first. Identify registers, classical bits, qubit order, gate sequence, inversion/control structure, and expected count-key format. Don't start translating before you know what you're translating.
2. Pick the execution API before writing the kernel (§1). The choice determines the kernel's return type and bit-ordering conventions.
3. Move complex Python out of the kernel. Precompute angles, operation records, control lists, swap schedules, and parameter bindings in normal Python, then pass simple lists/scalars in as kernel arguments.
4. Probe before relying on a kernel-language feature. Write a 10-line test in `/tmp/probe.py` and run it. CUDA-Q kernels need source-code introspection, so `python3 -c "..."` and stdin scripts fail at decoration time — use a written file. The compiler errors are far more informative than the docs.
5. Run at smallest valid size first. Compare raw count keys (not just aggregate fidelity) against the reference implementation. Scale up only after small cases match.
6. When a CUDA-Q lowering error depends on input size or value, split the path into a dedicated kernel. A `@cudaq.kernel` containing unrelated runtime-dispatch branches can lower differently than a method-specific kernel.

#### Debugging checklist

- Counts correct but keys reversed: fix at the port boundary (qubit-allocation convention or Python-side bitstring formatting), not by changing the algorithm.
- Fidelity around 0.4-0.6 with mass on bit-shifted neighbors: suspect (a) IQFT iteration direction, (b) qubit-reversal mismatch between QPE control assignments and the IQFT input register, (c) wrong controlled-gate flavor (e.g., `rz.ctrl` vs `r1.ctrl` inconsistency).
- Fidelity exactly 0.0: most often endianness. Verify with a 2-3 qubit deterministic probe (e.g., all marked states or all input integers).
- Only some widths or input values fail: inspect integer arithmetic inside the kernel; try precomputing the offending values in Python and passing them in.
- Sample-time `RuntimeError: Could not successfully apply argument synth`: a `cudaq.control` is wrapping a kernel that contains `cudaq.adjoint`. Hand-roll the inverse (§3.7).
- JIT error `'quake.extract_ref' op invalid constant index value`: a loop bound parameter could theoretically exceed an allocated qvector size. Clamp explicitly inside the kernel (§2's pitfall list).
- AttributeError on `result.keys()`: convert `SampleResult` to a dict: `counts = {k: v for k, v in result.items()}`.
- Mid-circuit measurement output looks wrong: compare `sample(..., explicit_measurements=True)` against a `cudaq.run` kernel that returns encoded per-shot values; pick whichever matches the expected semantics.
- Sparse-control dispatch starts to need dynamic list construction: precompute a swap-gather schedule (§3.2) rather than reaching for builder mode.

### 6. Validating a new algorithm

For new algorithms (no source-framework reference):

1. Pick a small case with a known analytic answer (e.g., Bernstein-Vazirani on a 3-bit secret).
2. Verify the output bitstring matches the analytic expectation at 100% probability (no noise model).
3. Scale up. Use `cudaq.get_state` to check intermediate distributions when feasible.

When task accuracy depends on a specific CUDA-Q version, check `cudaq.__version__` and the official CUDA-Q documentation for that version before relying on remembered API behavior. (Porting from a source framework uses a stricter validation gate — see the qiskit-to-cudaq skill.)

### 7. References

1. CUDA-Q 0.14 documentation: <https://nvidia.github.io/cuda-quantum/0.14.0/>
2. CUDA-Q examples (canonical references for `cudaq.run` + `List[bool]` + mid-circuit measurement): `docs/sphinx/examples/python/measuring_kernels.py` and `sample_to_run_migration.py` in the CUDA-Q 0.14 source tree. <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.14.0/docs/sphinx/examples/python>
3. CUDA-Q Academic: <https://github.com/NVIDIA/cuda-q-academic> — worked QPE / VQE / QAOA in `@cudaq.kernel` form.
4. 0.14 source-tree tests: `python/tests/kernel/test_kernel_features.py` — confirms specific kernel-language constructs (mid-circuit `mz`, conditionals, returns).

---

## Examples

- `/cudaq-guide` — print the onboarding menu and ask the user which topic to
  explore.
- `/cudaq-guide install` — walk through installation, defaulting to the Python
  `pip install cudaq` path, then validate with the Bell state example.
- `/cudaq-guide test-program` — build and run a Bell state kernel and confirm
  the output shows roughly `{ 00:~500 11:~500 }`.
- `/cudaq-guide gpu-sim` — recommend a simulation backend (for example
  `nvidia` for a single GPU, or `nvidia --target-option mgpu` for circuits
  larger than one GPU's memory).
- `/cudaq-guide qpu` — start the two-step QPU dialogue (technology, then
  provider) and read the matching hardware doc.
- `/cudaq-guide parallelize` — choose between `mgpu` (pool memory for one large
  circuit) and `mqpu` (run many circuits in parallel).
- `/cudaq-guide author` — author a `@cudaq.kernel`: pick the execution API
  (`sample`/`run`/`get_state`/`observe`), respect the kernel-language subset,
  apply the recurring patterns, and validate at the smallest configuration
  first. For porting Qiskit code, use the `qiskit-to-cudaq` skill.

---

## Limitations

- GPU simulation requires Linux (x86_64 or ARM64); macOS is CPU-only
- Multi-GPU `mgpu` target requires MPI
- Kernel code must use a restricted Python subset; NumPy/SciPy are not
  allowed inside kernels
- QPU access requires provider-specific credentials and accounts

## Troubleshooting

- Import error after `pip install cudaq`: Ensure Python 3.10+ and a
  supported OS (Linux or macOS)
- No GPU detected: Verify CUDA Toolkit is installed and `nvidia-smi`
  shows your GPU; fall back to `qpp-cpu`
- Kernel compile error: Check that only supported Python constructs are
  used inside `@cudaq.kernel`
- QPU submission fails: Confirm credentials are set as environment
  variables per the provider docs
