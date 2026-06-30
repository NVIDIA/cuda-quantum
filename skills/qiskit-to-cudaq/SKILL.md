---
name: "qiskit-to-cudaq"
title: "Qiskit to CUDA-Q"
description: "Port Qiskit Python code to CUDA-Q (0.14.x, @cudaq.kernel decorator mode). Covers the full Qiskit->CUDA-Q gate-translation table; bit-ordering / count-key conventions vs Qiskit; floating-point precision defaults (CUDA-Q fp32 vs Qiskit fp64) and how to match them for cross-framework comparison; porting disciplines (preserve the source algorithm; no source-framework runtime dependency); framework-decoupling patterns (recursive constructor -> pure-Python emitter; sys.meta_path verification; the gate-recorder class for deeply nested .inverse()/.control(k) constructors); and the port-validation gate (compare raw count keys, not just fidelity). Builds on the cudaq-guide skill's Authoring section for execution-API selection, kernel-language constraints, and shared kernel patterns. Use when porting Qiskit (or another framework) code to CUDA-Q."
version: "1.0.0"
author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
tags: [cuda-quantum, quantum-computing, qiskit, porting, migration, kernels, nvidia]
tools: [Read, Glob, Grep]
license: "Apache-2.0"
compatibility: "Python 3.10+"
metadata:
    author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
    tags:
        - cuda-quantum
        - quantum-computing
        - qiskit
        - porting
        - migration
        - nvidia
    languages:
        - python
    domain: "quantum"
---

# Qiskit -> CUDA-Q

Reference for porting Qiskit Python code to CUDA-Q. Targets the 0.14.x Python
API and decorator mode (`@cudaq.kernel`); re-verify version-specific behavior
against the installed version (`cudaq.__version__`).

This skill is porting-specific. The foundations it relies on — execution-API
selection, the kernel-language subset, silent-failure pitfalls, the runtime-
arity gate-encoding (§3.1) / contiguous-swap (§3.2) / statevector-distribution
(§3.3) / variational-loop (§3.4) / mid-circuit-measurement (§3.5) / variable-
shape-return (§3.6) / hand-rolled-inverse (§3.7) patterns, resource metrics, and
the general debugging workflow — live in the cudaq-guide skill's
Authoring section (`/cudaq-guide author`). Read that first; this skill adds
the Qiskit-facing pieces (§1, §2, §3, §4.1, §4.2, §4.3) on top.

---

## Porting disciplines

These extend the authoring defaults in `cudaq-guide`.

Preserve the source algorithm when porting.
Do not change the high-level quantum algorithm (e.g., remove mid-circuit measurements, substitute a coherent unroll for an iterative QPE, swap an oracle decomposition) without explicit user permission. If a workaround is needed, find one that preserves the original quantum logic.

Don't introduce restrictions vs the source reference unless the user directs.
Specific failure modes to avoid:

- Hardcoded register sizes (`cudaq.qvector(10)` literal) when `cudaq.qvector(num_qubits)` with an int parameter works fine. Register-dimension ints are lowered as runtime values; they don't bake into the compilation.
- Fixed-arity gate dispatchers (`if nc == 0 / 1 / 2`) that silently drop higher-arity multi-controlled gates the source emits. Use the runtime-arity list-comprehension pattern in §3.1 (cudaq-guide) instead.
- `NotImplementedError` for source-framework flags that turn out to be no-ops in CUDA-Q (e.g., a `parameterized=True` flag — a `@cudaq.kernel` is already parameterized over its runtime float arguments, no symbolic-vs-bound distinction). Accept the flag and proceed.
- Per-shape kernel factories (`exec` + `linecache.cache`) used when a simpler int-parameter kernel works. Use the factory only when a fixed-length return-list literal is genuinely tied to a parameter (§3.6, cudaq-guide).

If a cap is genuinely required by a CUDA-Q-language constraint, raise `NotImplementedError` with a clear message describing what's needed to lift it. Never silently drop a gate or mis-compute.

A CUDA-Q port should not depend on the source framework at runtime.
The resulting cudaq module must run with the source framework uninstalled.

- Don't `import qiskit` (or any source-framework package) from a cudaq kernel module.
- Don't `importlib`-load a source-framework module to reach helpers that happen to be pure-Python.
- Don't build a `QuantumCircuit` at port time and walk `qc.data` to extract a gate sequence — translate the construction directly into Python (see §4.1 and §4.2).
- Extract pure-Python helpers (analyzers, post-selection, problem generators) into a new sibling module that imports only `numpy` and stdlib.
- Verify with the `sys.meta_path` blocker recipe in §4.2.

Validate at the smallest valid configuration first.
Run your port at 2-4 qubits and compare raw count keys (not just aggregate fidelity) against the source implementation before scaling. Bit-ordering and IQFT-direction bugs produce ~50% fidelities that look plausible at a glance.

---

## 1. Qiskit -> CUDA-Q gate translation

| Qiskit | CUDA-Q (decorator mode) | Notes |
|---|---|---|
| `qc.h(q)`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg` | `h(q)`, `x(q)`, `y(q)`, `z(q)`, `s(q)`, `t(q)`, `s.adj(q)`, `t.adj(q)` | direct |
| `qc.rx/ry/rz(θ, q)` | `rx(θ, q)`, `ry(θ, q)`, `rz(θ, q)` | direct |
| `qc.p(θ, q)` (phase) | `r1(θ, q)` | exact: both `diag(1, e^iθ)` |
| `qc.cx(c, t)` | `cx(c, t)` or `x.ctrl(c, t)` | direct |
| `qc.cy/cz(c, t)` | `y.ctrl(c, t)`, `z.ctrl(c, t)` | direct |
| `qc.ch(c, t)` | `h.ctrl(c, t)` | direct |
| `qc.crx/cry/crz(θ, c, t)` | `rx.ctrl(θ, c, t)`, `ry.ctrl`, `rz.ctrl` | direct. `rz.ctrl` is asymmetric in control/target — match Qiskit's mathematical definition exactly. |
| `qc.cp(θ, c, t)` (controlled-phase) | `r1.ctrl(θ, c, t)` | exact: both apply `diag(1, 1, 1, e^iθ)`. Symmetric in control/target. |
| `qc.rzz(θ, i, j)` | `cx(qubits[i], qubits[j]); rz(θ, qubits[j]); cx(qubits[i], qubits[j])` | no native `rzz` in 0.14 |
| `qc.rxx(θ, i, j)` | wrap rzz decomposition in `H⊗H` on both sides | |
| `qc.ryy(θ, i, j)` | wrap with `Sdg⊗Sdg · H⊗H` on both sides | |
| `qc.mcx([c1, c2, …], t)` | `x.ctrl([c1, c2, ...], t)` or variadic `x.ctrl(c1, c2, ..., t)` | works for arbitrary arity |
| `qc.mcry(θ, [c1, c2, …], t)` | `ry.ctrl(θ, [c1, c2, ...], t)` or variadic | |
| `qc.mcp(θ, [c1, c2, …], t)` (multi-controlled phase) | `r1.ctrl(θ, [c1, c2, ...], t)` or variadic | works for arbitrary arity; preferred over `cp`/`crz` chains for many-control phase |
| `MCXGate(num_ctrl_qubits=k, ctrl_state="01010")` (open controls) | wrap each |0⟩-control position in X gates before and after the controlled operation | no native `ctrl_state` argument |
| `qc.swap(a, b)` | `swap(a, b)` | direct |
| `qc.cswap(c, a, b)` | `swap.ctrl(c, a, b)` | direct; also accepts list/variadic controls for multi-control swap |
| `qc.measure(q, c)` (final-circuit) | `mz(qubits)` and dispatch through `cudaq.sample` | qubit-indexing convention §2 |
| `qc.measure(q, c)` (mid-circuit) | `b = mz(q)`; reuse `b` in `if`/`return`; dispatch through `cudaq.run` with `-> List[bool]` | §3.5 (cudaq-guide) |
| `qc.reset(q)` | `reset(q)` | unconditional in noiseless sim |
| `qc.append(U, qubits)` | implement U as `@cudaq.kernel`; call `my_kernel(qubits, ...)` directly | |
| `qc.inverse()` | `cudaq.adjoint(K, *args)` at top level | hand-roll inverse (§3.7, cudaq-guide) if K will be wrapped in `cudaq.control` |
| `qc.control(n)` | `cudaq.control(K, ctrl_or_list, *args)` | propagates through every gate in K; K must not contain `cudaq.adjoint` |
| `qc.compose(other, qubits)` | direct function call: `other(qubits[a:b], ...)` | qubit slicing at the call site |
| `QuantumCircuit.parameter_binds=` with `ParameterVector` | pass current parameter values as kernel arguments at execution time | no symbolic-vs-bound distinction in CUDA-Q |

For asymmetric controlled rotations (`rz.ctrl`, `ry.ctrl`, `rx.ctrl`), the control and target positions matter — swapping them gives a different unitary. Match the source's mathematical definition exactly and verify against a small deterministic probe.

---

## 2. Bit ordering and count keys

CUDA-Q and Qiskit use different conventions for stringifying measurement outcomes. The cleanest discipline is to keep the ordering transformation at the port boundary (the cudaq kernel's qubit-allocation convention or the Python-side bitstring-formatting code), not scattered through the algorithm.

| API | Convention |
|---|---|
| `cudaq.sample` with `mz(qview)` | Count keys position bits in the order the qubits appear in the qview. `qview[0]` is leftmost. |
| `cudaq.sample` with `mz(qubits)` after multiple `cudaq.qvector` allocations | Bits are ordered by qubit allocation order (first-allocated qubit first), not `mz` call order. |
| `cudaq.sample(..., explicit_measurements=True)` | Bits are in measurement order (the order the `mz(q)` calls fired). Reverse or remap if matching a classical-register display order. |
| `cudaq.run` with `-> List[bool]` | The framework returns a list of per-shot lists. When joining shot bits into a count-key string, the natural choice is element-0 leftmost (`''.join('1' if b else '0' for b in shot)`). Arrange the return list so element 0 is the leftmost character of the desired count key. |
| `cudaq.get_state(K)[i]` | `format(i, f"0{N}b")` produces the same bitstring `cudaq.sample` would return from `mz(qubits)` on the same kernel. Both come from cudaq, no remapping. |
| Qiskit `qc.measure` + `get_counts` | Classical bit `c[0]` is displayed at the rightmost position of the count-key string. |

Mapping rules:

- To match a Qiskit count key like `"{cr_a}{c[N-1]}...{c[0]}"`, build the cudaq return list as `[a_bit, mz(qa[N-1]), mz(qa[N-2]), ..., mz(qa[0])]`.
- To match a Qiskit count key without an extra prefix register, allocate cudaq qubits so that `qubits[N-1-q]` corresponds to Qiskit's `qr[q]`. Then `mz(qubits)` produces qubit-0-leftmost which equals Qiskit's classical-bit-`(N-1)`-leftmost.
- When porting an oracle that bit-decomposes a `marked_item` integer in Python and passes the bits as `List[int]` into the kernel, remove the Qiskit reverse slice (`[::-1]`) — cudaq's convention puts `register[0]` on the left of the count key, opposite of Qiskit.

When count keys are wrong:

If counts are correct but keys are reversed, fix it at the port boundary (qubit-allocation convention or Python-side bitstring formatting), not by changing the algorithm.

If only some shot results look wrong, verify with a 2- or 3-qubit deterministic probe over all possible inputs and check exact keys, not just aggregate fidelity.

If matching Qiskit and a separate-classical-register prefix is present (e.g., `'1 0101'` with a space), note that cudaq does NOT separate classical registers with spaces in count keys.

---

## 3. Floating-point precision

CUDA-Q and Qiskit default to different statevector precisions. This matters whenever you compare numerical results across frameworks.

| Framework | Default backend | Default amplitude precision | Bytes per complex amplitude |
|---|---|---:|---:|
| CUDA-Q | `nvidia` (cuStateVec) | fp32 (single-precision complex) | 8 |
| Qiskit | `AerSimulator` (statevector / GPU) | fp64 (double-precision complex) | 16 |

Implications for cross-framework comparison:

- Fidelity floor. A cudaq-vs-qiskit Hellinger fidelity comparison at default settings is comparing single-precision sample statistics against double-precision exact distributions. Expect roughly `1e-6` to `1e-3` of precision-induced disagreement per amplitude on deep circuits — usually invisible in shot-noise but visible in exact-distribution comparisons at high amplitude counts.
- Deep / many-rotation circuits drift more. Long sequences of small-angle rotations (QPE, AE, deep Hamiltonian simulation) accumulate fp32 rounding. The drift is rarely large enough to flip an algorithmic outcome but can shift fidelities by a percent or two relative to the same circuit at fp64.
- Transpile-to-small-basis amplifies the drift. Decomposing a controlled-phase through `u3 + cx` chains introduces additional rotations whose rounding compounds. Native-gate emission (e.g., `r1.ctrl` directly) preserves precision over a transpile-through-`{u3, cx, swap}` path.

To match precisions when running the comparison:

```python
# CUDA-Q: switch to fp64
cudaq.set_target("nvidia", option="fp64")

# Qiskit Aer GPU: switch to fp32 (matching cudaq's default)
sim = AerSimulator(device="GPU", precision="single")
```

Switching cudaq to fp64 roughly doubles statevector memory (now 16 bytes per amplitude), halving the largest width that fits on the same GPU. Switching qiskit Aer to fp32 has the corresponding memory savings and matches cudaq's default behavior.

Recommendation: unless the goal is specifically to compare default-vs-default behavior, set both frameworks to the same precision (typically fp64 for fidelity-comparison work, fp32 for performance-benchmarking work) before measuring agreement. If you're not comparing across frameworks, the cudaq default of fp32 is a reasonable performance trade-off for most algorithms.

---

## 4. Framework-decoupling patterns

These build on the gate-encoding patterns in the cudaq-guide Authoring section (§3.1 runtime-arity control list, §3.7 hand-rolled inverse).

### 4.1 Recursive constructor -> flat-sequence emitter

CUDA-Q kernels cannot recurse. Source-framework references often use Python recursion to build gate sequences (uniformly-controlled rotations, Walsh-Hadamard angle expansions, recursive multi-controlled decompositions). Port the recursion as a pure-Python emitter that appends directly to output arrays — not by building a `QuantumCircuit` and walking its data (which would reintroduce a source-framework runtime dependency).

```python
# Pure-Python module — no source-framework import.
def ucr_sequence_pure(n, theta):
    """Flat (ry_angles, ctrl_indices) for the uniformly-controlled-RY gate."""
    ry_angles, ctrl_indices = [], []

    def recurse(qubit_idx_list, theta_slice):
        if len(qubit_idx_list) == 1:
            ry_angles.append(float(theta_slice[0]))
            ctrl_indices.append(qubit_idx_list[0])
            ry_angles.append(float(theta_slice[1]))
        else:
            half = len(theta_slice) // 2
            recurse(qubit_idx_list[1:], theta_slice[:half])
            ctrl_indices.append(qubit_idx_list[0])
            recurse(qubit_idx_list[1:], theta_slice[half:])

    recurse(list(range(n)), list(theta))
    ctrl_indices.append(0)  # final wrapper gate
    return ry_angles, ctrl_indices

@cudaq.kernel
def apply_ucr_ry(qubits: cudaq.qview, anc: cudaq.qubit,
                 ry_angles: List[float], ctrl_indices: List[int]):
    for i in range(len(ry_angles)):
        ry(ry_angles[i], anc)
        if i < len(ctrl_indices):
            cx(qubits[ctrl_indices[i]], anc)
```

Validate that the emitter is bit-exact: implement once with the source-framework reference temporarily available, compare arrays element-wise (`max |Δ| = 0.00e+00`) for a few representative inputs, then drop the source-framework dependency.

If the source uses `sympy.combinatorics.GrayCode` to enumerate Gray codes, replace with the closed form `gray(i) = i ^ (i >> 1)` — bit-exact, no sympy dependency.

### 4.2 Pure-Python helper extraction + sys.meta_path verification

When the source-framework module exposes pure-Python helpers (analyzers, problem generators, post-selection logic) but its top-level `import qiskit` / `import qiskit_aer` pulls the framework into the cudaq path at import time, create a sibling module that contains only those helpers and imports only `numpy` + Python stdlib. The cudaq port imports from this new module. Leave the source-framework module untouched.

What to put in the new sibling module:

- Problem generators that are already pure numpy but live in a framework-importing module: copy verbatim.
- Post-selection / fidelity analyzers that operate on `dict[str, int]` count keys: copy verbatim. Typically only need `numpy.linalg`.
- Recursive constructors that built a source-framework circuit: rewrite as pure-Python emitters (§4.1).
- Gray-code / combinatorial helpers sourced from `sympy`: rewrite using closed-form math (`gray(i) = i ^ (i >> 1)`).

Verification gate. Add a `sys.meta_path` finder that raises on `import qiskit` / `qiskit_aer` / `qiskit_ibm_runtime` / `sympy`, install it before any of the cudaq port's imports, then run the port at a small representative configuration:

```python
import sys, importlib.abc

class _BlockSourceFramework(importlib.abc.MetaPathFinder):
    BLOCKED = ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "sympy")
    def find_spec(self, name, path, target=None):
        for blocked in self.BLOCKED:
            if name == blocked or name.startswith(blocked + "."):
                raise ModuleNotFoundError(f"refusing to import {name}")
        return None

sys.meta_path.insert(0, _BlockSourceFramework())
# ...now import and run the cudaq port at a small configuration...
```

If the smoke run passes with the blocker active, the cudaq port is genuinely framework-free.

### 4.3 Gate-recorder class for deeply-nested constructors

When the source framework builds circuits through several nested levels of sub-circuits with `.inverse()` and `.control(k)` chains at every level (modular arithmetic chains, oracle assemblies, multi-level decompositions), the inline `if/elif` dispatch from §3.1 gets unwieldy. Encapsulate the Python-side gate emission in a class whose methods mirror the source framework's constructors one-for-one:

```python
class GateRecorder:
    """Append gate records into parallel arrays. Each record is one gate."""

    H, X, CX, CSWAP, R1, CR1, CCR1, CRZ = range(1, 9)  # op-kind dispatch codes

    def __init__(self):
        self.op_kind, self.q1, self.q2, self.q3, self.angle = [], [], [], [], []

    def _emit(self, kind, q1, q2, q3, angle):
        self.op_kind.append(int(kind))
        self.q1.append(int(q1))
        self.q2.append(int(q2))
        self.q3.append(int(q3))
        self.angle.append(float(angle))

    # Primitive gates (one per op-kind the kernel handles)
    def h(self, q):                   self._emit(self.H, q, 0, 0, 0.0)
    def x(self, q):                   self._emit(self.X, q, 0, 0, 0.0)
    def cx(self, c, t):               self._emit(self.CX, c, t, 0, 0.0)
    def cswap(self, c, a, b):         self._emit(self.CSWAP, c, a, b, 0.0)
    def r1(self, angle, t):           self._emit(self.R1, t, 0, 0, angle)
    def cr1(self, angle, c, t):       self._emit(self.CR1, c, t, 0, angle)
    def ccr1(self, angle, c1, c2, t): self._emit(self.CCR1, c1, c2, t, angle)
    def crz(self, angle, c, t):       self._emit(self.CRZ, c, t, 0, angle)

    def extend_inverse(self, sub):
        """Append the inverse of `sub` (another GateRecorder): walk its
        records in reverse with negated rotation angles. h/x/cx/cswap are
        self-inverse, so their angle stays 0.0."""
        for i in range(len(sub.op_kind) - 1, -1, -1):
            kind = sub.op_kind[i]
            angle = (0.0 if kind in (self.H, self.X, self.CX, self.CSWAP)
                     else -sub.angle[i])
            self._emit(kind, sub.q1[i], sub.q2[i], sub.q3[i], angle)

    # Composite — one method per source-framework constructor
    def my_subroutine(self, n, param, qubit_offset, inverse=False):
        # ...emit records that implement this constructor at qubit_offset...
        pass
```

Each `def some_sub(num_qubits, ...)` constructor in the source framework becomes a method `def some_sub(self, num_qubits, ..., qubit_offset)` that takes the qubit-offset(s) it should write into. The source's `qc.append(sub_gate, qubits)` pattern becomes "call the method with the right qubit offsets" — trace through register layouts once, then write the offsets out explicitly per call site.

For `.inverse()` calls, build the forward sub-sequence into a fresh recorder, then `parent.extend_inverse(sub)`:

```python
sub = GateRecorder()
sub.my_subroutine(n, param_inv, ...)
parent.extend_inverse(sub)
```

Pick the op-kind dispatch set to match the gates the source actually uses. A useful default for modular-arithmetic-style circuits (1-2 qubit operations + single-control phase gates):

```
1 = h           5 = r1(angle, t)               (qiskit `p`)
2 = x           6 = r1.ctrl(angle, c, t)       (qiskit `cp`)
3 = cx          7 = r1.ctrl(angle, c1, c2, t)  (qiskit 2-control `mcp`)
4 = swap.ctrl   8 = rz.ctrl(angle, c, t)       (qiskit `crz`)
```

The kernel walker has the same structure as §3.1 but with a richer op-kind menu. Combined with the runtime-arity control list from §3.1 and the per-iteration-with-offsets pattern, the recorder handles essentially any nested-constructor source framework. Native-gate dispatch like this avoids per-circuit transpile cost and the precision loss of transpiling through a small basis like `{u3, cx, swap}`.

---

## 5. Port validation gate

For ports from a source framework:

1. Run the source implementation for the same inputs to capture a baseline (raw count keys, statevector if available).
2. Run the cudaq implementation with the same deterministic inputs.
3. Compare raw count keys, not only summary metrics. A 0.95 polarization fidelity can hide a key-ordering bug that produces uniformly wrong but balanced counts.
4. Test at least one smaller, one nominal, and one larger configuration when feasible.
5. Include stochastic tolerance only when the algorithm or shot count requires it (i.e., when shot noise is expected to dominate).
6. Re-run any previously-failing configuration after any change.

When stuck: copy from the right source. Sub-circuit conventions ("the IQFT," "the oracle") can vary by iteration direction, qubit order, or gate flavor across codebases. When porting a sub-circuit, copy from the source-framework module you're actually porting from — not from a different one's CUDA-Q port. Cross-implementation copy/paste introduces subtle inconsistencies that produce ~50% fidelity bugs.

When task accuracy depends on a specific CUDA-Q version, check `cudaq.__version__` and the official CUDA-Q documentation for that version before relying on remembered API behavior.

---

## References

1. CUDA-Q 0.14 documentation: <https://nvidia.github.io/cuda-quantum/0.14.0/>
2. CUDA-Q examples (canonical references for `cudaq.run` + `List[bool]` + mid-circuit measurement): `docs/sphinx/examples/python/measuring_kernels.py` and `sample_to_run_migration.py` in the CUDA-Q 0.14 source tree. <https://github.com/NVIDIA/cuda-quantum/tree/releases/v0.14.0/docs/sphinx/examples/python>
3. CUDA-Q Academic: <https://github.com/NVIDIA/cuda-q-academic> — worked QPE / VQE / QAOA in `@cudaq.kernel` form.
4. 0.14 source-tree tests: `python/tests/kernel/test_kernel_features.py` — confirms specific kernel-language constructs (mid-circuit `mz`, conditionals, returns).
5. Companion skill: cudaq-guide (`/cudaq-guide author`) — execution-API selection, kernel-language subset, shared kernel patterns (§3.1–§3.7), resource metrics, debugging workflow.
