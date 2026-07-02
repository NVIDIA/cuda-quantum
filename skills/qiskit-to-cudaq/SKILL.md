---
name: "qiskit-to-cudaq"
title: "Qiskit to CUDA-Q"
description: "Use when porting Qiskit Python circuits to CUDA-Q kernels while preserving algorithms and validation fidelity."
version: "1.0.0"
author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
tags: [cuda-quantum, quantum-computing, qiskit, porting, migration, kernels, nvidia]
tools: [Read, Glob, Grep]
license: "Apache-2.0"
compatibility: "Python 3.10+"
metadata:
    author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
    short-description: "Port Qiskit circuits to CUDA-Q"
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

# Qiskit to CUDA-Q

## Purpose

Use this skill to port Qiskit Python code, or code with Qiskit-style circuit
construction, to CUDA-Q Python kernels. The goal is a framework-free CUDA-Q port
that preserves the source quantum algorithm, matches source behavior at small
test sizes, and documents any unavoidable CUDA-Q limitations.

## Prerequisites

- Python 3.10+.
- CUDA-Q installed in the target environment. Check the runtime with:
  `python -c "import cudaq; print(getattr(cudaq, '__version__', 'unknown'))"`.
- Access to the source implementation and a way to run or inspect its expected
  behavior.
- For validation against Qiskit, Qiskit/Aer must be installed in the validation
  environment. The final CUDA-Q port itself must not require Qiskit.
- When using CUDA-Q documentation or repository MCP connectors, verify the
  connector is available before relying on it; otherwise use local docs or the
  source tree.

## Workflow

1. Read the source circuit construction and identify the exact algorithm,
   qubit/register layout, measurement behavior, and any framework helpers.
2. Preserve the high-level quantum algorithm. Do not replace mid-circuit
   measurement, QPE structure, oracle definitions, or decomposition strategy
   without explicit user permission.
3. Select the CUDA-Q execution pattern:
   - Use `cudaq.sample` for final-measurement sampling.
   - Use `cudaq.run` when mid-circuit measurement values must be returned or
     used per shot.
   - Use runtime-argument kernels instead of generated per-size kernels unless
     CUDA-Q requires a fixed-length return shape.
4. Translate gates and subcircuits. For detailed gate mappings, ordering rules,
   precision guidance, and helper-extraction patterns, read
   [references/porting-reference.md](references/porting-reference.md).
5. Remove runtime source-framework dependencies from the CUDA-Q port. Extract
   pure helpers into framework-free modules.
6. Validate with small deterministic inputs before scaling. Compare raw count
   keys and distributions, not just aggregate fidelity.
7. Re-run any previously failing configurations after every fix.

## Core Rules

- Keep the source algorithm intact unless the user approves a change.
- Do not introduce fixed qubit caps, fixed control arities, or source-framework
  imports unless they are genuinely unavoidable and documented.
- Prefer native CUDA-Q gates (`r1.ctrl`, `x.ctrl`, `swap.ctrl`, etc.) over
  transpiling through Qiskit.
- Keep bit-order conversion at the port boundary: allocation order,
  measurement return list, or final count-key formatting.
- Match floating-point precision when comparing CUDA-Q and Qiskit results if
  fidelity differences matter.
- Accept source flags that become no-ops in CUDA-Q when doing so preserves
  source-compatible behavior.

## When to Read the Reference

Read [references/porting-reference.md](references/porting-reference.md) when
you need any of the following:

- Qiskit-to-CUDA-Q gate translation table.
- Bit-ordering and count-key conventions.
- CUDA-Q fp32 vs Qiskit fp64 precision implications.
- Pure-Python helper extraction and import-blocker validation.
- Recursive-constructor emitters or gate-recorder patterns.
- Detailed port validation checklist and external CUDA-Q references.

## Limitations

- Guidance targets CUDA-Q 0.14.x decorator-mode Python APIs. Re-check behavior
  against the installed CUDA-Q version for version-sensitive features.
- Some CUDA-Q kernel-language constructs are constrained compared with normal
  Python; use the companion `cudaq-guide` skill for core CUDA-Q authoring
  constraints and shared kernel patterns.
- CUDA-Q and Qiskit differ in default precision and count-key display order.
  Apparent fidelity or bitstring mismatches may be convention differences.
- Hardware-target behavior, available backends, and target options depend on
  the local CUDA-Q installation.
- This skill does not guarantee equivalent performance; it focuses on
  correctness-preserving ports.

## Troubleshooting

Use this format when diagnosing failures:

- **Error:** `ModuleNotFoundError: qiskit` from a CUDA-Q path.
  **Cause:** The port still imports the source framework.
  **Solution:** Move pure helpers into a framework-free module and verify with
  the import-blocker pattern in the reference.

- **Error:** Fidelity looks plausible but raw keys are reversed.
  **Cause:** Qiskit and CUDA-Q count-key ordering differ.
  **Solution:** Fix allocation, return-list order, or formatting at the port
  boundary. Do not alter the algorithm.

- **Error:** Deep-circuit fidelity differs between frameworks.
  **Cause:** CUDA-Q and Qiskit may be using different floating-point precision.
  **Solution:** Match precision before comparing, then rerun the smallest
  failing deterministic case.

- **Error:** A multi-controlled operation works for small controls but fails or
  silently changes behavior at higher arity.
  **Cause:** The port used a fixed-arity dispatcher.
  **Solution:** Use CUDA-Q control-list patterns for arbitrary arity.

- **Error:** MCP documentation or repository lookup fails.
  **Cause:** Connector unavailable, stale, or transiently failing.
  **Solution:** Verify the connector/resource list, retry transient failures
  once, then fall back to local docs/source or official CUDA-Q docs. Do not
  change the port based on unverified MCP results.

## References

- [Detailed porting reference](references/porting-reference.md)
- CUDA-Q 0.14 documentation: <https://nvidia.github.io/cuda-quantum/0.14.0/>
- CUDA-Q Academic examples: <https://github.com/NVIDIA/cuda-q-academic>
- Companion skill: `cudaq-guide` (`/cudaq-guide author`) for CUDA-Q authoring
  patterns, kernel-language constraints, execution APIs, and debugging workflow.
