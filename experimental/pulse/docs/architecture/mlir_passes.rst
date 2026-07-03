.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

MLIR Passes
===========

cudaq-pulse implements several C++ MLIR passes that optimize and
validate pulse programs. All passes operate on the in-memory
``mlir::ModuleOp`` and are invoked through ``mlir::PassManager``.

pulse-verify
------------

**Source:** ``core/mlir/transforms/PulseVerify.cpp``

Validates structural and semantic correctness of a pulse program:

- **Waveform validity**: Ensures all waveform durations are positive.
- **Monotone time ordering**: After scheduling, verifies that operations
  on each line are ordered by non-decreasing ``pulse.time`` attributes.

The verify pass runs early in the pipeline and raises a diagnostic
error for any violation, preventing malformed programs from reaching
later stages.

pulse-canonicalize
------------------

**Source:** ``core/mlir/transforms/Canonicalize.cpp``

Applies peephole simplifications using MLIR's greedy rewrite infrastructure:

- **Remove single-operand sync**: A ``pulse.sync`` with only one line
  operand is a no-op and is eliminated.
- **Remove redundant sync**: Consecutive ``pulse.sync`` operations on
  the same set of lines are deduplicated.

pulse-virtual-z
---------------

**Source:** ``core/mlir/transforms/VirtualZ.cpp``

Implements the *virtual-Z gate* optimization. Phase shifts
(``pulse.shift_phase``) that precede drive operations are commuted
forward and absorbed into the drive's waveform phase, eliminating
the need for a physical frame rotation.

This is a standard optimization in superconducting qubit control that
reduces the number of operations without changing the physical program.

pulse-fusion
------------

**Source:** ``core/mlir/transforms/Fusion.cpp``

Merges adjacent ``pulse.drive`` operations on the same line into a
single drive with a combined waveform (using ``pulse.wf_add``).
This reduces operation count and can improve scheduling density.

Fusion is only applied when both drives use the same tone and have
compatible timing.

pulse-schedule-alap
-------------------

**Source:** ``core/mlir/transforms/ScheduleAlap.cpp``

Implements As-Late-As-Possible (ALAP) scheduling. Assigns concrete
``pulse.time`` integer attributes to every operation by:

1. Building a dependency graph from data flow and sync constraints
2. Assigning times in reverse topological order, pushing operations
   as late as possible while respecting dependencies
3. Normalizing all times so the earliest operation starts at ``t=0``

loop-invariant-code-motion
--------------------------

Uses MLIR's built-in ``-loop-invariant-code-motion`` pass to hoist
waveform construction and other loop-invariant operations out of
``scf.for`` loop bodies. Particularly effective for dynamical
decoupling sequences where the same waveform is applied in every
iteration.
