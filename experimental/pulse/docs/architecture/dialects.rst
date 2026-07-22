.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

MLIR Dialects
=============

cudaq-pulse defines three MLIR dialects that form a layered IR stack.
Each dialect has full round-trip fidelity (parse -> print -> parse).

Pulse Dialect
-------------

The core dialect for pulse-level quantum programming. Defined in
``core/mlir/include/cudaq-pulse/Dialect/Pulse/PulseOps.td``.

**Types:**

- ``!pulse.qudit`` -- quantum degree of freedom
- ``!pulse.line`` -- drive or readout channel (linear resource)
- ``!pulse.tone`` -- frequency/phase reference for a channel
- ``!pulse.waveform`` -- envelope shape

**Operations:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operation
     - Description
   * - ``pulse.qudit_alloc``
     - Allocate a qudit resource
   * - ``pulse.get_drive_line``
     - Obtain drive line and tone for a qubit
   * - ``pulse.get_readout_line``
     - Obtain readout line and tone for a qubit
   * - ``pulse.drive``
     - Play a waveform on a drive line
   * - ``pulse.readout``
     - Acquire through a readout line
   * - ``pulse.wait``
     - Idle delay on a line
   * - ``pulse.sync``
     - Synchronize multiple lines
   * - ``pulse.shift_phase``
     - Relative phase offset on a tone
   * - ``pulse.set_phase``
     - Absolute phase on a tone
   * - ``pulse.shift_frequency``
     - Relative frequency offset on a tone
   * - ``pulse.set_frequency``
     - Absolute frequency on a tone
   * - ``pulse.gaussian``
     - Gaussian envelope waveform
   * - ``pulse.square``
     - Flat-top envelope waveform
   * - ``pulse.drag``
     - DRAG envelope waveform
   * - ``pulse.cosine``
     - Raised-cosine envelope
   * - ``pulse.tanh_ramp``
     - Hyperbolic tangent ramp
   * - ``pulse.gaussian_square``
     - Gaussian-edge flat-top waveform
   * - ``pulse.custom_waveform``
     - User-defined callable envelope
   * - ``pulse.custom_samples_waveform``
     - Pre-computed sample array
   * - ``pulse.wf_add``
     - Element-wise waveform addition
   * - ``pulse.wf_sub``
     - Element-wise waveform subtraction
   * - ``pulse.wf_mul``
     - Element-wise waveform multiplication
   * - ``pulse.wf_scale``
     - Scalar-waveform multiplication
   * - ``pulse.wf_neg``
     - Waveform negation

QOp Dialect
-----------

Backend-agnostic quantum operator algebra. Defined in
``core/mlir/include/cudaq-pulse/Dialect/QOp/``.

The QOp dialect represents the physics of pulse programs as
Hamiltonians, Lindbladians, and time-evolution operators. It serves
as the intermediate representation between pulse-level programming
and simulator-specific APIs.

**Key operations:**

- ``qop.hamiltonian`` -- define a Hamiltonian operator
- ``qop.lindbladian`` -- define Lindblad dissipator channels
- ``qop.evolve`` -- time-evolve a quantum state

CuDensityMat Dialect
--------------------

Wrapper for NVIDIA's cuDensityMat GPU-accelerated density matrix
solver. Defined in ``core/mlir/include/cudaq-pulse/Dialect/CuDensityMat/``.

Maps quantum operator algebra to concrete cuDensityMat API calls.

**Key operations:**

- ``cudm.state_create`` -- allocate GPU quantum state
- ``cudm.operator_create`` -- construct operator on GPU
- ``cudm.evolve`` -- GPU-accelerated time evolution
- ``cudm.state_destroy`` -- deallocate GPU state
