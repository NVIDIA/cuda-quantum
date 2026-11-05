..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

GPU Execution
=============

.. warning::

   GPU execution is a **research preview**. It is not production software and
   carries no API, numerical, compatibility, or stability guarantee.

cudaq-pulse contains an experimental lowering and runtime path intended to
connect compiled kernels to NVIDIA cuDensityMat. This path remains active
research and should not be treated as a production simulator.

Build Configuration
-------------------

With pulse enabled, setting ``CUDENSITYMAT_ROOT`` makes CMake use CUDA-Q's
``FindcuDensityMat.cmake`` module and automatically build the runtime as part
of the ``pulse`` target. CMake fails if the explicitly requested SDK is not
available. There is no separate pulse GPU option.

.. :spellcheck-disable:

.. code-block:: bash

   cmake -S pulse -B build-gpu -G Ninja \
     -DCUDAQ_BUILD_TESTS=ON \
     -DCUDENSITYMAT_ROOT=/path/to/cuquantum
   cmake --build build-gpu --parallel --target pulse
   cmake --build build-gpu --target check-pulse-gpu
   export CUDAQ_PULSE_BUILD_DIR="$PWD/build-gpu"
   export PATH="$PWD/build-gpu/bin:$PATH"
   export PYTHONPATH="$PWD/build-gpu/python${PYTHONPATH:+:$PYTHONPATH}"

.. :spellcheck-enable:

Pipeline Overview
-----------------

The target-aware public execution path is:

1. Call a ``@pulse.kernel`` to produce traced pulse IR.
2. ``pulse.evolve(..., target=...)`` verifies, optimizes, and schedules it.
3. The native pipeline lowers Pulse -> QOp -> CuDensityMat -> LLVM IR.
4. The JIT compiles the LLVM IR and executes it with cuDensityMat.

``pulse.compile()`` also returns a ``CompiledKernel`` whose
``lower_to_llvm()`` and ``run()`` methods expose the native pipeline directly.
That path has no ``Target`` parameter and therefore represents unitary,
two-level execution with pulse amplitudes already expressed in radians per
nanosecond. Use ``pulse.evolve`` when target calibration, coupling, or T1/T2
metadata matters.

MLIR Lowering
-------------

To inspect the lowered LLVM IR:

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   compiled_kernel = pulse.compile(my_kernel, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5e9})
   llvm_ir = compiled_kernel.lower_to_llvm()
   print(llvm_ir)

.. :spellcheck-enable:

The lowering passes through three dialect conversions:

**Pulse -> QOp**
   Converts supported drive operations and waveforms into backend-agnostic
   Hamiltonian and Lindblad operator algebra. Readout and acquisition are
   rejected because this preview does not implement measurement simulation.

**QOp -> CuDensityMat**
   Maps operator algebra to cuDensityMat API calls (state creation,
   operator construction, time evolution).

**CuDensityMat -> LLVM**
   Lowers cuDensityMat operations to LLVM IR with runtime library calls.

GPU Simulation
--------------

To execute on a GPU (requires cuDensityMat runtime):

.. :spellcheck-disable:

.. code-block:: python

   import math
   import cudaq_pulse as pulse
   from cudaq_pulse.targets import Qubit, Target

   target = Target(
       name="one-qubit-demo",
       qubits={
           0: Qubit(
               index=0,
               frequency_hz=5.0e9,
               anharmonicity_hz=-200.0e6,
               t1_us=50.0,
               t2_star_us=30.0,
               drive_params={"amplitude_scale_rad_per_ns": 1.0},
           )
       },
   )

   @pulse.kernel
   def x_gate(qubit):
       drive_line, tone = get_drive_line(qubit)
       # 40 virtual units at 2 GHz is 20 ns.
       drive(drive_line, square(40, math.pi / 20.0), tone)

   ir = x_gate(pulse.qudit_ref())
   result = pulse.evolve(
       ir,
       target=target,
       t_start=0.0,
       t_end=20.0,
       num_steps=200,
       integrator="rk4",
   )
   print(result.final_state)

.. :spellcheck-enable:

``t_start`` and ``t_end`` are in nanoseconds. The ``integrator`` argument
selects the time-evolution scheme used by the cuDensityMat runtime path:

- ``rk1``, ``rk2``, ``rk4`` -- fixed-step explicit Runge-Kutta methods.
- ``magnus`` -- a fixed-step Magnus expansion evaluated as a truncated Taylor
  series of the propagator; preserves structure well for smoothly varying
  drives.
- ``crank_nicolson`` -- a fixed-step implicit predictor-corrector scheme that
  is more stable for stiff dynamics.

All five schemes reuse the same boundary-safe sampling that keeps piecewise
constant pulse segments from being sampled across a discontinuity.

Requirements:

- NVIDIA GPU with compute capability 7.0+
- cuDensityMat library (part of NVIDIA cuQuantum SDK)
- The matching ``llc`` and ``clang`` tools on ``PATH`` or in
  ``CUDAQ_PULSE_LLVM_BIN``. A source build normally satisfies this by adding
  the CUDA-Q LLVM ``bin`` directory to ``PATH``.

Physical Model
--------------

The current native execution lowering is a two-level transmon model in one
rotating frame per qubit. Drive envelopes are converted to X/Y Hamiltonian
coefficients in radians per nanosecond; tone frequency and phase determine the
rotating-frame detuning and quadrature. A target may supply
``amplitude_scale_rad_per_ns`` explicitly, or pulse infers a scale from its
calibrated Gaussian/DRAG pi-pulse parameters when available.

Target relaxation and dephasing data produce Lindblad collapse operators.
Coupling edges are modeled as always-on XX terms and residual crosstalk as
always-on ZZ terms. ``anharmonicity_hz`` is retained as calibration metadata
but is not part of this two-level model; leakage requires a future multilevel
lowering.

Unsupported Execution Features
------------------------------

The execution lowering fails explicitly for readout/acquisition, observables,
neutral-atom targets, multilevel models, unspecialized numeric parameters,
arbitrary Python waveform callbacks, and waveform-algebra nodes. Built-in
waveforms and ``custom_samples`` are supported. The callback runtime currently
supports at most 128 drive operations per compiled module.

CMake registers link/descriptor smoke checks and numerical GPU tests for
single-qubit drives, T1 decay, idle evolution, two-qubit coupling, frame and
I/Q modulation, an eight-qubit ladder register, closed-system physics
validation, quantum-algorithm building blocks, integrator parity across
``rk4``/``magnus``/``crank_nicolson``, and the public compile/JIT path. It
enables the numerical tests only when ``nvidia-smi`` reports a GPU and the
CUDA runtime can access at least one device. Otherwise the ``check-pulse-gpu``
target reports them disabled and retains the CPU-safe SDK linkage check when
cuDensityMat is installed. These tests provide research regression coverage;
they do not establish production numerical accuracy.

The ``run()`` method and returned state representation are experimental and
will evolve with the lowering and runtime implementation.
