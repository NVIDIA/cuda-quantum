.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

GPU Execution
=============

.. warning::

   GPU execution is a **preview** capability and is still being ported. The
   APIs and lowering pipeline described here are experimental, not yet wired
   into the default build, and subject to change. The kernel-to-MLIR compiler
   path (:doc:`compilation`, :doc:`passes`) is the stable, front-facing story.

cudaq-pulse can lower compiled kernels all the way to GPU execution
via the NVIDIA cuDensityMat solver.

Pipeline Overview
-----------------

The full execution pipeline from kernel to GPU:

1. ``pulse.compile()`` produces a ``CompiledKernel`` with Pulse-dialect MLIR
2. ``lower_to_llvm()`` runs the three-stage lowering:
   Pulse -> QOp -> CuDensityMat -> LLVM IR
3. ``run()`` JIT-compiles the LLVM IR and executes on GPU

MLIR Lowering
-------------

To inspect the lowered LLVM IR:

.. code-block:: python

   import cudaq_pulse as pulse

   compiled_kernel = pulse.compile(my_kernel, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5e9})
   llvm_ir = compiled_kernel.lower_to_llvm()
   print(llvm_ir)

The lowering passes through three dialect conversions:

**Pulse -> QOp**
   Converts pulse operations (drive, readout, waveforms) into
   backend-agnostic quantum operator algebra (Hamiltonians, Lindbladians).

**QOp -> CuDensityMat**
   Maps operator algebra to cuDensityMat API calls (state creation,
   operator construction, time evolution).

**CuDensityMat -> LLVM**
   Lowers cuDensityMat operations to LLVM IR with runtime library calls.

GPU Simulation
--------------

To execute on a GPU (requires cuDensityMat runtime):

.. code-block:: python

   import cudaq_pulse as pulse

   compiled_kernel = pulse.compile(my_kernel, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5e9})
   results = compiled_kernel.run(qubit_count=1)

Requirements:

- NVIDIA GPU with compute capability 7.0+
- cuDensityMat library (part of NVIDIA cuQuantum SDK)
- ``MLIR_DIR`` environment variable or ``mlir-translate`` on ``PATH``

The ``run()`` method performs JIT compilation and returns the simulated
quantum state vector as a numpy array.
