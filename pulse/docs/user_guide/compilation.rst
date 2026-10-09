..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Compilation
===========

``pulse.compile()`` is the single public entry point for compiling
a ``@pulse.kernel`` function into a scheduled, optimized MLIR module.

Basic Usage
-----------

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def my_kernel(qubit):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(40, 0.3, 10.0), tone)

   compiled_kernel = pulse.compile(my_kernel, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5.0e9})

.. :spellcheck-enable:

The ``compile()`` Function
--------------------------

.. autofunction:: cudaq_pulse.compile
   :noindex:

Parameters
~~~~~~~~~~

``kernel_fn``
   A ``@pulse.kernel``-decorated function.

``args``
   Positional arguments -- typically ``pulse.qudit_ref()`` objects matching
   the kernel's parameters.

``clock_ghz`` *(float, default 2.0)*
   System clock frequency in GHz. Determines the physical time
   corresponding to one clock cycle.

``qubit_freq_hz`` *(dict[int, float], optional)*
   Mapping from qubit index to qubit frequency in Hz. Used for
   rotating-frame calculations and scheduling.

``schedule`` *(str, default "alap")*
   Scheduling policy. Options: ``"asap"``, ``"alap"``, ``"rcp"``,
   ``"alap_rcp"``. ASAP and ALAP preserve data and physical-line
   dependencies; RCP variants also enforce ``MachineModel`` resource limits.

``passes`` *(sequence of str, optional)*
   Optimization passes to run. Defaults to ``("verify", "virtual_z", "fusion")``.
   Pass an empty tuple ``()`` to skip all passes.

   Available passes: ``"verify"``, ``"canonicalize"``, ``"virtual_z"``,
   ``"fusion"``, ``"licm"``.

``machine`` *(MachineModel, optional)*
   Machine model for resource-constrained scheduling (RCP).

CompiledKernel
--------------

``pulse.compile()`` returns a ``CompiledKernel`` object:

.. autoclass:: cudaq_pulse.CompiledKernel
   :noindex:
   :members:
   :undoc-members:

Key properties and methods:

``.mlir``
   The Pulse-dialect MLIR text representation (lazily rendered).

``.module``
   The in-memory ``PulseModule`` (MLIR ``ModuleOp``).

``.metrics``
   A ``CompileMetrics`` dataclass with per-stage timing.

``.lower_to_llvm()``
   Run the full MLIR lowering pipeline (Pulse -> QOp -> CuDensityMat -> LLVM).

``.run()``
   JIT-compile and execute the unitary, two-level lowering on GPU via
   cuDensityMat. Use :func:`cudaq_pulse.evolve` for target-aware T1/T2,
   coupling, crosstalk, and drive-calibration metadata.

CompileMetrics
--------------

.. autoclass:: cudaq_pulse.CompileMetrics
   :noindex:
   :members:
   :undoc-members:

Fields (all in milliseconds):

- ``trace_ms`` -- bytecode tracing and packed buffer construction
- ``ffi_ms`` -- C++ MLIR module construction from packed buffer
- ``passes_ms`` -- optimization pass execution time
- ``schedule_ms`` -- scheduling pass time
- ``total_ms`` -- sum of all stages
- ``op_count`` -- number of MLIR operations in the module

Customizing the Pass Pipeline
-----------------------------

To run a custom set of passes:

.. :spellcheck-disable:

.. code-block:: python

   compiled_kernel = pulse.compile(
       my_kernel,
       [pulse.qudit_ref()],
       qubit_freq_hz={0: 5e9},
       passes=("verify", "canonicalize", "virtual_z", "fusion", "licm"),
   )

.. :spellcheck-enable:

To skip passes entirely (useful for benchmarking):

.. :spellcheck-disable:

.. code-block:: python

   compiled_kernel = pulse.compile(
       my_kernel,
       [pulse.qudit_ref()],
       qubit_freq_hz={0: 5e9},
       passes=(),
   )

.. :spellcheck-enable:
