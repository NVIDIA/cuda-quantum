.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Writing Transform Passes
========================

Beyond the built-in optimization pipeline that ``pulse.compile()`` runs, you
can transform pulse programs directly in Python. Passes operate on a
lightweight ``Program`` / ``Op`` intermediate representation, so authoring and
composing your own transform is just writing a plain Python function.

.. note::

   The ``cudaq_pulse.passes`` surface is experimental and may change without
   notice. For the standard "compile a kernel end-to-end" path, prefer
   :doc:`compilation`.

The Program IR
--------------

A ``Program`` is a flat, ordered list of ``Op`` records over SSA-style
``Value`` handles. You can build one with the fluent ``ProgramBuilder``:

.. code-block:: python

   from cudaq_pulse.passes import ProgramBuilder

   builder = ProgramBuilder("rabi", clock_ghz=2.0)
   line, tone = builder.get_drive_line(qubit=0, freq_hz=5.0e9)
   envelope = builder.gaussian(duration_vtu=64, amplitude=0.5, sigma=16.0)
   builder.drive(line, envelope, tone)

   program = builder.build()

Applying Built-in Passes
------------------------

Every optimization pass is a ``Program -> Program`` function, so passes compose
by ordinary function application:

.. code-block:: python

   from cudaq_pulse.passes import (
       run_canonicalize,
       run_virtual_z,
       run_fusion,
   )

   program = run_canonicalize(program)
   program = run_virtual_z(program)
   program = run_fusion(program)

Schedulers return a list of timed events plus metrics rather than a new
``Program``:

.. code-block:: python

   from cudaq_pulse.passes import schedule_alap

   events, metrics = schedule_alap(program)
   print(f"total {metrics.total_length_vtu:.0f} VTU, "
         f"idle {metrics.idle_fraction:.1%}")

Writing Your Own Pass
---------------------

A custom transform is just a function that takes a ``Program`` and returns a
new one. This example drops any zero-amplitude drives:

.. code-block:: python

   from cudaq_pulse.passes import Program, OpKind
   from dataclasses import replace

   def drop_zero_amplitude(program: Program) -> Program:
       kept = [
           op for op in program.ops
           if not (op.kind == OpKind.DRIVE and op.attrs.get("amplitude") == 0)
       ]
       return replace(program, ops=kept)

   program = drop_zero_amplitude(program)

Because passes are pure ``Program -> Program`` functions, they are trivial to
unit test and to interleave with the built-ins in any order.

Emitting MLIR
-------------

Once a program has been transformed, lower it to Pulse-dialect MLIR:

.. code-block:: python

   from cudaq_pulse.passes import program_to_pulse_mlir

   mlir = program_to_pulse_mlir(program)
   print(mlir)

See ``examples/10_scheduling_comparison.py`` and
``examples/11_loop_optimizations.py`` for complete, runnable pass walkthroughs.
