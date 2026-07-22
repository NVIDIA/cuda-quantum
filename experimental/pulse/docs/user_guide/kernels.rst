.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Pulse Kernels
=============

The ``@pulse.kernel`` decorator is the entry point for writing
pulse-level quantum programs. It captures the decorated function's
Python bytecode and traces it into an intermediate representation
that can be compiled to MLIR.

Import Convention
-----------------

A single import gives you everything:

.. code-block:: python

   import cudaq_pulse as pulse

Inside ``@pulse.kernel`` functions, DSL operations (``drive``,
``gaussian``, ``get_drive_line``, etc.) are used as bare names.
Infrastructure stays behind the ``pulse.`` prefix:

.. code-block:: python

   @pulse.kernel
   def rabi_oscillation(qubit):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(64, 0.5, 16.0), tone)

   compiled_kernel = pulse.compile(rabi_oscillation, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5.0e9})

Defining a Kernel
-----------------

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def my_kernel(qubit):
       drive_line, tone = get_drive_line(qubit)
       waveform = gaussian(40, 0.3, 10.0)
       drive(drive_line, waveform, tone)

Kernel arguments are **qudit references** -- opaque handles representing
quantum degrees of freedom. They are created outside the kernel using
``pulse.qudit_ref()`` or ``pulse.qvec_ref(n)`` and passed in when compiling.

Qudit Allocation
----------------

**Single qudit:**

.. code-block:: python

   qubit = pulse.qudit_ref()

**Vector of qudits:**

.. code-block:: python

   qubits = pulse.qvec_ref(4)
   qubit_0 = qubits[0]
   qubit_1 = qubits[1]

Control Flow
------------

Kernels support a subset of Python control flow that can be captured
at trace time:

**For loops** with compile-time integer bounds:

.. code-block:: python

   @pulse.kernel
   def echo_sequence(qubit):
       drive_line, tone = get_drive_line(qubit)
       for i in range(5):
           drive(drive_line, gaussian(40, 0.3, 10.0), tone)
           wait(drive_line, 20)

Loops are captured as rolled ``scf.for`` operations in the IR, preserving
loop structure for downstream optimization passes.

**If/else** with compile-time conditions:

.. code-block:: python

   @pulse.kernel
   def conditional_pulse(qubit, use_drag):
       drive_line, tone = get_drive_line(qubit)
       if use_drag:
           waveform = drag(40, 0.3, 10.0, 0.5)
       else:
           waveform = gaussian(40, 0.3, 10.0)
       drive(drive_line, waveform, tone)

Unsupported Patterns
--------------------

The following Python constructs are **not** supported inside kernels
and will raise ``CompilationError``:

- ``while`` loops
- Nested function definitions or closures
- List comprehensions or generator expressions
- ``try`` / ``except`` blocks
- Calls to arbitrary Python functions (only ``cudaq_pulse`` ops are allowed)
- Dynamic loop bounds (bounds must be known at trace time)
