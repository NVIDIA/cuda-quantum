..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Pulse Operations
================

All operations listed here are used as bare names inside ``@pulse.kernel``
functions. They are intercepted during bytecode tracing and lowered
to MLIR operations in the Pulse dialect.

Channel Access
--------------

.. function:: get_drive_line(qubit) -> (line, tone)

   Obtain the drive channel for a qubit. Returns a ``(line, tone)`` pair:
   the *line* is passed to ``drive()``, ``wait()``, and ``sync()``; the
   *tone* is passed to phase and frequency ops.

.. function:: get_readout_line(qubit) -> (line, tone)

   Obtain the readout channel for a qubit. Same return convention.

Scheduling Operations
---------------------

.. function:: drive(line, waveform, tone)

   Play a waveform envelope on a drive line at the given tone frequency.

.. function:: readout(line, waveform, tone)

   Acquire a measurement through a readout line.

.. function:: wait(line, duration)

   Insert an idle delay of *duration* clock cycles on a line.

.. function:: sync(line1, line2, ...)

   Synchronize two or more lines to a common time point.
   All lines are padded to the latest time among them.

Phase and Frequency Control
---------------------------

These operations modify the rotating frame of a tone channel.

.. function:: shift_phase(tone, phase)

   Add a relative phase offset (radians) to the tone.

.. function:: set_phase(tone, phase)

   Set the absolute phase (radians) of the tone.

.. function:: shift_frequency(tone, frequency)

   Add a relative frequency offset (Hz) to the tone.

.. function:: set_frequency(tone, frequency)

   Set the absolute frequency (Hz) of the tone.

Waveform Constructors
---------------------

Each constructor returns a waveform value that can be passed to
``drive()`` or combined with waveform arithmetic.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``gaussian(duration, amplitude, sigma)``
     - Gaussian envelope with standard deviation *sigma*
   * - ``square(duration, amplitude)``
     - Constant-amplitude (flat-top) envelope
   * - ``drag(duration, amplitude, sigma, beta)``
     - DRAG pulse for leakage suppression
   * - ``cosine(duration, amplitude)``
     - Raised-cosine envelope
   * - ``tanh_ramp(duration, amplitude, sigma)``
     - Hyperbolic tangent ramp from zero to the requested amplitude
   * - ``gaussian_square(duration, amplitude, sigma, width)``
     - Flat-top pulse with Gaussian rise/fall edges
   * - ``custom(duration, envelope_fn)``
     - Named user-defined envelope callback for compiler experimentation
   * - ``custom_samples(samples)``
     - Waveform from a non-empty sequence of at most 253 real-valued samples

For ``gaussian_square``, ``width`` is the flat-top width in virtual time units;
``duration - width`` must be a positive even integer so the two edges have the
same integer duration.

The GPU execution lowering supports the built-in constructors and
``custom_samples``. Arbitrary ``custom`` callbacks must be specialized to
samples by a future pass and are currently rejected before execution.

Waveform Arithmetic
-------------------

Waveforms can be combined algebraically inside kernels:

.. :spellcheck-disable:

.. code-block:: python

   @pulse.kernel
   def combined_waveform(qubit):
       drive_line, tone = get_drive_line(qubit)
       envelope_a = gaussian(64, 0.5, 16.0)
       envelope_b = gaussian(64, 0.3, 10.0)
       combined = wf_add(envelope_a, envelope_b)
       scaled = wf_scale(0.5, envelope_a)
       inverted = wf_neg(envelope_a)

.. :spellcheck-enable:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Operation
   * - ``wf_add(left, right)``
     - ``left + right``
   * - ``wf_sub(left, right)``
     - ``left - right``
   * - ``wf_mul(left, right)``
     - ``left * right`` (element-wise)
   * - ``wf_scale(scalar, waveform)``
     - ``scalar * waveform``
   * - ``wf_neg(waveform)``
     - ``-waveform``

Waveform algebra is available in the compiler IR, but the GPU execution
lowering does not yet evaluate algebra nodes and rejects them explicitly.
