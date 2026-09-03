Rotation Synthesis (Clifford+T)
===============================

.. _rotation_synthesis:

Fault-tolerant quantum hardware runs only a small, discrete set of gates —
typically ``H``, ``S``, ``T``, ``X`` and ``CNOT``, the *Clifford+T* set.
Arbitrary rotations such as ``rz(0.6)`` are not among them, so before a kernel
can run on such hardware every rotation has to be replaced by a sequence of
gates that approximates it.

CUDA-Q does this with grid synthesis (Ross and Selinger, `arXiv:1403.2975
<https://arxiv.org/abs/1403.2975>`__). Given an angle and a tolerance
``epsilon``, it finds a short Clifford+T sequence :math:`U` satisfying
:math:`\|R_z(\theta) - U\| \le \epsilon` in the operator norm.

There are two ways to use it. ``cudaq.synth`` is a Python API for approximating
individual rotations, and is shown first. Synthesizing a whole kernel is done by
a compiler pass, which is opt-in. It is not part of any default target pipeline,
and the targets that run it are benchmarking targets for estimating how many T
gates an algorithm would cost.

Synthesizing a rotation
-----------------------

To approximate a rotation, use ``cudaq.synth.gridsynth``. Note that the
``synth`` submodule is imported explicitly:

.. literalinclude:: ../../snippets/python/using/examples/synthesis/rotation_synthesis.py
     :language: python
     :start-after: [Begin Single]
     :end-before: [End Single]

.. code-block:: text

    T count: 102
    gates:   HTHTHTSHTHTSHTSHTHTHTSHTSHTHTSHT...
    error:   9.665e-11

``gridsynth`` returns a ``CliffordTSequence``. Its ``t_count`` is the number of
T gates, and ``str()`` gives the gate string over ``{H, S, T, X, W}``, where
``W`` is the global phase :math:`\omega = e^{i\pi/4}`. The gates are listed in
matrix-multiplication order, so as a circuit they apply right to left.

``cudaq.synth.rz_error`` reports the error the sequence actually achieves. It
is computed exactly in arbitrary precision, so it does not depend on the
precision of any simulator, and it never exceeds the ``epsilon`` you asked for.

A ``CliffordTSequence`` can also be built directly from a gate string, which is
useful for inspecting Clifford+T circuits that came from somewhere else.
``normalized`` rewrites a sequence into Matsumoto-Amano normal form. An exactly
equal sequence with the smallest possible T count.

.. literalinclude:: ../../snippets/python/using/examples/synthesis/rotation_synthesis.py
     :language: python
     :start-after: [Begin Sequence]
     :end-before: [End Sequence]

.. code-block:: text

    imported:   TST (T count 2)
    normalized: SS (T count 0)
    already normal form: True

``gridsynth`` already returns sequences in normal form, so normalizing its
output changes nothing.

To use a synthesized sequence in a circuit, ``to_kernel`` builds a kernel that
takes a single qubit, ready for ``apply_call``:

.. literalinclude:: ../../snippets/python/using/examples/synthesis/rotation_synthesis.py
     :language: python
     :start-after: [Begin Kernel]
     :end-before: [End Kernel]

.. code-block:: text

    synthesized: { 0:908 1:92 }
    exact rz:    { 0:908 1:92 }

The synthesized sequence reproduces the exact rotation. Note that ``to_kernel``
drops the ``W`` phase factors, so the kernel it returns equals
:math:`R_z(\theta)` only up to a global phase. That phase has no effect when
the kernel is used on its own, but it becomes an observable relative phase if
the kernel is made the target of a controlled operation. The
``compiler-bench-ftqc-clifford-t`` target does not have this limitation. It
emits the phase explicitly, so the circuit it produces is exactly equal to the
original.

Estimating the T count of a kernel
----------------------------------

The T gate is the expensive resource on fault-tolerant hardware, so the T count
of a circuit is the number usually worth measuring. The
``compiler-bench-ftqc-clifford-t`` target runs synthesis over a whole kernel so
that count can be read off directly. It reduces ``rx``, ``ry`` and ``r1``
rotations to ``rz``, synthesizes each one, and optimizes the result. Its
``epsilon`` argument sets the per-rotation tolerance and defaults to ``1e-10``.

.. literalinclude:: ../../snippets/python/using/examples/synthesis/rotation_synthesis.py
     :language: python
     :start-after: [Begin Target]
     :end-before: [End Target]

.. code-block:: text

    only Clifford+T: True
    T count:         208

This is a benchmarking target for resource counting, not a path for compiling
kernels to run on hardware. It is also Python only. An ``nvq++`` build accepts
the target but does not run the synthesis.

Rotation angles must be compile-time constants when synthesis runs. A kernel
whose angle comes from a runtime argument has to be specialized first.

Choosing epsilon
----------------

``epsilon`` trades accuracy against cost. A tighter tolerance means more T
gates and more compile time; the T count grows roughly with
:math:`\log(1/\epsilon)`, while compile time grows much faster and becomes
significant below about ``1e-30``.

Synthesis results are cached per distinct angle, so a kernel that applies the
same rotation a hundred times pays the cost once.

Synthesis is randomized. Repeated calls with the same angle and tolerance
return sequences with the same T count, but not necessarily the same gates. Pass
``seed`` to make a run reproducible, as the example above does. The remaining
arguments to ``gridsynth`` are work budgets that trade compile time against T
count; see the :doc:`Python API reference </api/languages/python_api>` for what
each one bounds.

Dependencies
------------

Rotation synthesis uses exact arbitrary-precision arithmetic, provided by the
GMP and MPFR libraries. These ship with every CUDA-Q binary distribution, so no
action is needed to use the feature. See :ref:`dynamic-linking-gmp-mpfr` for
details on how they are packaged and how to substitute your own builds.
