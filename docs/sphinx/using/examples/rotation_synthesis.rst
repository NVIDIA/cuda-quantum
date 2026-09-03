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

Synthesizing a whole kernel
---------------------------

The ``compiler-bench-ftqc-clifford-t`` target compiles a kernel all the way
down to Clifford+T. It reduces ``rx``, ``ry`` and ``r1`` rotations to ``rz``,
synthesizes each one, and optimizes the result. Its ``epsilon`` argument sets
the per-rotation tolerance and defaults to ``1e-10``.

.. literalinclude:: ../../snippets/python/using/examples/synthesis/rotation_synthesis.py
     :language: python
     :start-after: [Begin Target]
     :end-before: [End Target]

.. code-block:: text

    only Clifford+T: True
    T count:         208

Because the T gate is the expensive resource on fault-tolerant hardware, the
T count is the number to watch. Pairing the target with
:doc:`cudaq.estimate_resources </api/languages/python_api>` reports that count
without a full state-vector simulation.

Rotation angles must be compile-time constants when synthesis runs. A kernel
whose angle comes from a runtime argument has to be specialized first.

Synthesizing a single rotation
------------------------------

To approximate one rotation directly, use ``cudaq.synth.gridsynth``. Note that
the ``synth`` submodule is imported explicitly:

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
``compiler-bench-ftqc-clifford-t`` target does not have this limitation: it
emits the phase explicitly, so the circuit it produces is exactly equal to the
original.

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
