:orphan:

.. _sample-to-run-migration:

Migrating Kernels from ``sample`` to ``run``
*********************************************

Introduction
============

Starting with CUDA-Q 0.14.0, ``sample`` no longer supports kernels that branch
on measurement results (measurement-dependent control flow). Kernels containing
patterns such as ``if mz(q):`` or ``if (result) { ... }`` where ``result``
comes from a measurement must now use ``run`` instead.

This breaking change creates a clearer API separation:

- Use ``sample`` for **aggregate measurement statistics** (counts dictionaries).
- Use ``run`` for **shot-by-shot execution** with measurement-dependent control
  flow and individual return values.

For the full API specification, see the :ref:`sample <cudaq-sample-spec>` and
:ref:`run <cudaq-run-spec>` sections in the Algorithmic Primitives documentation.
For a usage guide, see :doc:`Running your first CUDA-Q Program <basics/run_kernel>`.


What Still Works with ``sample``
================================

Kernels without measurement-dependent control flow continue to work exactly as
before. This includes implicit measurements, explicit measurements without
conditionals, partial qubit measurement, mid-circuit measurement for
reset patterns, and the ``explicit_measurements`` option (which concatenates all
measurement results in execution order rather than re-measuring at the end of
the kernel -- see the :ref:`sample specification <cudaq-sample-spec>` for
details).

.. tab:: Python

    .. literalinclude:: ../examples/python/sample_to_run_migration.py
        :language: python
        :start-after: [Begin Sample_Works]
        :end-before: [End Sample_Works]

.. tab:: C++

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Sample_Works]
        :end-before: [End Sample_Works]

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Sample_Works_Run]
        :end-before: [End Sample_Works_Run]


What No Longer Works
====================

Kernels that branch on measurement results can no longer be used with
``sample`` or ``sample_async``. Attempting to do so will raise a runtime error.

This includes both inline conditionals on measurements and conditionals on
variables holding measurement results:

.. tab:: Python

    .. code-block:: python

        @cudaq.kernel
        def kernel():
            q = cudaq.qvector(2)
            h(q[0])
            r = mz(q[0])
            if r:               # ERROR
                x(q[1])

        cudaq.sample(kernel)    # raises RuntimeError

.. tab:: C++

    .. code-block:: cpp

        auto kernel = []() __qpu__ {
            cudaq::qvector q(2);
            h(q[0]);
            auto r = mz(q[0]);
            if (r) {            // ERROR
                x(q[1]);
            }
        };

        cudaq::sample(kernel);  // throws std::runtime_error

The error message will read:

.. code-block:: text

    `cudaq::sample` and `cudaq::sample_async` no longer support kernels that
    branch on measurement results. Kernel '<name>' uses conditional feedback.
    Use `cudaq::run` or `cudaq::run_async` instead. See CUDA-Q documentation
    for migration guide.


How to Migrate
==============

Migrating a kernel from ``sample`` to ``run`` requires three changes.

Step 1: Add a return type to the kernel
-----------------------------------------

``run`` requires kernels to return a non-void value. Instead of relying on
implicit measurement at the end of the circuit, explicitly ``return`` the
measurement results you need.

.. tab:: Python

    .. code-block:: python

        # Before (no return type, used with sample)
        @cudaq.kernel
        def kernel():
            q = cudaq.qvector(2)
            h(q[0])
            r = mz(q[0])
            if r:
                x(q[1])

        # After (returns a value, used with run)
        @cudaq.kernel
        def kernel() -> bool:
            q = cudaq.qvector(2)
            h(q[0])
            r = mz(q[0])
            if r:
                x(q[1])
            return mz(q[1])

.. tab:: C++

    .. code-block:: cpp

        // Before (void kernel, used with sample)
        auto kernel = []() __qpu__ {
            cudaq::qvector q(2);
            h(q[0]);
            auto r = mz(q[0]);
            if (r) { x(q[1]); }
        };

        // After (returns a value, used with run)
        struct kernel {
          auto operator()() __qpu__ {
            cudaq::qvector q(2);
            h(q[0]);
            auto r = mz(q[0]);
            if (r) { x(q[1]); }
            return mz(q[1]);
          }
        };

Step 2: Replace ``sample`` with ``run``
-----------------------------------------

.. tab:: Python

    .. code-block:: python

        # Before
        counts = cudaq.sample(kernel, shots_count=1000)

        # After
        results = cudaq.run(kernel, shots_count=1000)

.. tab:: C++

    .. code-block:: cpp

        // Before
        auto counts = cudaq::sample(1000, kernel);

        // After
        auto results = cudaq::run(1000, kernel{});

.. note::

    The default ``shots_count`` for ``run`` is 100, compared to 1000 for
    ``sample``. Specify ``shots_count`` explicitly if you need a particular
    number of shots.

Step 3: Update result processing
----------------------------------

``sample`` returns a ``sample_result`` (a counts dictionary mapping bit strings
to frequencies). ``run`` returns a list (Python) or ``std::vector`` (C++) of
individual return values -- one per shot. If you need a counts-dictionary view,
you can reconstruct it from the individual results:

.. tab:: Python

    .. literalinclude:: ../examples/python/sample_to_run_migration.py
        :language: python
        :start-after: [Begin Result_Processing]
        :end-before: [End Result_Processing]

.. tab:: C++

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Result_Processing]
        :end-before: [End Result_Processing]


Migration Examples
==================

Example 1: Simple conditional logic
-------------------------------------

A kernel that measures one qubit and conditionally applies a gate on another.

.. tab:: Python

    .. literalinclude:: ../examples/python/sample_to_run_migration.py
        :language: python
        :start-after: [Begin Example1]
        :end-before: [End Example1]

.. tab:: C++

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Example1]
        :end-before: [End Example1]

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Example1Run]
        :end-before: [End Example1Run]

Example 2: Returning multiple measurement results
---------------------------------------------------

A kernel that performs multiple mid-circuit measurements with conditional logic
and returns all results as a list. When returning a ``std::vector<bool>`` in
C++, pre-allocate the result vector and assign elements individually for
broadest target compatibility.

.. tab:: Python

    .. literalinclude:: ../examples/python/sample_to_run_migration.py
        :language: python
        :start-after: [Begin Example2]
        :end-before: [End Example2]

.. tab:: C++

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Example2]
        :end-before: [End Example2]

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Example2Run]
        :end-before: [End Example2Run]

Example 3: Quantum teleportation
----------------------------------

Teleportation of a qubit state requires conditional corrections based on 
Bell-basis measurements.

.. tab:: Python

    .. literalinclude:: ../examples/python/sample_to_run_migration.py
        :language: python
        :start-after: [Begin Example3]
        :end-before: [End Example3]

.. tab:: C++

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Example3]
        :end-before: [End Example3]

    .. literalinclude:: ../examples/cpp/sample_to_run_migration.cpp
        :language: cpp
        :start-after: [Begin Example3Run]
        :end-before: [End Example3Run]


When to Use ``sample`` vs. ``run``
====================================

**Use** ``sample`` **when:**

- You want aggregate measurement statistics (histograms).
- Your kernel has no measurement-dependent control flow.
- You only need final measurement distributions.
- You are using the ``explicit_measurements`` option, which concatenates all
  measurement results in execution order rather than re-measuring qubits at the
  end of the kernel. See the :ref:`sample specification <cudaq-sample-spec>`
  for details.

**Use** ``run`` **when:**

- You need shot-by-shot measurement values.
- Your kernel has conditionals based on measurement results.
- You want to return computed values from the kernel.
- You need to store or analyze individual shot data.


Additional Notes
================

- Users of ``sample_async`` with conditional-feedback kernels should migrate to
  ``run_async``. See the :ref:`run specification <cudaq-run-spec>` for the
  asynchronous API.

- ``run`` supports a variety of return types including scalars, vectors/lists,
  and user-defined data structures. See the
  :ref:`run specification <cudaq-run-spec>` for the complete list of supported
  types and their requirements.

- Assigning measurement results to named variables in kernels passed to
  ``sample`` is deprecated and will be removed in a future release. Use ``run``
  to retrieve individual measurement results.
