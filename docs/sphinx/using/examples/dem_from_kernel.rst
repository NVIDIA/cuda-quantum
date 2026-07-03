Detector Error Models
=====================

.. _dem_from_kernel:

A *detector error model* (DEM) is a detector error matrix (capturing which
detectors each error mechanism flips) together with a noise model that
assigns a likelihood to each error mechanism. It is the input a decoder needs
to infer which errors occurred from a circuit's measurement record. 

In CUDA-Q you declare the parity checks (*detectors*) and *logical observables* 
directly inside a kernel, then extract the DEM with
:code:`cudaq::dem_from_kernel` (C++) or :code:`cudaq.dem_from_kernel` (Python)
as text in Stim's standard
`.dem file format <https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md>`__,
which ``stim.DetectorErrorModel`` parses back into a decoder-ready model. The
measurements that feed the declarations are *measurement handles*
(:doc:`measuring_kernels`).

Three kernel-side declarations are available in both C++ and Python : 
``detector(m0, m1, ...)`` declares one detector as a parity
constraint over the given measurements; ``detectors(prev, curr)`` declares ``N``
detectors by pairing two equal-length handle vectors element-wise (the
standard form for cross-round detectors); and
``logical_observable(m0, m1, ...)`` declares a logical observable.

The example below is a three-qubit bit-flip memory experiment: each round
measures the data qubits and pairs them with the previous round via
``detectors``, with a final ``logical_observable`` reading out the register.
In-kernel ``apply_noise`` seeds the error mechanisms. Each call applies a
single-qubit bit-flip channel (``cudaq::x_error`` in C++, ``cudaq.XError`` in
Python) that applies a Pauli ``X`` with the given probability, so a flipped
data qubit shows up as a parity change in the next ``detectors`` pair. 
See the :doc:`C++ </api/languages/cpp_api>` and 
:doc:`Python </api/languages/python_api>` API references for ``apply_noise``
and the other predefined noise channels.

.. tab:: Python

   .. literalinclude:: ../../snippets/python/using/examples/dem/dem_from_kernel.py
        :language: python
        :start-after: [Begin Kernel]
        :end-before: [End Kernel]

.. tab:: C++

   .. literalinclude:: ../../snippets/cpp/using/examples/dem/dem_from_kernel.cpp
        :language: cpp
        :start-after: [Begin Kernel]
        :end-before: [End Kernel]

Pass the kernel (and a noise model) to ``dem_from_kernel`` to extract the DEM.

.. tab:: Python

   .. literalinclude:: ../../snippets/python/using/examples/dem/dem_from_kernel.py
        :language: python
        :start-after: [Begin Generate]
        :end-before: [End Generate]

.. tab:: C++

   .. literalinclude:: ../../snippets/cpp/using/examples/dem/dem_from_kernel.cpp
        :language: cpp
        :start-after: [Begin Generate]
        :end-before: [End Generate]

The ``.dem`` text is a list of independent error mechanisms. Each
``error(p) D... L...`` line gives one mechanism: its probability ``p``, the
detectors it flips (its *symptoms*, ``D``), and the logical observables it
flips (its *frame changes*, ``L``).

Output DEM: With two rounds and three data qubits, there are six independent
``error`` mechanisms: one bit-flip per qubit per round at the in-kernel
probability ``0.01`` (printed at full floating-point precision). Each error
flips one detector together with the logical observable ``L0``:

.. code-block:: text

    error(0.01000000000000000021) D0 L0
    error(0.01000000000000000021) D1 L0
    error(0.01000000000000000021) D2 L0
    error(0.01000000000000000021) D3 L0
    error(0.01000000000000000021) D4 L0
    error(0.01000000000000000021) D5 L0

DEM Options
-----------

``dem_from_kernel`` accepts optional parameters that are forwarded to the Stim
error analyzer (C++: ``cudaq::dem_options`` struct; Python: keyword arguments).
All options default to ``False`` / ``0``.

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Option
     - Description
   * - ``decompose_errors``
     - Decompose hyper-edge error mechanisms into pairs of two-detector edges.
       Required when feeding the DEM to minimum-weight perfect matching decoders.
   * - ``fold_loops``
     - Fold loop bodies in the circuit for a more compact DEM. CUDA-Q kernels
       are compiled to a flat (loop-free) Stim circuit, so this option has no
       effect in practice. 
   * - ``allow_gauge_detectors``
     - Allow detectors whose parity is not determined by the circuit.
   * - ``approximate_disjoint_errors_threshold``
     - Threshold in [0, 1] for approximating disjoint-error products.
       Set to ``0.0`` (the default) to disable approximation.
   * - ``ignore_decomposition_failures``
     - When decomposition fails for an error mechanism, insert it into the DEM
       undecomposed (as a hyper-edge) instead of raising an exception. Only
       relevant when ``decompose_errors`` is ``True``.
   * - ``block_decomposition_from_introducing_remnant_edges``
     - Prevent the decomposer from introducing remnant edges that would
       otherwise be needed to satisfy the decomposition.

Hyper-edges appear when a single fault trips both an ``X``-type and a ``Z``-type parity
check. The circuit below prepares a Bell pair (the ``+1`` eigenstate of both
``XX`` and ``ZZ``) and measures each stabilizer with its own ancilla. A ``Y``
error on a data qubit anti-commutes with both checks and flips the data
readout, so one mechanism flips three detectors at once. The accompanying
single-qubit ``X`` and ``Z`` errors seed the graph-like edges that the
hyper-edge decomposes into (since ``Y = X · Z``):

.. tab:: Python

   .. literalinclude:: ../../snippets/python/using/examples/dem/dem_from_kernel.py
        :language: python
        :start-after: [Begin Options Kernel]
        :end-before: [End Options Kernel]

.. tab:: C++

   .. literalinclude:: ../../snippets/cpp/using/examples/dem/dem_from_kernel.cpp
        :language: cpp
        :start-after: [Begin Options Kernel]
        :end-before: [End Options Kernel]

Generate the DEM with and without ``decompose_errors``:

.. tab:: Python

   Pass any option as a keyword argument after the kernel arguments:

   .. literalinclude:: ../../snippets/python/using/examples/dem/dem_from_kernel.py
        :language: python
        :start-after: [Begin Options]
        :end-before: [End Options]

.. tab:: C++

   Construct a ``cudaq::dem_options`` value and pass it as the third
   argument (after the noise model):

   .. literalinclude:: ../../snippets/cpp/using/examples/dem/dem_from_kernel.cpp
        :language: cpp
        :start-after: [Begin Options]
        :end-before: [End Options]

Without decomposition the ``Y`` error is a single three-detector hyper-edge
(``D0 D1 D2``), printed alongside the graph-like ``X`` edge (``D0 D2``) and
``Z`` edge (``D1``):

.. code-block:: text

    error(0.02000000000000000042) D0 D1 D2
    error(0.01000000000000000021) D0 D2
    error(0.01000000000000000021) D1

With ``decompose_errors=True`` the hyper-edge is written as the product of
two graph-like components, separated by ``^``:

.. code-block:: text

    error(0.01000000000000000021) D0 D2
    error(0.02000000000000000042) D0 D2 ^ D1
    error(0.01000000000000000021) D1

Measurement Matrices
--------------------

The DEM describes how error mechanisms flip detectors, but decoders often also
need to know how the raw measurement record maps onto the detectors and logical
observables. ``dem_from_kernel`` can return that mapping alongside the DEM text
as two sparse binary *measurement matrices*, ``m2d`` and ``m2o``:

* ``m2d`` has shape ``(num_detectors, num_measurements)``. Entry
  ``m2d[d, m] == 1`` means measurement ``m`` contributes to detector ``d``.
* ``m2o`` has shape ``(num_observables, num_measurements)``. Entry
  ``m2o[k, m] == 1`` means measurement ``m`` contributes to observable ``k``.

In both matrices the columns are indexed by measurement in chronological order.

.. tab:: Python

   Pass ``return_measurement_matrices=True``. The function then returns a
   3-tuple ``(dem_text, m2d, m2o)`` instead of a plain string, where ``m2d``
   and ``m2o`` are ``scipy.sparse.csr_matrix`` objects with binary entries:

   .. literalinclude:: ../../snippets/python/using/examples/dem/dem_from_kernel.py
        :language: python
        :start-after: [Begin Measurement Matrices]
        :end-before: [End Measurement Matrices]

.. tab:: C++

   Use the overloads that accept ``cudaq::M2DSparseMatrix`` and
   ``cudaq::M2OSparseMatrix`` output references. Each ``rows[i]`` lists the
   chronological measurement indices contributing to that detector or
   observable, and ``num_measurements`` gives the column count:

   .. literalinclude:: ../../snippets/cpp/using/examples/dem/dem_from_kernel.cpp
        :language: cpp
        :start-after: [Begin Measurement Matrices]
        :end-before: [End Measurement Matrices]

Both matrices are computed in the same pass as the DEM, so requesting them adds
no additional circuit execution. They can be combined with any of the DEM
options above (for example ``decompose_errors=True``).

Limitations
------------

* **Stabilizer (Clifford) circuits only.** The DEM formalism requires detectors
  to be deterministic under noise-free execution, which is only well defined for
  Clifford circuits; a non-Clifford gate raises a diagnostic.
* **No measurement-conditional control flow.** Branching on a measurement result
  changes the measurement count shot-to-shot and breaks the detector matrix
  model; such kernels are rejected.
* **Independent Pauli noise.** Each error mechanism is assumed independent.
* **Pre-decomposition.** The DEM reflects the abstract kernel circuit, not the
  hardware-decomposed circuit.
