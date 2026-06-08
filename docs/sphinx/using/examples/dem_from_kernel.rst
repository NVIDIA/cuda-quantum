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

Limitations
------------

* **Stabilizer (Clifford) circuits only.** The DEM formalism requires detectors
  to be deterministic under noise-free execution, which is only well defined for
  Clifford circuits; a non-Clifford gate raises a diagnostic.
* **No measurement-conditional control flow.** Branching on a measurement result
  changes the measurement count shot-to-shot and breaks the detector matrix
  model; such kernels are rejected.
* **Independent Pauli noise.** Each error mechanism is assumed independent..
* **Pre-decomposition.** The DEM reflects the abstract kernel circuit, not the
  hardware-decomposed circuit.
