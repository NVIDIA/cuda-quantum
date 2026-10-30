:repository: https://github.com/eclipse-qrisp/Qrisp
:maintainers: The Eclipse Qrisp developers
:license: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
:install: qrisp[cudaq]
:example: example.py
:summary: High-level quantum programming framework whose programs compile to native CUDA-Q kernels.

Eclipse Qrisp
=============

Overview
--------

`Eclipse Qrisp <https://www.qrisp.eu>`__ is a high-level quantum programming
framework. Programs are written with typed quantum variables such as
``QuantumFloat`` and ``QuantumModulus`` instead of individual qubits and gates.
Arithmetic on them uses ordinary Python operators such as ``+=`` and ``*``.
Matrices such as Hamiltonians are represented as block encodings, the building
blocks of Hamiltonian simulation and other algorithms based on quantum signal
processing.

The ``@cudaq_kernel`` decorator from the
`qrisp-cudaq <https://github.com/eclipse-qrisp/qrisp-cudaq>`__ backend compiles
a Qrisp function to the Quake MLIR dialect and returns a native CUDA-Q kernel.
The kernel is passed to ``cudaq.run`` like a kernel defined with
``@cudaq.kernel`` and runs on CUDA-Q's simulators, including the
GPU-accelerated state-vector simulator. Kernels that do not use measurement
results inside the kernel can be compiled with
``@cudaq_kernel(execution_mode="sample")`` and passed to ``cudaq.sample``.

The `Qrisp CUDA-Q tutorial <https://qrisp.eu/general/tutorial/CUDAQ.html>`__
and API reference are part of the Qrisp documentation at
`qrisp.eu <https://www.qrisp.eu>`__.

Installation
------------

.. code-block:: bash

   pip install "qrisp[cudaq]"

This installs Qrisp together with qrisp-cudaq and CUDA-Q. It requires Python
3.11 to 3.13 on Linux (x86_64 and aarch64) or on macOS with Apple silicon.

Example
-------

The kernel below adds two ``QuantumFloat`` variables. The first, ``a``, is put
in a uniform superposition of 0 to 7. The second, ``b``, is initialized to the
integer ``k`` passed at runtime, here 5. The in-place addition ``b += a``
entangles the two variables, so each shot returns a random value of ``a``
together with ``b = a + k``.

.. literalinclude:: example.py
   :language: python
   :start-after: [Begin Documentation]
   :end-before: [End Documentation]
