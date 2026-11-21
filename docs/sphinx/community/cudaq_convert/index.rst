:repository: https://github.com/QuantumRaul/cudaq-convert
:maintainers: Raul Martinez
:license: Apache-2.0
:install: cudaq-convert==0.1.0
:example: example.py
:summary: Convert OpenQASM 2.0/3.0 programs and Qiskit circuits into CUDA-Q kernels.

cudaq-convert
=============

Overview
--------

``cudaq-convert`` converts quantum circuits written in **OpenQASM 2.0/3.0** or
built with **Qiskit** into CUDA-Q kernels, so existing circuits can be
simulated and extended with CUDA-Q without rewriting them by hand.

The OpenQASM translator is written in pure Python and does not depend on
Qiskit. It dispatches on the ``OPENQASM <version>;`` header and supports:

- **OpenQASM 2.0**: the full ``qelib1.inc`` gate set, user-defined ``gate``
  blocks (expanded recursively), register broadcasting, and
  ``measure``/``reset``/``barrier``/``opaque``.
- **OpenQASM 3.0**: ``qubit[N]``/``bit[N]`` declarations, ``c = measure q;``
  assignments, the built-in ``U(...)`` and ``gphase``, and
  ``include "stdgates.inc";``.
- **Qiskit**: the full standard gate library. Custom and composite gates fall
  back to their ``definition``, expanded recursively.

The public API has three functions:

- ``from_qasm(path)``: parses an OpenQASM 2.0/3.0 file into a CUDA-Q kernel.
- ``from_qasm_str(source)``: parses OpenQASM 2.0/3.0 source text.
- ``from_qiskit(circuit)``: converts a Qiskit ``QuantumCircuit`` into a CUDA-Q
  kernel.

These functions replace the deprecated ``cudaq.contrib.from_qiskit``.

Installation
------------

.. code-block:: bash

   pip install cudaq-convert

The package requires ``cudaq >= 0.14.2``, ``qiskit >= 1.0`` and Python 3.11
to 3.13.

Example
-------

The following example converts an OpenQASM 2.0 program and a Qiskit circuit
into CUDA-Q kernels, then samples them.

.. literalinclude:: example.py
   :language: python
   :start-after: [Begin Documentation]
   :end-before: [End Documentation]

Known limitations
-----------------

- OpenQASM 3.0 gate modifiers (``ctrl @``, ``negctrl @``, ``inv @``,
  ``pow(n) @``) and classical control flow (``if``, ``for``, ``while``,
  ``def``) are not supported yet. They raise ``NotImplementedError``.
- CUDA-Q bitstrings are big-endian (qubit 0 is the leftmost bit), while Qiskit
  bitstrings are little-endian. Reverse the bitstring when comparing results
  between the two.
