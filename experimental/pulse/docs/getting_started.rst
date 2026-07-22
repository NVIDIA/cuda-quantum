.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Getting Started
===============

.. note::

   cudaq-pulse is **experimental**. APIs may change without notice and carry
   no stability guarantee.

Prerequisites
-------------

- Python 3.10+
- numpy

For building the MLIR bindings from source:

- LLVM/MLIR 22+
- CMake 3.24+
- Ninja
- nanobind

For GPU simulation (preview, in active porting):

- NVIDIA GPU with compute capability 7.0+
- cuDensityMat (part of the cuQuantum SDK)

Installation
------------

cudaq-pulse lives in ``experimental/pulse`` within CUDA-Q. Build it from
source:

.. code-block:: bash

   cd experimental/pulse
   mkdir build && cd build
   cmake .. -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
     -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
   ninja

Then put the Python frontend and the built bindings on your ``PYTHONPATH``:

.. code-block:: bash

   export PYTHONPATH=core/frontend:build/core/mlir/bindings

Preview wheels (``pip install cudaq-pulse``) may be published later.

Hello World
-----------

Define a pulse kernel, compile it, and inspect the generated MLIR:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def rabi_oscillation(qubit):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(64, 0.5, 16.0), tone)

   compiled_kernel = pulse.compile(rabi_oscillation, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5.0e9})
   print(compiled_kernel.mlir)
   print(f"Compiled in {compiled_kernel.metrics.total_ms:.1f} ms")

The Compiler Pipeline
---------------------

cudaq-pulse is a Python-first compiler pipeline with four stages:

1. **Write a kernel in Python** -- the ``@pulse.kernel`` DSL
   (``get_drive_line``, ``drive``, ``gaussian``, ``wait``, ``sync``, ...).
2. **Compile to MLIR** -- ``pulse.compile()`` returns a ``CompiledKernel``
   whose ``.mlir`` is the scheduled Pulse dialect. See :doc:`user_guide/compilation`.
3. **Write transform passes in Python and apply them** -- passes are plain
   ``Program -> Program`` functions; compose the built-ins or author your own.
   See :doc:`user_guide/passes`.
4. **Emit** -- lower the transformed program back to MLIR with
   ``program_to_pulse_mlir``.

GPU simulation via NVIDIA cuDensityMat is a preview capability; see
:doc:`user_guide/gpu_execution`.

IDE Setup
---------

For the best experience, add a ``pyrightconfig.json`` to your project root:

.. code-block:: json

   {
       "reportUndefinedVariable": "warning"
   }

This downgrades bare-name diagnostics inside ``@pulse.kernel`` functions
from errors to warnings. See :doc:`user_guide/kernels` for details.

Running Tests
-------------

.. code-block:: bash

   cd experimental/pulse
   PYTHONPATH=core/frontend:build/core/mlir/bindings python -m pytest tests/ -q
