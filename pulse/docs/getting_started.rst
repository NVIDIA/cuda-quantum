..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Getting Started
===============

.. warning::

   CUDA-Q pulse is **research-preview software**, not production software or a
   product-supported CUDA-Q feature. APIs, IR, runtime behavior, build options,
   and numerical behavior may change incompatibly or be removed without
   notice. No stability or production-readiness guarantee is provided.

Prerequisites
-------------

- Python 3.10+
- numpy

For building the MLIR bindings from source:

- The ``cudaq`` and ``cudaq-devel`` wheels, which supply the CUDA-Q and
  LLVM/MLIR toolchain
- nanobind 2.12+
- CMake and Ninja
- pytest, Hypothesis, and LLVM lit for tests

For GPU simulation (research preview):

- NVIDIA GPU with compute capability 7.0+
- cuDensityMat (part of the cuQuantum SDK)

Build from Source
-----------------

CUDA-Q pulse lives in the top-level ``pulse`` directory but is a standalone
CMake project: it is not part of the CUDA-Q build. It compiles against the
CUDA-Q toolchain distributed as Python wheels, so there is no LLVM to build and
no submodule to initialize.

- ``cudaq-devel`` provides the headers, CMake packages, MLIR/LLVM archives and
  ``mlir-tblgen``.
- ``cudaq`` (the runtime wheel) provides ``libcudaqMLIR``, the single shared
  MLIR/LLVM instance the pulse Python extension resolves its symbols from.

Both wheels must come from the same CUDA-Q revision.

.. :spellcheck-disable:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cuda-quantum.git
   cd cuda-quantum

   python3 -m venv .venv-pulse
   source .venv-pulse/bin/activate
   python -m pip install cudaq cudaq-devel
   python -m pip install "nanobind>=2.12" cmake ninja pytest hypothesis lit numpy

   cmake -S pulse -B build-pulse -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build-pulse --parallel

.. :spellcheck-enable:

Note the ``-S pulse``: the configure entry point is the ``pulse`` directory,
not the repository root.

Pulse locates the wheels through ``site.getsitepackages()`` of the interpreter
CMake picks up, so run CMake from the environment the wheels were installed
into, or pass ``-DPython3_EXECUTABLE=/path/to/venv/bin/python``. To build
against a CUDA-Q installation that is not a wheel, point CMake at it with
``-DCMAKE_PREFIX_PATH=/path/to/cudaq/prefix``. nanobind is discovered the same
way and can be overridden with
``-Dnanobind_DIR="$(python -m nanobind --cmake_dir)"``.

The explicit ``--target pulse`` aggregate target remains available for scripts
that prefer a named target.

Like CUDA-Q itself, the complete pulse package is staged under
``build-pulse/python``. Put that single directory on your ``PYTHONPATH``:

.. :spellcheck-disable:

.. code-block:: bash

   export PATH="$PWD/build-pulse/bin:$PATH"
   export PYTHONPATH="$PWD/build-pulse/python${PYTHONPATH:+:$PYTHONPATH}"

.. :spellcheck-enable:

Build the cuDensityMat GPU Runtime
----------------------------------

The experimental GPU runtime is not needed for compiler-only use. Point
``CUDENSITYMAT_ROOT`` at a cuQuantum SDK containing
``include/cudensitymat.h`` and ``lib/libcudensitymat.so``. CMake then discovers
cuDensityMat and automatically adds the runtime to the ``pulse`` target; there
is no second pulse feature flag:

.. :spellcheck-disable:

.. code-block:: bash

   cmake -S pulse -B build-gpu -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDAQ_BUILD_TESTS=ON \
     -DCUDENSITYMAT_ROOT=/path/to/cuquantum \
     -DPython3_EXECUTABLE="$PWD/.venv-pulse/bin/python"
   cmake --build build-gpu --parallel --target pulse
   export CUDAQ_PULSE_BUILD_DIR="$PWD/build-gpu"
   export PATH="$PWD/build-gpu/bin:$PATH"
   export PYTHONPATH="$PWD/build-gpu/python${PYTHONPATH:+:$PYTHONPATH}"

.. :spellcheck-enable:

When ``CUDENSITYMAT_ROOT`` is set, configuration fails if the CUDA Toolkit or
cuDensityMat dependency cannot be found. Without cuDensityMat, pulse remains
usable in compiler-only mode. The runtime links the discovered SDK library and
records its location in the runtime search path.

Hello World
-----------

Define a pulse kernel, compile it, and inspect the generated MLIR:

.. :spellcheck-disable:

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

.. :spellcheck-enable:

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

.. :spellcheck-disable:

.. code-block:: json

   {
       "reportUndefinedVariable": "warning"
   }

.. :spellcheck-enable:

This downgrades bare-name diagnostics inside ``@pulse.kernel`` functions
from errors to warnings. See :doc:`user_guide/kernels` for details.

Running Tests
-------------

.. :spellcheck-disable:

.. code-block:: bash

   cmake --build build-pulse --target check-pulse
   ctest --test-dir build-pulse -L pulse --output-on-failure

.. :spellcheck-enable:

For a GPU-enabled build, run the cuDensityMat linkage, descriptor, and
numerical regression tests:

.. :spellcheck-disable:

.. code-block:: bash

   cmake --build build-gpu --target check-pulse-gpu

.. :spellcheck-enable:

The checks validate SDK linkage and basic descriptors, then exercise
single-qubit drive evolution, T1 decay, two-qubit XX coupling, and the public
compile/JIT path. CMake enables the numerical tests only when ``nvidia-smi``
reports a GPU and the CUDA runtime can access at least one device. Otherwise
the target reports them disabled and retains the CPU-safe SDK linkage check
when cuDensityMat is installed. These are research regression tests, not a
production numerical-accuracy qualification.

Docker Environment
------------------

Build the turnkey image from the CUDA-Q repository root:

.. :spellcheck-disable:

.. code-block:: bash

   docker build -f pulse/docker/Dockerfile -t cudaq-pulse-preview .
   docker run --rm -it cudaq-pulse-preview

.. :spellcheck-enable:

The Dockerfile installs the ``cudaq`` and ``cudaq-devel`` wheels into a
virtual environment and builds pulse against them; it does not rebuild LLVM.

Documentation
-------------

Enable the Sphinx target when configuring pulse, then build it:

.. :spellcheck-disable:

.. code-block:: bash

   cmake -S pulse -B build-pulse -G Ninja \
     -DCUDAQ_PULSE_BUILD_DOCS=ON
   cmake --build build-pulse --target pulse-docs

.. :spellcheck-enable:

Generated HTML is written to ``build-pulse/docs/html``.
