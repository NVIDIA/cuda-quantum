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

- The standard CUDA-Q source-build dependencies and git submodules
- LLVM/MLIR from the CUDA-Q toolchain
- CMake and Ninja
- pytest, Hypothesis, and LLVM lit for tests

For GPU simulation (research preview):

- NVIDIA GPU with compute capability 7.0+
- cuDensityMat (part of the cuQuantum SDK)

Build from Source
-----------------

CUDA-Q pulse lives in the top-level ``pulse`` directory and is integrated into
the CUDA-Q CMake build. It is disabled by default. Clone CUDA-Q with submodules,
build the exact LLVM revision pinned by that checkout, then enable the research
package explicitly with ``CUDAQ_ENABLE_PROJECTS=pulse``. Pulse is deliberately
excluded from the stable ``CUDAQ_ALL_PROJECTS`` catalog; selecting it alone
configures pulse without the stable CUDA-Q projects.

.. code-block:: bash

   git clone --recursive https://github.com/NVIDIA/cuda-quantum.git
   cd cuda-quantum

   export LLVM_SOURCE="$PWD/tpls/llvm"
   export LLVM_INSTALL_PREFIX="$PWD/.cudaq-llvm"
   export LLVM_PROJECTS="clang;lld;mlir"
   bash scripts/build_llvm.sh -j "$(nproc)"

   python3 -m venv .venv-pulse
   source .venv-pulse/bin/activate
   python -m pip install pytest hypothesis lit numpy
   cmake -S . -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDAQ_ENABLE_PROJECTS=pulse \
     -DLLVM_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/llvm"
   cmake --build build --parallel

Pulse does not need LLVM's Python bindings, so the LLVM build above omits them.
No separate nanobind installation or prefix is required: like CUDA-Q's Python
build, pulse uses ``tpls/nanobind/cmake`` from the recursive checkout by
default. To use the CMake package shipped in a pip-installed nanobind wheel
instead:

.. code-block:: bash

   python -m pip install nanobind
   cmake -S . -B build -G Ninja \
     -DCUDAQ_ENABLE_PROJECTS=pulse \
     -DLLVM_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/llvm" \
     -Dnanobind_DIR="$(python -m nanobind --cmake_dir)"

For this pulse-only configuration, the default build contains only pulse. It
does not build stable CUDA-Q utilities such as ``CircuitCheck`` or their test
harnesses. The explicit ``--target pulse`` aggregate target remains available
for scripts that prefer a named target.

Like CUDA-Q itself, the complete pulse package is staged under
``build/python``. Put that single directory on your ``PYTHONPATH``:

.. code-block:: bash

   export PATH="$PWD/build/bin:$PATH"
   export PYTHONPATH="$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}"

To build pulse alongside stable CUDA-Q projects, list each desired project:

.. code-block:: bash

   cmake -S . -B build -G Ninja \
     -DCUDAQ_ENABLE_PROJECTS="cudaq;runtime;python;pulse" \
     -DLLVM_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/llvm"
   cmake --build build --parallel

Build the cuDensityMat GPU Runtime
----------------------------------

The experimental GPU runtime is not needed for compiler-only use. Point
``CUDENSITYMAT_ROOT`` at a cuQuantum SDK containing
``include/cudensitymat.h`` and ``lib/libcudensitymat.so``. CMake then discovers
cuDensityMat and automatically adds the runtime to the ``pulse`` target; there
is no second pulse feature flag:

.. code-block:: bash

   cmake -S . -B build-gpu -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DCUDAQ_BUILD_TESTS=ON \
     -DCUDAQ_ENABLE_PROJECTS=pulse \
     -DCUDENSITYMAT_ROOT=/path/to/cuquantum \
     -DLLVM_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/llvm" \
     -DPython_EXECUTABLE="$PWD/.venv-pulse/bin/python"
   cmake --build build-gpu --parallel --target pulse
   export CUDAQ_PULSE_BUILD_DIR="$PWD/build-gpu"
   export PATH="$PWD/build-gpu/bin:$LLVM_INSTALL_PREFIX/bin:$PATH"
   export PYTHONPATH="$PWD/build-gpu/python${PYTHONPATH:+:$PYTHONPATH}"

When ``CUDENSITYMAT_ROOT`` is set, configuration fails if the CUDA Toolkit or
cuDensityMat dependency cannot be found. Without cuDensityMat, pulse remains
usable in compiler-only mode. The runtime links the discovered SDK library and
records its location in the runtime search path.

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

   cmake --build build --target check-pulse
   ctest --test-dir build -L pulse --output-on-failure

For a GPU-enabled build, run the cuDensityMat linkage, descriptor, and
numerical regression tests:

.. code-block:: bash

   cmake --build build-gpu --target check-pulse-gpu

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

.. code-block:: bash

   docker build -f pulse/docker/Dockerfile -t cudaq-pulse-preview .
   docker run --rm -it cudaq-pulse-preview

The Dockerfile inherits CUDA-Q's ``llvm-main`` development image, where the
pinned LLVM/MLIR toolchain is already installed; it does not rebuild LLVM.

Documentation
-------------

Enable the Sphinx target when configuring CUDA-Q, then build it:

.. code-block:: bash

   cmake -S . -B build -G Ninja \
     -DCUDAQ_ENABLE_PROJECTS=pulse \
     -DCUDAQ_PULSE_BUILD_DOCS=ON \
     -DLLVM_DIR="$LLVM_INSTALL_PREFIX/lib/cmake/llvm"
   cmake --build build --target pulse-docs

Generated HTML is written to ``build/pulse/docs/html``.
