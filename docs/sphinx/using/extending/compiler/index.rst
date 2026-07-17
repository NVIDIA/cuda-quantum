CUDA-Q compiler development
***************************

.. toctree::
   :hidden:
   :maxdepth: 1

   Compiler IR <cudaq_ir>
   CUDA-Q dialect documentation <dialect_reference>
   External compiler pass plugins <pass_plugins>

CUDA-Q compiles C++ and Python quantum kernels using the
`MLIR compiler infrastructure <https://mlir.llvm.org/>`_. The language frontends
construct mixed-dialect MLIR modules. Compiler passes analyze, transform, and
lower those modules before the compiler translates or emits the representation
needed for local execution or the selected backend.

The C++ frontend uses Clang to find quantum kernels and builds MLIR for them
while Clang emits LLVM IR for the surrounding host program. The Python language
frontend constructs equivalent MLIR through its Python bridge, either by
lowering a decorated Python AST or by using the kernel builder. Both Python
paths run a target-independent preparation pipeline before a kernel is compiled
for execution.

These modules use several dialects rather than representing kernels as sequences
of standalone circuit instructions. :doc:`Quake </_mdgen/Dialects/Quake>`
represents quantum operations and values, :doc:`CC </_mdgen/Dialects/CC>`
represents classical constructs needed by CUDA-Q kernels, and the `upstream MLIR
dialects <https://mlir.llvm.org/docs/Dialects/>`_ provide functions, arithmetic,
control flow, and lower-level forms.

CUDA-Q registers individual passes and reusable pass pipelines with MLIR's
`pass infrastructure <https://mlir.llvm.org/docs/PassManagement/>`_. CUDA-Q
normally builds each target's compilation pipeline from shared compiler steps
and target-specific lowering. A target can instead supply a complete pass
pipeline.

.. rubric:: At a glance

* Read :doc:`cudaq_ir` to understand the dialects and IR forms that a
  transformation consumes or produces. For Quake's reference and value models,
  see the :doc:`Quake semantic specification
  <../../../specification/quake-dialect>`.
* Browse :doc:`dialect_reference` for the generated operation and type
  documentation for the CUDA-Q dialects.
* See :doc:`pass_plugins` for the existing external pass plugin interface.

.. rubric:: Code organization

The C++ AST bridge is under ``cudaq/lib/Frontend/nvqpp`` and is driven by
``cudaq-quake``. The Python AST bridge and builder are
``python/cudaq/kernel/ast_bridge.py`` and
``python/cudaq/kernel/kernel_builder.py``.

Quake, CC, and QEC declarations are under
``cudaq/include/cudaq/Optimizer/Dialect``. Code generation helper declarations
are under ``cudaq/include/cudaq/Optimizer/CodeGen``. Built-in transformations
and lowering passes are implemented under ``cudaq/lib/Optimizer/Transforms``
and ``cudaq/lib/Optimizer/CodeGen``. Their shared pipelines are defined in the
corresponding ``Pipelines.cpp`` files.

``cudaq-opt`` parses and runs registered MLIR passes. ``cudaq-translate`` owns
the standalone translation path, while ``cudaq-target-conf`` reads target
configuration for the C++ driver. Representative lit tests are grouped under
``cudaq/test/AST-Quake``, ``cudaq/test/Transforms``, and
``cudaq/test/Translate``. Python MLIR regression tests are under
``python/tests/mlir``, with broader frontend behavior tested under
``python/tests/kernel``.
