.. _external-compiler-pass-plugins:

External compiler pass plugins
******************************

``cudaq-opt`` can load a custom MLIR operation pass from a shared library. This
keeps the pass outside CUDA-Q's built-in pass catalog and production pipelines
while allowing it to transform CUDA-Q IR with the same MLIR APIs used by
built-in passes. See :doc:`Developing compiler passes <mlir_pass>` for guidance
on choosing an operation anchor, defining the accepted IR, and testing a
transformation.

CUDA-Q currently builds and tests pass plugins within a CUDA-Q development
build. A plugin must use CUDA-Q, LLVM, and MLIR headers and libraries compatible
with the ``cudaq-opt`` binary that loads it. Rebuild the plugin when those
dependencies change.

Implement and register the pass
===============================

A plugin pass is an MLIR operation pass with a textual argument. Include
``cudaq/Support/Plugin.h`` and place ``CUDAQ_REGISTER_MLIR_PASS`` at file scope
after the pass definition. The macro exports the CUDA-Q plugin entry point and
registers one default-constructible pass.

The complete example below is included from the source compiled by the plugin
test target. It replaces each ``quake.h`` operation with
``quake.s`` so that the regression can observe the transformation. This example
demonstrates plugin registration and does not preserve circuit semantics.

.. literalinclude:: ../../../../../cudaq/test/plugin/CustomPassPlugin.cpp
   :language: cpp

Build the plugin
================

The tested CMake target uses LLVM's pass-plugin helper and depends on the
generated Quake dialect headers:

.. :spellcheck-disable:

.. literalinclude:: ../../../../../cudaq/test/plugin/CMakeLists.txt
   :language: cmake

Build that target from a configured CUDA-Q build tree:

.. code:: bash

   cmake --build build --target CustomPassPlugin

.. :spellcheck-enable:

Load and test the plugin
========================

Load the shared library before naming its registered pass. This Linux command
uses the paths produced by the in-tree build:

.. :spellcheck-disable:

.. code:: bash

   build/bin/cudaq-opt input.qke \
     --load-cudaq-plugin build/lib/CustomPassPlugin.so \
     --cudaq-custom-pass

.. :spellcheck-enable:

A corresponding regression test loads the plugin into ``cudaq-opt`` and checks
the transformed IR. The pass is registered only for that invocation and is not
added to a CUDA-Q compilation pipeline.
