.. Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.
   SPDX-License-Identifier: Apache-2.0

cudaq-pulse
===========

**cudaq-pulse** is a pulse-level MLIR dialect and programming model for
quantum control. The core workflow is a compiler pipeline entirely in Python:

1. Write a pulse kernel with the ``@pulse.kernel`` DSL.
2. Compile it to Pulse-dialect MLIR with ``pulse.compile()``.
3. Write and apply transform passes over the pulse program.
4. Emit MLIR (and lower further) from the transformed program.

.. note::

   cudaq-pulse is **experimental**. APIs may change without notice and carry
   no stability guarantee.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/kernels
   user_guide/operations
   user_guide/compilation
   user_guide/passes

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/pipeline
   architecture/dialects
   architecture/mlir_passes

.. toctree::
   :maxdepth: 1
   :caption: Resources

   examples

.. toctree::
   :maxdepth: 1
   :caption: Preview / Experimental

   user_guide/gpu_execution


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
