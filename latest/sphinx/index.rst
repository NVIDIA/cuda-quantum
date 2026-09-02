************
CUDA-Q
************

Welcome to the CUDA-Q documentation page!

CUDA-Q streamlines hybrid application development and promotes productivity and scalability
in quantum computing. It offers a unified programming model designed for a hybrid
setting |---| that is, CPUs, GPUs, and QPUs working together. CUDA-Q contains support for 
programming in Python and in C++.

You are browsing the documentation for |version| version of CUDA-Q. You can find documentation for all released versions :doc:`here <versions>`.

**CUDA-Q** is a programming model and toolchain for using quantum acceleration in heterogeneous computing architectures available in C++ and Python.

Research Preview: Pulse-Level Programming
-----------------------------------------

CUDA-Q pulse is a new research-preview package for programming at the pulse level. It provides a Python
kernel DSL, pulse and operator dialects, compiler passes, and an experimental GPU execution path. Because it
is a research preview, its APIs and behavior may change incompatibly or be removed without notice, and it is
not a product-supported CUDA-Q feature.

See the `CUDA-Q pulse README <https://github.com/NVIDIA/cuda-quantum/blob/feature/pulse/pulse/README.md>`__ for a
quick example, build instructions, and the current scope and limitations.

.. toctree::
   :caption: Contents
   :maxdepth: 2

      Quick Start <using/quick_start.rst>
      Basics <using/basics/basics.rst>
      Examples <using/examples/examples.rst>
      Applications <using/applications.rst>
      Backends <using/backends/backends.rst>
      Dynamics <using/dynamics.rst>
      Realtime <using/realtime.rst>
      CUDA-QX <using/cudaqx/cudaqx.rst>
      Installation <using/install/install.rst>
      Integration <using/integration/integration.rst>
      Extending <using/extending/extending.rst>
      Specifications <specification/index.rst>
      API Reference <api/api.rst>
      Other Versions <versions.rst>

.. |---|   unicode:: U+2014 .. EM DASH
   :trim:
