Troubleshooting
-----------------

Debugging and Verbose Simulation Output
+++++++++++++++++++++++++++++++++++++++++

One helpful mechanism of debugging CUDA-Q simulation execution is
the :code:`CUDAQ_LOG_LEVEL` environment variable. For any CUDA-Q
executable, just prepend this and turn it on:

.. tab:: Python

  .. code-block:: bash

      CUDAQ_LOG_LEVEL=info python3 file.py

.. tab:: C++

    .. code-block:: bash

      CUDAQ_LOG_LEVEL=info ./a.out

Similarly, one may write the IR to their console or to a file before remote
submission. This may be done through the :code:`CUDAQ_DUMP_JIT_IR` environment
variable. For any CUDA-Q executable, just prepend as follows:

.. tab:: Python

  .. code-block:: bash

      CUDAQ_DUMP_JIT_IR=1 python3 file.py
      # or
      CUDAQ_DUMP_JIT_IR=<output_filename> python3 file.py

.. tab:: C++

  .. code-block:: bash

      CUDAQ_DUMP_JIT_IR=1 ./a.out
      # or
      CUDAQ_DUMP_JIT_IR=<output_filename> ./a.out

Python Stack-Traces
++++++++++++++++++++++++

CUDA-Q does not parse command-line options when it is imported. To use CUDA-Q
runtime options, call :func:`cudaq.parse_args` explicitly before running the
rest of the program.

.. code-block:: python

    import cudaq

    cudaq.parse_args()

When :func:`cudaq.parse_args` processes Python command-line options, Python
stack-traces are suppressed by default to keep runtime errors concise. To show
the full stack-trace for debugging, pass
:code:`--cudaq-full-stack-trace` when invoking your script.

.. code-block:: bash

    python3 program.py --cudaq-full-stack-trace

This flag can be combined with other CUDA-Q Python runtime options such as
:code:`--target`, :code:`--target-option`, and :code:`--emulate`.

.. code-block:: bash

    python3 program.py --target nvidia --target-option fp64 --cudaq-full-stack-trace

LLVM-based Python packages
++++++++++++++++++++++++++

Older CUDA-Q releases can expose their bundled LLVM symbols to other Python
extensions. On Linux, importing Numba or Triton after CUDA-Q can then abort the
process during JIT compilation or report duplicate LLVM command-line options.
For affected releases, import Numba or Triton before importing CUDA-Q. If the
application cannot control import order, run CUDA-Q in a separate process.

CUDA-Q keeps its LLVM-dependent runtime and backend libraries local. Importing
``cudaq`` exposes only the LLVM-free kernel runtime (``libnvqir``,
``libcudaq-common``, ``libcudaq-kernel-runtime``, and their dependencies) globally.
This allows shared libraries containing ``nvq++``-compiled kernels to be loaded
with ``dlopen`` or ``ctypes.CDLL`` after importing CUDA-Q, even when compiled with
``nvq++ -c -fPIC`` and linked without explicit CUDA-Q dependencies. Kernel launches
use the active CUDA-Q target without exposing the compiler's LLVM symbols
globally.

External native extensions using C++ runtime APIs must still link to the CUDA-Q
libraries they use, including ``cudaq::cudaq-em-default`` when using the default
execution manager. Compiler plugins should link to ``cudaq::MLIR``. They must not
rely on ``import cudaq`` placing those dependencies in the process-wide symbol
scope.
