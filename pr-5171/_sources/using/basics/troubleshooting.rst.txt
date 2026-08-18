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

When CUDA-Q parses Python command-line options via :func:`cudaq.parse_args`,
Python stack-traces are suppressed by default to keep runtime errors concise.
To show the full stack-trace for debugging, pass
:code:`--cudaq-full-stack-trace` when invoking your script.

.. code-block:: bash

    python3 program.py --cudaq-full-stack-trace

This flag can be combined with other CUDA-Q Python runtime options such as
:code:`--target`, :code:`--target-option`, and :code:`--emulate`.

.. code-block:: bash

    python3 program.py --target nvidia --target-option fp64 --cudaq-full-stack-trace

If your application parses CUDA-Q command-line arguments explicitly, call
:func:`cudaq.parse_args` before running the rest of the program so the flag is
recognized.
