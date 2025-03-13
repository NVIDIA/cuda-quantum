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