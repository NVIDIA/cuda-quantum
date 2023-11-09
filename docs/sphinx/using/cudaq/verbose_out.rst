
Debugging and Verbose Simulation Output
---------------------------------------
One helpful mechanism of debugging CUDA Quantum simulation execution is
the :code:`CUDAQ_LOG_LEVEL` environment variable. For any CUDA Quantum
executable, just prepend this and turn it on:

.. code-block:: console

    CUDAQ_LOG_LEVEL=info ./a.out
    # python
    CUDAQ_LOG_LEVEL=info python3 file.py

Similarly, one may write the IR to their console or to a file before remote
submission. This may be done through the :code:`CUDAQ_DUMP_JIT_IR` environment
variable. For any CUDA Quantum executable, just prepend as follows:

.. code-block:: console

    CUDAQ_DUMP_JIT_IR=1 ./a.out
    # or
    CUDAQ_DUMP_JIT_IR=<filename> ./a.out

These will work for both codes in C++ and Python.
