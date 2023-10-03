
Debugging and Verbose Simulation Output
---------------------------------------
One helpful mechanism of debugging CUDA Quantum simulation execution is 
the :code:`CUDAQ_LOG_LEVEL` environment variable. For any CUDA Quantum 
executable, just prepend this and turn it on:

.. code-block:: console 

  CUDAQ_LOG_LEVEL=info ./a.out 

This will work for both codes in C++ and Python. 
