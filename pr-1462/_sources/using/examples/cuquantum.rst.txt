Simulations with cuQuantum
-----------------------------------

CUDA-Q provides support for cuQuantum-accelerated state vector and tensor network 
simulations. Let's take a look at an example that is too large for a standard CPU-only simulator, but 
can be trivially simulated via a NVIDIA GPU-accelerated backend:

.. tab:: Python

   .. literalinclude:: ../../examples/python/cuquantum_backends.py
      :language: python

   Here we generate a GHZ state on 28 qubits. The built-in cuQuantum state vector
   backend is selected by default if a local GPU is present. Alternatively, the
   target may be manually set through the `cudaq.set_target("nvidia")` command.

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/basics/cuquantum_backends.cpp
      :language: cpp

   Here we generate a GHZ state on 28 qubits. To run with the built-in cuQuantum state 
   vector support, we pass the :code:`--target nvidia` flag at compile time:

   .. code:: bash 

      nvq++ --target nvidia cuquantum_backends.cpp -o ghz.x
      ./ghz.x

   Alternatively, we can set the environment variable `CUDAQ_DEFAULT_SIMULATOR` to `nvidia`.