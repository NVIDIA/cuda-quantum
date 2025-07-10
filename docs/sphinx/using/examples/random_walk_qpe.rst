Random Walk Phase Estimation
----------------------------

CUDA-Q allows us to express complex quantum algorithms that incorporate non-trivial control flow and conditional 
quantum instruction invocation. This example demonstrates a quantum phase estimation algorithm that uses a random 
walk approach to iteratively refine the phase estimation.

.. tab:: Python

   The Python implementation demonstrates how to build a quantum kernel with control flow and measurement-dependent 
   operations. The kernel performs a series of controlled rotations and measurements to estimate a phase value.

   .. literalinclude:: ../../examples/python/random_walk_qpe.py
      :language: python

.. tab:: C++

   The C++ implementation shows the same algorithm expressed as a CUDA-Q kernel with a functor structure. 
   Note the similar quantum operations but with C++-specific syntax for control flow and measurements.

   .. literalinclude:: ../../applications/cpp/random_walk_qpe.cpp
      :language: cpp

   To compile and execute this code, run the following:

   .. code:: bash 

      nvq++ random_walk_qpe.cpp -o qpe.x
      ./qpe.x
