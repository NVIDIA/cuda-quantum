Bernstein-Vazirani
--------------------------------

The Bernstein-Vazirani algorithm aims to identify the bitstring encoded in a given function. 

For the original source of this algorithm, see 
`this publication <https://epubs.siam.org/doi/10.1137/S0097539796300921>`__.

In this example, we generate a random bitstring and encode it into an inner-product oracle, 
and define a kernel for the Bernstein-Vazirani algorithm.  Then, we simulate the kernel and return the most probable bitstring from its execution.


If all goes well, the state measured with the highest probability should be our hidden bitstring!

.. tab:: Python

   .. literalinclude:: ../../examples/python/bernstein_vazirani.py
      :language: python

.. tab:: C++

   .. literalinclude:: ../../examples/cpp/basics/bernstein_vazirani.cpp
      :language: cpp
