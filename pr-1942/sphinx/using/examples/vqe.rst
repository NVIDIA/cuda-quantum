Variational Quantum Eigensolver
--------------------------------

The Variational Quantum Eigensolver (VQE) algorithm, originally proposed in
`this publication <https://arxiv.org/abs/1304.3061>`__, 
is a hybrid algorithm that can make use of both quantum and classical resources.

Let's take a look at how we can use CUDA-Q's built-in `vqe` module to run our own custom VQE routines! 
Given a parameterized quantum kernel, a system spin Hamiltonian, and one of CUDA-Q's optimizers, 
`cudaq.vqe` will find and return the optimal set of parameters that minimize the energy, <Z>, of the system.

The code block below represents the contents of a file titled `simple_vqe.py`. 

.. tab:: Python

   .. literalinclude:: ../../examples/python/simple_vqe.py
      :language: python

.. tab:: C++
   
   .. literalinclude:: ../../examples/cpp/algorithms/vqe_h2.cpp
      :language: cpp

Let's look at a more advanced variation of the previous example.

As an alternative to `cudaq.vqe`, we can also use the `cudaq.optimizers` suite on its own to write custom variational algorithm routines. Much of this can be slightly modified for use with third-party optimizers, such as `scipy`.

.. literalinclude:: ../../examples/python/advanced_vqe.py
   :language: python