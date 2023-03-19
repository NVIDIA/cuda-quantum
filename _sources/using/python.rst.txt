
CUDA Quantum in Python
======================

Examples for running CUDA Quantum programs written in Python on the statevector simulator included in the `cudaq` package.

Introduction
--------------------------------

Welcome to CUDA Quantum! We're going to take a look at how to construct quantum programs through CUDA Quantum's `Kernel` API.

When you create a `Kernel` and invoke its methods, a quantum program is constructed that can then be executed by calling, for example, `cudaq::sample`. Let's take a closer look!

.. literalinclude:: ../examples/python/intro.py
   :language: python

Bernstein-Vazirani
--------------------------------

Bernstein Vazirani is an algorithm for finding the bitstring encoded in a given function. 

For the original source of this algorithm, see: https://epubs.siam.org/doi/10.1137/S0097539796300921

In this example, we generate a random bitstring, encode it into an inner-product oracle, then we simulate the kernel and return the most probable bitstring from its execution.

If all goes well, the state measured with the highest probability should be our hidden bitstring!

.. literalinclude:: ../examples/python/bernstein_vazirani.py
   :language: python

Variational Quantum Eigensolver
--------------------------------

Let's take a look at how we can use CUDA Quantum's built-in `vqe` module to run our own custom VQE routines! Given a parameterized quantum kernel, a system spin Hamiltonian, and one of CUDA Quantum's optimizers, `cudaq.vqe` will find and return the optimal set of parameters that minimize the energy, <Z>, of the system.

.. literalinclude:: ../examples/python/simple_vqe.py
   :language: python

Let's look at a more advanced examples.

As an alternative to `cudaq.vqe`, we can also use the `cudaq.optimizers` suite on its own to write custom variational algorithm routines. Much of this can be slightly modified for use with third-party optimizers, such as scipy.

.. literalinclude:: ../examples/python/advanced_vqe.py
   :language: python

Quantum Approximate Optimization Algorithm
-------------------------------------------

Let's now see how we can leverage the VQE algorithm to compute the maxcut of a rectangular graph.

.. literalinclude:: ../examples/python/qaoa_maxcut.py
   :language: python