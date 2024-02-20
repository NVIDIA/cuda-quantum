
CUDA Quantum in Python
======================

.. _python-examples-landing-page:

Welcome to CUDA Quantum! On this page we will illustrate CUDA Quantum in Python with several examples. 
We first introduce how to code a kernel and execute it on a state vector simulator before moving onto 
common algorithms such as Bernstein-Vazirani, Variational Quantum Eigensolver, and QAOA. The next sections 
demonstrate how to run noisy simulations on the density matrix simulator and how to execute quantum kernels on real hardware.

Introduction
--------------------------------

We're going to take a look at how to construct quantum programs through CUDA Quantum's `Kernel` API.

When you create a `Kernel` and invoke its methods, a quantum program is constructed that can then be executed by calling, for example, `cudaq::sample`. Let's take a closer look!

.. literalinclude:: ../examples/python/intro.py
   :language: python

Bernstein-Vazirani
--------------------------------

The Bernstein-Vazirani algorithm aims to identify the bitstring encoded in a given function. 

For the original source of this algorithm, see 
`this publication <https://epubs.siam.org/doi/10.1137/S0097539796300921>`__.

In this example, we generate a random bitstring and encode it into an inner-product oracle, 
and define a kernel for the Bernstein-Vazirani algorithm.  Then, we simulate the kernel and return the most probable bitstring from its execution.


If all goes well, the state measured with the highest probability should be our hidden bitstring!

.. literalinclude:: ../examples/python/bernstein_vazirani.py
   :language: python

Variational Quantum Eigensolver
--------------------------------

The Variational Quantum Eigensolver (VQE) algorithm, originally proposed in
`this publication <https://arxiv.org/abs/1304.3061>`__, 
is a hybrid algorithm that can make use of both quantum and classical resources.

Let's take a look at how we can use CUDA Quantum's built-in `vqe` module to run our own custom VQE routines! 
Given a parameterized quantum kernel, a system spin Hamiltonian, and one of CUDA Quantum's optimizers, 
`cudaq.vqe` will find and return the optimal set of parameters that minimize the energy, <Z>, of the system.

The code block below represents the contents of a file titled `simple_vqe.py`. 

.. literalinclude:: ../examples/python/simple_vqe.py
   :language: python

Let's look at a more advanced variation of the previous example.

As an alternative to `cudaq.vqe`, we can also use the `cudaq.optimizers` suite on its own to write custom variational algorithm routines. Much of this can be slightly modified for use with third-party optimizers, such as `scipy`.

.. literalinclude:: ../examples/python/advanced_vqe.py
   :language: python

Quantum Approximate Optimization Algorithm
-------------------------------------------

Let's now see how we can implement the Quantum Approximate Optimization Algorithm (QAOA) to compute the Max-Cut of a rectangular graph by leveraging 
`cudaq:vqe`. For more on the QAOA algorithm and the Max-Cut problem, refer to 
`this paper <https://arxiv.org/abs/1411.4028>`__.

.. literalinclude:: ../examples/python/qaoa_maxcut.py
   :language: python

Noisy Simulation
-----------------

CUDA Quantum makes it simple to model noise within the simulation of your quantum program.
Let's take a look at the various built-in noise models we support, before concluding with a brief example of a custom noise model constructed from user-defined Kraus Operators.

The following code illustrates how to run a simulation with depolarization noise.

.. literalinclude:: ../examples/python/noise_depolarization.py
   :language: python

The following code illustrates how to run a simulation with amplitude damping noise.

.. literalinclude:: ../examples/python/noise_amplitude_damping.py
   :language: python

The following code illustrates how to run a simulation with bit-flip noise.

.. literalinclude:: ../examples/python/noise_bit_flip.py
   :language: python

The following code illustrates how to run a simulation with phase-flip noise.

.. literalinclude:: ../examples/python/noise_phase_flip.py
   :language: python

The following code illustrates how to run a simulation with a custom noise model.

.. literalinclude:: ../examples/python/noise_kraus_operator.py
   :language: python

.. _python-examples-for-hardware-providers:

Using Quantum Hardware Providers
-----------------------------------

CUDA Quantum contains support for using a set of hardware providers (Quantinuum, IonQ, and IQM). 
For more information about executing quantum kernels on different hardware backends, please take a look at :doc:`backends/hardware`.

The following code illustrates how to run kernels on Quantinuum's backends.

.. literalinclude:: ../examples/python/providers/quantinuum.py
   :language: python

The following code illustrates how to run kernels on IonQ's backends.

.. literalinclude:: ../examples/python/providers/ionq.py
   :language: python

The following code illustrates how to run kernels on IQM's backends.

.. literalinclude:: ../examples/python/providers/iqm.py
   :language: python
