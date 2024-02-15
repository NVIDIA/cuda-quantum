CUDA Quantum Basics
*******************

.. _cudaq-basics-landing-page:

What is a CUDA Quantum kernel?
-------------------------------

Quantum kernels are defined as functions that are executed on a quantum processing unit (QPU) or
a simulator. They generalize quantum circuits and provide a new abstraction for quantum programming.
Quantum kernels can be combined with classical functions to create quantum-classical applications
that can be executed on a heterogeneous system of QPUs, GPUs, and CPUs to solve real-world problems.

What’s the difference between a quantum kernel and a quantum circuit?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every quantum circuit is a kernel, but not every quantum kernel is a circuit. For instance, a quantum
kernel can be built up from other kernels, allowing us to interpret a large quantum program as a sequence
of subroutines or subcircuits.  

Moreover, since quantum kernels are functions, there is more expressibility available compared to a
standard quantum circuit. We can not only parameterize the kernel, but can also apply classical controls
(`if`, `for`, `while`, etc.). As functions, quantum kernels can return void, Boolean values, integers,
floating point numbers, and vectors of Boolean values. Conditional statements on quantum memory and qubit
measurements can be included in quantum kernels to enable dynamic circuits and fast feedback, particularly
useful for quantum error correction. 

How do I build and run a quantum kernel?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a quantum kernel has been defined in a program, it can be executed using the `sample` or the `observe` primitives.
Let’s take a closer look at how to build and execute a quantum kernel with CUDA Quantum.


Building your first CUDA Quantum Program
-----------------------------------------

Todo 

Running your first CUDA Quantum Program
----------------------------------------

Todo

Language Fundamentals
----------------------

Todo