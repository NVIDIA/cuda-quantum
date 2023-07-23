
CUDA Quantum in C++
======================

Welcome to CUDA Quantum!
This is a introduction by example for using CUDA Quantum in C++. 

Introduction
--------------------------------

Welcome to CUDA Quantum! We're going to take a look at how to construct quantum programs using CUDA Quantum kernel expressions.

CUDA Quantum kernels are any typed callable in the language that is annotated with the :code:`__qpu__` attribute. Let's take a look at a very 
simple "Hello World" example, specifically a CUDA Quantum kernel that prepares a GHZ state on a programmer-specified number of qubits. 

.. literalinclude:: ../examples/cpp/basics/static_kernel.cpp
    :language: cpp

Here we see that we can define a custom :code:`struct` that is templated on a :code:`size_t` parameter. 
Our kernel expression is free to use this template parameter in the allocation of a 
compile-time-known register of qubits. Within the kernel, we are free to apply various quantum operations, 
like a Hadamard on qubit 0 :code:`h(q[0])`. Controlled operations are **modifications** of single-qubit 
operations, like the :code:`x<cudaq::ctrl>(q[0],q[1])` operation to affect a controlled-X. We 
can measure single qubits or entire registers. 

In this example we are interested in sampling the final state produced by this CUDA Quantum kernel. 
To do so, we leverage the generic :code:`cudaq::sample` function, which returns a data type 
encoding the qubit measurement strings and the corresponding number of times that string 
was observed (here the default number of shots is used, :code:`1000`).

To compile and execute this code, we run the following 

.. code:: bash 

    nvq++ static_kernel.cpp -o ghz.x
    ./ghz.x

Computing Expectation Values 
------------------------------

CUDA Quantum provides generic library functions enabling one to compute expectation values 
of quantum spin operators with respect to a parameterized CUDA Quantum kernel. Let's take a look 
at an example of this:

.. literalinclude:: ../examples/cpp/basics/expectation_values.cpp
    :language: cpp

Here we define a parameterized CUDA Quantum kernel, a callable type named :code:`ansatz` that takes as 
input a single angle :code:`theta`. This angle is used as part of a single :code:`ry` rotation. 

In host code, we define a Hamiltonian operator we are interested in via the CUDA Quantum :code:`spin_op` type. 
CUDA Quantum provides a generic function :code:`cudaq::observe` which takes a parameterized 
kernel, the :code:`spin_op` whose expectation value we wish to compute, and the runtime 
parameters at which we evaluate the parameterized kernel. 

The return type of this function is an :code:`cudaq::observe_result` which contains all the data 
from the execution, but is trivially convertible to a double, resulting in the expectation value we are interested in. 

To compile and execute this code, we run the following 

.. code:: bash 

    nvq++ expectation_values.cpp -o exp_vals.x 
    ./exp_vals.x 

Multi-control Synthesis 
-------------------------

Now let's take a look at how CUDA Quantum allows one to control a general unitary 
on an arbitrary number of control qubits. For this scenario, our general unitary can be described 
by another pre-defined CUDA Quantum kernel expression. Let's take a look at the following example:

.. literalinclude:: ../examples/cpp/basics/multi_controlled_operations.cpp
    :language: cpp

In this example, we show 2 distinct ways for generating a Toffoli operation. The first one in host code 
is the definition of a CUDA Quantum lambda that synthesizes a Toffoli via the general multi-control functionality 
for any single-qubit quantum operation :code:`x<cudaq::ctrl>(q[0], q[1], q[2])`.

The second way to generate a Toffoli operation starts with a kernel that takes another kernel as input. 
CUDA Quantum exposes a way to synthesize a control on any general unitary described as another kernel - 
the :code:`cudaq::control()` call. Here we take as input a kernel that applies an X operation to 
the given qubit. Within the :code:`control` call, we specify two control qubits, and the final target qubit.
This call requires trailing parameters that serve as the arguments for the applied kernel (:code:`apply_x` takes 
a single target qubit).

To compile and execute this code, we run the following 

.. code:: bash 

    nvq++ multi_controlled_operations.cpp -o mcx.x
    ./mcx.x

Simulations with cuQuantum
-----------------------------------

CUDA Quantum provides native support for cuQuantum-accelerated state vector and tensor network 
simulations. Let's take a look at an example that is too large for a standard CPU-only simulator, but 
can be trivially simulated via a NVIDIA GPU-accelerated backend:

.. literalinclude:: ../examples/cpp/basics/cuquantum_backends.cpp
    :language: cpp

Here we generate a GHZ state on 30 qubits. To run with the built-in cuQuantum state 
vector support, we pass the :code:`--target nvidia` flag at compile time:

.. code:: bash 

    nvq++ --target nvidia cuquantum_backends.cpp -o ghz.x
    ./ghz.x

.. _cpp-examples-for-hardware-providers:

Using Quantum Hardware Providers
-----------------------------------

CUDA Quantum contains support for using a set of hardware providers. 
For more information about executing quantum kernels on different hardware backends, please take a look at :doc:`hardware`.

The following code illustrates how run kernels on Quantinuum's backends.

.. literalinclude:: ../examples/cpp/providers/quantinuum.cpp
    :language: cpp

The following code illustrates how run kernels on IonQ's backends.

.. literalinclude:: ../examples/cpp/providers/ionq.cpp
    :language: cpp
