Quick Start
*******************************************

**NVIDIA CUDA Quantum**

CUDA Quantum streamlines hybrid application development and promotes productivity and scalability
in quantum algorithm research. It offers a unified programming model designed for a hybrid
setting--that is, CPUs, GPUs, and QPUs working together, and consists of language extensions
for Python and C++ and a system level toolchain that enables application acceleration.

Learn more about CUDA Quantum’s key benefits here [FIXME: Link to CUDA Quantum Marketing page]

If you have already installed and configured CUDA Quantum, we encourage you to move directly to our
:ref:`Basics Section <cudaq-basics-landing-page>`.


Python
-------

To install the latest stable release of the CUDA Quantum Python API:  

.. code-block:: bash

    pip install cuda-quantum 

For further information, see the `CUDA Quantum project <https://pypi.org/project/cuda-quantum/>`_ on PyPI.

We also offer a fully featured CUDA Quantum installation, including all C++ and Python tools, via our
`Docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-quantum>`_. For further
installation methods and resources, see our :ref:`Installation Guide <install-cuda-quantum>`.

You should now be able to import CUDA Quantum and start building quantum programs in Python!

To test your installation, create a file titled `first_program.py`, containing the following code:

.. code-block:: Python

    import cudaq

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        h(qubit)
        mz(qubit)

    result = cudaq.sample(kernel)
    print(result)  # { 1:500 0:500 }

You may now execute this file as you do any other Python program. For example, from the command line:

.. code-block:: bash

    python3 first_program.py

The Hadamard gate places the qubit in a superposition state, giving a roughly 50/50 mixture
of measurments in the `|0>` and `|1>` states.

Now that you have succesfully run your first program, you are ready to move on to our :ref:`Basics Section <cudaq-basics-landing-page>`.


C++
----

To install CUDA Quantum for C++, TODO ...

We also offer a fully featured CUDA Quantum installation, including all C++ and Python tools, via our
`Docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-quantum>`_. For further
installation methods and resources, see our :ref:`Installation Guide <install-cuda-quantum>`.

You should now have the CUDA Quantum compiler toolchain available via the command line and are ready to run your first quantum program!

To test your installation, create a file titled `first_program.cpp`, containing the following code:

.. code-block:: cpp

    __qpu__ void kernel() { 
      cudaq::qubit qubit; 
      h(qubit); 
      mz(qubit); 
    } 

    int main() { 
      auto result = cudaq::sample(kernel); 
      result.dump();  // { 1:500 0:500 }
    } 

You may now execute this file from the command line using the `nvq++` toolchain:

.. code-block:: bash

    nvq++ first_program.cpp && ./a.out

The Hadamard gate places the qubit in a superposition state, giving a roughly 50/50 mixture
of measurments in the `|0>` and `|1>` states.

Now that you have succesfully run your first program, you are ready to move on to our :ref:`Basics Section <cudaq-basics-landing-page>`.
