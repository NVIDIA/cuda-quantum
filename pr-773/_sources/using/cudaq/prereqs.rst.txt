Prerequisites for a CUDA Quantum Code 
-------------------------------------
CUDA Quantum is a library-based, single-source, C++ language extension. As such, before
developing any CUDA Quantum source files, one must understand that each file should 
begin with appropriate include headers. Core CUDA Quantum support (quantum memory types,
kernel definitions, sampling) can be included with 

.. code-block:: cpp 

  #include <cudaq.h>

Defining spin operators will require 

.. code-block:: cpp 

  #include <cudaq/spin_op.h>
  -or- 
  #include <cudaq/algorithm.h>

CUDA Quantum spin operator observation will also be made available by :code:`<cudaq/algorithm.h>`.
For the available CUDA Quantum optimizers and gradients

.. code-block:: cpp 

  #include <cudaq/optimizers.h>
  #include <cudaq/gradients.h>