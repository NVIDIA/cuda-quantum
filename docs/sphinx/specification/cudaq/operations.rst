Quantum Intrinsic Operations
****************************
In an effort to support low-level quantum programming and build foundational
quantum kernels for large-scale applications, CUDA Quantum defines the quantum
intrinsic operation as an abstraction
for device-specific single-qudit unitary operations. A quantum intrinsic
operation is modeled via a standard C++ function with a unique instruction name and
general function operands. These operands can model classical rotation
parameters or units of quantum information (e.g. the :code:`cudaq::qudit`).
The syntax for these operations is 

.. code-block:: cpp 

    void INST_NAME( PARAM...?, qudit<N>&...);

where :code:`INST_NAME` is the name of the instruction, :code:`qudit<N>&...` indicates one many
:code:`cudaq::qudit` instances, and :code:`PARAM...?` indicates optional parameters of 
floating point type (e.g. :code:`double`, :code:`float`). All intrinsic operations should 
start with a base declaration targeting a single :code:`cudaq::qudit`, and overloads
should be provided that take more than one :code:`cudaq::qudit` instances to model the application
of that instruction on all provided :code:`cudaq::qudits`, e.g. :code:`void x(cudaq::qubit&)` and
:code:`x(cudaq::qubit&, cudaq::qubit&, cudaq::qubit&)`, modeling the NOT operation on a single 
:code:`cudaq::qubit` or on multiple :code:`cudaq::qubit`. 
