CUDA Quantum C++ API
******************************

Operators 
=============

.. doxygenclass:: cudaq::spin_op
    :members:

Quantum
=========

.. doxygenvariable:: cudaq::dyn

.. doxygenclass:: cudaq::qudit
    :members:

.. doxygenclass:: cudaq::qreg
    :members:

.. doxygenclass:: cudaq::qspan
    :members:

.. doxygentypedef:: cudaq::qubit
    
Common
=========

.. doxygenclass:: cudaq::observe_result
    :members:

.. doxygenclass:: cudaq::ExecutionContext
    :members:

.. doxygenclass:: cudaq::details::future
    :members:

.. doxygenclass:: cudaq::async_result
    :members:


.. doxygenstruct:: cudaq::ExecutionResult
    :members:

.. doxygenclass:: cudaq::sample_result
    :members:

.. doxygentypedef:: cudaq::State

.. doxygenclass:: cudaq::registry::RegisteredType
    :members:

.. doxygenclass:: cudaq::complex_matrix
    :members:

.. doxygenclass:: cudaq::Resources

.. doxygentypedef:: cudaq::complex_matrix::value_type

Noise Modeling 
================
.. doxygentypedef:: cudaq::complex

.. doxygenstruct:: cudaq::kraus_op
    :members:

.. doxygenclass:: cudaq::kraus_channel
    :members:

.. doxygenclass:: cudaq::amplitude_damping_channel
    :members:

.. doxygenclass:: cudaq::bit_flip_channel
    :members:

.. doxygenclass:: cudaq::phase_flip_channel
    :members:

.. doxygenclass:: cudaq::depolarization_channel
    :members:

.. doxygenclass:: cudaq::noise_model
    :members:

Kernel Builder
===============

.. doxygenclass:: cudaq::kernel_builder
    :members:

.. doxygenclass:: cudaq::QuakeValue
    :members:

.. doxygenclass:: cudaq::details::kernel_builder_base
    :members:

.. doxygenclass:: cudaq::details::KernelBuilderType
    :members:

Algorithms
===========

.. doxygenclass:: cudaq::optimizer
    :members:

.. doxygenclass:: cudaq::optimizable_function
    :members:

.. doxygentypedef:: cudaq::optimization_result

.. doxygenclass:: cudaq::state
    :members:

.. doxygenclass:: cudaq::gradient
    :members:

.. doxygenclass:: cudaq::gradients::central_difference
    :members:

.. doxygenclass:: cudaq::gradients::parameter_shift
    :members:


Platform
=========

.. doxygenclass:: cudaq::QPU
    :members:

.. doxygenclass:: cudaq::quantum_platform
    :members:

.. doxygentypedef:: cudaq::QuantumTask

.. doxygentypedef:: cudaq::QubitConnectivity

.. doxygentypedef:: cudaq::QubitEdge

.. doxygentypedef:: cudaq::KernelExecutionTask

Namespaces 
===========

.. doxygennamespace:: cudaq
    :desc-only:

.. doxygennamespace:: cudaq::details
    :desc-only:

.. doxygennamespace:: cudaq::registry
    :desc-only:
