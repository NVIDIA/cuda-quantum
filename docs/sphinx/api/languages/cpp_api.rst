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

.. doxygenclass:: cudaq::qvector
    :members:

.. doxygenclass:: cudaq::qspan
    :members:

.. doxygenclass:: cudaq::qview
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

.. doxygenclass:: cudaq::SimulationState

.. doxygenclass:: cudaq::CusvState

.. doxygenclass:: cudaq::registry::RegisteredType
    :members:

.. doxygenclass:: cudaq::complex_matrix
    :members:

.. doxygenclass:: cudaq::Trace

.. doxygenfunction:: cudaq::range(ElementType total)
.. doxygenfunction:: cudaq::range(ElementType begin, ElementType end, ElementType step)

.. doxygenfunction:: cudaq::draw(QuantumKernel &&kernel, Args&&... args)

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

.. doxygenclass:: cudaq::gradients::forward_difference
    :members:

Platform
=========

.. doxygenclass:: cudaq::QPU
    :members:

.. doxygenclass:: cudaq::BaseRemoteRESTQPU

.. doxygenclass:: cudaq::BaseRemoteSimulatorQPU

.. doxygenclass:: cudaq::BaseNvcfSimulatorQPU    

.. doxygenclass:: cudaq::quantum_platform
    :members:

.. doxygentypedef:: cudaq::QuantumTask

.. doxygentypedef:: cudaq::QubitConnectivity

.. doxygentypedef:: cudaq::QubitEdge

.. doxygentypedef:: cudaq::KernelExecutionTask

Utilities
=========

.. doxygenfunction:: cudaq::range(std::size_t)
    
Namespaces 
===========

.. doxygennamespace:: cudaq
    :desc-only:

.. doxygenfunction:: cudaq::num_available_gpus
.. doxygenfunction:: cudaq::set_random_seed
.. doxygenfunction:: cudaq::set_noise
.. doxygenfunction:: cudaq::unset_noise

.. doxygennamespace:: cudaq::details
    :desc-only:

.. doxygennamespace:: cudaq::registry
    :desc-only:

.. doxygennamespace:: cudaq::mpi
    :desc-only:

.. doxygenfunction:: cudaq::mpi::initialize()
.. doxygenfunction:: cudaq::mpi::initialize(int argc, char **argv)
.. doxygenfunction:: cudaq::mpi::is_initialized
.. doxygenfunction:: cudaq::mpi::finalize
.. doxygenfunction:: cudaq::mpi::rank
.. doxygenfunction:: cudaq::mpi::num_ranks
.. doxygenfunction:: cudaq::mpi::all_gather(std::vector<double> &global, const std::vector<double> &local)
.. doxygenfunction:: cudaq::mpi::all_gather(std::vector<int> &global, const std::vector<int> &local)
.. doxygenfunction:: cudaq::mpi::all_reduce(const T&, const Func&)
.. doxygenfunction:: cudaq::mpi::all_reduce(const T &localValue, const BinaryFunction &function)
.. doxygenfunction:: cudaq::mpi::broadcast(std::vector<double> &data, int rootRank)
.. doxygenfunction:: cudaq::mpi::broadcast(std::string &data, int rootRank)
