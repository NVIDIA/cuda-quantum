CUDA-Q C++ API
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

.. doxygenstruct:: cudaq::observe_options
    :members:

.. doxygenfunction:: cudaq::observe(const observe_options &options, QuantumKernel &&kernel, spin_op H, Args &&...args)
.. doxygenfunction:: cudaq::observe(std::size_t shots, QuantumKernel &&kernel, spin_op H, Args &&...args)
.. doxygenfunction:: cudaq::observe(QuantumKernel &&kernel, spin_op H, Args &&...args)
.. doxygenfunction:: cudaq::observe(QuantumKernel &&kernel, const SpinOpContainer &termList, Args &&...args)

.. doxygenclass:: cudaq::ExecutionContext
    :members:

.. doxygenclass:: cudaq::details::future
    :members:

.. doxygenclass:: cudaq::async_result
    :members:

.. doxygentypedef:: async_sample_result


.. doxygenstruct:: cudaq::ExecutionResult
    :members:

.. doxygenclass:: cudaq::sample_result
    :members:

.. doxygenstruct:: cudaq::sample_options
    :members:

.. doxygenfunction:: cudaq::sample(const sample_options &options, QuantumKernel &&kernel, Args &&...args)
.. doxygenfunction:: cudaq::sample(std::size_t shots, QuantumKernel &&kernel, Args &&...args)
.. doxygenfunction:: cudaq::sample(QuantumKernel &&kernel, Args&&... args)

.. doxygenclass:: cudaq::SimulationState

.. doxygenstruct:: cudaq::SimulationState::Tensor
    :members:

.. doxygenenum:: cudaq::SimulationState::precision

.. doxygenenum:: cudaq::simulation_precision

.. doxygentypedef:: cudaq::tensor

.. doxygentypedef:: cudaq::TensorStateData

.. doxygentypedef:: cudaq::state_data

.. doxygenclass:: cudaq::CusvState

.. doxygenclass:: nvqir::MPSSimulationState

.. doxygenclass:: nvqir::TensorNetSimulationState

.. doxygenclass:: cudaq::RemoteSimulationState

.. doxygenclass:: cudaq::registry::RegisteredType
    :members:

.. doxygenclass:: cudaq::complex_matrix
    :members:

.. doxygenclass:: cudaq::Trace

.. doxygenfunction:: cudaq::range(ElementType total)
.. doxygenfunction:: cudaq::range(ElementType begin, ElementType end, ElementType step)

.. doxygenfunction:: cudaq::draw(QuantumKernel &&kernel, Args&&... args)

.. doxygenfunction:: cudaq::get_state(QuantumKernel &&kernel, Args&&... args)

.. doxygenclass:: cudaq::Resources

.. doxygentypedef:: cudaq::complex_matrix::value_type

Noise Modeling 
================

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

.. doxygenenum:: cudaq::noise_model_type

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

.. doxygenclass:: cudaq::FermioniqBaseQPU

.. doxygenclass:: cudaq::OrcaRemoteRESTQPU 

.. doxygenclass:: cudaq::QuEraBaseQPU

.. doxygenclass:: cudaq::quantum_platform
    :members:

.. doxygenstruct:: cudaq::RemoteCapabilities
    :members:

.. doxygenclass:: cudaq::SerializedCodeExecutionContext

.. doxygentypedef:: cudaq::QuantumTask

.. doxygentypedef:: cudaq::QubitConnectivity

.. doxygentypedef:: cudaq::QubitEdge

.. doxygentypedef:: cudaq::KernelExecutionTask

.. doxygenstruct:: cudaq::KernelThunkResultType

.. doxygentypedef:: cudaq::KernelThunkType

Utilities
=========

.. doxygentypedef:: cudaq::complex

.. doxygentypedef:: cudaq::real 

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

.. doxygennamespace:: cudaq::orca
    :desc-only:

.. doxygenfunction:: cudaq::orca::sample(std::vector<std::size_t> &input_state, std::vector<std::size_t> &loop_lengths, std::vector<double> &bs_angles, int n_samples = 10000, std::size_t qpu_id = 0)
.. doxygenfunction:: cudaq::orca::sample(std::vector<std::size_t> &input_state, std::vector<std::size_t> &loop_lengths, std::vector<double> &bs_angles, std::vector<double> &ps_angles, int n_samples = 10000, std::size_t qpu_id = 0)
.. doxygenfunction:: cudaq::orca::sample_async(std::vector<std::size_t> &input_state, std::vector<std::size_t> &loop_lengths, std::vector<double> &bs_angles, int n_samples = 10000, std::size_t qpu_id = 0)
.. doxygenfunction:: cudaq::orca::sample_async(std::vector<std::size_t> &input_state, std::vector<std::size_t> &loop_lengths, std::vector<double> &bs_angles, std::vector<double> &ps_angles, int n_samples = 10000, std::size_t qpu_id = 0)
