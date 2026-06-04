CUDA-Q C++ API
******************************

Operators
=============

.. doxygenclass:: cudaq::scalar_callback
    :members:

.. doxygenclass:: cudaq::scalar_operator
    :members:

.. doxygenstruct:: cudaq::commutation_relations

.. doxygenclass:: cudaq::matrix_callback

.. doxygenclass:: cudaq::diag_matrix_callback

.. cpp:type:: csr_spmatrix = std::tuple<std::vector<std::complex<double>>, std::vector<std::size_t>, std::vector<std::size_t>>

    Alias for a tuple containing vectors for complex values, indices, and sizes.

    The tuple elements are:

    - ``std::vector<std::complex<double>>``: Complex values.
    - ``std::vector<std::size_t>``: Indices.
    - ``std::vector<std::size_t>``: Sizes.

.. cpp:type:: mdiag_sparse_matrix = std::pair<std::vector<std::complex<double>>, std::vector<std::int64_t>>

    Alias for a pair, in the multi-diagonal representation, containing vectors for complex values and diagonal offsets.  

.. doxygenclass:: cudaq::operator_handler

.. doxygenclass:: cudaq::mdiag_operator_handler

.. doxygenclass:: cudaq::spin_handler
    :members:

.. doxygenclass:: cudaq::fermion_handler

.. doxygenclass:: cudaq::boson_handler

.. doxygenclass:: cudaq::matrix_handler
    :members:

.. doxygenclass:: cudaq::product_op
    :members:

.. doxygenclass:: cudaq::sum_op
    :members:

.. cpp:type:: cudaq::spin_op

.. cpp:type:: cudaq::spin_op_term

.. cpp:type:: cudaq::fermion_op

.. cpp:type:: cudaq::fermion_op_term

.. cpp:type:: cudaq::boson_op

.. cpp:type:: cudaq::boson_op_term

.. cpp:type:: cudaq::matrix_op

.. cpp:type:: cudaq::matrix_op_term

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

.. doxygenfunction:: cudaq::observe(const observe_options &options, QuantumKernel &&kernel, const spin_op &H, Args &&...args)
.. doxygenfunction:: cudaq::observe(std::size_t shots, QuantumKernel &&kernel, const spin_op &H, Args &&...args)
.. doxygenfunction:: cudaq::observe(QuantumKernel &&kernel, const spin_op &H, Args &&...args)
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

.. doxygenfunction:: cudaq::run(std::size_t shots, QuantumKernel &&kernel, ARGS &&...args)
.. doxygenfunction:: cudaq::run(std::size_t shots, cudaq::noise_model &noise_model, QuantumKernel &&kernel, ARGS &&...args)
.. doxygenfunction:: cudaq::run_async(std::size_t qpu_id, std::size_t shots, QuantumKernel &&kernel, ARGS &&...args)
.. doxygenfunction:: cudaq::run_async(std::size_t qpu_id, std::size_t shots, cudaq::noise_model &noise_model, QuantumKernel &&kernel, ARGS &&...args)

.. doxygenclass:: cudaq::measure_handle

.. doxygentypedef:: cudaq::measure_result

The measurement operations ``mz`` / ``mx`` / ``my`` return a
``cudaq::measure_handle`` (also available under the alias
``cudaq::measure_result``) rather than a bare classical value. Inside a kernel the
handle converts to a classical value implicitly (the compiler intercepts the
coercion), which defers discrimination. The helpers below discriminate a
vector of handles.

.. cpp:function:: std::vector<bool> cudaq::to_bools(const std::vector<measure_result> &results)

    Discriminate a vector of measurement handles into their boolean outcomes.
    Used inside a kernel (where the compiler intercepts the call); at host
    scope the handles must already have been discriminated.

.. cpp:function:: std::int64_t cudaq::to_integer(const std::vector<measure_result> &bits)
.. cpp:function:: std::int64_t cudaq::to_integer(const std::vector<bool> &bits)
.. cpp:function:: std::int64_t cudaq::to_integer(const std::string &bitstring)

    Interpret a sequence of measurement outcomes as a little-endian unsigned
    integer (bit ``i`` has weight ``2^i``). Overloads accept a vector of
    handles, a vector of ``bool``, or a bitstring such as ``"011"``.

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

.. doxygenclass:: cudaq::QPUState

.. doxygenclass:: cudaq::registry::RegisteredType
    :members:

.. doxygenclass:: cudaq::complex_matrix
    :members:

.. doxygenclass:: cudaq::Trace

.. doxygenfunction:: cudaq::range(ElementType total)
.. doxygenfunction:: cudaq::range(ElementType begin, ElementType end, ElementType step)

.. doxygenfunction:: cudaq::get_state(QuantumKernel &&kernel, Args&&... args)

.. doxygenclass:: cudaq::Resources

.. doxygentypedef:: cudaq::complex_matrix::value_type

Noise Modeling 
================

.. cpp:function:: template <typename Channel, typename... Args> void cudaq::apply_noise(Args&&... args)

    This function is a type-safe injection of noise into a quantum kernel,
    occurring precisely at the call site of the function invocation. The
    function should be called inside CUDA-Q kernels (those annotated with
    `__qpu__`). The functionality is only supported for simulation targets, so
    it is automatically (and silently) stripped from any programs submitted to
    hardware targets.

    :`tparam` Channel: A subtype of :cpp:class:`cudaq::kraus_channel` that
        implements/defines the desired noise mechanisms as Kraus channels (e.g.
        :cpp:class:`cudaq::depolarization2`). If you want to use a custom
        :cpp:class:`cudaq::kraus_channel` (i.e. not built-in to CUDA-Q), it must
        first be registered *outside the kernel* with
        `:cpp:func:cudaq::noise_model::register_channel`, like this:

        .. code-block:: cpp

            struct my_custom_kraus_channel_subtype : public ::cudaq::kraus_channel {
              static constexpr std::size_t num_parameters = 1;
              static constexpr std::size_t num_targets = 1;

              my_custom_kraus_channel_subtype(const std::vector<cudaq::real> &params) {
                  std::vector<cudaq::complex> k0v{std::sqrt(1 - params[0]), 0, 0,
                                                  std::sqrt(1 - params[0])},
                      k1v{0, std::sqrt(params[0]), std::sqrt(params[0]), 0};
                  push_back(cudaq::kraus_op(k0v));
                  push_back(cudaq::kraus_op(k1v));
                  validateCompleteness();
                  generateUnitaryParameters();
              }
              REGISTER_KRAUS_CHANNEL("my_custom_kraus_channel_subtype");
            };

            cudaq::noise_model noise;
            noise.register_channel<my_custom_kraus_channel_subtype>();

    :`param` `args`: The precise argument pack depend on the concrete `Channel` being
        used. The arguments are a concatenated list of parameters and targets.
        For example, to apply a 2-qubit depolarization channel, which has
        `num_parameters = 1` and `num_targets = 2`, one would write the call
        like this:

        .. code-block:: cpp

            cudaq::qubit q, r;
            cudaq::apply_noise<cudaq::depolarization2>(/*probability=*/0.1, q, r);

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

.. doxygenclass:: cudaq::x_error
    :members:

.. doxygenclass:: cudaq::y_error
    :members:

.. doxygenclass:: cudaq::z_error
    :members:

.. doxygenclass:: cudaq::amplitude_damping
    :members:

.. doxygenclass:: cudaq::phase_damping
    :members:

.. doxygenclass:: cudaq::pauli1
    :members:

.. doxygenclass:: cudaq::pauli2
    :members:

.. doxygenclass:: cudaq::depolarization1
    :members:

.. doxygenclass:: cudaq::depolarization2
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

.. doxygenfunction:: cudaq::dem_from_kernel(QuantumKernel &&kernel, const cudaq::noise_model *noise, Args &&...args)
.. doxygenfunction:: cudaq::dem_from_kernel(QuantumKernel &&kernel, Args &&...args)

Quantum Error Correction
========================

These kernel-side declarations annotate a circuit with detectors and logical
observables so a detector error model (DEM) can be extracted with
``cudaq::dem_from_kernel``. They are intercepted by the compiler
inside ``__qpu__`` kernels and lower to the ``qec`` dialect; for
backends that do not consume them they are erased before execution.

.. cpp:function:: template <typename... MeasArgs> void cudaq::detector(MeasArgs &&...ms)

    Declare one detector as the parity of the referenced measurements. Each
    argument is a ``cudaq::measure_result`` or a
    ``std::vector<cudaq::measure_result>``. A detector is a parity constraint
    that is deterministic under noise-free execution.

.. cpp:function:: void cudaq::detectors(const std::vector<measure_result> &prev, const std::vector<measure_result> &curr)

    Declare N detectors at once by pairing two equal-length handle vectors
    element-wise (``detector(prev[i], curr[i])`` for each ``i``); the standard
    form for cross-round detectors. Length agreement is enforced at runtime.

.. cpp:function:: template <typename... MeasArgs> void cudaq::logical_observable(MeasArgs &&...ms)
.. cpp:function:: void cudaq::logical_observable(const std::vector<measure_result> &ms, std::size_t observable_index)

    Declare one logical observable as the parity of the referenced
    measurements. The variadic form uses observable index ``0``; codes with
    ``k`` logical qubits use the ``(vector, observable_index)`` overload to
    declare observables ``0..k-1``.

Platform
=========

.. doxygenclass:: cudaq::QPU
    :members:

.. doxygenclass:: cudaq::BaseRemoteRESTQPU

.. doxygenclass:: cudaq::AnalogRemoteRESTQPU    

.. doxygenclass:: cudaq::FermioniqQPU

.. doxygenclass:: cudaq::OrcaRemoteRESTQPU

.. doxygenclass:: cudaq::quantum_platform
    :members:

.. doxygenstruct:: cudaq::RemoteCapabilities
    :members:

.. doxygentypedef:: cudaq::QuantumTask

.. doxygentypedef:: cudaq::QubitConnectivity

.. doxygentypedef:: cudaq::QubitEdge

.. doxygentypedef:: cudaq::KernelExecutionTask

.. doxygenstruct:: cudaq::KernelThunkResultType

.. doxygentypedef:: cudaq::KernelThunkType

.. doxygenstruct:: cudaq::CodeGenConfig

.. doxygenstruct:: cudaq::RuntimeTarget

Utilities
=========

.. doxygentypedef:: cudaq::complex

.. doxygentypedef:: cudaq::real 

.. doxygenfunction:: cudaq::range(std::size_t)

.. doxygenfunction:: cudaq::contrib::draw(QuantumKernel &&kernel, Args&&... args)

.. doxygenfunction:: cudaq::contrib::get_unitary_cmat(QuantumKernel &&kernel, Args&&... args)
    
Namespaces 
===========

.. doxygennamespace:: cudaq
    :desc-only:

.. doxygenfunction:: cudaq::num_available_gpus
.. doxygenfunction:: cudaq::set_random_seed
.. doxygenfunction:: cudaq::set_noise
.. doxygenfunction:: cudaq::unset_noise
    
.. doxygennamespace:: cudaq::contrib
    :desc-only:

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

PTSBE
=====

The ``cudaq::ptsbe`` namespace implements Pre-Trajectory Sampling with Batch
Execution (PTSBE). For a conceptual overview and usage tutorial see
:doc:`../../using/examples/ptsbe`.

.. doxygennamespace:: cudaq::ptsbe
    :desc-only:

Sampling Functions
-------------------

.. doxygenfunction:: cudaq::ptsbe::sample(const cudaq::noise_model &noise, std::size_t shots, QuantumKernel &&kernel, Args &&...args)
.. doxygenfunction:: cudaq::ptsbe::sample(const sample_options &options, QuantumKernel &&kernel, Args &&...args)
.. doxygenfunction:: cudaq::ptsbe::sample_async(const cudaq::noise_model &noise, std::size_t shots, QuantumKernel &&kernel, Args &&...args)
.. doxygenfunction:: cudaq::ptsbe::sample_async(const sample_options &options, QuantumKernel &&kernel, Args &&...args)

----

Options
--------

.. doxygenstruct:: cudaq::ptsbe::sample_options
    :members:

.. doxygenstruct:: cudaq::ptsbe::PTSBEOptions
    :members:

----

Result Type
------------

.. doxygenclass:: cudaq::ptsbe::sample_result
    :members:

----

Trajectory Sampling Strategies
--------------------------------

.. doxygenstruct:: cudaq::ptsbe::detail::NoisePoint
    :members:

.. doxygenclass:: cudaq::ptsbe::PTSSamplingStrategy
    :members:

.. doxygenclass:: cudaq::ptsbe::ProbabilisticSamplingStrategy
    :members:

.. doxygenclass:: cudaq::ptsbe::OrderedSamplingStrategy
    :members:

.. doxygenclass:: cudaq::ptsbe::ExhaustiveSamplingStrategy
    :members:

.. doxygentypedef:: cudaq::ptsbe::TrajectoryPredicate

.. doxygenclass:: cudaq::ptsbe::ConditionalSamplingStrategy
    :members:

----

Shot Allocation Strategy
-------------------------

See :ref:`ptsbe-shot-allocation` for full API details.

----

Execution Data
---------------

.. doxygentypedef:: cudaq::ptsbe::PTSBETrace

.. doxygenstruct:: cudaq::ptsbe::PTSBEExecutionData
    :members:

.. doxygenstruct:: cudaq::ptsbe::TraceInstruction
    :members:

.. doxygenenum:: cudaq::ptsbe::TraceInstructionType

----

Trajectory and Selection Types
--------------------------------

.. doxygenclass:: cudaq::KrausTrajectoryBuilder
    :members:

.. doxygenstruct:: cudaq::KrausTrajectory
    :members:

.. doxygenstruct:: cudaq::KrausSelection
    :members:
