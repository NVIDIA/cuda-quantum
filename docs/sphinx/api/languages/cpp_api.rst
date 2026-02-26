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

.. _classcudaq_1_1ptsbe_1_1sample__result:

**cudaq::`ptsbe`::sample_result** — Result type returned by ``ptsbe::sample()``, extending `cudaq::sample_result` with optional execution data (trace and per-trajectory info). See ``PTSBESampleResult.h`` and ``PTSBEExecutionData.h``.

.. doxygenstruct:: cudaq::sample_options
    :members:

.. doxygenfunction:: cudaq::sample(const sample_options &options, QuantumKernel &&kernel, Args &&...args)
.. doxygenfunction:: cudaq::sample(std::size_t shots, QuantumKernel &&kernel, Args &&...args)
.. doxygenfunction:: cudaq::sample(QuantumKernel &&kernel, Args&&... args)

.. doxygenfunction:: cudaq::run(std::size_t shots, QuantumKernel &&kernel, ARGS &&...args)
.. doxygenfunction:: cudaq::run(std::size_t shots, cudaq::noise_model &noise_model, QuantumKernel &&kernel, ARGS &&...args)
.. doxygenfunction:: cudaq::run_async(std::size_t qpu_id, std::size_t shots, QuantumKernel &&kernel, ARGS &&...args)
.. doxygenfunction:: cudaq::run_async(std::size_t qpu_id, std::size_t shots, cudaq::noise_model &noise_model, QuantumKernel &&kernel, ARGS &&...args)

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

    :tparam Channel: A subtype of :cpp:class:`cudaq::kraus_channel` that
        implements/defines the desired noise mechanisms as Kraus channels (e.g.
        :cpp:class:`cudaq::depolarization2`). If you want to use a custom
        :cpp:class:`cudaq::kraus_channel` (i.e. not built-in to CUDA-Q), it must
        first be registered *outside the kernel* with
        :cpp:func:`cudaq::noise_model::register_channel`, like this:

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

    :param args: The precise argument pack depend on the concrete `Channel` being
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

Platform
=========

.. doxygenclass:: cudaq::QPU
    :members:

.. doxygenclass:: cudaq::BaseRemoteRESTQPU

.. doxygenclass:: cudaq::BaseRemoteSimulatorQPU

.. doxygenclass:: cudaq::AnalogRemoteRESTQPU    

.. doxygenclass:: cudaq::FermioniqBaseQPU

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
:doc:`../../using/ptsbe`.

.. cpp:namespace:: cudaq::ptsbe

Sampling Functions
-------------------

.. cpp:function:: template <typename QuantumKernel, typename... Args> \
                  sample_result \
                  sample(const sample_options& options, QuantumKernel&& kernel, Args&&... args)

   Sample a quantum kernel using PTSBE.

   :tparam QuantumKernel: A CUDA-Q kernel callable.
   :tparam Args: Kernel argument types.
   :param options: Execution options (shots, noise model, PTSBE configuration).
   :param kernel: The kernel to execute.
   :param args: Arguments forwarded to the kernel.
   :returns: Aggregated ``sample_result``.

   .. code-block:: cpp

      #include "cudaq/ptsbe/PTSBESample.h"

      cudaq::ptsbe::sample_options opts;
      opts.shots           = 10'000;
      opts.noise           = noise_model;
      opts.ptsbe.max_trajectories = 200;

      auto result = cudaq::ptsbe::sample(opts, bell);
      result.dump();


.. cpp:function:: template <typename QuantumKernel, typename... Args> \
                  cudaq::async_sample_result \
                  sample_async(const sample_options& options, QuantumKernel&& kernel, Args&&... args, std::size_t qpu_id = 0)

   Asynchronous variant of :cpp:func:`sample`. Returns a
   ``std::future<sample_result>``.

   .. code-block:: cpp

      auto future = cudaq::ptsbe::sample_async(opts, bell);
      auto result = future.get();

----

Options
--------

.. cpp:struct:: sample_options

   Top-level options passed to :cpp:func:`sample`.

   .. cpp:member:: std::size_t shots = 1000

      Total number of measurement shots.

   .. cpp:member:: cudaq::noise_model noise

      Noise model describing gate-level error channels.

   .. cpp:member:: PTSBEOptions ptsbe

      PTSBE-specific configuration (trajectories, strategy, allocation).


.. cpp:struct:: PTSBEOptions

   PTSBE-specific execution configuration.

   .. cpp:member:: bool return_execution_data = false

      When ``true``, the returned result contains a
      :cpp:struct:`PTSBEExecutionData` payload with the circuit trace,
      trajectory details, and per-trajectory measurement counts.

   .. cpp:member:: std::optional<std::size_t> max_trajectories = std::nullopt

      Maximum number of unique trajectories to generate. ``std::nullopt``
      defaults to the shot count.

   .. cpp:member:: std::shared_ptr<PTSSamplingStrategy> strategy = nullptr

      Trajectory sampling strategy. ``nullptr`` uses the default
      :cpp:class:`ProbabilisticSamplingStrategy`.

   .. cpp:member:: ShotAllocationStrategy shot_allocation

      Shot allocation strategy. Defaults to
      :cpp:enumerator:`ShotAllocationStrategy::Type::PROPORTIONAL`.

----

Result Type
------------

.. cpp:class:: sample_result : public cudaq::sample_result

   Extends :cpp:class:`cudaq::sample_result` with an optional
   :cpp:struct:`PTSBEExecutionData` payload.

   .. cpp:function:: bool has_execution_data() const

      Return ``true`` if execution data is attached.

   .. cpp:function:: const PTSBEExecutionData& execution_data() const

      Return the attached execution data.

      :throws std::runtime_error: If no execution data is available.

   .. cpp:function:: void set_execution_data(PTSBEExecutionData data)

      Attach execution data to this result.

----

Trajectory Sampling Strategies
--------------------------------

.. cpp:class:: PTSSamplingStrategy

   Abstract base class for trajectory sampling strategies.

   .. cpp:function:: virtual std::vector<cudaq::KrausTrajectory> generateTrajectories(std::span<const cudaq::KrausTrajectory> noise_points, std::size_t max_trajectories) const = 0

      Generate up to *max_trajectories* unique trajectories from the noise
      space.

   .. cpp:function:: virtual const char* name() const = 0

      Return the strategy name.

   .. cpp:function:: virtual std::unique_ptr<PTSSamplingStrategy> clone() const = 0

      Return a deep copy of this strategy.


.. cpp:class:: ProbabilisticSamplingStrategy : public PTSSamplingStrategy

   Randomly samples unique trajectories weighted by probability.

   .. cpp:function:: explicit ProbabilisticSamplingStrategy(std::uint64_t seed = 0)

      :param seed: Random seed. ``0`` uses the global CUDA-Q seed if set,
          otherwise ``std::random_device``.

   .. code-block:: cpp

      #include "cudaq/ptsbe/strategies/ProbabilisticSamplingStrategy.h"

      opts.ptsbe.strategy =
          std::make_shared<cudaq::ptsbe::ProbabilisticSamplingStrategy>(/*seed=*/42);


.. cpp:class:: OrderedSamplingStrategy : public PTSSamplingStrategy

   Selects the top-*T* trajectories by probability (descending order).

   .. code-block:: cpp

      #include "cudaq/ptsbe/strategies/OrderedSamplingStrategy.h"

      opts.ptsbe.max_trajectories = 100;
      opts.ptsbe.strategy =
          std::make_shared<cudaq::ptsbe::OrderedSamplingStrategy>();


.. cpp:class:: ExhaustiveSamplingStrategy : public PTSSamplingStrategy

   Enumerates every possible trajectory in lexicographic order.


.. cpp:class:: ConditionalSamplingStrategy : public PTSSamplingStrategy

   Samples trajectories that satisfy a user-supplied predicate.

   .. cpp:type:: TrajectoryPredicate = std::function<bool(const cudaq::KrausTrajectory&)>

   .. cpp:function:: explicit ConditionalSamplingStrategy(TrajectoryPredicate predicate, std::uint64_t seed = 0)

      :param predicate: Returns ``true`` for trajectories to include.
      :param seed: Random seed. ``0`` uses the global CUDA-Q seed.

   .. code-block:: cpp

      #include "cudaq/ptsbe/strategies/ConditionalSamplingStrategy.h"

      // Only single-error trajectories
      opts.ptsbe.strategy =
          std::make_shared<cudaq::ptsbe::ConditionalSamplingStrategy>(
              [](const cudaq::KrausTrajectory &t) {
                return t.countErrors() <= 1;
              });

----

Shot Allocation Strategy
-------------------------

.. cpp:struct:: ShotAllocationStrategy

   Controls how shots are distributed across selected trajectories.

   .. cpp:enum-class:: Type

      .. cpp:enumerator:: PROPORTIONAL

         *(default)* Multinomial sampling weighted by trajectory probability.
         Total is always exactly ``total_shots``.

      .. cpp:enumerator:: UNIFORM

         Equal shots per trajectory.

      .. cpp:enumerator:: LOW_WEIGHT_BIAS

         More shots to low-error trajectories.
         Weight: ``(1 + error_count)^(-bias_strength) * probability``.

      .. cpp:enumerator:: HIGH_WEIGHT_BIAS

         More shots to high-error trajectories.
         Weight: ``(1 + error_count)^(+bias_strength) * probability``.

   .. cpp:member:: Type type = Type::PROPORTIONAL

   .. cpp:member:: double bias_strength = 2.0

      Exponent for the biased strategies.

   .. cpp:member:: std::uint64_t seed = 0

      Random seed for the multinomial draw. ``0`` uses the global CUDA-Q seed.

   .. cpp:function:: explicit ShotAllocationStrategy(Type t, double bias = 2.0, std::uint64_t seed = 0)

   .. code-block:: cpp

      #include "cudaq/ptsbe/ShotAllocationStrategy.h"

      opts.ptsbe.shot_allocation = cudaq::ptsbe::ShotAllocationStrategy(
          cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
          /*bias=*/3.0);

----

Execution Data
---------------

.. cpp:struct:: PTSBEExecutionData

   Full execution trace attached to the result when
   ``return_execution_data = true``.

   .. cpp:member:: std::vector<TraceInstruction> instructions

      Ordered circuit operations (``PTSBETrace``, alias for
      ``std::vector<TraceInstruction>``).

   .. cpp:member:: std::vector<cudaq::KrausTrajectory> trajectories

      Trajectories that were sampled and executed.

   .. cpp:function:: std::size_t count_instructions(TraceInstructionType type, std::optional<std::string> name = std::nullopt) const

      Count instructions of the given type, optionally filtered by name.

   .. cpp:function:: std::optional<std::reference_wrapper<const cudaq::KrausTrajectory>> get_trajectory(std::size_t trajectory_id) const

      Look up a trajectory by ID. Returns ``std::nullopt`` if not found.


.. cpp:struct:: TraceInstruction

   A single operation in the PTSBE execution trace.

   .. cpp:member:: TraceInstructionType type

   .. cpp:member:: std::string name

      Operation name (e.g. ``"h"``, ``"depolarizing"``, ``"mz"``).

   .. cpp:member:: std::vector<std::size_t> targets

   .. cpp:member:: std::vector<std::size_t> controls

   .. cpp:member:: std::vector<double> params

   .. cpp:member:: std::optional<cudaq::kraus_channel> channel

      Populated only for ``Noise`` instructions.


.. cpp:enum-class:: TraceInstructionType

   .. cpp:enumerator:: Gate
   .. cpp:enumerator:: Noise
   .. cpp:enumerator:: Measurement

----

Trajectory and Selection Types
--------------------------------

.. cpp:namespace:: cudaq

.. cpp:struct:: KrausTrajectory

   One complete assignment of Kraus operators across all circuit noise sites.

   .. cpp:member:: std::size_t trajectory_id = 0

   .. cpp:member:: std::vector<KrausSelection> kraus_selections

      Ordered by ``circuit_location`` (ascending).

   .. cpp:member:: double probability = 0.0

      Product of the selected Kraus operator probabilities at each site.

   .. cpp:member:: std::size_t num_shots = 0

      Shots allocated to this trajectory.

   .. cpp:member:: std::size_t multiplicity = 1

      Draw count before deduplication.

   .. cpp:member:: CountsDictionary measurement_counts

      Per-trajectory measurement outcomes (populated after execution).

   .. cpp:function:: std::size_t countErrors() const

      Return the number of non-identity Kraus selections (error weight).

   .. cpp:function:: bool isOrdered() const

      Return ``true`` if ``kraus_selections`` are sorted by
      ``circuit_location``.


.. cpp:struct:: KrausSelection

   The choice of a specific Kraus operator at one noise site.

   .. cpp:member:: std::size_t circuit_location = 0

      Index of the noise site in the circuit instruction sequence.

   .. cpp:member:: std::vector<std::size_t> qubits

   .. cpp:member:: std::string op_name

      Gate name after which this noise occurs (e.g. ``"h"``).

   .. cpp:member:: KrausOperatorType kraus_operator_index = KrausOperatorType::IDENTITY

      Selected Kraus operator index. ``IDENTITY`` (0) means no error.


.. cpp:enum-class:: KrausOperatorType : std::size_t

   .. cpp:enumerator:: IDENTITY = 0

      The identity (no-error) Kraus operator.

   Values ≥ 1 correspond to actual error operators from the noise channel,
   indexed in the order they appear in the :cpp:class:`cudaq::kraus_channel`.
