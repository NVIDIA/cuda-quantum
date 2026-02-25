.. |:spellcheck-disable:| replace:: \

PTSBE API Reference
********************

.. _ptsbe_api:

This page documents the public API for Pre-Trajectory Sampling with Batch
Execution (PTSBE). For a conceptual overview and usage tutorial see
:doc:`../using/ptsbe`.

.. contents:: Contents
   :local:
   :depth: 2

----

Python API — ``cudaq.ptsbe``
=============================

Sampling Functions
-------------------

.. py:function:: cudaq.ptsbe.sample(kernel, *args, shots_count=1000, noise_model=None, max_trajectories=None, sampling_strategy=None, shot_allocation=None, return_execution_data=False)

   Sample a quantum kernel using Pre-Trajectory Sampling with Batch Execution.

   Pre-samples *T* unique noise trajectories from the circuit's noise model
   and batches circuit executions by unique trajectory. Each trajectory is
   simulated as a pure-state circuit; results are merged into a single
   :class:`~cudaq.SampleResult`.

   When any argument is a list (broadcast mode), the kernel is executed for
   each element of the list and a list of results is returned.

   :param kernel: The quantum kernel to execute. Must be a static circuit
       with no mid-circuit measurements or measurement-dependent conditional
       logic.
   :param args: Positional arguments forwarded to the kernel.
   :param int shots_count: Total number of measurement shots to distribute
       across all trajectories. Default: ``1000``.
   :param noise_model: Noise model describing gate-level error channels.
       Noise can also be injected inside the kernel via
       ``cudaq.apply_noise()``; both can be combined. Default: ``None``
       (no noise).
   :type noise_model: :class:`cudaq.NoiseModel` or ``None``
   :param max_trajectories: Maximum number of unique trajectories to
       generate. ``None`` defaults to ``shots_count``. Setting an explicit
       limit (e.g. 500) enables trajectory reuse and is strongly recommended
       for large shot counts.
   :type max_trajectories: int or ``None``
   :param sampling_strategy: Strategy used to select trajectories from the
       noise space. ``None`` uses the default
       :class:`~cudaq.ptsbe.ProbabilisticSamplingStrategy`.
   :type sampling_strategy: :class:`~cudaq.ptsbe.PTSSamplingStrategy` or ``None``
   :param shot_allocation: Strategy used to distribute shots across the
       selected trajectories. ``None`` uses
       :attr:`~cudaq.ptsbe.ShotAllocationStrategy.Type.PROPORTIONAL`.
   :type shot_allocation: :class:`~cudaq.ptsbe.ShotAllocationStrategy` or ``None``
   :param bool return_execution_data: When ``True``, attaches the full
       execution trace (circuit instructions, trajectory specifications, and
       per-trajectory measurement counts) to the returned result. Default:
       ``False``.

   :returns: Aggregated measurement outcomes. In broadcast mode, a list of
       results is returned.
   :rtype: :class:`cudaq.ptsbe.PTSBESampleResult`

   :raises RuntimeError: If the kernel contains mid-circuit measurements,
       conditional feedback, unsupported noise channels, or invalid arguments.

   .. code-block:: python

      import cudaq
      from cudaq import ptsbe

      cudaq.set_target("nvidia")

      @cudaq.kernel
      def bell():
          q = cudaq.qvector(2)
          h(q[0])
          cx(q[0], q[1])
          mz(q)

      noise = cudaq.NoiseModel()
      noise.add_channel("h", [0], cudaq.DepolarizationChannel(0.01))

      result = ptsbe.sample(bell, shots_count=10_000,
                            noise_model=noise, max_trajectories=200)
      print(result)


.. py:function:: cudaq.ptsbe.sample_async(kernel, *args, shots_count=1000, noise_model=None, max_trajectories=None, sampling_strategy=None, shot_allocation=None, return_execution_data=False)

   Asynchronous variant of :func:`~cudaq.ptsbe.sample`. Submits the job
   without blocking and returns a future.

   All parameters are identical to :func:`~cudaq.ptsbe.sample`.

   :returns: A future whose ``.get()`` method returns the
       :class:`~cudaq.ptsbe.PTSBESampleResult`.
   :rtype: :class:`~cudaq.AsyncSampleResult`

   :raises: Any exception raised during kernel execution is captured and
       re-raised when ``.get()`` is called on the returned future.

   .. code-block:: python

      future = ptsbe.sample_async(bell, shots_count=10_000, noise_model=noise)
      # ... do other work ...
      result = future.get()

----

Result Type
------------

.. py:class:: cudaq.ptsbe.PTSBESampleResult

   Extends :class:`cudaq.SampleResult` with an optional
   :class:`~cudaq.ptsbe.PTSBEExecutionData` payload produced when
   ``return_execution_data=True``.

   .. py:method:: has_execution_data() -> bool

      Return ``True`` if execution data is attached to this result.

   .. py:method:: execution_data() -> PTSBEExecutionData

      Return the attached :class:`~cudaq.ptsbe.PTSBEExecutionData`.

      :raises RuntimeError: If no execution data is available. Check
          :meth:`has_execution_data` first.

----

Trajectory Sampling Strategies
--------------------------------

.. py:class:: cudaq.ptsbe.PTSSamplingStrategy

   Abstract base class for trajectory sampling strategies. Subclass and
   implement :meth:`generate_trajectories` to define a custom strategy.

   .. py:method:: generate_trajectories(noise_points, max_trajectories) -> list[KrausTrajectory]

      Generate up to *max_trajectories* unique trajectories from the given
      noise points.

      :param noise_points: Noise site information extracted from the circuit.
      :param int max_trajectories: Upper bound on the number of trajectories.
      :returns: List of unique :class:`~cudaq.ptsbe.KrausTrajectory` objects.

   .. py:method:: name() -> str

      Return a human-readable name for this strategy.


.. py:class:: cudaq.ptsbe.ProbabilisticSamplingStrategy(seed=0)

   Randomly samples unique trajectories weighted by their occurrence
   probability. Produces a representative cross-section of the noise space.
   Duplicate trajectories are discarded.

   :param int seed: Random seed for reproducibility. ``0`` uses the global
       CUDA-Q seed if set, otherwise a random device seed.

   .. code-block:: python

      strategy = ptsbe.ProbabilisticSamplingStrategy(seed=42)
      result = ptsbe.sample(bell, shots_count=10_000,
                            noise_model=noise,
                            sampling_strategy=strategy)


.. py:class:: cudaq.ptsbe.OrderedSamplingStrategy()

   Selects the top-*T* trajectories sorted by probability in descending
   order. Ensures the highest-probability noise realizations are always
   represented. Best when the noise space is dominated by a small number
   of likely error patterns.

   .. code-block:: python

      result = ptsbe.sample(bell, shots_count=10_000,
                            noise_model=noise,
                            max_trajectories=100,
                            sampling_strategy=ptsbe.OrderedSamplingStrategy())


.. py:class:: cudaq.ptsbe.ExhaustiveSamplingStrategy()

   Enumerates every possible trajectory in lexicographic order. Produces a
   complete representation of the noise space. Only practical when the noise
   space is small (few noise sites and low Kraus operator count).


.. py:class:: cudaq.ptsbe.ConditionalSamplingStrategy(predicate, seed=0)

   Samples trajectories that satisfy a user-supplied predicate function.
   Useful for targeted studies such as restricting to single-qubit error
   events or trajectories below a probability threshold.

   :param predicate: A callable ``(KrausTrajectory) -> bool`` that returns
       ``True`` for trajectories to include.
   :param int seed: Random seed. ``0`` uses the global CUDA-Q seed.

   .. code-block:: python

      # Keep only trajectories with at most one error
      strategy = ptsbe.ConditionalSamplingStrategy(
          predicate=lambda traj: traj.count_errors() <= 1,
          seed=42,
      )
      result = ptsbe.sample(bell, shots_count=10_000,
                            noise_model=noise,
                            sampling_strategy=strategy)

----

Shot Allocation Strategy
-------------------------

.. py:class:: cudaq.ptsbe.ShotAllocationStrategy(type=ShotAllocationStrategy.Type.PROPORTIONAL, bias_strength=2.0, seed=0)

   Controls how the total shot count is distributed across the selected
   trajectories after trajectory sampling.

   :param type: Allocation strategy type.
   :type type: :class:`~cudaq.ptsbe.ShotAllocationStrategy.Type`
   :param float bias_strength: Exponent used by the biased strategies.
       Higher values produce stronger bias. Default: ``2.0``.
   :param int seed: Random seed used by probabilistic allocation (PROPORTIONAL
       and biased strategies). ``0`` uses the global CUDA-Q seed.

   .. py:class:: Type

      .. py:attribute:: PROPORTIONAL

         *(default)* Shots are allocated via multinomial sampling weighted by
         trajectory probability. The total is always exactly ``shots_count``
         and every trajectory with non-zero probability receives a fair share.

      .. py:attribute:: UNIFORM

         Equal shots per trajectory regardless of probability.

      .. py:attribute:: LOW_WEIGHT_BIAS

         Biases more shots toward trajectories with fewer errors (lower Kraus
         weight). Weight formula:
         ``(1 + error_count)^(-bias_strength) * probability``.

      .. py:attribute:: HIGH_WEIGHT_BIAS

         Biases more shots toward trajectories with more errors. Weight
         formula: ``(1 + error_count)^(+bias_strength) * probability``.

   .. code-block:: python

      alloc = ptsbe.ShotAllocationStrategy(
          ptsbe.ShotAllocationStrategy.Type.LOW_WEIGHT_BIAS,
          bias_strength=3.0,
      )
      result = ptsbe.sample(bell, shots_count=10_000,
                            noise_model=noise,
                            shot_allocation=alloc)

----

Execution Data
---------------

.. py:class:: cudaq.ptsbe.PTSBEExecutionData

   Container for the full PTSBE execution trace. Returned by
   :meth:`~cudaq.ptsbe.PTSBESampleResult.execution_data` when
   ``return_execution_data=True``.

   .. py:attribute:: instructions
      :type: list[TraceInstruction]

      Ordered list of circuit operations: gates, noise channel locations, and
      terminal measurements.

   .. py:attribute:: trajectories
      :type: list[KrausTrajectory]

      The trajectories that were sampled and executed.

   .. py:method:: count_instructions(type, name=None) -> int

      Count instructions of the given :class:`~cudaq.ptsbe.TraceInstructionType`,
      optionally filtered by operation name.

   .. py:method:: get_trajectory(trajectory_id)

      Look up a trajectory by its ID. Returns ``None`` if not found.


.. py:class:: cudaq.ptsbe.TraceInstruction

   A single operation in the PTSBE execution trace.

   .. py:attribute:: type
      :type: TraceInstructionType

      Whether this instruction is a ``Gate``, ``Noise``, or ``Measurement``
      (see :class:`~cudaq.ptsbe.TraceInstructionType`).

   .. py:attribute:: name
      :type: str

      Operation name (e.g. ``"h"``, ``"cx"``, ``"depolarizing"``, ``"mz"``).

   .. py:attribute:: targets
      :type: list[int]

      Target qubit indices.

   .. py:attribute:: controls
      :type: list[int]

      Control qubit indices. Empty for non-controlled operations.

   .. py:attribute:: params
      :type: list[float]

      Gate rotation angles or noise channel parameters.

   .. py:attribute:: channel

      The noise channel (``cudaq.KrausChannel``), or ``None``.
      Populated only for ``Noise`` instructions.


.. py:class:: cudaq.ptsbe.TraceInstructionType

   Discriminator enum for :class:`~cudaq.ptsbe.TraceInstruction` entries.

   .. py:attribute:: Gate

      A unitary quantum gate (H, X, CNOT, RX, …).

   .. py:attribute:: Noise

      A noise channel injection point.

   .. py:attribute:: Measurement

      A terminal measurement operation.


.. py:class:: cudaq.ptsbe.KrausTrajectory

   One complete assignment of Kraus operators across all noise sites in the
   circuit.

   .. py:attribute:: trajectory_id
      :type: int

      Unique identifier assigned during trajectory sampling.

   .. py:attribute:: probability
      :type: float

      Product of the probabilities of the selected Kraus operators at each
      noise site.

   .. py:attribute:: num_shots
      :type: int

      Number of measurement shots allocated to this trajectory.

   .. py:attribute:: multiplicity
      :type: int

      Number of times this trajectory was drawn before deduplication.

   .. py:attribute:: kraus_selections
      :type: list[KrausSelection]

      Ordered list of Kraus operator choices, one per noise site.

   .. py:method:: count_errors() -> int

      Return the number of non-identity Kraus operators in this trajectory
      (the *error weight*).


.. py:class:: cudaq.ptsbe.KrausSelection

   The choice of a specific Kraus operator at one noise site.

   .. py:attribute:: circuit_location
      :type: int

      Index of the noise site in the circuit's instruction sequence.

   .. py:attribute:: qubits
      :type: list[int]

      Qubits affected by this noise operation.

   .. py:attribute:: op_name
      :type: str

      Name of the gate after which this noise occurs (e.g. ``"h"``).

   .. py:attribute:: kraus_operator_index
      :type: int

      Index of the selected Kraus operator. ``0`` is the identity
      (no error); values ≥ 1 represent actual error operators.

   .. py:attribute:: is_error
      :type: bool

      ``True`` if the selected Kraus operator is not the identity
      (i.e. an actual error occurred at this noise site).

----

C++ API — ``cudaq::ptsbe``
============================

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
.. |:spellcheck-enable:| replace:: \
