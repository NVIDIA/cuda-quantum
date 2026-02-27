CUDA-Q Python API
******************************

.. automodule:: cudaq

Program Construction
=============================

.. autofunction:: cudaq::make_kernel
.. [SKIP_TEST]: Reason - AttributeError: module 'cudaq' has no attribute 'from_state'
.. .. autofunction:: cudaq::from_state

.. autoclass:: cudaq::PyKernel
.. autoclass:: cudaq::Kernel

    .. automethod:: qalloc
    .. automethod:: __str__
    .. automethod:: __call__

    .. automethod:: x
    .. automethod:: cx
    .. automethod:: y
    .. automethod:: cy
    .. automethod:: z
    .. automethod:: cz
    .. automethod:: h
    .. automethod:: ch
    .. automethod:: s
    .. automethod:: sdg
    .. automethod:: cs
    .. automethod:: t
    .. automethod:: tdg
    .. automethod:: ct
    .. automethod:: rx
    .. automethod:: crx
    .. automethod:: ry
    .. automethod:: cry
    .. automethod:: rz
    .. automethod:: crz
    .. automethod:: r1
    .. automethod:: cr1
    .. automethod:: swap
    .. automethod:: cswap
    .. automethod:: exp_pauli
    .. automethod:: mx
    .. automethod:: my
    .. automethod:: mz
    .. automethod:: for_loop
    .. automethod:: adjoint
    .. automethod:: control
    .. automethod:: apply_call
    .. automethod:: u3

.. autoclass:: cudaq::PyKernelDecorator
    :members:
    :special-members: __str__, __call__

.. autofunction:: kernel
    
Kernel Execution
=============================

.. autofunction:: cudaq::sample
.. autofunction:: cudaq::sample_async
.. autofunction:: cudaq::run
.. autofunction:: cudaq::run_async    
.. autofunction:: cudaq::observe
.. autofunction:: cudaq::observe_async
.. autofunction:: cudaq::get_state
.. autofunction:: cudaq::get_state_async
.. autofunction:: cudaq::vqe
.. autofunction:: cudaq::draw
.. autofunction:: cudaq::translate
.. autofunction:: cudaq::estimate_resources

Backend Configuration
=============================

.. autofunction:: cudaq::has_target
.. autofunction:: cudaq::get_target
.. autofunction:: cudaq::get_targets
.. autofunction:: cudaq::set_target
.. autofunction:: cudaq::reset_target
.. autofunction:: cudaq::set_noise
.. autofunction:: cudaq::unset_noise
.. autofunction:: cudaq::register_set_target_callback
.. autofunction:: cudaq::unregister_set_target_callback

.. function:: cudaq.apply_noise(error_type, parameters..., targets...)

    This function is a type-safe injection of noise into a quantum kernel,
    occurring precisely at the call site of the function invocation. The
    function should be called inside CUDA-Q kernels (those annotated with
    `@cudaq.kernel`). The functionality is only supported for simulation targets, so
    it is automatically (and silently) stripped from any programs submitted to
    hardware targets.

    :param error_type: A subtype of :class:`cudaq.KrausChannel` that
        implements/defines the desired noise mechanisms as Kraus channels (e.g.
        :class:`cudaq.Depolarization2`). If you want to use a custom
        :class:`cudaq.KrausChannel` (i.e. not built-in to CUDA-Q), it must
        first be registered *outside the kernel* with `register_channel`, like
        this:

        .. code-block:: python

            class CustomNoiseChannel(cudaq.KrausChannel):
                num_parameters = 1
                num_targets = 1

            def __init__(self, params: list[float]):
                cudaq.KrausChannel.__init__(self)
                # Example: Create Kraus ops based on params
                p = params[0]
                k0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]],
                            dtype=np.complex128)
                k1 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]],
                            dtype=np.complex128)

                # Create KrausOperators and add to channel
                self.append(cudaq.KrausOperator(k0))
                self.append(cudaq.KrausOperator(k1))

                self.noise_type = cudaq.NoiseModelType.Unknown

            noise = cudaq.NoiseModel()
            noise.register_channel(CustomNoiseChannel)

    :param parameters: The precise argument pack depend on the concrete
        :class:`cudaq.KrausChannel` being used. The arguments are a concatenated
        list of parameters and targets.  For example, to apply a 2-qubit
        depolarization channel, which has `num_parameters = 1` and `num_targets =
        2`, one would write the call like this:

        .. code-block:: python

            q, r = cudaq.qubit(), cudaq.qubit()
            cudaq.apply_noise(cudaq.Depolarization2, 0.1, q, r)

    :param targets: The target qubits on which to apply the noise


.. automethod:: cudaq::initialize_cudaq
.. automethod:: cudaq::num_available_gpus
.. automethod:: cudaq::set_random_seed

Dynamics
=============================

.. autofunction:: cudaq::evolve
.. autofunction:: cudaq::evolve_async

.. autoclass:: cudaq::Schedule
.. autoclass:: cudaq.dynamics.integrator.BaseIntegrator

.. autoclass:: cudaq.dynamics.helpers.InitialState
.. autoclass:: cudaq.InitialStateType
.. autoclass:: cudaq.IntermediateResultSave

Operators
=============================

.. autoclass:: cudaq.operators.OperatorSum
.. autoclass:: cudaq.operators.ProductOperator
.. autoclass:: cudaq.operators.ElementaryOperator
.. autoclass:: cudaq.operators.ScalarOperator
   :members:

.. autoclass:: cudaq.operators.RydbergHamiltonian
    :members:
    :special-members: __init__

.. autoclass:: cudaq.SuperOperator
   :members:

.. automethod:: cudaq.operators.define
.. automethod:: cudaq.operators.instantiate

Spin Operators
-----------------------------
.. autoclass:: cudaq.operators.spin.SpinOperator
   :members:
.. autoclass:: cudaq.operators.spin.SpinOperatorTerm
   :members:
.. autoclass:: cudaq.operators.spin.SpinOperatorElement
   :members:

.. automodule:: cudaq.spin
    :imported-members:
    :members:

Fermion Operators
-----------------------------
.. autoclass:: cudaq.operators.fermion.FermionOperator
   :members:
.. autoclass:: cudaq.operators.fermion.FermionOperatorTerm
   :members:
.. autoclass:: cudaq.operators.fermion.FermionOperatorElement
   :members:

.. automodule:: cudaq.fermion
    :imported-members:
    :members:

Boson Operators
-----------------------------
.. autoclass:: cudaq.operators.boson.BosonOperator
   :members:
.. autoclass:: cudaq.operators.boson.BosonOperatorTerm
   :members:
.. autoclass:: cudaq.operators.boson.BosonOperatorElement
   :members:

.. automodule:: cudaq.boson
    :imported-members:
    :members:

General Operators
-----------------------------
.. autoclass:: cudaq.operators.MatrixOperator
   :members:
.. autoclass:: cudaq.operators.MatrixOperatorTerm
   :members:
.. autoclass:: cudaq.operators.MatrixOperatorElement
   :members:

.. automodule:: cudaq.operators.custom
    :imported-members:
    :members:

Data Types
=============================

.. autoclass:: cudaq::SimulationPrecision
    :members:
    
.. autoclass:: cudaq::Target
    :members:

.. autoclass:: cudaq::State
    :members:

.. autoclass:: cudaq::Tensor

.. autoclass:: cudaq::QuakeValue

    .. automethod:: __add__
    .. automethod:: __radd__
    .. automethod:: __sub__
    .. automethod:: __rsub__
    .. automethod:: __neg__
    .. automethod:: __mul__
    .. automethod:: __rmul__
    .. automethod:: __getitem__
    .. automethod:: slice

.. autoclass:: cudaq::qubit
.. autoclass:: cudaq::qreg
.. autoclass:: cudaq::qvector

.. autoclass:: cudaq::ComplexMatrix
    :members:
    :special-members: __getitem__, __str__

.. autoclass:: cudaq::SampleResult
    :members:
    :special-members: __getitem__, __len__, __iter__

.. autoclass:: cudaq::AsyncSampleResult
    :members:

.. autoclass:: cudaq::ObserveResult
    :members:

.. autoclass:: cudaq::AsyncObserveResult
    :members:

.. autoclass:: cudaq::AsyncStateResult
    :members:

.. autoclass:: cudaq::OptimizationResult
    :members:

.. autoclass:: cudaq::EvolveResult
    :members:

.. autoclass:: cudaq::AsyncEvolveResult
    :members:

.. autoclass:: cudaq::Resources
    :members:

Optimizers
-----------------
.. |:spellcheck-disable:| replace:: \

.. py:method:: optimize(dimensions: int, function) -> tuple[float, list[float]]
   :noindex:

   Run the optimization procedure.

   :param dimensions: The number of parameters to optimize
   :param function: The objective function to minimize
   :returns: tuple of (optimal_value, optimal_parameters)

.. py:method:: requires_gradients() -> bool
   :noindex:

   Check whether this optimizer requires gradient information.

   :returns: True if gradients required, False otherwise

.. autoclass:: cudaq.optimizers::GradientDescent
    :members:
    :exclude-members: optimize, requires_gradients

.. autoclass:: cudaq.optimizers::COBYLA
    :members:
    :exclude-members: optimize, requires_gradients

.. autoclass:: cudaq.optimizers::NelderMead
    :members:
    :exclude-members: optimize, requires_gradients

.. autoclass:: cudaq.optimizers::LBFGS
    :members:
    :exclude-members: optimize, requires_gradients

.. autoclass:: cudaq.optimizers::Adam
    :members:
    :exclude-members: optimize, requires_gradients

.. autoclass:: cudaq.optimizers::SGD
    :members:
    :exclude-members: optimize, requires_gradients

.. autoclass:: cudaq.optimizers::SPSA
    :members:
    :exclude-members: optimize, requires_gradients

.. |:spellcheck-enable:| replace:: \

Gradients
-----------------

.. autoclass:: cudaq.gradients::gradient
    :members:

.. autoclass:: cudaq.gradients::CentralDifference
    :members:

.. autoclass:: cudaq.gradients::ForwardDifference
    :members:

.. autoclass:: cudaq.gradients::ParameterShift
    :members:

Noisy Simulation
-----------------

.. autoclass:: cudaq::NoiseModel
    :members:
    :exclude-members: register_channel
    :special-members: __init__

.. autoclass:: cudaq::BitFlipChannel
    :members:
    :special-members: __init__

.. autoclass:: cudaq::PhaseFlipChannel
    :members:
    :special-members: __init__

.. autoclass:: cudaq::DepolarizationChannel
    :members:
    :special-members: __init__

.. autoclass:: cudaq::AmplitudeDampingChannel
    :members:
    :special-members: __init__

.. autoclass:: cudaq::PhaseDamping

.. autoclass:: cudaq::XError

.. autoclass:: cudaq::YError

.. autoclass:: cudaq::ZError

.. autoclass:: cudaq::Pauli1

.. autoclass:: cudaq::Pauli2

.. autoclass:: cudaq::Depolarization1

.. autoclass:: cudaq::Depolarization2

.. autoclass:: cudaq::KrausChannel
    :members:
    :special-members: __getitem__

.. autoclass:: cudaq::KrausOperator
    :members:

MPI Submodule
=============================

.. automethod:: cudaq.mpi::initialize
.. automethod:: cudaq.mpi::rank
.. automethod:: cudaq.mpi::num_ranks
.. automethod:: cudaq.mpi::all_gather
.. automethod:: cudaq.mpi::broadcast
.. automethod:: cudaq.mpi::is_initialized
.. automethod:: cudaq.mpi::finalize

ORCA Submodule
=============================

.. automethod:: cudaq.orca::sample

PTSBE Submodule
=============================

.. _ptsbe_api:

The ``cudaq.ptsbe`` submodule implements Pre-Trajectory Sampling with Batch
Execution (PTSBE). For a conceptual overview and usage tutorial see
:doc:`../../using/ptsbe`.

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
   :param `args`: Positional arguments forwarded to the kernel.
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

   :raises `RuntimeError`: If the kernel contains mid-circuit measurements,
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
