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
    .. automethod:: c_if
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

.. autoclass:: cudaq.optimizers::optimizer

.. autoclass:: cudaq.optimizers::GradientDescent
    :members:

.. autoclass:: cudaq.optimizers::COBYLA
    :members:

.. autoclass:: cudaq.optimizers::NelderMead
    :members:

.. autoclass:: cudaq.optimizers::LBFGS
    :members:

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
