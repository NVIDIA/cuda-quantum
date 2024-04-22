CUDA Quantum Python API
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

.. autoclass:: cudaq::PyKernelDecorator
    :members:
    :special-members: __str__, __call__

.. autofunction:: kernel
    
Kernel Execution
=============================

.. autofunction:: cudaq::sample
.. autofunction:: cudaq::sample_async
.. autofunction:: cudaq::observe
.. autofunction:: cudaq::observe_async
.. autofunction:: cudaq::get_state
.. autofunction:: cudaq::get_state_async
.. autofunction:: cudaq::vqe
.. autofunction:: cudaq::draw    

Backend Configuration
=============================

.. autofunction:: cudaq::has_target
.. autofunction:: cudaq::get_target
.. autofunction:: cudaq::get_targets
.. autofunction:: cudaq::set_target
.. autofunction:: cudaq::reset_target
.. autofunction:: cudaq::set_noise
.. autofunction:: cudaq::unset_noise
.. automethod:: cudaq::initialize_cudaq
.. automethod:: cudaq::num_available_gpus
.. automethod:: cudaq::set_random_seed

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

.. autoclass:: cudaq::SpinOperator
    :members:

    .. automethod:: __eq__
    .. automethod:: __add__
    .. automethod:: __radd__
    .. automethod:: __sub__
    .. automethod:: __rsub__
    .. automethod:: __mul__
    .. automethod:: __rmul__
    .. automethod:: __iter__
        
.. autofunction:: cudaq::spin.i
.. autofunction:: cudaq::spin.x
.. autofunction:: cudaq::spin.y
.. autofunction:: cudaq::spin.z

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