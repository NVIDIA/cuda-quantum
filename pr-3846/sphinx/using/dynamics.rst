Dynamics Simulation 
+++++++++++++++++++++

.. _dynamics:

CUDA-Q enables the design, simulation and execution of quantum dynamics via 
the ``evolve`` API. Specifically, this API allows us to solve the time evolution 
of quantum systems or models. In the simulation mode, CUDA-Q provides the ``dynamics``
backend target, which is based on the cuQuantum library, optimized for performance and scale
on NVIDIA GPU.

Quick Start
^^^^^^^^^^^^

In the example below, we demonstrate a simple time evolution simulation workflow comprising of the 
following steps:

1. Define a quantum system model

A quantum system model is defined by a Hamiltonian. 
For example, a superconducting `transmon <https://en.wikipedia.org/wiki/Transmon>`_ qubit can be modeled by the following Hamiltonian

.. math:: 
    
    H = \frac{\omega_z}{2} \sigma_z + \omega_x \cos(\omega_d t)\sigma_x,

where :math:`\sigma_z` and :math:`\sigma_x` are Pauli Z and X operators, respectively.

Using CUDA-Q `operator`, the above time-dependent Hamiltonian can be set up as follows.

.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Transmon]
        :end-before: [End Transmon]

  In particular, `ScalarOperator` provides an easy way to model arbitrary time-dependent control signals.
  Details about CUDA-Q `operator`, including builtin operators that it supports can be found :ref:`here <operators>`.

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin Transmon]
        :end-before: [End Transmon]

  Details about CUDA-Q `operator`, including builtin operators that it supports can be found :ref:`here <operators>`.

2. Setup the evolution simulation

The below code snippet shows how to simulate the time-evolution of the above system
with `cudaq.evolve`.

.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Evolve]
        :end-before: [End Evolve]

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin Evolve]
        :end-before: [End Evolve]

Specifically, we need to set up the simulation by providing:

- The system model in terms of a Hamiltonian as well as any decoherence terms, so-called `collapse_operators`.

- The dimensionality of component systems in the model. CUDA-Q `evolve` allows users to model arbitrary multi-level systems, such as photonic Fock space.

- The initial quantum state.

- The time schedule, aka time steps, of the evolution.

- Any 'observable' operator that we want to measure the expectation value with respect to the evolving state.


.. note::

    By default, `evolve` will only return the final state and expectation values.
    To save intermediate results (at each time step specified in the schedule),
    the `store_intermediate_results` flag must be set to `True`.

3. Retrieve and plot the results

Once the simulation is complete, we can retrieve the final state and the expectation values
as well as intermediate values at each time step (with `store_intermediate_results=cudaq.IntermediateResultSave.ALL`).

.. note::
    
    Storing intermediate states can be memory-intensive, especially for large systems.
    If you only need the intermediate expectation values, you can set `store_intermediate_results` to 
    `cudaq.IntermediateResultSave.EXPECTATION_VALUES` (Python) / `cudaq::IntermediateResultSave::ExpectationValue` (C++) instead.

For example, we can plot the Pauli expectation value for the above simulation as follows.

.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Plot]
        :end-before: [End Plot]

  In particular, for each time step, `evolve` captures an array of expectation values, one for each
  observable. Hence, we convert them into sequences for plotting purposes.

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin Print]
        :end-before: [End Print]

Examples that illustrate how to use the ``dynamics`` target are available 
in the `CUDA-Q repository <https://github.com/NVIDIA/cuda-quantum/tree/main/docs/sphinx/examples/python/dynamics>`__. 

Operator
^^^^^^^^^^

.. _operators:

CUDA-Q provides builtin definitions for commonly-used operators, 
such as the ladder operators (:math:`a` and :math:`a^\dagger`) of a harmonic oscillator, 
the Pauli spin operators for a two-level system, etc.

Here is a list of those operators.

.. list-table:: Builtin Operators
        :widths: 20 50 
        :header-rows: 1

        *   - Name
            - Description
        *   - `identity`
            - Identity operator
        *   - `zero`
            - Zero or null operator
        *   - `annihilate`
            - Bosonic annihilation operator (:math:`a`)
        *   - `create`
            - Bosonic creation operator (:math:`a^\dagger`)
        *   - `number`
            - Number operator of a bosonic mode (equivalent to :math:`a^\dagger a`)
        *   - `parity`
            - Parity operator of a bosonic mode (defined as :math:`e^{i\pi a^\dagger a}`)
        *   - `displace`
            - Displacement operator of complex amplitude :math:`\alpha` (`displacement`). It is defined as :math:`e^{\alpha a^\dagger - \alpha^* a}`.  
        *   - `squeeze`
            - Squeezing operator of complex squeezing amplitude :math:`z` (`squeezing`). It is defined as :math:`\exp(\frac{1}{2}(z^*a^2 - z a^{\dagger 2}))`.
        *   - `position`
            - Position operator (equivalent to :math:`(a^\dagger + a)/2`)
        *   - `momentum`
            - Momentum operator (equivalent to :math:`i(a^\dagger - a)/2`)
        *   - `spin.x`
            - Pauli :math:`\sigma_x` operator
        *   - `spin.y`
            - Pauli :math:`\sigma_y` operator
        *   - `spin.z`
            - Pauli :math:`\sigma_z` operator
        *   - `spin.plus`
            - Pauli raising (:math:`\sigma_+`) operator
        *   - `spin.minus`
            - Pauli lowering (:math:`\sigma_-`) operator

As an example, let's look at the Jaynes-Cummings model, which describes 
the interaction between a two-level atom and a light (Boson) field.

Mathematically, the Hamiltonian can be expressed as

.. math:: 
    
    H = \omega_c a^\dagger a + \omega_a \frac{\sigma_z}{2} + \frac{\Omega}{2}(a\sigma_+ + a^\dagger \sigma_-).

This Hamiltonian can be converted to CUDA-Q `Operator` representation with

.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Jaynes-Cummings]
        :end-before: [End Jaynes-Cummings]

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin Jaynes-Cummings]
        :end-before: [End Jaynes-Cummings]

In the above code snippet, we map the cavity light field to degree index 1 and the two-level atom to degree index 0. 
The description of composite quantum system dynamics is independent from the Hilbert space of the system components.
The latter is specified by the dimension map that is provided to the `cudaq.evolve` call. 

Builtin operators support both dense and multi-diagonal sparse formats. 
Depending on the sparsity of operator matrix and/or the sub-system dimension, CUDA-Q will
either use the dense or multi-diagonal data formats for optimal performance.

Specifically, the following environment variable options are applicable to the :code:`dynamics` target. 
Any environment variables must be set prior to setting the target or running "`import cudaq`".

.. list-table:: **Additional environment variable options for the `dynamics` target**
  :widths: 20 30 50

  * - Option
    - Value
    - Description
  * - ``CUDAQ_DYNAMICS_MIN_MULTIDIAGONAL_DIMENSION``
    - Non-negative number
    - The minimum sub-system dimension on which the operator acts to activate multi-diagonal data format. For example, if a minimum dimension configuration of `N` is set, all operators acting on degrees of freedom (sub-system) whose dimension is less than or equal to `N` would always use the dense format. The final data format to be used depends on the next configuration. The default is 4.
  * - ``CUDAQ_DYNAMICS_MAX_DIAGONAL_COUNT_FOR_MULTIDIAGONAL``
    - Non-negative number
    - The maximum number of diagonals for multi-diagonal representation. If the operator matrix has more diagonals than this value, the dense format will be used. Default is 1, i.e., operators with only one diagonal line (center, lower, or upper) will use the multi-diagonal sparse storage. 

Time-Dependent Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _time_dependent:

In the previous examples of operator construction, we assumed that the systems under consideration were described by time-independent Hamiltonian. 
However, we may want to simulate systems whose Hamiltonian operators have explicit time dependence.

CUDA-Q provides multiple ways to construct time-dependent operators.

1. Time-dependent coefficient

CUDA-Q `ScalarOperator` can be used to wrap a Python/C++ function that returns the coefficient value at a specific time.

As an example, we will look at a time-dependent Hamiltonian of the form :math:`H = H_0 + f(t)H_1`, 
where :math:`f(t)` is the time-dependent driving strength given as :math:`cos(\omega t)`.

The following code sets up the problem

.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Hamiltonian]
        :end-before: [End Hamiltonian]

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin Hamiltonian]
        :end-before: [End Hamiltonian]

2. Time-dependent operator

We can also construct a time-dependent operator from a function that returns a complex matrix representing the time dynamics of 
that operator.

As an example, let's looks at the `displacement operator <https://en.wikipedia.org/wiki/Displacement_operator>`__. It can be defined as follows:


.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin DefineOp]
        :end-before: [End DefineOp]

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin DefineOp]
        :end-before: [End DefineOp]

The defined operator is parameterized by the `displacement` amplitude. To create simulate the evolution of an 
operator under a time dependent displacement amplitude, we can define how the amplitude changes in time:

.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Schedule1]
        :end-before: [End Schedule1]

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin Schedule1]
        :end-before: [End Schedule1]

Let's say we want to add a squeezing term to the system operator. We can independently vary the squeezing 
amplitude and the displacement amplitude by instantiating a schedule with a custom function that returns 
the desired value for each parameter: 

.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Schedule2]
        :end-before: [End Schedule2]

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin Schedule2]
        :end-before: [End Schedule2]

Compile and Run C++ program

.. tab:: C++

    .. code:: bash 

        nvq++ --target dynamics dynamics.cpp -o dynamics && ./dynamics

Super-operator Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _generic_rhs:

In the previous examples, we assumed that the system dynamics is driven by a `Lindblad` master equation, which is specified by the Hamiltonian operator and the collapse operators.

However, we may want to simulate an arbitrary state evolution equation, whereby the right-hand-side of the differential equation is provided as a generic super-operator.

CUDA-Q provides a `SuperOperator` (Python) / `super_op` (C++) class that can be used to represent the right-hand-side of the evolution equation. A super-operator can be constructed as a linear combination (sum) of left and/or right multiplication actions of `Operator` instances.

As an example, we will look at specifying the Schrodinger's equation :math:`\frac{d|\Psi\rangle}{dt} = -i H |\Psi\rangle` as a super-operator.

.. tab:: Python

  .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin SuperOperator]
        :end-before: [End SuperOperator]

.. tab:: C++

  .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin SuperOperator]
        :end-before: [End SuperOperator]

The super-operator, once constructed, can be used in the `evolve` API instead of the Hamiltonian and collapse operators as shown in the above examples.

Numerical Integrators
^^^^^^^^^^^^^^^^^^^^^^^^

.. _integrators:

For Python, CUDA-Q provides a set of numerical integrators, to be used with the ``dynamics``
backend target.

.. list-table:: Numerical Integrators
        :widths: 20 50 
        :header-rows: 1

        *   - Name
            - Description
        *   - `RungeKuttaIntegrator`
            - Explicit 4th-order Runge-Kutta method (default integrator)
        *   - `ScipyZvodeIntegrator`
            - Complex-valued variable-coefficient ordinary differential equation solver (provided by SciPy)
        *   - `CUDATorchDiffEqDopri5Integrator`
            - Runge-Kutta of order 5 of Dormand-Prince-Shampine (provided by `torchdiffeq`) 
        *   - `CUDATorchDiffEqAdaptiveHeunIntegrator`
            - Runge-Kutta of order 2 (provided by `torchdiffeq`) 
        *   - `CUDATorchDiffEqBosh3Integrator`
            - Runge-Kutta of order 3 of Bogacki-Shampine (provided by `torchdiffeq`) 
        *   - `CUDATorchDiffEqDopri8Integrator`
            - Runge-Kutta of order 8 of Dormand-Prince-Shampine (provided by `torchdiffeq`)  
        *   - `CUDATorchDiffEqEulerIntegrator`
            - Euler method (provided by `torchdiffeq`) 
        *   - `CUDATorchDiffEqExplicitAdamsIntegrator`
            - Explicit Adams-Bashforth method (provided by `torchdiffeq`) 
        *   - `CUDATorchDiffEqImplicitAdamsIntegrator`
            - Implicit Adams-Bashforth-Moulton method (provided by `torchdiffeq`) 
        *   - `CUDATorchDiffEqMidpointIntegrator`
            - Midpoint method (provided by `torchdiffeq`) 
        *   - `CUDATorchDiffEqRK4Integrator`
            - Fourth-order Runge-Kutta with 3/8 rule (provided by `torchdiffeq`) 
     
.. note::
    To use Torch-based integrators, users need to install `torchdiffeq` (e.g., with `pip install torchdiffeq`).
    This is an optional dependency of CUDA-Q, thus will not be installed by default.

.. note::

    If you are using CUDA 12.8 on Blackwell, you may need to install nightly torch.

    See :ref:`Blackwell Torch Dependencies <blackwell-torch-dependences>` for more information.

.. warning:: 
    Torch-based integrators require a CUDA-enabled Torch installation. Depending on your platform (e.g., `aarch64`),
    the default Torch pip package may not have CUDA support. 

    The below command can be used to verify your installation:

    .. code:: bash

        python3 -c "import torch; print(torch.version.cuda)"

    If the output is a '`None`' string, it indicates that your Torch installation does not support CUDA.
    In this case, you need to install a CUDA-enabled Torch package via other mechanisms, e.g., building Torch from source or
    using their Docker images.

For C++, CUDA-Q provides Runge-Kutta integrator, to be used with the ``dynamics``
backend target.

.. list-table:: Numerical Integrators
        :widths: 20 50
        :header-rows: 1

        *   - Name
            - Description
        *   - `runge_kutta`
            - 1st-order (Euler method), 2nd-order (Midpoint method), and 4th-order (classical Runge-Kutta method).

Batch simulation
^^^^^^^^^^^^^^^^^

.. _cudensitymat_batching:

CUDA-Q ``dynamics`` target supports batch simulation, which allows users to run multiple simulations simultaneously.
This batching capability applies to (1) multiple initial states and/or (2) multiple Hamiltonians.

Batching can significantly improve performance when simulating many small identical system dynamics, e.g., parameter sweeping or tomography. 

For example, we can simulate the time evolution of multiple initial states with the same Hamiltonian as follows:

.. tab:: Python

    .. literalinclude:: ../snippets/python/using/backends/dynamics_state_batching.py
        :language: python
        :start-after: [Begin State Batching]
        :end-before: [End State Batching]

.. tab:: C++

    .. literalinclude:: ../snippets/cpp/using/backends/dynamics_state_batching.cpp
        :language: cpp
        :start-after: [Begin State Batching]
        :end-before: [End State Batching]

Similarly, we can also batch simulate the time evolution of multiple Hamiltonians as follows:

.. tab:: Python

    .. literalinclude:: ../snippets/python/using/backends/dynamics_operator_batching.py
        :language: python
        :start-after: [Begin Operator Batching]
        :end-before: [End Operator Batching]

    In this example, we show the most generic batching capability, where each Hamiltonian in the batch corresponds to a specific initial state.
    In other words, the vector of Hamiltonians and the vector of initial states are of the same length.
    If only one initial state is provided, it will be used for all Hamiltonians in the batch.

.. tab:: C++

    .. literalinclude:: ../snippets/cpp/using/backends/dynamics_operator_batching.cpp
        :language: cpp
        :start-after: [Begin Operator Batching]
        :end-before: [End Operator Batching]

The results of the batch simulation will be returned as a list of evolve result objects, one for each Hamiltonian in the batch.
For example, we can extract the time evolution results of the expectation values for each Hamiltonian in the batch as follows:

.. tab:: Python

    .. literalinclude:: ../snippets/python/using/backends/dynamics_operator_batching.py
        :language: python
        :start-after: [Begin Batch Results]
        :end-before: [End Batch Results]

    The expectation values are returned as a list of lists, where each inner list corresponds to the expectation values for the observables at each time step for the respective Hamiltonian in the batch.

    .. code:: bash     

        all_exp_val_x = [[0.0, ...], [0.0, ...], ..., [0.0, ...]]
        all_exp_val_y = [[0.0, ...], [0.0, ...], ..., [0.0, ...]]
        all_exp_val_z = [[1.0, ...], [1.0, ...], ..., [1.0, ...]]

.. tab:: C++

    .. literalinclude:: ../snippets/cpp/using/backends/dynamics_operator_batching.cpp
        :language: cpp
        :start-after: [Begin Batch Results]
        :end-before: [End Batch Results]

    The expectation values are returned as a list of lists, where each inner list corresponds to the expectation values for the observables at each time step for the respective Hamiltonian in the batch.

    .. code:: bash     

        all_exp_val_x = [[0.0, ...], [0.0, ...], ..., [0.0, ...]]
        all_exp_val_y = [[0.0, ...], [0.0, ...], ..., [0.0, ...]]
        all_exp_val_z = [[1.0, ...], [1.0, ...], ..., [1.0, ...]]    


Collapse operators and super-operators can also be batched in a similar manner. 
Specifically, if the `collapse_operators` parameter is a nested list of operators (:math:`\{\{L\}_1, \{\{L\}_2, ...\}`), 
then each set of collapsed operators in the list will be applied to the corresponding Hamiltonian in the batch.


In order for all Hamiltonians to be batched, they must have the same structure, i.e., same number of product terms and those terms must act on the same degrees of freedom.
The order of the terms in the Hamiltonian does not matter, nor do the coefficient values/callback functions and the specific operators on those product terms.
Here are a couple of examples of Hamiltonians that can or cannot be batched:

.. list-table:: 
    :widths: 50 50 50
    :header-rows: 1

    *   - First Hamiltonian
        - Second Hamiltonian
        - Batchable?
    *   - :math:`H_1 = \omega_1 \sigma_z(0)`
        - :math:`H_2 = \omega_2 \sigma_z(0)` 
        - Yes (different coefficients, same operator)
    *   - :math:`H_1 = \omega_z \sigma_z(0) + \cos(\omega_xt) \sigma_x(1)`
        - :math:`H_2 = \omega_z \sigma_z(0) + \sin(\omega_xt)  \sigma_x(1)`
        - Yes (same structure, different callback coefficients)
    *   - :math:`H_1 = \omega_z \sigma_z(0) + \cos(\omega_xt) \sigma_x(1)`
        - :math:`H_2 = \omega_z \sigma_z(0) + \cos(\omega_xt) \sigma_y(1)`
        - Yes (different operators on the same degree of freedom)
    *   - :math:`H_1 = \omega_z \sigma_z(0) + \cos(\omega_xt) \sigma_x(1)`
        - :math:`H_2 = \omega_z \sigma_z(0) + \cos(\omega_xt) \sigma_x(1) + \cos(\omega_yt) \sigma_y(1)`
        - No (different number of product terms)
    *   - :math:`H_1 = \omega_z \sigma_z(0) + \cos(\omega_xt) \sigma_{xx}(0, 1)`
        - :math:`H_2 = \omega_z \sigma_z(0) + \cos(\omega_xt) \sigma_x(0)\sigma_x(1)`
        - No (different structures, two-body operators vs. tensor product of single-body operators)

When the Hamiltonians are **not** `batchable`, CUDA-Q will still run the simulations, but each Hamiltonian will be simulated separately in a sequential manner.
CUDA-Q will log a warning "The input Hamiltonian and collapse operators are not compatible for batching. Running the simulation in non-batched mode.", when that happens.

.. note::

    Depending on the number of Hamiltonian operators together with factors such as the integrator, schedule step size, and whether intermediate results are stored, the batch simulation can be memory-intensive.
    If you encounter out-of-memory issues, the `max_batch_size` parameter can be used to limit the number of Hamiltonians that are batched together in one run. 
    For example, if you set `max_batch_size=2`, then we will run the simulations in batches of 2 Hamiltonians at a time, i.e., the first two Hamiltonians will be simulated together, then the next two, and so on.

    .. tab:: Python

        .. literalinclude:: ../snippets/python/using/backends/dynamics_operator_batching.py
            :language: python
            :start-after: [Begin Batch Size]
            :end-before: [End Batch Size]

    .. tab:: C++

        .. literalinclude:: ../snippets/cpp/using/backends/dynamics_operator_batching.cpp
            :language: cpp
            :start-after: [Begin Batch Size]
            :end-before: [End Batch Size]


Multi-GPU Multi-Node Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _cudensitymat_mgmn:

CUDA-Q ``dynamics`` target supports parallel execution on multiple GPUs. 
To enable parallel execution, the application must initialize MPI as follows.


.. tab:: Python

    .. literalinclude:: ../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin MPI]
        :end-before: [End MPI]

    .. code:: bash 

        mpiexec -np <N> python3 program.py 
  
  where ``N`` is the number of processes.

.. tab:: C++

    .. literalinclude:: ../snippets/cpp/using/backends/dynamics.cpp
        :language: cpp
        :start-after: [Begin MPI]
        :end-before: [End MPI]

    .. code:: bash 

        nvq++ --target dynamics example.cpp -o a.out 
        mpiexec -np <N> a.out
  
  where ``N`` is the number of processes.

By initializing the MPI execution environment (via `cudaq.mpi.initialize()`) in the application code and
invoking it via an MPI launcher, we have activated the multi-node multi-GPU feature of the ``dynamics`` target.
Specifically, it will detect the number of processes (GPUs) and distribute the computation across all available GPUs.


.. note::
    The number of MPI processes must be a power of 2, one GPU per process.

.. note::
    Not all integrators are capable of handling distributed state. Errors will be raised if parallel execution is activated 
    but the selected integrator does not support distributed state. 

.. note::
    When running batched simulations in a multi-GPU multi-node environment, the batch size will be automatically divided by the number of MPI processes.
    Hence, the batch size needs to be divisible by the number of processes. For example, if the original batch size is 8 and there are 4 MPI processes, 
    then each process (GPU) will simulate a batch size of 2. Errors will be raised if the batch size is not divisible by the number of processes.

    Each process will return its own set of results. The user is responsible for gathering the results from all processes if needed.

Examples
^^^^^^^^^^^^^
The :ref:`Dynamics Examples <dynamics_examples>` section of the docs contains a number of excellent dynamics examples demonstrating how to simulate basic physics models, specific qubit modalities, and utilize multi-GPU multi-Node capabilities.


