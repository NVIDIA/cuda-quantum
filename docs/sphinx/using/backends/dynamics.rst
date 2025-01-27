Dynamics Simulation 
+++++++++++++++++++++

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

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Transmon]
        :end-before: [End Transmon]

In particular, `ScalarOperator` provides an easy way to model arbitrary time-dependent control signals.   
Details about CUDA-Q `operator`, including builtin operators that it supports can be found :ref:`here <operators>`.

2. Setup the evolution simulation

The below code snippet shows how to simulate the time-evolution of the above system
with `cudaq.evolve`.

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
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
as well as intermediate values at each time step (with `store_intermediate_results=True`).

For example, we can plot the Pauli expectation value for the above simulation as follows.

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Plot]
        :end-before: [End Plot]

In particular, for each time step, `evolve` captures an array of expectation values, one for each  
observable. Hence, we convert them into sequences for plotting purposes.

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

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Jaynes-Cummings]
        :end-before: [End Jaynes-Cummings]

In the above code snippet, we map the cavity light field to degree index 1 and the two-level atom to degree index 0. 
The description of composite quantum system dynamics is independent from the Hilbert space of the system components.
The latter is specified by the dimension map that is provided to the `cudaq.evolve` call. 


Time-Dependent Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _time_dependent:

In the previous examples of operator construction, we assumed that the systems under consideration were described by time-independent Hamiltonian. 
However, we may want to simulate systems whose Hamiltonian operators have explicit time dependence.

CUDA-Q provides multiple ways to construct time-dependent operators.

1. Time-dependent coefficient

CUDA-Q `ScalarOperator` can be used to wrap a Python function that returns the coefficient value at a specific time.

As an example, we will look at a time-dependent Hamiltonian of the form :math:`H = H_0 + f(t)H_1`, 
where :math:`f(t)` is the time-dependent driving strength given as :math:`cos(\omega t)`.

The following code sets up the problem

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Hamiltonian]
        :end-before: [End Hamiltonian]

2. Time-dependent operator

We can also construct a time-dependent operator from a function that returns a complex matrix representing the time dynamics of 
that operator.

As an example, let's looks at the `displacement operator <https://en.wikipedia.org/wiki/Displacement_operator>`__. It can be defined as follows:


.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin DefineOp]
        :end-before: [End DefineOp]

The defined operator is parameterized by the `displacement` amplitude. To create simulate the evolution of an 
operator under a time dependent displacement amplitude, we can define how the amplitude changes in time:

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Schedule1]
        :end-before: [End Schedule1]

Let's say we want to add a squeezing term to the system operator. We can independently vary the squeezing 
amplitude and the displacement amplitude by instantiating a schedule with a custom function that returns 
the desired value for each parameter: 

.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin Schedule2]
        :end-before: [End Schedule2]

Numerical Integrators
^^^^^^^^^^^^^^^^^^^^^^^^

.. _integrators:

CUDA-Q provides a set of numerical integrators, to be used with the ``dynamics``
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

.. warning:: 
    Torch-based integrators require a CUDA-enabled Torch installation. Depending on your platform (e.g., `aarch64`),
    the default Torch pip package may not have CUDA support. 

    The below command can be used to verify your installation:

    .. code:: bash

        python3 -c "import torch; print(torch.version.cuda)"

    If the output is a '`None`' string, it indicates that your Torch installation does not support CUDA.
    In this case, you need to install a CUDA-enabled Torch package via other mechanisms, e.g., building Torch from source or
    using their Docker images.

Multi-GPU Multi-Node Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _cudensitymat_mgmn:

CUDA-Q ``dynamics`` target supports parallel execution on multiple GPUs. 
To enable parallel execution, the application must initialize MPI as follows.


.. tab:: Python

  .. literalinclude:: ../../snippets/python/using/backends/dynamics.py
        :language: python
        :start-after: [Begin MPI]
        :end-before: [End MPI]

  .. code:: bash 

        mpiexec -np <N> python3 program.py 
  
  where ``N`` is the number of processes.


By initializing the MPI execution environment (via `cudaq.mpi.initialize()`) in the application code and
invoking it via an MPI launcher, we have activated the multi-node multi-GPU feature of the ``dynamics`` target.
Specifically, it will detect the number of processes (GPUs) and distribute the computation across all available GPUs.


.. note::
    The number of MPI processes must be a power of 2, one GPU per process.

.. note::
    Not all integrators are capable of handling distributed state. Errors will be raised if parallel execution is activated 
    but the selected integrator does not support distributed state. 

.. warning:: 
    As of cuQuantum version 24.11, there are a couple of `known limitations <https://docs.nvidia.com/cuda/cuquantum/24.11.0/cudensitymat/index.html>`__ for parallel execution:

    - Computing the expectation value of a mixed quantum state is not supported. Thus, `collapse_operators` are not supported if expectation calculation is required.

    - Some combinations of quantum states and quantum many-body operators are not supported. Errors will be raised in those cases. 

