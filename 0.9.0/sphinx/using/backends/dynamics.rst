CUDA-Q Dynamics 
*********************************

CUDA-Q enables the design, simulation and execution of quantum dynamics via 
the ``evolve`` API. Specifically, this API allows us to solve the time evolution 
of quantum systems or models. In the simulation mode, CUDA-Q provides the ``dynamics``
backend target, which is based on the cuQuantum library, optimized for performance and scale
on NVIDIA GPU.

Quick Start
+++++++++++

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
    
    .. code:: python

        from cudaq.operator import *

        # Qubit Hamiltonian
        hamiltonian = 0.5 * omega_z * spin.z(0)
        # Add modulated driving term to the Hamiltonian
        hamiltonian += omega_x * ScalarOperator(
            lambda t: np.cos(omega_d * t)) * spin.x(0)

In particular, `ScalarOperator` provides an easy way to model arbitrary time-dependent control signals.   
Details about CUDA-Q `operator`, including builtin operators that it supports can be found :ref:`here <operators>`.

2. Setup the evolution simulation

The below code snippet shows how to simulate the time-evolution of the above system
with `cudaq.evolve`.

.. tab:: Python
    
    .. code:: python

        # Set the target to our dynamics simulator
        cudaq.set_target("dynamics")

        # Dimensions of sub-systems: a single two-level system.
        dimensions = {0: 2}

        # Initial state of the system (ground state).
        rho0 = cudaq.State.from_data(
            cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

        # Schedule of time steps.
        steps = np.linspace(0, t_final, n_steps)
        schedule = Schedule(steps, ["t"])

        # Run the simulation.
        evolution_result = evolve(hamiltonian,
                                dimensions,
                                schedule,
                                rho0,
                                observables=[spin.x(0),
                                            spin.y(0),
                                            spin.z(0)],
                                collapse_operators=[],
                                store_intermediate_results=True)



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
    
    .. code:: python

        get_result = lambda idx, res: [
            exp_vals[idx].expectation() for exp_vals in res.expectation_values()
        ]

        import matplotlib.pyplot as plt

        plt.plot(steps, get_result(0, evolution_result))
        plt.plot(steps, get_result(1, evolution_result))
        plt.plot(steps, get_result(2, evolution_result))
        plt.ylabel("Expectation value")
        plt.xlabel("Time")
        plt.legend(("Sigma-X", "Sigma-Y", "Sigma-Z"))


In particular, for each time step, `evolve` captures an array of expectation values, one for each  
observable. Hence, we convert them into sequences for plotting purposes.


Operator
+++++++++++

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
    
    .. code:: python

        hamiltonian = omega_c * operators.create(1) * operators.annihilate(1) + (omega_a/2) * spin.z(0) + (Omega/2) (operators.annihilate(1) * spin.plus(0) + operators.create(1)*spin.minus(0))


In the above code snippet, we map the cavity light field to degree index 1 and the two-level atom to degree index 0. 
The description of composite quantum system dynamics is independent from the Hilbert space of the system components.
The latter is specified by the dimension map that is provided to the `cudaq.evolve` call. 


Time-Dependent Dynamics
++++++++++++++++++++++++

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
    
    .. code:: python
        
        import numpy as np
        # Define the static (drift) and control terms 
        H0 = spin.z(0)
        H1 = spin.x(0)
        H = H0 + ScalarOperator(lambda t: np.cos(omega * t)) * H1

2. Time-dependent operator

We can also construct a time-dependent operator from a function that returns a complex matrix representing the time dynamics of 
that operator.

As an example, let's revisit the above example, whereby we now define a time-dependent operator :math:`H_1(t) = cos(\omega t) \sigma_X`.

.. tab:: Python
    
    .. code:: python
        
        import numpy as np
        
        # A function that returns H1 matrix as a function of time
        def H1_matrix(t):
            return np.cos(omega * t) * np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
        # Define and register the time-dependent operator
        # This operator is expected to be applied to a two-level sub-system (dimension = 2).
        ElementaryOperator.define("H1", [2], H1_matrix)
        
        # Construct the Hamiltonian terms
        H0 = spin.z(0)
        H1 = ElementaryOperator("H1", [0])
        # Total Hamiltonian
        H = H0 + H1


Numerical Integrators
++++++++++++++++++++++

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