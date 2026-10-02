***************************************
Quantum Chemistry with CUDA-Q
***************************************

CUDA-Q provides a :mod:`cudaq.chemistry` module for quantum chemistry
applications, enabling users to create molecular Hamiltonians and run
variational quantum algorithms like VQE.

Prerequisites
=============

The chemistry module requires :mod:`openfermion` and :mod:`openfermionpyscf`
to be installed:

.. code-block:: bash

   pip install openfermionpyscf

These packages are used to compute molecular integrals and generate the
Hamiltonian from molecular geometry.

Creating Molecular Hamiltonians
===============================

The primary entry point is :func:`cudaq.chemistry.create_molecular_hamiltonian`:

.. code-block:: python

   import cudaq

   # Define molecular geometry (atom, (x, y, z))
   geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7474))]

   # Create the Hamiltonian
   hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
       geometry,
       basis='sto-3g',        # Basis set
       multiplicity=1,        # Spin multiplicity
       charge=0               # Molecular charge
   )

   print(hamiltonian)  # cudaq.SpinOperator

The function returns a tuple containing:

- A :class:`cudaq.SpinOperator` representing the molecular Hamiltonian
- A molecular data object containing properties like ``n_electrons`` and ``n_orbitals``

Active Space Approximations
---------------------------

To reduce the number of qubits, you can freeze core orbitals by specifying
an active space:

.. code-block:: python

   hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
       geometry,
       basis='sto-3g',
       multiplicity=1,
       charge=0,
       n_active_electrons=2,  # Number of electrons in active space
       n_active_orbitals=4    # Number of spatial orbitals in active space
   )

This will freeze the core orbitals and only include the specified number
of active electrons and orbitals in the Hamiltonian.

Available Basis Sets
--------------------

Common basis sets supported by PySCF include:

- ``sto-3g`` (minimal)
- ``6-31g`` (split-valence)
- ``6-31g*`` (with polarization)
- ``cc-pvdz`` (correlation-consistent)
- ``cc-pvtz``

Refer to the PySCF documentation for a complete list.

Built-in Ansatz Kernels
=======================

CUDA-Q provides built-in kernels for common variational ansatzes in
:mod:`cudaq.kernels`:

UCCSD (Unitary Coupled Cluster Singles and Doubles)
----------------------------------------------------

.. code-block:: python

   from cudaq.kernels import uccsd, uccsd_num_parameters

   # Get the number of parameters
   num_params = uccsd_num_parameters(num_electrons, num_qubits)

   @cudaq.kernel
   def ansatz(thetas: list[float]):
       q = cudaq.qvector(num_qubits)
       # Hartree-Fock initial state
       for i in range(num_electrons):
           x(q[i])
       # Apply UCCSD
       uccsd(q, thetas, num_electrons, num_qubits)

HWE (Hardware Efficient Ansatz)
-------------------------------

.. code-block:: python

   from cudaq.kernels import hwe, num_hwe_parameters

   # Get the number of parameters
   num_params = num_hwe_parameters(num_qubits, num_layers)

   @cudaq.kernel
   def ansatz(thetas: list[float]):
       q = cudaq.qvector(num_qubits)
       # Hartree-Fock initial state
       for i in range(num_electrons):
           x(q[i])
       # Apply HWE
       hwe(q, num_qubits, num_layers, thetas)

Running VQE
===========

Once you have a Hamiltonian and an ansatz, you can run VQE using
CUDA-Q's optimizers:

.. code-block:: python

   import numpy as np
   from cudaq.kernels import uccsd, uccsd_num_parameters
   from cudaq import optimizers

   # Create Hamiltonian
   geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7474))]
   hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
       geometry, 'sto-3g', 1, 0)
   num_electrons = data.n_electrons
   num_qubits = 2 * data.n_orbitals

   # Build ansatz
   num_params = uccsd_num_parameters(num_electrons, num_qubits)

   @cudaq.kernel
   def ansatz(thetas: list[float]):
       q = cudaq.qvector(num_qubits)
       for i in range(num_electrons):
           x(q[i])
       uccsd(q, thetas, num_electrons, num_qubits)

   # Define objective function
   def objective(x):
       exp_val = cudaq.observe(ansatz, hamiltonian, x).expectation()
       return exp_val

   # Run optimization
   optimizer = cudaq.optimizers.COBYLA()
   energy, params = optimizer.optimize(num_params, objective)

   print(f"Ground state energy: {energy} Hartree")

Complete Example: H2 Ground State
=================================

.. code-block:: python

   import cudaq
   import numpy as np
   from cudaq.kernels import uccsd, uccsd_num_parameters
   from cudaq import optimizers

   # H2 molecule
   geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7474))]
   hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
       geometry, 'sto-3g', 1, 0)

   num_electrons = data.n_electrons
   num_qubits = 2 * data.n_orbitals

   num_params = uccsd_num_parameters(num_electrons, num_qubits)

   @cudaq.kernel
   def ansatz(thetas: list[float]):
       q = cudaq.qvector(num_qubits)
       for i in range(num_electrons):
           x(q[i])
       uccsd(q, thetas, num_electrons, num_qubits)

   def objective(x):
       return cudaq.observe(ansatz, hamiltonian, x).expectation()

   optimizer = cudaq.optimizers.COBYLA()
   optimizer.max_iterations = 100
   energy, params = optimizer.optimize(num_params, objective)

   print(f"H2 ground state energy: {energy:.6f} Hartree")
   print(f"Exact FCI energy: {data.fci_energy:.6f} Hartree")

Reference
=========

.. currentmodule:: cudaq.chemistry

.. autofunction:: create_molecular_hamiltonian

.. currentmodule:: cudaq.kernels

.. autofunction:: uccsd
.. autofunction:: uccsd_num_parameters
.. autofunction:: hwe
.. autofunction:: num_hwe_parameters