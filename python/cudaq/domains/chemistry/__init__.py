# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


def tryImport():
    """Import `openfermion` and `openfermionpyscf`."""
    try:
        import openfermion, openfermionpyscf
    except ImportError as Error:
        raise ImportError("This feature requires openfermionpyscf. ") from Error

    return openfermion, openfermionpyscf


def create_molecular_hamiltonian(geometry: list,
                                 basis='sto-3g',
                                 multiplicity=1,
                                 charge=0,
                                 n_active_electrons=None,
                                 n_active_orbitals=None):
    '''
    Create the molecular Hamiltonian corresponding to the provided 
    geometry, basis set, multiplicity, and charge.  One can also specify the 
    number of active electrons and orbitals, thereby approximating the 
    molecular Hamiltonian and freezing core orbitals. This function delegates 
    to the `OpenFermion-PySCF` package and will throw an error if that module is 
    not available.

    Arguments: 
      geometry: The geometry should be provided as a list of tuples, 
        where each tuple element contains the atom name and a tuple
        of atom coordinates, e.g. [('H', (0.,0.,0.)), ('H', (0.,0.,.7474))].
      basis: The basis set as a string.
      multiplicity: The spin multiplicity as an int.
      charge: The total charge of the molecular system as an int.
      n_active_electrons: The number of electrons in the active space as an int.
      n_active_orbitals: The number of spatial orbitals in the active space. 

    Returns: 
      A tuple containing the `cudaq.SpinOperator` representation for the molecular 
      Hamiltonian and the raw molecular data. 
    '''
    of, ofpyscf = tryImport()
    molecule = ofpyscf.run_pyscf(of.MolecularData(geometry, basis, multiplicity,
                                                  charge),
                                 run_fci=True)
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(
            range(n_core_orbitals, n_core_orbitals + n_active_orbitals))
    hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices, active_indices=active_indices)
    spin_op = of.jordan_wigner(hamiltonian)
    from cudaq import SpinOperator
    return SpinOperator(spin_op), molecule


def __internal_cpp_create_molecular_hamiltonian(geometry: list,
                                                basis='sto-3g',
                                                multiplicity=1,
                                                charge=0,
                                                n_active_electrons=None,
                                                n_active_orbitals=None):
    '''
    Internal function meant for integration with CUDA Quantum C++. 
    (Does not require `import cudaq`)
    
    Create the molecular Hamiltonian corresponding to the provided 
    geometry, basis set, multiplicity, and charge.  One can also specify the 
    number of active electrons and orbitals, thereby approximating the 
    molecular Hamiltonian and freezing core orbitals. This function delegates 
    to the `OpenFermion-PySCF` package and will throw an error if that module is 
    not available.

    Arguments: 
      geometry: The geometry should be provided as a list of tuples, 
        where each tuple element contains the atom name and a tuple
        of atom coordinates, e.g. [('H', (0.,0.,0.)), ('H', (0.,0.,.7474))].
      basis: The basis set as a string.
      multiplicity: The spin multiplicity as an int.
      charge: The total charge of the molecular system as an int.
      n_active_electrons: The number of electrons in the active space as an int.
      n_active_orbitals: The number of spatial orbitals in the active space. 

    Returns: 
      A tuple containing the Hamiltonian representation for the molecular 
      Hamiltonian and the raw molecular data. 
    '''
    of, ofpyscf = tryImport()
    molecule = ofpyscf.run_pyscf(of.MolecularData(geometry, basis, multiplicity,
                                                  charge),
                                 run_fci=True)
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(
            range(n_core_orbitals, n_core_orbitals + n_active_orbitals))
    hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices, active_indices=active_indices)
    spin_op = of.jordan_wigner(hamiltonian)
    return spin_op, molecule
