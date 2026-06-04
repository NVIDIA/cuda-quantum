import numpy as np

from ipie.utils.from_pyscf import (load_from_pyscf_chkfile,
                                   generate_hamiltonian, copy_LPX_to_LXmn,
                                   generate_wavefunction_from_mo_coeff)


def signature_permutation(orbital_list):
    """
    Returns the signature of the permutation in orbital_list
    """
    if len(orbital_list) == 1:
        return 1

    transposition_count = 0
    for index, element in enumerate(orbital_list):
        for next_element in orbital_list[index + 1:]:
            if element > next_element:
                transposition_count += 1

    return (-1)**transposition_count


def get_coeff_wf(final_state_vector, n_active_elec, spin=0, thres=1e-6):
    """
    :`param` `final_state_vector`: State vector from a VQE simulation
    :`param` `n_active_elec`: Number of total electrons in active space
    :`param` `spin`: spin
    :`param` `thres`: Threshold for coefficients to keep from VQE wavefunction
    :returns: Input for `ipie` trial: coefficients, list of occupied alpha, list of occupied bets
    """
    n_qubits = int(np.log2(final_state_vector.size))
    n_elec = [(n_active_elec + spin) // 2, (n_active_elec - spin) // 2]

    coeff = []
    occas = []
    occbs = []
    for j, val in enumerate(final_state_vector):
        if abs(val) > thres:
            ket = np.binary_repr(j, width=n_qubits)
            alpha_ket = ket[::2]
            beta_ket = ket[1::2]
            occ_alpha = np.where([int(_) for _ in alpha_ket])[0]
            occ_beta = np.where([int(_) for _ in beta_ket])[0]
            occ_orbitals = np.append(2 * occ_alpha, 2 * occ_beta + 1)

            if (len(occ_alpha) == n_elec[0]) and (len(occ_beta) == n_elec[1]):
                coeff.append(signature_permutation(occ_orbitals) * val)
                occas.append(occ_alpha)
                occbs.append(occ_beta)

    coeff = np.array(coeff, dtype=complex)
    ixs = np.argsort(np.abs(coeff))[::-1]
    coeff = coeff[ixs]
    occas = np.array(occas)[ixs]
    occbs = np.array(occbs)[ixs]

    return coeff, occas, occbs


def gen_ipie_input_from_pyscf_chk(pyscf_chkfile: str,
                                  verbose: bool = True,
                                  chol_cut: float = 1e-5,
                                  ortho_ao: bool = False,
                                  mcscf: bool = False,
                                  linear_dep_thresh: float = 1e-8,
                                  num_frozen_core: int = 0):
    """Generate AFQMC data from PYSCF (molecular) simulation.
        Adapted from `ipie`.`utils`.from_`pyscf`: returns Hamiltonian and wavefunction instead of writing on files
    """
    if mcscf:
        scf_data = load_from_pyscf_chkfile(pyscf_chkfile, base="mcscf")
    else:
        scf_data = load_from_pyscf_chkfile(pyscf_chkfile)
    mol = scf_data["mol"]
    hcore = scf_data["hcore"]
    ortho_ao_mat = scf_data["X"]
    mo_coeffs = scf_data["mo_coeff"]
    mo_occ = scf_data["mo_occ"]
    if ortho_ao:
        basis_change_matrix = ortho_ao_mat
    else:
        basis_change_matrix = mo_coeffs

        if isinstance(mo_coeffs, list) or len(mo_coeffs.shape) == 3:
            if verbose:
                print("# UHF mo coefficients found and ortho-ao == False. Using"
                      " alpha mo coefficients for basis transformation.")
            basis_change_matrix = mo_coeffs[0]
    ham = generate_hamiltonian(
        mol,
        mo_coeffs,
        hcore,
        basis_change_matrix,
        chol_cut=chol_cut,
        num_frozen_core=num_frozen_core,
        verbose=False,
    )
    # write_Hamiltonian(ham.H1[0], copy_`LPX`_to_`LXmn`(ham.`chol`), ham.`ecore`, filename=Hamiltonian_file)
    ipie_ham = (ham.H1[0], copy_LPX_to_LXmn(ham.chol), ham.ecore)
    nelec = (mol.nelec[0] - num_frozen_core, mol.nelec[1] - num_frozen_core)
    if verbose:
        print(f"# Number of electrons in simulation: {nelec}")
    if mcscf:
        # `ci`_`coeffs` = `scf`_data["`ci_coeffs`"]
        # `occa` = `scf`_data["`occa`"]
        # `occb` = `scf`_data["`occb`"]
        # write_wavefunction((`ci_coeffs`, `occa`, `occb`), wavefunction_file)
        return ipie_ham
    else:
        wavefunction = generate_wavefunction_from_mo_coeff(
            mo_coeffs,
            mo_occ,
            basis_change_matrix,
            nelec,
            ortho_ao=ortho_ao,
            num_frozen_core=num_frozen_core,
        )
        # write_wavefunction(wavefunction, wavefunction_file)
        return ipie_ham, wavefunction
