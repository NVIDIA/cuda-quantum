import numpy as np
from qchem.hamiltonian import jordan_wigner_pe
from functools import reduce
from qchem.cppe_lib import PolEmbed


def generate_pe_spin_ham_restricted(v_pe):

    # Total number of qubits equals the number of spin molecular orbitals
    nqubits = 2 * v_pe.shape[0]

    # Initialization
    spin_pe_op = np.zeros((nqubits, nqubits))

    for p in range(nqubits // 2):
        for q in range(nqubits // 2):

            # p & q have the same spin <a|a>= <b|b>=1
            # <a|b>=<b|a>=0 (orthogonal)
            spin_pe_op[2 * p, 2 * q] = v_pe[p, q]
            spin_pe_op[2 * p + 1, 2 * q + 1] = v_pe[p, q]

    return spin_pe_op


def pe_operator(dm, mol, mo_coeff, potential, elec_only=False, tolerance=1e-12):

    # `dm` are in atomic orbitals
    mype = PolEmbed(mol, potential)
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm, elec_only)

    #convert from `ao` to `mo`
    V_pe_mo = reduce(np.dot, (mo_coeff.T, V_pe, mo_coeff))

    obi_pe = generate_pe_spin_ham_restricted(V_pe_mo)
    spin_pe_ham = jordan_wigner_pe(obi_pe, tolerance)

    return spin_pe_ham, E_pe, V_pe_mo


def pe_operator_as(dm,
                   mol,
                   mo_coeff,
                   potential,
                   mc,
                   elec_only=False,
                   tolerance=1e-12):

    # `dm` is in atomic orbitals
    mype = PolEmbed(mol, potential)
    E_pe, V_pe, V_es, V_ind = mype.get_pe_contribution(dm, elec_only)
    #convert from `ao` to `mo`
    V_pe_mo = reduce(np.dot, (mo_coeff.T, V_pe, mo_coeff))
    V_pe_cas = V_pe_mo[mc.ncore:mc.ncore + mc.ncas, mc.ncore:mc.ncore + mc.ncas]

    obi_pe = generate_pe_spin_ham_restricted(V_pe_cas)
    spin_pe_ham = jordan_wigner_pe(obi_pe, tolerance)

    return spin_pe_ham, E_pe, V_pe, V_pe_cas
