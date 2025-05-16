from cudaq import spin
from qchem.uccsd import uccsd_get_excitation_list, add_single_excitation, add_double_excitation
import numpy as np


def get_uccsd_pool(nelectrons, n_qubits, spin=0):

    singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta = \
        uccsd_get_excitation_list(nelectrons, n_qubits, spin)

    n_alpha_singles = len(singles_alpha)
    n_beta_singles = len(singles_beta)
    n_mixed_doubles = len(doubles_mixed)
    n_alpha_doubles = len(doubles_alpha)
    n_beta_doubles = len(doubles_beta)

    pool_op = []

    for i in range(n_alpha_singles):
        p_alpha_occ, q_alpha_virt = singles_alpha[i]
        add_single_excitation(pool_op, p_alpha_occ, q_alpha_virt)

    for i in range(n_beta_singles):
        p_beta_occ, q_beta_virt = singles_beta[i]
        add_single_excitation(pool_op, p_beta_occ, q_beta_virt)

    for i in range(n_mixed_doubles):
        p_alpha_occ, q_beta_occ, r_beta_virt, s_alpha_virt = doubles_mixed[i]
        add_double_excitation(pool_op, p_alpha_occ, q_beta_occ, r_beta_virt,
                              s_alpha_virt)

    for i in range(n_alpha_doubles):
        p_alpha_occ, q_alpha_occ, r_alpha_virt, s_alpha_virt = doubles_alpha[i]
        add_double_excitation(pool_op, p_alpha_occ, q_alpha_occ, r_alpha_virt,
                              s_alpha_virt)

    for i in range(n_beta_doubles):
        p_beta_occ, q_beta_occ, r_beta_virt, s_beta_virt = doubles_beta[i]

        add_double_excitation(pool_op, p_beta_occ, q_beta_occ, r_beta_virt,
                              s_beta_virt)

    return pool_op
