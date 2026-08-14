import cudaq
from cudaq import spin


def uccsd_get_excitation_list(n_electrons: int, n_qubits: int, spin_mult=0):

    if n_qubits % 2 != 0:
        raise ValueError(
            "Total number of spin molecular orbitals (number of qubits) should be even."
        )
    else:
        n_spatial_orbitals = n_qubits // 2

    singles_alpha = []
    singles_beta = []
    doubles_mixed = []
    doubles_alpha = []
    doubles_beta = []

    if spin_mult > 0:
        n_occupied_beta = (n_electrons - spin_mult) // 2
        n_occupied_alpha = n_electrons - n_occupied_beta
        n_virtual_alpha = n_spatial_orbitals - n_occupied_alpha
        n_virtual_beta = n_spatial_orbitals - n_occupied_beta

        occupied_alpha_indices = [i * 2 for i in range(n_occupied_alpha)]
        virtual_alpha_indices = [
            i * 2 + n_electrons + 1 for i in range(n_virtual_alpha)
        ]

        occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied_beta)]
        virtual_beta_indices = [
            i * 2 + 1 + n_electrons - 1 for i in range(n_virtual_beta)
        ]

    elif n_electrons % 2 == 0 and spin_mult == 0:
        n_occupied = n_electrons // 2
        n_virtual = n_spatial_orbitals - n_occupied

        occupied_alpha_indices = [i * 2 for i in range(n_occupied)]
        virtual_alpha_indices = [i * 2 + n_electrons for i in range(n_virtual)]

        occupied_beta_indices = [i * 2 + 1 for i in range(n_occupied)]
        virtual_beta_indices = [
            i * 2 + 1 + n_electrons for i in range(n_virtual)
        ]

    else:
        raise ValueError("spin should be zero or larger than zero.")

    #occupied orbital is alpha, virtual orbital is alpha
    for p in occupied_alpha_indices:
        for q in virtual_alpha_indices:
            singles_alpha.append((p, q))

    # occupied orbital is beta and virtual orbital is beta
    for p in occupied_beta_indices:
        for q in virtual_beta_indices:
            singles_beta.append((p, q))

    #Mixed spin double excitation
    for p in occupied_alpha_indices:
        for q in occupied_beta_indices:
            for r in virtual_beta_indices:
                for s in virtual_alpha_indices:
                    doubles_mixed.append((p, q, r, s))

    # same spin double excitation
    n_occ_alpha = len(occupied_alpha_indices)
    n_occ_beta = len(occupied_beta_indices)
    n_virt_alpha = len(virtual_alpha_indices)
    n_virt_beta = len(virtual_beta_indices)

    for p in range(n_occ_alpha - 1):
        for q in range(p + 1, n_occ_alpha):
            for r in range(n_virt_alpha - 1):
                for s in range(r + 1, n_virt_alpha):

                    # Same spin: all alpha
                    doubles_alpha.append((occupied_alpha_indices[p],occupied_alpha_indices[q],\
                                     virtual_alpha_indices[r],virtual_alpha_indices[s]))

    for p in range(n_occ_beta - 1):
        for q in range(p + 1, n_occ_beta):
            for r in range(n_virt_beta - 1):
                for s in range(r + 1, n_virt_beta):

                    # Same spin: all beta
                    doubles_beta.append((occupied_beta_indices[p],occupied_beta_indices[q],\
                                     virtual_beta_indices[r],virtual_beta_indices[s]))

    return singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta


#######################################


def uccsd_parameter_size(n_electrons: int, n_qubits: int, spin_mult=0):
    # Compute the size of theta parameters for all UCCSD excitation.

    singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta = \
        uccsd_get_excitation_list(n_electrons, n_qubits, spin_mult)

    length_alpha_singles = len(singles_alpha)
    length_beta_singles = len(singles_beta)
    length_mixed_doubles = len(doubles_mixed)
    length_alpha_doubles = len(doubles_alpha)
    length_beta_doubles = len(doubles_beta)

    singles = length_alpha_singles + length_beta_singles
    doubles = length_mixed_doubles + length_alpha_doubles + length_beta_doubles
    total = singles + doubles

    return singles, doubles, total


###################################################


def add_single_excitation(op, p_occ, q_virt):

    parity = 1.0
    for i in range(p_occ + 1, q_virt):
        parity *= spin.z(i)

    c = 0.5
    op.append(c * spin.y(p_occ) * parity * spin.x(q_virt) -
              c * spin.x(p_occ) * parity * spin.y(q_virt))


##################################


def add_double_excitation(op, p_occ, q_occ, r_virt, s_virt):

    temp = []
    if (p_occ < q_occ) and (r_virt < s_virt):
        i_occ, j_occ = p_occ, q_occ
        a_virt, b_virt = r_virt, s_virt

    elif (p_occ > q_occ) and (r_virt > s_virt):
        i_occ, j_occ = q_occ, p_occ
        a_virt, b_virt = s_virt, r_virt

    elif (p_occ < q_occ) and (r_virt > s_virt):
        i_occ, j_occ = p_occ, q_occ
        a_virt, b_virt = s_virt, r_virt

    elif (p_occ > q_occ) and (r_virt < s_virt):
        i_occ, j_occ = q_occ, p_occ
        a_virt, b_virt = r_virt, s_virt

    parity_a = 1.0
    parity_b = 1.0

    for i in range(i_occ + 1, j_occ):
        parity_a *= spin.z(i)

    for i in range(a_virt + 1, b_virt):
        parity_b *= spin.z(i)

    c = 1.0 / 8.0
    temp_op = c * spin.x(i_occ) * parity_a * spin.x(j_occ) * spin.x(
        a_virt) * parity_b * spin.y(b_virt)
    temp_op += c * spin.x(i_occ) * parity_a * spin.x(j_occ) * spin.y(
        a_virt) * parity_b * spin.x(b_virt)
    temp_op += c * spin.x(i_occ) * parity_a * spin.y(j_occ) * spin.y(
        a_virt) * parity_b * spin.y(b_virt)
    temp_op += c * spin.y(i_occ) * parity_a * spin.x(j_occ) * spin.y(
        a_virt) * parity_b * spin.y(b_virt)
    temp_op -= c * spin.x(i_occ) * parity_a * spin.y(j_occ) * spin.x(
        a_virt) * parity_b * spin.x(b_virt)
    temp_op -= c * spin.y(i_occ) * parity_a * spin.x(j_occ) * spin.x(
        a_virt) * parity_b * spin.x(b_virt)
    temp_op -= c * spin.y(i_occ) * parity_a * spin.y(j_occ) * spin.x(
        a_virt) * parity_b * spin.y(b_virt)
    temp_op -= c * spin.y(i_occ) * parity_a * spin.y(j_occ) * spin.y(
        a_virt) * parity_b * spin.x(b_virt)

    op.append(temp_op)


########################################


def get_uccsd_op(nelectrons,
                 n_qubits,
                 spin_mult=0,
                 only_singles=False,
                 only_doubles=False):

    singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta = \
        uccsd_get_excitation_list(nelectrons, n_qubits, spin_mult)

    n_alpha_singles = len(singles_alpha)
    n_beta_singles = len(singles_beta)
    n_mixed_doubles = len(doubles_mixed)
    n_alpha_doubles = len(doubles_alpha)
    n_beta_doubles = len(doubles_beta)

    pool_op_single = []
    pool_op_double = []

    if not only_singles and not only_doubles:

        for i in range(n_alpha_singles):
            p_alpha_occ, q_alpha_virt = singles_alpha[i]
            add_single_excitation(pool_op_single, p_alpha_occ, q_alpha_virt)

        for i in range(n_beta_singles):
            p_beta_occ, q_beta_virt = singles_beta[i]
            add_single_excitation(pool_op_single, p_beta_occ, q_beta_virt)

        for i in range(n_mixed_doubles):
            p_alpha_occ, q_beta_occ, r_beta_virt, s_alpha_virt = doubles_mixed[
                i]
            add_double_excitation(pool_op_double, p_alpha_occ, q_beta_occ,
                                  r_beta_virt, s_alpha_virt)

        for i in range(n_alpha_doubles):
            p_alpha_occ, q_alpha_occ, r_alpha_virt, s_alpha_virt = doubles_alpha[
                i]
            add_double_excitation(pool_op_double, p_alpha_occ, q_alpha_occ,
                                  r_alpha_virt, s_alpha_virt)

        for i in range(n_beta_doubles):
            p_beta_occ, q_beta_occ, r_beta_virt, s_beta_virt = doubles_beta[i]

            add_double_excitation(pool_op_double, p_beta_occ, q_beta_occ,
                                  r_beta_virt, s_beta_virt)

    elif only_singles and not only_doubles:
        pool_op = []

        for i in range(n_alpha_singles):
            p_alpha_occ, q_alpha_virt = singles_alpha[i]
            add_single_excitation(pool_op_single, p_alpha_occ, q_alpha_virt)

        for i in range(n_beta_singles):
            p_beta_occ, q_beta_virt = singles_beta[i]
            add_single_excitation(pool_op_single, p_beta_occ, q_beta_virt)

    elif not only_singles and only_doubles:
        pool_op = []

        for i in range(n_mixed_doubles):
            p_alpha_occ, q_beta_occ, r_beta_virt, s_alpha_virt = doubles_mixed[
                i]
            add_double_excitation(pool_op_double, p_alpha_occ, q_beta_occ,
                                  r_beta_virt, s_alpha_virt)

        for i in range(n_alpha_doubles):
            p_alpha_occ, q_alpha_occ, r_alpha_virt, s_alpha_virt = doubles_alpha[
                i]
            add_double_excitation(pool_op_double, p_alpha_occ, q_alpha_occ,
                                  r_alpha_virt, s_alpha_virt)

        for i in range(n_beta_doubles):
            p_beta_occ, q_beta_occ, r_beta_virt, s_beta_virt = doubles_beta[i]

            add_double_excitation(pool_op_double, p_beta_occ, q_beta_occ,
                                  r_beta_virt, s_beta_virt)

    # Convert the list of operators to cudaq.`pauli_word`
    word_single = []
    word_double = []
    coef_single = []
    coef_double = []

    for i in range(len(pool_op_single)):

        op_i = pool_op_single[i]
        for term in op_i:
            coef_single.append(term.evaluate_coefficient().real)
            word_single.append(term.get_pauli_word(n_qubits))

    for i in range(len(pool_op_double)):

        op_i = pool_op_double[i]
        for term in op_i:
            coef_double.append(term.evaluate_coefficient().real)
            word_double.append(term.get_pauli_word(n_qubits))

    # Return the list of operators
    if only_singles and not only_doubles:
        return word_single, coef_single

    elif not only_singles and only_doubles:
        return word_double, coef_double

    elif not only_singles and not only_doubles:
        return word_single, word_double, coef_single, coef_double


###################################################


@cudaq.kernel
def uccsd_circuit_single(qubits: cudaq.qview, theta: list[float],
                         words_single: list[cudaq.pauli_word],
                         coef_single: list[float]):
    """
    UCCSD circuit implementation for only single excitation.
    """

    n_qubits = qubits.size()

    count = 0
    for i in range(0, len(words_single), 2):
        exp_pauli(coef_single[i] * theta[count], qubits, words_single[i])
        exp_pauli(coef_single[i + 1] * theta[count], qubits,
                  words_single[i + 1])
        count += 1


@cudaq.kernel
def uccsd_circuit_double(qubits: cudaq.qview, theta: list[float],
                         words_double: list[cudaq.pauli_word],
                         coef_double: list[float]):
    """
    UCCSD circuit implementation for only double excitation.
    """

    n_qubits = qubits.size()

    count = 0
    for i in range(0, len(words_double), 8):
        exp_pauli(coef_double[i] * theta[count], qubits, words_double[i])
        exp_pauli(coef_double[i + 1] * theta[count], qubits,
                  words_double[i + 1])
        exp_pauli(coef_double[i + 2] * theta[count], qubits,
                  words_double[i + 2])
        exp_pauli(coef_double[i + 3] * theta[count], qubits,
                  words_double[i + 3])
        exp_pauli(coef_double[i + 4] * theta[count], qubits,
                  words_double[i + 4])
        exp_pauli(coef_double[i + 5] * theta[count], qubits,
                  words_double[i + 5])
        exp_pauli(coef_double[i + 6] * theta[count], qubits,
                  words_double[i + 6])
        exp_pauli(coef_double[i + 7] * theta[count], qubits,
                  words_double[i + 7])
        count += 1


@cudaq.kernel
def uccsd_circuit(qubits: cudaq.qview, theta: list[float],
                  words_single: list[cudaq.pauli_word],
                  coef_single: list[float],
                  words_double: list[cudaq.pauli_word],
                  coef_double: list[float]):
    """
    UCCSD circuit implementation for both single and double excitation.
    """

    n_qubits = qubits.size()

    count = 0
    for i in range(0, len(words_single), 2):
        exp_pauli(coef_single[i] * theta[count], qubits, words_single[i])
        exp_pauli(coef_single[i + 1] * theta[count], qubits,
                  words_single[i + 1])
        count += 1

    for i in range(0, len(words_double), 8):
        exp_pauli(coef_double[i] * theta[count], qubits, words_double[i])
        exp_pauli(coef_double[i + 1] * theta[count], qubits,
                  words_double[i + 1])
        exp_pauli(coef_double[i + 2] * theta[count], qubits,
                  words_double[i + 2])
        exp_pauli(coef_double[i + 3] * theta[count], qubits,
                  words_double[i + 3])
        exp_pauli(coef_double[i + 4] * theta[count], qubits,
                  words_double[i + 4])
        exp_pauli(coef_double[i + 5] * theta[count], qubits,
                  words_double[i + 5])
        exp_pauli(coef_double[i + 6] * theta[count], qubits,
                  words_double[i + 6])
        exp_pauli(coef_double[i + 7] * theta[count], qubits,
                  words_double[i + 7])
        count += 1
