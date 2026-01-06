"""
    Contains the class with the VQE using the quantum-number-preserving ansatz
"""
import numpy as np
import cudaq
from cudaq import spin as spin_op


class VQE(object):
    """
        Implements the quantum-number-preserving ansatz from Anselmetti et al. NJP 23 (2021)
    """

    def __init__(self, n_qubits, num_active_electrons, spin, options):
        self.n_qubits = n_qubits
        self.n_layers = options.get('n_vqe_layers', 1)
        self.number_of_Q_blocks = n_qubits // 2 - 1
        self.num_params = 2 * self.number_of_Q_blocks * self.n_layers
        self.options = options
        num_active_orbitals = n_qubits // 2

        # number of alpha and beta electrons in the active space
        num_active_electrons_alpha = (num_active_electrons + spin) // 2
        num_active_electrons_beta = (num_active_electrons - spin) // 2

        # Define the initial state for the VQE as a list
        # [n_1, n_2, ....]
        # where n_j=(0,1,2) is the occupation of j-`th` the orbital

        n_alpha_vec = [1] * num_active_electrons_alpha + [0] * (
            num_active_orbitals - num_active_electrons_alpha)
        n_beta_vec = [1] * num_active_electrons_beta + [0] * (
            num_active_orbitals - num_active_electrons_beta)
        init_mo_occ = [n_a + n_b for n_a, n_b in zip(n_alpha_vec, n_beta_vec)]

        self.init_mo_occ = init_mo_occ
        self.final_state_vector_best = None
        self.best_vqe_params = None
        self.best_vqe_energy = None
        self.target = "nvidia"
        self.initial_x_gates_pos = self.prepare_initial_circuit()

    def prepare_initial_circuit(self):
        """
        Creates a list with the position of the X gates that should be applied to the initial |00...0>
        state to set the number of electrons and the spin correctly
        """
        x_gates_pos_list = []
        if self.init_mo_occ is not None:
            for idx_occ, occ in enumerate(self.init_mo_occ):
                if int(occ) == 2:
                    x_gates_pos_list.extend([2 * idx_occ, 2 * idx_occ + 1])
                elif int(occ) == 1:
                    x_gates_pos_list.append(2 * idx_occ)

        return x_gates_pos_list

    def layers(self):
        """
            Generates the QNP ansatz circuit and returns the  kernel and the optimization parameters thetas

            `params`: list/`np`.array
            [theta_0, ..., theta_{M-1}, phi_0, ..., phi_{M-1}]
            where M is the total number of blocks = layer * (n_qubits/2 - 1)

            returns: kernel
                     thetas
        """
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        number_of_blocks = self.number_of_Q_blocks

        # changed self.target to `nvidia`` to pass CI job, as it expects a string
        # literal
        # cudaq.set_target("`nvidia`")  # `nvidia` or `nvidia-mgpu`

        kernel, thetas = cudaq.make_kernel(list)
        # Allocate n qubits.
        qubits = kernel.qalloc(n_qubits)

        for init_gate_position in self.initial_x_gates_pos:
            kernel.x(qubits[init_gate_position])

        count_params = 0
        for idx_layer in range(n_layers):
            for starting_block_num in [0, 1]:
                for idx_block in range(starting_block_num, number_of_blocks, 2):
                    qubit_list = [qubits[2 * idx_block + j] for j in range(4)]

                    # PX gates decomposed in terms of standard gates
                    # and NO controlled Y rotations.
                    # See Appendix E1 of Anselmetti et al New J. Phys. 23 (2021) 113010

                    a, b, c, d = qubit_list
                    kernel.cx(d, b)
                    kernel.cx(d, a)
                    kernel.rz(parameter=-np.pi / 2, target=a)
                    kernel.s(b)
                    kernel.h(d)
                    kernel.cx(d, c)
                    kernel.cx(b, a)
                    kernel.ry(parameter=(1 / 8) * thetas[count_params],
                              target=c)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params],
                              target=d)
                    kernel.rz(parameter=+np.pi / 2, target=a)
                    kernel.cz(a, d)
                    kernel.cx(a, c)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params],
                              target=d)
                    kernel.ry(parameter=(+1 / 8) * thetas[count_params],
                              target=c)
                    kernel.cx(b, c)
                    kernel.cx(b, d)
                    kernel.rz(parameter=+np.pi / 2, target=b)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params],
                              target=c)
                    kernel.ry(parameter=(+1 / 8) * thetas[count_params],
                              target=d)
                    kernel.cx(a, c)
                    kernel.cz(a, d)
                    kernel.ry(parameter=(-1 / 8) * thetas[count_params],
                              target=c)
                    kernel.ry(parameter=(1 / 8) * thetas[count_params],
                              target=d)
                    kernel.cx(d, c)
                    kernel.h(d)
                    kernel.cx(d, b)
                    kernel.s(d)
                    kernel.rz(parameter=-np.pi / 2, target=b)
                    kernel.cx(b, a)
                    count_params += 1

                    # Orbital rotation
                    kernel.fermionic_swap(np.pi, b, c)
                    kernel.givens_rotation((-1 / 2) * thetas[count_params], a,
                                           b)
                    kernel.givens_rotation((-1 / 2) * thetas[count_params], c,
                                           d)
                    kernel.fermionic_swap(np.pi, b, c)
                    count_params += 1

        return kernel, thetas

    def get_state_vector(self, param_list):
        """
        Returns the state vector generated by the ansatz with parameters given by `param`_list
        """
        kernel, thetas = self.layers()
        state = convert_state_big_endian(
            np.array(
                cudaq.StateMemoryView(cudaq.get_state(kernel, param_list),
                                      dtype=complex)))
        return state

    def execute(self, hamiltonian):
        """
        Run VQE
        """
        options = self.options
        mpi_support = options.get("mpi_support", False)
        return_final_state_vec = options.get("return_final_state_vec", False)

        if mpi_support:
            cudaq.mpi.initialize()
            print('# mpi is initialized? ', cudaq.mpi.is_initialized())
            num_ranks = cudaq.mpi.num_ranks()
            rank = cudaq.mpi.rank()
            print('# rank', rank, 'num_ranks', num_ranks)

        optimizer = cudaq.optimizers.COBYLA()
        initial_parameters = options.get('initial_parameters')
        if initial_parameters:
            optimizer.initial_parameters = initial_parameters
        else:
            optimizer.initial_parameters = np.random.rand(self.num_params)

        kernel, thetas = self.layers()
        maxiter = options.get('maxiter', 100)
        optimizer.max_iterations = options.get('maxiter', maxiter)
        optimizer_type = "cudaq"
        callback_energies = []

        def eval(theta):
            """
            Compute the energy by cudaq.observe
            """
            exp_val = cudaq.observe(kernel, hamiltonian, theta).expectation()

            callback_energies.append(exp_val)
            return exp_val

        if optimizer_type == "cudaq":
            print("# Using cudaq optimizer")
            energy_optimized, best_parameters = optimizer.optimize(
                self.num_params, eval)

            # We add here the energy core
            energy_core = options.get('energy_core', 0.)
            total_opt_energy = energy_optimized + energy_core
            callback_energies = [en + energy_core for en in callback_energies]

            print("# Num Params:", self.num_params)
            print("# Qubits:", self.n_qubits)
            print("# N_layers:", self.n_layers)
            print("# Energy after the VQE:", total_opt_energy)

            result = {
                "energy_optimized": total_opt_energy,
                "best_parameters": best_parameters,
                "callback_energies": callback_energies
            }

            if return_final_state_vec:
                result["state_vec"] = self.get_state_vector(best_parameters)
            return result

        else:
            print(f"# Optimizer {optimizer_type} not implemented")
            exit()


def convert_state_big_endian(state_little_endian):

    state_big_endian = 0. * state_little_endian

    n_qubits = int(np.log2(state_big_endian.size))
    for j, val in enumerate(state_little_endian):
        little_endian_pos = np.binary_repr(j, n_qubits)
        big_endian_pos = little_endian_pos[::-1]
        int_big_endian_pos = int(big_endian_pos, 2)
        state_big_endian[int_big_endian_pos] = state_little_endian[j]

    return state_big_endian


def from_string_to_cudaq_spin(pauli_string, qubit):
    if pauli_string.lower() in ('id', 'i'):
        return 1
    elif pauli_string.lower() == 'x':
        return spin_op.x(qubit)
    elif pauli_string.lower() == 'y':
        return spin_op.y(qubit)
    elif pauli_string.lower() == 'z':
        return spin_op.z(qubit)


def get_cudaq_hamiltonian(jw_hamiltonian):
    """ Converts an `openfermion` QubitOperator Hamiltonian into a `cudaq.SpinOperator` Hamiltonian

    """

    hamiltonian_cudaq = 0.0
    energy_core = 0.0
    for ham_term in jw_hamiltonian:
        [(operators, ham_coeff)] = ham_term.terms.items()
        if len(operators):
            cuda_operator = 1.0
            for qubit_index, pauli_op in operators:
                cuda_operator *= from_string_to_cudaq_spin(
                    pauli_op, qubit_index)
        else:
            cuda_operator = 0.0  #from_string_to_cudaq_spin('id', 0)
            energy_core = ham_coeff
        cuda_operator = ham_coeff * cuda_operator
        hamiltonian_cudaq += cuda_operator

    return hamiltonian_cudaq, energy_core


def get_cudaq_operator(jw_hamiltonian):
    """ Converts an `openfermion` QubitOperator Hamiltonian into a `cudaq.SpinOperator` Hamiltonian

    """

    hamiltonian_cudaq = 0.0
    for ham_term in jw_hamiltonian:
        [(operators, ham_coeff)] = ham_term.terms.items()
        if len(operators):
            cuda_operator = 1.0
            for qubit_index, pauli_op in operators:
                cuda_operator *= from_string_to_cudaq_spin(
                    pauli_op, qubit_index)
        else:
            cuda_operator = from_string_to_cudaq_spin('id', 0)
        if abs(ham_coeff.imag) < 1e-8:
            cuda_operator = ham_coeff.real * cuda_operator
        else:
            print(
                "In function get_cudaq_operator can convert only real operator to cuda_operator"
            )
            exit()
        hamiltonian_cudaq += cuda_operator

    return hamiltonian_cudaq
