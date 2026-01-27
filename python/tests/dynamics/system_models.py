# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import uuid
import cudaq
from cudaq.operators import *
from cudaq.operators.boson import annihilate, create, number
from cudaq.dynamics import *
from cudaq.dynamics.integrators import *
import numpy as np
import cupy as cp


class TestSystem:

    def run_tests(self, integrator):
        pass


class TestCavityModel(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial state
        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        decay_rate = 0.1
        evolution_result = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi0,
            observables=[hamiltonian],
            collapse_operators=[np.sqrt(decay_rate) * annihilate(0)],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        expectation_values = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]
        expected_answer = (N - 1) * np.exp(-decay_rate * steps)
        np.testing.assert_allclose(expected_answer, expectation_values, 1e-3)


class TestCavityModelTimeDependentHam(TestSystem):

    def run_tests(self, integrator):
        hamiltonian = ScalarOperator(lambda t: 1.0) * number(0)
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        dimensions = {0: N}
        # initial state
        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        decay_rate = 0.1
        evolution_result = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi0,
            observables=[number(0)],
            collapse_operators=[np.sqrt(decay_rate) * annihilate(0)],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        expectation_values = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]
        expected_answer = (N - 1) * np.exp(-decay_rate * steps)
        np.testing.assert_allclose(expected_answer, expectation_values, 1e-3)


class TestCavityModelTimeDependentCollapseOp(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial state
        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        decay_rate = 0.1
        decay_op = ScalarOperator(
            lambda t: np.sqrt(decay_rate * np.exp(-t))) * annihilate(0)
        evolution_result = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi0,
            observables=[hamiltonian],
            collapse_operators=[decay_op],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        expectation_values = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]
        expected_answer = [
            (N - 1) * np.exp(-decay_rate * (1.0 - np.exp(-t))) for t in steps
        ]
        np.testing.assert_allclose(expected_answer, expectation_values, 1e-3)


class TestCompositeSystems(TestSystem):
    dimensions = {0: 2, 1: 10}
    a = annihilate(1)
    a_dag = create(1)
    sm = annihilate(0)
    sm_dag = create(0)
    hamiltonian = 2 * np.pi * number(1) + 2 * np.pi * number(
        0) + 2 * np.pi * 0.25 * (sm * a_dag + sm_dag * a)
    steps = np.linspace(0, 10, 201)
    tol = 0.1
    # Expected results (from qutips)
    exp_val_cavity_photon_count_expected = [
        5., 4.94466224, 4.83283916, 4.67577546, 4.49005953, 4.29554749,
        4.11293268, 3.9612583, 3.8556717, 3.80568503, 3.81413758, 3.87696871,
        3.98380677, 4.119292, 4.26495877, 4.40144402, 4.51075509, 4.5783315,
        4.59466257, 4.55628415, 4.46605353, 4.33269238, 4.16966358, 3.99353126,
        3.82200631, 3.67190972, 3.55728994, 3.48790565, 3.46823929, 3.49713171,
        3.56806657, 3.67005033, 3.78896882, 3.90925939, 4.01567932, 4.0949908,
        4.13736878, 4.13736771, 4.09437611, 4.01251319, 3.90000245, 3.76811486,
        3.62981024, 3.49824988, 3.38534406, 3.30049419, 3.24967193, 3.23490971,
        3.25425315, 3.30215402, 3.37024071, 3.44836692, 3.52580579, 3.59244887,
        3.63987998, 3.66220929, 3.65658578, 3.62334812, 3.56581635, 3.48976516,
        3.40265612, 3.31272469, 3.22803342, 3.15559767, 3.10067871, 3.06631096,
        3.05310468, 3.05933038, 3.08125612, 3.11368675, 3.15063275, 3.1860258,
        3.21439575, 3.23143717, 3.23440854, 3.22232951, 3.19596697, 3.15762627,
        3.11078307, 3.05960893, 3.00845174, 2.96133274, 2.92151698, 2.89120046,
        2.87134207, 2.86164934, 2.86070807, 2.86623131, 2.87538992, 2.88518025,
        2.89278426, 2.89588, 2.89287058, 2.88301077, 2.8664239, 2.84401594,
        2.8173043, 2.78818813, 2.75869224, 2.7307165, 2.7058207, 2.68506741,
        2.66893719, 2.65732093, 2.64958357, 2.64468636, 2.64134786, 2.63822099,
        2.6340634, 2.62788048, 2.61902601, 2.60725121, 2.59270021, 2.57585723,
        2.55745565, 2.53836366, 2.51946286, 2.50153562, 2.48517492, 2.47072635,
        2.45826687, 2.44762032, 2.4384044, 2.43010036, 2.42213454, 2.41395968,
        2.40512535, 2.39532878, 2.38444085, 2.37250556, 2.35971525, 2.34636682,
        2.33280716, 2.31937488, 2.30634939, 2.29391212, 2.28212631, 2.27093667,
        2.26018816, 2.24966021, 2.23911009, 2.22831915, 2.21713378, 2.20549565,
        2.19345582, 2.18117105, 2.16888258, 2.15688043, 2.14545821, 2.13486571,
        2.12526463, 2.11669586, 2.10906232, 2.10213054, 2.09555172, 2.0888997,
        2.08172175, 2.07359592, 2.06418712, 2.05329515, 2.04088803, 2.02711568,
        2.01230232, 1.99691669, 1.98152412, 1.96672526, 1.9530884, 1.9410839,
        1.93102743, 1.92303989, 1.91702838, 1.91269067, 1.90954432, 1.90697522,
        1.90430203, 1.90084857, 1.89601656, 1.88935027, 1.88058606, 1.86968122,
        1.85681903, 1.84238946, 1.82694775, 1.81115554, 1.79571095, 1.78127546,
        1.76840533, 1.75749503, 1.74873845, 1.74211201, 1.73738078, 1.73412685,
        1.73179624, 1.72975898, 1.72737601, 1.72406529, 1.71935987, 1.71295289,
        1.70472415, 1.69474653, 1.68327208
    ]
    exp_val_atom_excitation = [
        0., 0.03045109, 0.11780874, 0.25108517, 0.41387855, 0.58642725,
        0.74803001, 0.87953535, 0.96560131, 0.99645888, 0.96898079, 0.88694284,
        0.7604683, 0.60473357, 0.43810635, 0.27994732, 0.14834104, 0.05802209,
        0.01873597, 0.03421434, 0.10186891, 0.21321772, 0.35498156, 0.51070411,
        0.6626958, 0.79406976, 0.89063374, 0.94242526, 0.94472216, 0.89843557,
        0.80985324, 0.68978436, 0.55222219, 0.41268334, 0.28644233, 0.18684052,
        0.12386477, 0.10316084, 0.12555568, 0.18713662, 0.27985547, 0.39256731,
        0.51237723, 0.62612287, 0.72182968, 0.78997817, 0.82443994, 0.82300562,
        0.78745257, 0.72317102, 0.63840972, 0.54323895, 0.44836217, 0.36391703,
        0.29839564, 0.25779982, 0.24511453, 0.26014132, 0.2996911, 0.35809709,
        0.42797215, 0.50111569, 0.56945882, 0.62594185, 0.66522823, 0.68418739,
        0.68210374, 0.66060442, 0.62333276, 0.57541793, 0.52281172, 0.47157516,
        0.42720022, 0.3940388, 0.3748966, 0.37082755, 0.38113878, 0.40359115,
        0.43476063, 0.4705079, 0.50649584, 0.53869281, 0.56380504, 0.57959403,
        0.58505045, 0.58041472, 0.56705332, 0.54721485, 0.52370308, 0.49951097,
        0.4774606, 0.45989093, 0.4484257, 0.44384291, 0.44605353, 0.45418324,
        0.46674017, 0.48184203, 0.49747097, 0.51172419, 0.52303019, 0.53030814,
        0.53305529, 0.53135769, 0.52582921, 0.51749167, 0.50761545, 0.49754321,
        0.48851944, 0.48154649, 0.47728249, 0.47599038, 0.47754042, 0.4814612,
        0.48702902, 0.49338136, 0.49963785, 0.50501315, 0.50890781, 0.51096746,
        0.51110527, 0.50948779, 0.50648915, 0.50262218, 0.49845744, 0.49454187,
        0.49132814, 0.48912313, 0.48806118, 0.4881037, 0.48906316, 0.49064622,
        0.49250793, 0.49430987, 0.49577127, 0.49670815, 0.49705414, 0.49686134,
        0.49628218, 0.49553555, 0.49486368, 0.49448591, 0.49455746, 0.49513883,
        0.49618114, 0.49752929, 0.49894268, 0.50013061, 0.50079728, 0.50068936,
        0.49964055, 0.49760484, 0.49467433, 0.49107836, 0.48716303, 0.4833538,
        0.48010488, 0.47784194, 0.47690569, 0.47750332, 0.47967437, 0.4832762,
        0.48799066, 0.49335313, 0.49879971, 0.50372832, 0.50756624, 0.50983607,
        0.51021311, 0.50856635, 0.50497846, 0.49974248, 0.49333362, 0.48636152,
        0.47950633, 0.47344646, 0.46878562, 0.46598766, 0.46532612, 0.46685443,
        0.47039961, 0.47558062, 0.48184869, 0.48854565, 0.49497336, 0.50046693,
        0.50446361, 0.50656003, 0.50655194, 0.5044521, 0.50048532, 0.49506098,
        0.48872704, 0.48211041, 0.47585027, 0.47053189, 0.46662808, 0.46445353,
        0.46413733, 0.46561558, 0.46864443
    ]

    def run_test_simple(self, input_state, integrator):
        """
        test composite system
        """
        schedule = Schedule(self.steps, ["t"])
        evolution_result = cudaq.evolve(
            self.hamiltonian,
            self.dimensions,
            schedule,
            input_state,
            observables=[number(1), number(0)],
            collapse_operators=[np.sqrt(0.1) * self.a],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        exp_val_cavity_photon_count = []
        exp_val_atom_excitation = []

        for exp_vals in evolution_result.expectation_values():
            exp_val_cavity_photon_count.append(exp_vals[0].expectation())
            exp_val_atom_excitation.append(exp_vals[1].expectation())
        np.testing.assert_allclose(exp_val_cavity_photon_count,
                                   self.exp_val_cavity_photon_count_expected,
                                   atol=self.tol)
        np.testing.assert_allclose(exp_val_atom_excitation,
                                   self.exp_val_atom_excitation,
                                   atol=self.tol)

    def run_tests(self, integrator):
        qubit_state = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)
        cavity_state = cp.zeros((10, 10), dtype=cp.complex128)
        cavity_state[5][5] = 1.0
        rho0 = cudaq.State.from_data(cp.kron(cavity_state, qubit_state))
        self.run_test_simple(rho0, integrator)

        qubit_state = cp.array([1.0, 0.0], dtype=cp.complex128)
        cavity_state = cp.zeros(10, dtype=cp.complex128)
        cavity_state[5] = 1.0
        psi0 = cudaq.State.from_data(cp.kron(cavity_state, qubit_state))
        self.run_test_simple(psi0, integrator)


class TestCrossResonance(TestSystem):

    def run_tests(self, integrator):
        detuning = 100 * 2 * np.pi
        coupling_coeff = 7 * 2 * np.pi
        crosstalk_coeff = 0.2
        drive_strength = 20 * 2 * np.pi

        # Hamiltonian
        hamiltonian = detuning / 2 * spin.z(0) + coupling_coeff * (
            spin.minus(1) * spin.plus(0) +
            spin.plus(1) * spin.minus(0)) + drive_strength * spin.x(
                0) + crosstalk_coeff * drive_strength * spin.x(1)

        # Dimensions of sub-system
        dimensions = {0: 2, 1: 2}

        # Initial state of the system (ground state).
        rho0 = cudaq.State.from_data(
            cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

        # Two initial states: |00> and |10>.
        # We show the 'conditional' evolution when controlled qubit is in |1> state.
        psi_00 = cudaq.State.from_data(
            cp.array([1.0, 0.0, 0.0, 0.0], dtype=cp.complex128))
        psi_10 = cudaq.State.from_data(
            cp.array([0.0, 1.0, 0.0, 0.0], dtype=cp.complex128))

        # Schedule of time steps.
        steps = np.linspace(0.0, 1.0, 1001)
        schedule = Schedule(steps, ["t"])

        # Run the simulation.
        # Control bit = 0
        evolution_result_00 = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi_00,
            observables=[
                spin.x(0),
                spin.y(0),
                spin.z(0),
                spin.x(1),
                spin.y(1),
                spin.z(1)
            ],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        # Control bit = 1
        evolution_result_10 = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi_10,
            observables=[
                spin.x(0),
                spin.y(0),
                spin.z(0),
                spin.x(1),
                spin.y(1),
                spin.z(1)
            ],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        get_result = lambda idx, res: [
            exp_vals[idx].expectation()
            for exp_vals in res.expectation_values()
        ]
        results_00 = [
            get_result(0, evolution_result_00),
            get_result(1, evolution_result_00),
            get_result(2, evolution_result_00),
            get_result(3, evolution_result_00),
            get_result(4, evolution_result_00),
            get_result(5, evolution_result_00)
        ]
        results_10 = [
            get_result(0, evolution_result_10),
            get_result(1, evolution_result_10),
            get_result(2, evolution_result_10),
            get_result(3, evolution_result_10),
            get_result(4, evolution_result_10),
            get_result(5, evolution_result_10)
        ]

        def freq_from_crossings(sig):
            """
            Estimate frequency by counting zero crossings
            """
            crossings = np.where(np.diff(np.sign(sig)))[0]
            return 1.0 / np.mean(np.diff(crossings))

        freq_0 = freq_from_crossings(results_00[5])
        freq_1 = freq_from_crossings(results_10[5])
        np.testing.assert_allclose(freq_0, 2.0 * freq_1, atol=0.01)


class TestCallbackTensor(TestSystem):

    def run_tests(self, integrator):
        # Device parameters
        # Qubit resonant frequency
        nu_z = 10.0
        # Transverse term
        nu_x = 1.0
        # Harmonic driving frequency
        # Note: we chose a frequency slightly different from the resonant frequency to demonstrate the off-resonance effect.
        nu_d = 9.98

        def callback_tensor(t):
            return np.cos(2 * np.pi * nu_d * t) * np.array([[0., 1.], [1., 0.]],
                                                           dtype=np.complex128)

        # Let's define the control term as a callback tensor
        op_name = "control_term_" + str(uuid.uuid4())
        define(op_name, [2], callback_tensor)

        # Qubit Hamiltonian
        hamiltonian = 0.5 * 2 * np.pi * nu_z * spin.z(0)
        # Add modulated driving term to the Hamiltonian
        hamiltonian += 2 * np.pi * nu_x * instantiate(op_name, 0)

        # Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
        dimensions = {0: 2}

        # Initial state of the system (ground state).
        rho0 = cudaq.State.from_data(
            cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

        # Schedule of time steps.
        t_final = 0.5 / nu_x
        tau = .005
        n_steps = int(np.ceil(t_final / tau)) + 1
        steps1 = np.linspace(0, t_final, n_steps)
        schedule = Schedule(steps1, ["t"])

        # Run the simulation.
        # First, we run the simulation without any collapse operators (no decoherence).
        evolution_result = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            rho0,
            observables=[spin.x(0), spin.y(0), spin.z(0)],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        get_result = lambda idx, res: [
            exp_vals[idx].expectation()
            for exp_vals in res.expectation_values()
        ]
        ideal_results = [
            get_result(0, evolution_result),
            get_result(1, evolution_result),
            get_result(2, evolution_result)
        ]
        np.testing.assert_allclose(ideal_results[0][-1], 0, atol=0.1)
        np.testing.assert_allclose(ideal_results[1][-1], 0, atol=0.1)
        np.testing.assert_allclose(ideal_results[2][-1], -1, atol=0.1)


class TestBug3326(TestSystem):

    def case1(self, integrator):
        # This test is to reproduce the bug reported in issue #3326
        # Number of spins
        N = 3
        dimensions = {i: 2 for i in range(N)}

        # Observable
        observables = []
        for i in range(N):
            ob = spin.empty()
            ob += spin.z(i) / N
            observables.append(ob)

        H = spin.empty()
        for i in range(N):
            H += spin.x(i)
        for i in range(N - 1):
            H += spin.y(i) * spin.z(i + 1)

        steps = np.linspace(0.0, 1, 200)
        schedule = Schedule(steps, ["time"])

        gamma_dephasing = 0.05

        # Run the simulation
        evolution_result = cudaq.evolve(
            H,
            dimensions,
            schedule,
            initial_state=cudaq.dynamics.InitialState.ZERO,
            observables=observables,
            collapse_operators=[np.sqrt(gamma_dephasing) * spin.z(0)],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        assert len(evolution_result.expectation_values()) == len(steps)

    def case2(self, integrator):
        # This test is to reproduce the bug reported in issue #3326 (comment)
        w_q0 = 5
        w_q1 = 5
        w_tunc = 7

        anharmonicity = 0.21
        tunc_anharmonicity = 0.13
        N_tuncs = 3
        dim = {0: N_tuncs, 1: N_tuncs, 2: N_tuncs}

        H_q0 = w_q0 * boson.number(0) - anharmonicity * boson.number(0) * (
            boson.number(0) - 1) / 2
        H_q1 = w_q1 * boson.number(1) - anharmonicity * boson.number(1) * (
            boson.number(1) - 1) / 2
        H_tc = w_tunc * boson.create(2) * boson.annihilate(
            2) - tunc_anharmonicity * boson.number(2) * (boson.number(2) -
                                                         1) / 2
        H = H_tc + H_q0 + H_q1

        q0_state = cp.zeros((N_tuncs, N_tuncs), dtype=cp.complex128)
        q0_state[1, 1] = 1.0

        q1_state = cp.zeros((N_tuncs, N_tuncs), dtype=cp.complex128)
        q1_state[0, 0] = 1.0

        tc_state = cp.zeros((N_tuncs, N_tuncs), dtype=cp.complex128)
        tc_state[0, 0] = 1.0

        composite_state = cp.kron(cp.kron(tc_state, q1_state), q0_state)
        rho0 = cudaq.State.from_data(composite_state)

        steps = np.linspace(0, 10, 1000)
        schedule = Schedule(steps, ["t"])

        evolution_result = cudaq.evolve(
            H,
            dim,
            schedule,
            rho0,
            observables=[boson.number(0),
                         boson.number(1),
                         boson.number(2)],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        assert len(evolution_result.expectation_values()) == len(steps)

    def run_tests(self, integrator):
        self.case1(integrator)
        self.case2(integrator)


class TestInitialStateEnum(TestSystem):

    def run_tests(self, integrator):
        hamiltonian = 2.0 * np.pi * 0.1 * spin.x(0)
        steps = np.linspace(0, 1, 10)
        schedule = Schedule(steps, ["t"])
        dimensions = {0: 2}
        # initial state
        psi0 = cudaq.dynamics.InitialState.ZERO
        evolution_result = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi0,
            observables=[spin.z(0)],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        expectation_values = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]
        expected_answer = np.cos(4.0 * np.pi * 0.1 * steps)
        np.testing.assert_allclose(expected_answer, expectation_values, 1e-3)


class TestCavityModelBatchedInputState(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial states
        num_states = 4
        initial_states = []
        for i in range(num_states):
            psi0_ = cp.zeros(N, dtype=cp.complex128)
            psi0_[-(i + 1)] = 1.0
            initial_states.append(cudaq.State.from_data(psi0_))
        decay_rate = 0.1
        evolution_results = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            initial_states,
            observables=[hamiltonian],
            collapse_operators=[np.sqrt(decay_rate) * annihilate(0)],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        for i in range(num_states):
            expectation_values = [
                exp_vals[0].expectation()
                for exp_vals in evolution_results[i].expectation_values()
            ]
            expected_answer = (N - 1 - i) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer, expectation_values,
                                       1e-3)


class TestCavityModelSuperOperator(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial state
        rho0_ = cp.zeros(N * N, dtype=cp.complex128)
        rho0_[-1] = 1.0
        rho0_ = cudaq.State.from_data(rho0_)
        decay_rate = 0.1
        me_super_op = cudaq.SuperOperator()
        # Apply `-i[H, rho]` superop
        me_super_op += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
        me_super_op += cudaq.SuperOperator.right_multiply(1j * hamiltonian)
        L = np.sqrt(decay_rate) * annihilate(0)
        L_dagger = np.sqrt(decay_rate) * create(0)
        # Lindblad terms
        # L * rho * L_dagger
        me_super_op += cudaq.SuperOperator.left_right_multiply(L, L_dagger)
        # -0.5 * L_dagger * L * rho
        me_super_op += cudaq.SuperOperator.left_multiply(-0.5 * L_dagger * L)
        # -0.5 * rho * L_dagger * L
        me_super_op += cudaq.SuperOperator.right_multiply(-0.5 * L_dagger * L)

        evolution_result = cudaq.evolve(
            me_super_op,
            dimensions,
            schedule,
            rho0_,
            observables=[hamiltonian],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        expectation_values = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]
        expected_answer = (N - 1) * np.exp(-decay_rate * steps)
        np.testing.assert_allclose(expected_answer, expectation_values, 1e-3)


class TestInitialStateEnumSuperOperator(TestSystem):

    def run_tests(self, integrator):
        hamiltonian = 2.0 * np.pi * 0.1 * spin.x(0)
        steps = np.linspace(0, 1, 10)
        schedule = Schedule(steps, ["t"])
        dimensions = {0: 2}
        # initial state
        psi0 = cudaq.dynamics.InitialState.ZERO
        se_super_op = cudaq.SuperOperator()
        # Apply `-iH|psi>` superop
        se_super_op += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
        evolution_result = cudaq.evolve(
            se_super_op,
            dimensions,
            schedule,
            psi0,
            observables=[spin.z(0)],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        expectation_values = [
            exp_vals[0].expectation()
            for exp_vals in evolution_result.expectation_values()
        ]
        expected_answer = np.cos(4.0 * np.pi * 0.1 * steps)
        np.testing.assert_allclose(expected_answer, expectation_values, 1e-3)


class TestCavityModelBatchedInputStateSuperOperator(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial states
        num_states = 4
        initial_states = []
        for i in range(num_states):
            psi0_ = cp.zeros(N, dtype=cp.complex128)
            psi0_[-(i + 1)] = 1.0
            initial_states.append(cudaq.State.from_data(psi0_))
        decay_rate = 0.1

        me_super_op = cudaq.SuperOperator()
        # Apply `-i[H, rho]` superop
        me_super_op += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
        me_super_op += cudaq.SuperOperator.right_multiply(1j * hamiltonian)
        L = np.sqrt(decay_rate) * annihilate(0)
        L_dagger = np.sqrt(decay_rate) * create(0)
        # Lindblad terms
        # L * rho * L_dagger
        me_super_op += cudaq.SuperOperator.left_right_multiply(L, L_dagger)
        # -0.5 * L_dagger * L * rho
        me_super_op += cudaq.SuperOperator.left_multiply(-0.5 * L_dagger * L)
        # -0.5 * rho * L_dagger * L
        me_super_op += cudaq.SuperOperator.right_multiply(-0.5 * L_dagger * L)

        evolution_results = cudaq.evolve(
            me_super_op,
            dimensions,
            schedule,
            initial_states,
            observables=[hamiltonian],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        for i in range(num_states):
            expectation_values = [
                exp_vals[0].expectation()
                for exp_vals in evolution_results[i].expectation_values()
            ]
            expected_answer = (N - 1 - i) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer, expectation_values,
                                       1e-3)


class TestBatchedCavityModel(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        dimensions = {0: N}
        # initial state
        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        decay_rates = [0.05, 0.1, 0.15, 0.2]
        collapse_operators_list = [
            [np.sqrt(decay_rate) * annihilate(0)] for decay_rate in decay_rates
        ]
        hamiltonian_list = [number(0)] * len(decay_rates)
        initial_states = [psi0] * len(decay_rates)
        evolution_results = cudaq.evolve(
            hamiltonian_list,
            dimensions,
            schedule,
            initial_states,
            observables=[number(0)],
            collapse_operators=collapse_operators_list,
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        for i, decay_rate in enumerate(decay_rates):
            evolution_result = evolution_results[i]
            expectation_values = [
                exp_vals[0].expectation()
                for exp_vals in evolution_result.expectation_values()
            ]
            expected_answer = (N - 1) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer, expectation_values,
                                       1e-3)


class TestBatchedCavityModelBroadcastInputState(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        dimensions = {0: N}
        # initial state
        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        decay_rates = [0.05, 0.1, 0.15, 0.2]
        collapse_operators_list = [
            [np.sqrt(decay_rate) * annihilate(0)] for decay_rate in decay_rates
        ]
        hamiltonian_list = [number(0)] * len(decay_rates)
        evolution_results = cudaq.evolve(
            hamiltonian_list,
            dimensions,
            schedule,
            psi0,
            observables=[number(0)],
            collapse_operators=collapse_operators_list,
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        for i, decay_rate in enumerate(decay_rates):
            evolution_result = evolution_results[i]
            expectation_values = [
                exp_vals[0].expectation()
                for exp_vals in evolution_result.expectation_values()
            ]
            expected_answer = (N - 1) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer, expectation_values,
                                       1e-3)


class TestBatchedCavityModelTimeDependentHam(TestSystem):

    def run_tests(self, integrator):
        hamiltonian = ScalarOperator(lambda t: 1.0) * number(0)
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        dimensions = {0: N}
        # initial state
        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        decay_rates = [0.05, 0.1, 0.15, 0.2]
        collapsed_operators_list = [
            [np.sqrt(decay_rate) * annihilate(0)] for decay_rate in decay_rates
        ]
        hamiltonian_list = [hamiltonian] * len(decay_rates)
        initial_states = [psi0] * len(decay_rates)
        evolution_results = cudaq.evolve(
            hamiltonian_list,
            dimensions,
            schedule,
            initial_states,
            observables=[number(0)],
            collapse_operators=collapsed_operators_list,
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        for i, decay_rate in enumerate(decay_rates):
            evolution_result = evolution_results[i]
            expectation_values = [
                exp_vals[0].expectation()
                for exp_vals in evolution_result.expectation_values()
            ]
            expected_answer = (N - 1) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer, expectation_values,
                                       1e-3)


class TestBatchedCavityModelTimeDependentCollapseOp(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial state
        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        decay_rates = [0.05, 0.1]
        collapse_operators_list = [[
            ScalarOperator(lambda t, decay_rate=decay_rate: np.sqrt(
                decay_rate * np.exp(-t))) * annihilate(0)
        ] for decay_rate in decay_rates]
        hamiltonian_list = [hamiltonian] * len(decay_rates)
        initial_states = [psi0] * len(decay_rates)

        evolution_results = cudaq.evolve(
            hamiltonian_list,
            dimensions,
            schedule,
            initial_states,
            observables=[hamiltonian],
            collapse_operators=collapse_operators_list,
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())
        for i, decay_rate in enumerate(decay_rates):
            evolution_result = evolution_results[i]
            expectation_values = [
                exp_vals[0].expectation()
                for exp_vals in evolution_result.expectation_values()
            ]
            expected_answer = [
                (N - 1) * np.exp(-decay_rate * (1.0 - np.exp(-t)))
                for t in steps
            ]
            np.testing.assert_allclose(expected_answer, expectation_values,
                                       1e-3)


class TestBatchedCavityModelSuperOperator(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial state
        rho0_ = cp.zeros(N * N, dtype=cp.complex128)
        rho0_[-1] = 1.0
        rho0_ = cudaq.State.from_data(rho0_)
        decay_rates = [0.05, 0.1]
        me_super_op_lists = []
        for decay_rate in decay_rates:
            me_super_op = cudaq.SuperOperator()
            # Apply `-i[H, rho]` superop
            me_super_op += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
            me_super_op += cudaq.SuperOperator.right_multiply(1j * hamiltonian)
            L = np.sqrt(decay_rate) * annihilate(0)
            L_dagger = np.sqrt(decay_rate) * create(0)
            # Lindblad terms
            # L * rho * L_dagger
            me_super_op += cudaq.SuperOperator.left_right_multiply(L, L_dagger)
            # -0.5 * L_dagger * L * rho
            me_super_op += cudaq.SuperOperator.left_multiply(-0.5 * L_dagger *
                                                             L)
            # -0.5 * rho * L_dagger * L
            me_super_op += cudaq.SuperOperator.right_multiply(-0.5 * L_dagger *
                                                              L)

            # Add this super operator to the list
            me_super_op_lists.append(me_super_op)
        initial_states = [rho0_] * len(decay_rates)

        evolution_results = cudaq.evolve(
            me_super_op_lists,
            dimensions,
            schedule,
            initial_states,
            observables=[hamiltonian],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        for i, decay_rate in enumerate(decay_rates):
            evolution_result = evolution_results[i]
            expectation_values = [
                exp_vals[0].expectation()
                for exp_vals in evolution_result.expectation_values()
            ]
            expected_answer = (N - 1) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer, expectation_values,
                                       1e-3)


class TestBatchedCavityModelSuperOperatorBroadcastInputState(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial state
        rho0_ = cp.zeros(N * N, dtype=cp.complex128)
        rho0_[-1] = 1.0
        rho0 = cudaq.State.from_data(rho0_)
        decay_rates = [0.05, 0.1]
        me_super_op_lists = []
        for decay_rate in decay_rates:
            me_super_op = cudaq.SuperOperator()
            # Apply `-i[H, rho]` superop
            me_super_op += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
            me_super_op += cudaq.SuperOperator.right_multiply(1j * hamiltonian)
            L = np.sqrt(decay_rate) * annihilate(0)
            L_dagger = np.sqrt(decay_rate) * create(0)
            # Lindblad terms
            # L * rho * L_dagger
            me_super_op += cudaq.SuperOperator.left_right_multiply(L, L_dagger)
            # -0.5 * L_dagger * L * rho
            me_super_op += cudaq.SuperOperator.left_multiply(-0.5 * L_dagger *
                                                             L)
            # -0.5 * rho * L_dagger * L
            me_super_op += cudaq.SuperOperator.right_multiply(-0.5 * L_dagger *
                                                              L)

            # Add this super operator to the list
            me_super_op_lists.append(me_super_op)

        evolution_results = cudaq.evolve(
            me_super_op_lists,
            dimensions,
            schedule,
            rho0,
            observables=[hamiltonian],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        for i, decay_rate in enumerate(decay_rates):
            evolution_result = evolution_results[i]
            expectation_values = [
                exp_vals[0].expectation()
                for exp_vals in evolution_result.expectation_values()
            ]
            expected_answer = (N - 1) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer, expectation_values,
                                       1e-3)


class TestBatchedCavityModelWithBatchSize(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        dimensions = {0: N}
        # initial state
        psi0_ = cp.zeros(N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        decay_rates = [0.05, 0.1, 0.15, 0.2]
        collapse_operators_list = [
            [np.sqrt(decay_rate) * annihilate(0)] for decay_rate in decay_rates
        ]
        hamiltonian_list = [number(0)] * len(decay_rates)
        initial_states = [psi0] * len(decay_rates)
        evolution_results_1 = cudaq.evolve(
            hamiltonian_list,
            dimensions,
            schedule,
            initial_states,
            observables=[number(0)],
            collapse_operators=collapse_operators_list,
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator(),
            max_batch_size=1)

        evolution_results_2 = cudaq.evolve(
            hamiltonian_list,
            dimensions,
            schedule,
            initial_states,
            observables=[number(0)],
            collapse_operators=collapse_operators_list,
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator(),
            max_batch_size=2)

        for i, decay_rate in enumerate(decay_rates):
            evolution_result_1 = evolution_results_1[i]
            evolution_result_2 = evolution_results_2[i]
            expectation_values_1 = [
                exp_vals[0].expectation()
                for exp_vals in evolution_result_1.expectation_values()
            ]
            expectation_values_2 = [
                exp_vals[0].expectation()
                for exp_vals in evolution_result_2.expectation_values()
            ]
            expected_answer = (N - 1) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer, expectation_values_1,
                                       1e-3)
            np.testing.assert_allclose(expected_answer, expectation_values_2,
                                       1e-3)


class TestBatchedCavityModelSuperOperatorWithBatchSize(TestSystem):

    def run_tests(self, integrator):
        N = 10
        steps = np.linspace(0, 10, 101)
        schedule = Schedule(steps, ["t"])
        hamiltonian = number(0)
        dimensions = {0: N}
        # initial state
        rho0_ = cp.zeros(N * N, dtype=cp.complex128)
        rho0_[-1] = 1.0
        rho0_ = cudaq.State.from_data(rho0_)
        decay_rates = [0.05, 0.1]
        me_super_op_lists = []
        for decay_rate in decay_rates:
            me_super_op = cudaq.SuperOperator()
            # Apply `-i[H, rho]` superop
            me_super_op += cudaq.SuperOperator.left_multiply(-1j * hamiltonian)
            me_super_op += cudaq.SuperOperator.right_multiply(1j * hamiltonian)
            L = np.sqrt(decay_rate) * annihilate(0)
            L_dagger = np.sqrt(decay_rate) * create(0)
            # Lindblad terms
            # L * rho * L_dagger
            me_super_op += cudaq.SuperOperator.left_right_multiply(L, L_dagger)
            # -0.5 * L_dagger * L * rho
            me_super_op += cudaq.SuperOperator.left_multiply(-0.5 * L_dagger *
                                                             L)
            # -0.5 * rho * L_dagger * L
            me_super_op += cudaq.SuperOperator.right_multiply(-0.5 * L_dagger *
                                                              L)

            # Add this super operator to the list
            me_super_op_lists.append(me_super_op)
        initial_states = [rho0_] * len(decay_rates)

        evolution_results_batch_size_1 = cudaq.evolve(
            me_super_op_lists,
            dimensions,
            schedule,
            initial_states,
            observables=[hamiltonian],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator(),
            max_batch_size=1)

        evolution_results_batch_size_10 = cudaq.evolve(
            me_super_op_lists,
            dimensions,
            schedule,
            initial_states,
            observables=[hamiltonian],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator(),
            max_batch_size=10)

        for i, decay_rate in enumerate(decay_rates):
            evolution_result_batch_size_1 = evolution_results_batch_size_1[i]
            evolution_result_batch_size_10 = evolution_results_batch_size_10[i]
            # Check that the results are the same for both batch sizes
            expectation_values_batch_size_1 = [
                exp_vals[0].expectation() for exp_vals in
                evolution_result_batch_size_1.expectation_values()
            ]
            expectation_values_batch_size_10 = [
                exp_vals[0].expectation() for exp_vals in
                evolution_result_batch_size_10.expectation_values()
            ]
            expected_answer = (N - 1) * np.exp(-decay_rate * steps)
            np.testing.assert_allclose(expected_answer,
                                       expectation_values_batch_size_1, 1e-3)
            np.testing.assert_allclose(expected_answer,
                                       expectation_values_batch_size_10, 1e-3)


class TestMultiDegreeElemOp(TestSystem):

    def run_tests(self, integrator):
        detuning = 100 * 2 * np.pi
        coupling_coeff = 7 * 2 * np.pi
        crosstalk_coeff = 0.2
        drive_strength = 20 * 2 * np.pi

        # Hamiltonian
        sigma_minus = spin.minus(0).to_matrix()
        sigma_plus = spin.plus(0).to_matrix()

        def op_definition():
            return np.kron(sigma_minus, sigma_plus) + np.kron(
                sigma_plus, sigma_minus)

        cudaq.operators.define("coupling_op", (2, 2),
                               op_definition,
                               override=True)

        hamiltonian = detuning / 2 * spin.z(
            0) + coupling_coeff * cudaq.operators.instantiate(
                "coupling_op", [0, 1]) + drive_strength * spin.x(
                    0) + crosstalk_coeff * drive_strength * spin.x(1)

        # Dimensions of sub-system
        dimensions = {0: 2, 1: 2}

        # Initial state of the system (ground state).
        rho0 = cudaq.State.from_data(
            cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128))

        # Two initial states: |00> and |10>.
        # We show the 'conditional' evolution when controlled qubit is in |1> state.
        psi_00 = cudaq.State.from_data(
            cp.array([1.0, 0.0, 0.0, 0.0], dtype=cp.complex128))
        psi_10 = cudaq.State.from_data(
            cp.array([0.0, 1.0, 0.0, 0.0], dtype=cp.complex128))

        # Schedule of time steps.
        steps = np.linspace(0.0, 1.0, 1001)
        schedule = Schedule(steps, ["t"])

        # Run the simulation.
        # Control bit = 0
        evolution_result_00 = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi_00,
            observables=[
                spin.x(0),
                spin.y(0),
                spin.z(0),
                spin.x(1),
                spin.y(1),
                spin.z(1)
            ],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        # Control bit = 1
        evolution_result_10 = cudaq.evolve(
            hamiltonian,
            dimensions,
            schedule,
            psi_10,
            observables=[
                spin.x(0),
                spin.y(0),
                spin.z(0),
                spin.x(1),
                spin.y(1),
                spin.z(1)
            ],
            collapse_operators=[],
            store_intermediate_results=cudaq.IntermediateResultSave.
            EXPECTATION_VALUE,
            integrator=integrator())

        get_result = lambda idx, res: [
            exp_vals[idx].expectation()
            for exp_vals in res.expectation_values()
        ]
        results_00 = [
            get_result(0, evolution_result_00),
            get_result(1, evolution_result_00),
            get_result(2, evolution_result_00),
            get_result(3, evolution_result_00),
            get_result(4, evolution_result_00),
            get_result(5, evolution_result_00)
        ]
        results_10 = [
            get_result(0, evolution_result_10),
            get_result(1, evolution_result_10),
            get_result(2, evolution_result_10),
            get_result(3, evolution_result_10),
            get_result(4, evolution_result_10),
            get_result(5, evolution_result_10)
        ]

        def freq_from_crossings(sig):
            """
            Estimate frequency by counting zero crossings
            """
            crossings = np.where(np.diff(np.sign(sig)))[0]
            return 1.0 / np.mean(np.diff(crossings))

        freq_0 = freq_from_crossings(results_00[5])
        freq_1 = freq_from_crossings(results_10[5])
        np.testing.assert_allclose(freq_0, 2.0 * freq_1, atol=0.01)


class TestDensityMatrixIndexing(TestSystem):

    def run_tests(self, integrator):
        """
        Test that density matrix element access uses correct indexing.
        
        This is a regression test for a bug where the operator() function
        used the total dimension (dim*dim) instead of single-side dimension (dim)
        for bounds checking and linear index calculation.
        
        For a 2-qubit system (4x4 density matrix with 16 total elements):
        - Valid indices should be 0, 1, 2, 3
        - Accessing rho[i, j] should compute linear index as i * 4 + j
        - The bug computed i * 16 + j, causing out-of-bounds access
        """

        import pytest

        # 1-qubit system: 2x2 density matrix
        psi0_1q = cudaq.State.from_data(
            cp.array([1.0, 0.0], dtype=cp.complex128))
        hamiltonian_1q = 0.0 * spin.z(0)
        steps = np.linspace(0.0, 0.1, 2)
        schedule = Schedule(steps, ["t"])

        result_1q = cudaq.evolve(
            hamiltonian_1q, {0: 2},
            schedule,
            psi0_1q,
            collapse_operators=[spin.z(0)],
            store_intermediate_results=cudaq.IntermediateResultSave.ALL,
            integrator=integrator())

        rho_1q = result_1q.final_state()
        # For |0> initial state, density matrix is |0><0|
        # rho[0,0] = 1, rho[1,1] = 0
        assert abs(rho_1q[0, 0] - 1.0) < 1e-10
        assert abs(rho_1q[1, 1]) < 1e-10
        assert abs(rho_1q[0, 1]) < 1e-10
        assert abs(rho_1q[1, 0]) < 1e-10

        # Test out-of-bounds access is rejected
        with pytest.raises(RuntimeError, match="indices out of range"):
            _ = rho_1q[2, 0]

        # 2-qubit system: 4x4 density matrix
        psi0_2q = cudaq.State.from_data(
            cp.array([1.0, 0.0, 0.0, 0.0], dtype=cp.complex128))
        hamiltonian_2q = 0.0 * spin.z(0) + 0.0 * spin.z(1)

        result_2q = cudaq.evolve(
            hamiltonian_2q, {
                0: 2,
                1: 2
            },
            schedule,
            psi0_2q,
            collapse_operators=[spin.z(0)],
            store_intermediate_results=cudaq.IntermediateResultSave.ALL,
            integrator=integrator())

        rho_2q = result_2q.final_state()
        # For |00> initial state, density matrix is |00><00|
        # Only rho[0,0] = 1, all other elements = 0
        assert abs(rho_2q[0, 0] - 1.0) < 1e-10
        # These indices would cause out-of-bounds access with the old buggy code
        assert abs(rho_2q[1, 1]) < 1e-10
        assert abs(rho_2q[2, 2]) < 1e-10
        assert abs(rho_2q[3, 3]) < 1e-10
        # Off-diagonal elements
        assert abs(rho_2q[0, 3]) < 1e-10
        assert abs(rho_2q[3, 0]) < 1e-10

        # Test out-of-bounds access is rejected
        with pytest.raises(RuntimeError, match="indices out of range"):
            _ = rho_2q[4, 0]
        with pytest.raises(RuntimeError, match="indices out of range"):
            _ = rho_2q[0, 4]
        with pytest.raises(RuntimeError, match="indices out of range"):
            _ = rho_2q[4, 4]
