# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os

import pytest
import cudaq
from cudaq.operator import *
import numpy as np
import cupy as cp

cudaq.set_target("nvidia-dynamics")
ci_integrators = [RungeKuttaIntegrator, ScipyZvodeIntegrator, CUDATorchDiffEqDopri5Integrator]
all_integrators = [RungeKuttaIntegrator, ScipyZvodeIntegrator, CUDATorchDiffEqDopri5Integrator, CUDATorchDiffEqAdaptiveHeunIntegrator, CUDATorchDiffEqBosh3Integrator, CUDATorchDiffEqFixedAdamsIntegrator, CUDATorchDiffEqDopri8Integrator, CUDATorchDiffEqEulerIntegrator, CUDATorchDiffEqExplicitAdamsIntegrator, CUDATorchDiffEqMidpointIntegrator, CUDATorchDiffEqRK4Integrator]

# By default, only test representatives from each integrator collection.
all_integrator_classes = ci_integrators


class TestCavityDecay:
    N = 10
    dimensions = {0: N}
    a = operators.annihilate(0)
    a_dag = operators.create(0)
    kappa = 0.2
    steps = np.linspace(0, 10, 201)
    number = operators.number(0)
    tol = 1e-3

    @pytest.fixture(params=[pytest.param(np.sqrt(kappa) * a, id='const')])
    def const_c_ops(self, request):
        return request.param

    @pytest.mark.parametrize('integrator',
                             all_integrator_classes,
                             ids=all_integrator_classes)
    def test_simple(self, const_c_ops, integrator):
        """
        test simple constant decay, constant Hamiltonian
        """
        hamiltonian = self.number
        schedule = Schedule(self.steps, ["t"])
        # initial state
        psi0_ = cp.zeros(self.N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        evolution_result = evolve(hamiltonian,
                                  self.dimensions,
                                  schedule,
                                  psi0,
                                  observables=[hamiltonian],
                                  collapse_operators=[const_c_ops],
                                  store_intermediate_results=True,
                                  integrator=integrator())
        expt = []
        for exp_vals in evolution_result.expectation_values():
            expt.append(exp_vals[0].expectation())
        actual_answer = 9.0 * np.exp(-self.kappa * self.steps)
        np.testing.assert_allclose(actual_answer, expt, atol=self.tol)

    @pytest.mark.parametrize('integrator',
                             all_integrator_classes,
                             ids=all_integrator_classes)
    def test_td_ham(self, const_c_ops, integrator):
        """
        test time-dependent Hamiltonian with constant decay
        """
        hamiltonian = ScalarOperator(lambda t: 1.0) * self.number
        schedule = Schedule(self.steps, ["t"])
        # initial state
        psi0_ = cp.zeros(self.N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        evolution_result = evolve(hamiltonian,
                                  self.dimensions,
                                  schedule,
                                  psi0,
                                  observables=[self.number],
                                  collapse_operators=[const_c_ops],
                                  store_intermediate_results=True,
                                  integrator=integrator())
        expt = []
        for exp_vals in evolution_result.expectation_values():
            expt.append(exp_vals[0].expectation())
        actual_answer = 9.0 * np.exp(-self.kappa * self.steps)
        np.testing.assert_allclose(actual_answer, expt, atol=self.tol)

    @pytest.mark.parametrize('integrator',
                             all_integrator_classes,
                             ids=all_integrator_classes)
    def test_td_collapse_ops(self, integrator):
        """
        test time-dependent collapse operators
        """
        hamiltonian = self.number
        schedule = Schedule(self.steps, ["t"])
        # initial state
        psi0_ = cp.zeros(self.N, dtype=cp.complex128)
        psi0_[-1] = 1.0
        psi0 = cudaq.State.from_data(psi0_)
        c_op = ScalarOperator(
            lambda t: np.sqrt(self.kappa * np.exp(-t))) * self.a
        evolution_result = evolve(hamiltonian,
                                  self.dimensions,
                                  schedule,
                                  psi0,
                                  observables=[hamiltonian],
                                  collapse_operators=[c_op],
                                  store_intermediate_results=True,
                                  integrator=integrator())
        expt = []
        for exp_vals in evolution_result.expectation_values():
            expt.append(exp_vals[0].expectation())
        actual_answer = 9.0 * np.exp(-self.kappa * (1.0 - np.exp(-self.steps)))
        np.testing.assert_allclose(actual_answer, expt, atol=self.tol)


class TestCompositeSystems:
    dimensions = {0: 2, 1: 10}
    a = operators.annihilate(1)
    a_dag = operators.create(1)
    sm = operators.annihilate(0)
    sm_dag = operators.create(0)
    hamiltonian = 2 * np.pi * operators.number(
        1) + 2 * np.pi * operators.number(0) + 2 * np.pi * 0.25 * (sm * a_dag +
                                                                   sm_dag * a)
    qubit_state = cp.array([[1.0, 0.0], [0.0, 0.0]], dtype=cp.complex128)
    cavity_state = cp.zeros((10, 10), dtype=cp.complex128)
    cavity_state[5][5] = 1.0
    rho0 = cudaq.State.from_data(cp.kron(qubit_state, cavity_state))
    qubit_state = cp.array([1.0, 0.0], dtype=cp.complex128)
    cavity_state = cp.zeros(10, dtype=cp.complex128)
    cavity_state[5] = 1.0
    psi0 = cudaq.State.from_data(cp.kron(qubit_state, cavity_state))
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

    @pytest.mark.parametrize('integrator',
                             all_integrator_classes,
                             ids=all_integrator_classes)
    @pytest.mark.parametrize('input_state', [rho0, psi0])
    def test_simple(self, input_state, integrator):
        """
        test composite system
        """
        schedule = Schedule(self.steps, ["t"])
        evolution_result = evolve(
            self.hamiltonian,
            self.dimensions,
            schedule,
            input_state,
            observables=[operators.number(1),
                         operators.number(0)],
            collapse_operators=[np.sqrt(0.1) * self.a],
            store_intermediate_results=True,
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


class CavityModel:

    def __init__(self, n: int, delta: float, alpha0: float, kappa: float):
        self.n = n
        self.dimensions = {0: n}
        self.delta = delta
        self.alpha0 = alpha0
        self.kappa = kappa
        self.tlist = np.linspace(0.0, 0.3, 11)

    def hamiltonian(self):
        return self.delta * operators.number(0)

    def collapse_ops(self):
        return [np.sqrt(self.kappa) * operators.annihilate(0)]

    def initial_state(self):
        return coherent_state(self.n, self.alpha0)

    def observables(self):
        return [operators.position(0), operators.momentum(0)]

    def _alpha(self, t: float):
        return self.alpha0 * np.exp(-1j * self.delta * t - 0.5 * self.kappa * t)

    def expected_state(self, t: float):
        return coherent_dm(self.n, self._alpha(t)).ravel().get()

    def expect(self, t: float):
        alpha_t = self._alpha(t)
        exp_x = alpha_t.real
        exp_p = alpha_t.imag
        return [exp_x, exp_p]


class LossyQubitModel:

    def __init__(self, eps: float, omega: float, gamma: float):
        self.n = 2
        self.dimensions = {0: self.n}
        self.eps = eps
        self.omega = omega
        self.gamma = gamma
        self.tlist = np.linspace(0.0, 1.0, 11)

    def hamiltonian(self):
        return ScalarOperator(
            lambda t: self.eps * np.cos(self.omega * t)) * pauli.x(0)

    def collapse_ops(self):
        return [np.sqrt(self.gamma) * pauli.x(0)]

    def initial_state(self):
        return cp.array([1.0, 0.0], dtype=cp.complex128)

    def observables(self):
        return [pauli.x(0), pauli.y(0), pauli.z(0)]

    def _theta(self, t: float) -> float:
        return 2 * self.eps / self.omega * np.sin(self.omega * t)

    def _eta(self, t: float) -> float:
        return np.exp(-2 * self.gamma * t)

    def expected_state(self, t: float):
        theta = self._theta(t)
        eta = self._eta(t)
        rho_00 = 0.5 * (1 + eta * np.cos(theta))
        rho_11 = 0.5 * (1 - eta * np.cos(theta))
        rho_01 = 0.5j * eta * np.sin(theta)
        rho_10 = -0.5j * eta * np.sin(theta)
        return np.array([[rho_00, rho_01], [rho_10, rho_11]],
                        dtype=np.complex128).ravel()

    def expect(self, t: float):
        theta = self._theta(t)
        eta = self._eta(t)
        exp_x = 0
        exp_y = -eta * np.sin(theta)
        exp_z = eta * np.cos(theta)
        return np.array([exp_x, exp_y, exp_z]).real


all_models = [
    LossyQubitModel(eps=3.0, omega=10.0, gamma=1.0),
    CavityModel(n=8, delta=2 * np.pi, alpha0=0.5, kappa=2 * np.pi)
]

def sync_run(*args, **kwargs):
    return evolve(*args, **kwargs)

def async_run(*args, **kwargs):
    return evolve_async(*args, **kwargs).get()

@pytest.mark.parametrize('integrator', all_integrator_classes)
@pytest.mark.parametrize('model', all_models)
@pytest.mark.parametrize('runner', [sync_run, async_run])
def test_analytical_models(integrator, model, runner):
    rho0 = cudaq.State.from_data(model.initial_state())
    schedule = Schedule(model.tlist, ["t"])
    evolution_result = runner(model.hamiltonian(),
                              model.dimensions,
                              schedule,
                              rho0,
                              observables=model.observables(),
                              collapse_operators=model.collapse_ops(),
                              store_intermediate_results=True,
                              integrator=integrator())
    # Check expectation values
    for i, exp_vals in enumerate(evolution_result.expectation_values()):
        np.testing.assert_allclose([res.expectation() for res in exp_vals],
                                   model.expect(model.tlist[i]),
                                   atol=1e-3)
    # Check intermediate states
    for i, state in enumerate(evolution_result.intermediate_states()):
        np.testing.assert_allclose(state,
                                   model.expected_state(model.tlist[i]),
                                   atol=1e-3)


@pytest.mark.parametrize('integrator', all_integrator_classes)
def test_squeezing(integrator):

    def n_thermal(w: float, w_th: float):
        """
        Return the number of average photons in thermal equilibrium for a
            an oscillator with the given frequency and temperature.
        """
        if (w_th > 0) and np.exp(w / w_th) != 1.0:
            return 1.0 / (np.exp(w / w_th) - 1.0)
        else:
            return 0.0

    # Problem parameters
    w0 = 1.0 * 2 * np.pi
    gamma0 = 0.05
    # the temperature of the environment
    w_th = 0.0 * 2 * np.pi
    # the number of average excitations in the environment mode w0 at temperature w_th
    Nth = n_thermal(w0, w_th)
    # squeezing parameter for the environment
    r = 1.0
    theta = 0.1 * np.pi
    N = Nth * (np.cosh(r)**2 + np.sinh(r)**2) + np.sinh(r)**2
    sz_ss_analytical = -1 / (2 * N + 1)
    # System Hamiltonian
    hamiltonian = -0.5 * w0 * pauli.z(0)
    # Collapse operators
    c_ops = [
        np.sqrt(gamma0) * (pauli.minus(0) * np.cosh(r) +
                           pauli.plus(0) * np.sinh(r) * np.exp(1j * theta))
    ]

    # System dimension
    dimensions = {0: 2}
    # Start in an arbitrary superposition state
    psi0_ = cp.array([2j, 1.0], dtype=cp.complex128)
    psi0_ = psi0_ / cp.linalg.norm(psi0_)
    psi0 = cudaq.State.from_data(psi0_)
    # Simulation time points
    steps = np.linspace(0, 50, 1001)
    schedule = Schedule(steps, ["t"])

    # Run the simulation
    evolution_result = evolve(hamiltonian,
                              dimensions,
                              schedule,
                              psi0,
                              observables=[pauli.x(0),
                                           pauli.y(0),
                                           pauli.z(0)],
                              collapse_operators=c_ops,
                              store_intermediate_results=True,
                              integrator=integrator())

    exp_val_x = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    exp_val_y = [
        exp_vals[1].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    exp_val_z = [
        exp_vals[2].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    np.testing.assert_allclose(exp_val_z[-1], sz_ss_analytical, atol=1e-3)


@pytest.mark.parametrize('integrator', all_integrator_classes)
def test_cat_state(integrator):
    # Number of Fock levels
    N = 15
    # Kerr-nonlinearity
    chi = 1 * 2 * np.pi

    dimensions = {0: N}

    a = operators.annihilate(0)
    a_dag = operators.create(0)

    # Defining the Hamiltonian for the system (non-linear Kerr effect)
    hamiltonian = 0.5 * chi * a_dag * a_dag * a * a

    # we start with a coherent state with alpha=2.0
    # This will evolve into a cat state.
    rho0 = cudaq.State.from_data(coherent_state(N, 2.0))

    # Choose the end time at which the state evolves to the exact cat state.
    steps = np.linspace(0, 0.5 * chi / (2 * np.pi), 51)
    schedule = Schedule(steps, ["t"])

    # Run the simulation: observe the photon count, position and momentum.
    evolution_result = evolve(hamiltonian,
                              dimensions,
                              schedule,
                              rho0,
                              observables=[],
                              collapse_operators=[],
                              store_intermediate_results=False,
                              integrator=integrator())

    # The expected cat state: superposition of |alpla> and |-alpha> coherent states.
    expected_state = np.exp(1j * np.pi / 4) * coherent_state(N, -2.0j) + np.exp(
        -1j * np.pi / 4) * coherent_state(N, 2.0j)
    expected_state = expected_state / cp.linalg.norm(expected_state)
    final_state = evolution_result.final_state()
    overlap = final_state.overlap(expected_state)
    np.testing.assert_allclose(overlap, 1.0, atol=1e-3)


@pytest.mark.parametrize('integrator', all_integrator_classes)
def test_floquet_steady_state(integrator):
    # two-level system coupled with the external heat bath (fixed temperature)
    # (time-dependent Hamiltonian and multiple collapse operators)
    delta = (2 * np.pi) * 0.3
    eps_0 = (2 * np.pi) * 1.0
    A = (2 * np.pi) * 0.05
    w = (2 * np.pi) * 1.0
    kappa_1 = 0.15
    kappa_2 = 0.05
    sx = pauli.x(0)
    sz = pauli.z(0)
    sm = operators.annihilate(0)
    sm_dag = operators.create(0)
    dimensions = {0: 2}

    # Hamiltonian
    hamiltonian = -delta / 2.0 * sx - eps_0 / 2.0 * sz + A / 2.0 * ScalarOperator(
        lambda t: np.sin(w * t)) * sz

    psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

    # Thermal population
    n_th = 0.5

    # Collapse operators
    c_op_list = [
        np.sqrt(kappa_1 * (1 + n_th)) * sm,
        np.sqrt(kappa_1 * n_th) * sm_dag,
        np.sqrt(kappa_2) * sz
    ]

    # Schedule of time steps.
    steps = np.linspace(0, 50, 500)
    schedule = Schedule(steps, ["t"])

    # Run the simulation.
    # First, we run the simulation without any collapse operators (no decoherence).
    evolution_result = evolve(hamiltonian,
                              dimensions,
                              schedule,
                              psi0,
                              observables=[operators.number(0)],
                              collapse_operators=c_op_list,
                              store_intermediate_results=True,
                              integrator=integrator())

    expected_steady_state = 0.27
    n = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    # should converge to the steady-state after half of the duration
    for i in range(len(n) // 2, len(n)):
        # Note: the final population is still oscillating.
        np.testing.assert_allclose(n[i], expected_steady_state, atol=0.02)


@pytest.mark.parametrize('integrator', all_integrator_classes)
def test_landau_zener(integrator):
    # Define some shorthand operators
    sx = pauli.x(0)
    sz = pauli.z(0)
    sm = operators.annihilate(0)
    sm_dag = operators.create(0)

    dimensions = {0: 2}

    # System parameters
    gamma1 = 0.0001  # relaxation rate
    gamma2 = 0.005  # dephasing  rate
    delta = 0.5 * 2 * np.pi  # qubit pauli_x coefficient
    eps0 = 0.0 * 2 * np.pi  # qubit pauli_z coefficient
    A = 2.0 * 2 * np.pi  # time-dependent sweep rate

    # Hamiltonian
    hamiltonian = -delta / 2.0 * sx - eps0 / 2.0 * sz - A / 2.0 * ScalarOperator(
        lambda t: t) * sz

    # collapse operators: relaxation and dephasing
    c_op_list = [np.sqrt(gamma1) * sm, np.sqrt(gamma2) * sz]

    # Initial state of the system (ground state)
    psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

    # Schedule of time steps.
    steps = np.linspace(-20.0, 20.0, 5000)
    schedule = Schedule(steps, ["t"])

    # Run the simulation.
    evolution_result = evolve(hamiltonian,
                              dimensions,
                              schedule,
                              psi0,
                              observables=[operators.number(0)],
                              collapse_operators=c_op_list,
                              store_intermediate_results=True,
                              integrator=integrator())

    prob1 = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    analytical_result = 1 - np.exp(-np.pi * delta**2 / (2 * A))
    for idx, time in enumerate(steps):
        if time > 10.0:
            np.testing.assert_allclose(prob1[idx], analytical_result, atol=0.05)


@pytest.mark.parametrize('integrator', all_integrator_classes)
def test_cross_resonance(integrator):
    # Device parameters
    # Detuning between two qubits
    delta = 100 * 2 * np.pi
    # Static coupling between qubits
    J = 7 * 2 * np.pi
    # spurious electromagnetic crosstalk
    m2 = 0.2
    # Drive strength
    Omega = 20 * 2 * np.pi

    # Qubit Hamiltonian
    hamiltonian = delta / 2 * pauli.z(0) + J * (
        pauli.minus(1) * pauli.plus(0) + pauli.plus(1) *
        pauli.minus(0)) + Omega * pauli.x(0) + m2 * Omega * pauli.x(1)

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
        cp.array([0.0, 0.0, 1.0, 0.0], dtype=cp.complex128))

    # Schedule of time steps.
    steps = np.linspace(0.0, 1.0, 1001)
    schedule = Schedule(steps, ["t"])

    # Run the simulation.
    # Control bit = 0
    evolution_result_00 = evolve(hamiltonian,
                                 dimensions,
                                 schedule,
                                 psi_00,
                                 observables=[
                                     pauli.x(0),
                                     pauli.y(0),
                                     pauli.z(0),
                                     pauli.x(1),
                                     pauli.y(1),
                                     pauli.z(1)
                                 ],
                                 collapse_operators=[],
                                 store_intermediate_results=True,
                                 integrator=integrator())

    # Control bit = 1
    evolution_result_10 = evolve(hamiltonian,
                                 dimensions,
                                 schedule,
                                 psi_10,
                                 observables=[
                                     pauli.x(0),
                                     pauli.y(0),
                                     pauli.z(0),
                                     pauli.x(1),
                                     pauli.y(1),
                                     pauli.z(1)
                                 ],
                                 collapse_operators=[],
                                 store_intermediate_results=True,
                                 integrator=integrator())

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
