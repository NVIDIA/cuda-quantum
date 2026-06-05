import numpy as np
import pytest

import cudaq
import cudaq.contrib
from cudaq import ScalarOperator, Schedule, spin

scipy_integrate = pytest.importorskip("scipy.integrate")
scipy_linalg = pytest.importorskip("scipy.linalg")

if cudaq.num_available_gpus() == 0:
    pytest.skip("cudaq.contrib.propagator requires the dynamics backend",
                allow_module_level=True)


@pytest.fixture(autouse=True)
def dynamics_target():
    cudaq.set_target("dynamics")
    yield
    cudaq.reset_target()


_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _solve_propagator_reference(hamiltonian_matrix, t_final):
    dimension = hamiltonian_matrix(0.0).shape[0]

    def rhs(t, flattened_u):
        u = flattened_u.reshape((dimension, dimension))
        du = -1j * hamiltonian_matrix(t) @ u
        return du.reshape(-1)

    initial = np.eye(dimension, dtype=np.complex128).reshape(-1)
    solution = scipy_integrate.solve_ivp(
        rhs,
        (0.0, t_final),
        initial,
        rtol=1e-10,
        atol=1e-12,
    )

    assert solution.success
    return solution.y[:, -1].reshape((dimension, dimension))


def test_propagator_constant_z_hamiltonian():
    omega = 0.3
    t_final = 1.2

    hamiltonian = 0.5 * omega * spin.z(0)
    schedule = Schedule(np.linspace(0.0, t_final, 21), ["t"])

    computed = cudaq.contrib.propagator(hamiltonian, {0: 2}, schedule)

    expected = scipy_linalg.expm(-1j * 0.5 * omega * _Z * t_final)

    np.testing.assert_allclose(computed, expected, atol=1e-4)
    np.testing.assert_allclose(computed.conj().T @ computed,
                               np.eye(2),
                               atol=1e-4)


def test_propagator_constant_x_hamiltonian():
    t_final = 0.4

    hamiltonian = spin.x(0)
    schedule = Schedule(np.linspace(0.0, t_final, 21), ["t"])

    computed = cudaq.contrib.propagator(hamiltonian, {0: 2}, schedule)
    expected = scipy_linalg.expm(-1j * _X * t_final)

    np.testing.assert_allclose(computed, expected, atol=1e-4)


def test_propagator_batched_hamiltonians():
    t_final = 0.7
    omegas = [0.2, 0.5]

    hamiltonians = [0.5 * omega * spin.z(0) for omega in omegas]
    schedule = Schedule(np.linspace(0.0, t_final, 21), ["t"])

    computed = cudaq.contrib.propagator(
        hamiltonians,
        {0: 2},
        schedule,
        max_batch_size=2,
    )

    assert len(computed) == len(omegas)

    for propagator, omega in zip(computed, omegas):
        expected = scipy_linalg.expm(-1j * 0.5 * omega * _Z * t_final)
        np.testing.assert_allclose(propagator, expected, atol=1e-4)


def test_propagator_intermediate_results():
    omega = 0.3
    steps = np.linspace(0.0, 1.2, 7)

    hamiltonian = 0.5 * omega * spin.z(0)
    schedule = Schedule(steps, ["t"])

    computed = cudaq.contrib.propagator(
        hamiltonian,
        {0: 2},
        schedule,
        store_intermediate_results=True,
    )

    assert len(computed) == len(steps)

    for propagator, t in zip(computed, steps):
        expected = scipy_linalg.expm(-1j * 0.5 * omega * _Z * t)
        np.testing.assert_allclose(propagator, expected, atol=1e-4)


def test_propagator_commuting_time_dependent_hamiltonian():
    omega0 = 0.2
    omega1 = 0.4
    t_final = 1.1

    hamiltonian = (0.5 * omega0 * spin.z(0) +
                   0.5 * omega1 * ScalarOperator(lambda t: t) * spin.z(0))
    schedule = Schedule(np.linspace(0.0, t_final, 31), ["t"])

    computed = cudaq.contrib.propagator(hamiltonian, {0: 2}, schedule)

    phase_integral = omega0 * t_final + 0.5 * omega1 * t_final**2
    expected = scipy_linalg.expm(-1j * 0.5 * phase_integral * _Z)

    np.testing.assert_allclose(computed, expected, atol=1e-4)


def test_propagator_noncommuting_time_dependent_hamiltonian():
    ax = 0.2
    bz = 0.35
    t_final = 0.8

    hamiltonian = (ax * ScalarOperator(lambda t: t) * spin.x(0) +
                   bz * ScalarOperator(lambda t: 1.0 - t) * spin.z(0))
    schedule = Schedule(np.linspace(0.0, t_final, 41), ["t"])

    computed = cudaq.contrib.propagator(hamiltonian, {0: 2}, schedule)

    def hamiltonian_matrix(t):
        return ax * t * _X + bz * (1.0 - t) * _Z

    expected = _solve_propagator_reference(hamiltonian_matrix, t_final)

    np.testing.assert_allclose(computed, expected, atol=1e-4)


def _amplitude_damping_liouvillian(gamma):
    sigma_minus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_plus = sigma_minus.conj().T
    identity = np.eye(2, dtype=np.complex128)

    collapse_product = sigma_plus @ sigma_minus

    return (
        gamma * np.kron(sigma_minus.conj(), sigma_minus)
        - 0.5 * gamma * np.kron(identity, collapse_product)
        - 0.5 * gamma * np.kron(collapse_product.T, identity)
    )


def test_open_system_propagator_amplitude_damping():
    gamma = 0.2
    t_final = 0.7

    hamiltonian = 0.0 * spin.z(0)
    collapse_operators = [np.sqrt(gamma) * spin.minus(0)]
    schedule = Schedule(np.linspace(0.0, t_final, 21), ["t"])

    computed = cudaq.contrib.propagator(
        hamiltonian,
        {0: 2},
        schedule,
        collapse_operators=collapse_operators,
    )

    expected = scipy_linalg.expm(_amplitude_damping_liouvillian(gamma) *
                                 t_final)

    np.testing.assert_allclose(computed, expected, atol=1e-4)


def test_open_system_propagator_shape():
    gamma = 0.1
    t_final = 0.3

    hamiltonian = 0.0 * spin.z(0)
    collapse_operators = [np.sqrt(gamma) * spin.minus(0)]
    schedule = Schedule(np.linspace(0.0, t_final, 11), ["t"])

    computed = cudaq.contrib.propagator(
        hamiltonian,
        {0: 2},
        schedule,
        collapse_operators=collapse_operators,
    )

    assert computed.shape == (4, 4)
