import numpy as np
import pytest

import cudaq
import cudaq.contrib
from cudaq import Schedule, spin

scipy_linalg = pytest.importorskip("scipy.linalg")

if cudaq.num_available_gpus() == 0:
    pytest.skip("cudaq.contrib.propagator requires the dynamics backend",
                allow_module_level=True)


@pytest.fixture(autouse=True)
def dynamics_target():
    cudaq.set_target("dynamics")
    yield
    cudaq.reset_target()


def test_propagator_constant_z_hamiltonian():
    omega = 0.3
    t_final = 1.2

    hamiltonian = 0.5 * omega * spin.z(0)
    schedule = Schedule(np.linspace(0.0, t_final, 21), ["t"])

    computed = cudaq.contrib.propagator(hamiltonian, {0: 2}, schedule)

    z_matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    expected = scipy_linalg.expm(-1j * 0.5 * omega * z_matrix * t_final)

    np.testing.assert_allclose(computed, expected, atol=1e-4)
    np.testing.assert_allclose(computed.conj().T @ computed,
                               np.eye(2),
                               atol=1e-4)


def test_propagator_constant_x_hamiltonian():
    t_final = 0.4

    hamiltonian = spin.x(0)
    schedule = Schedule(np.linspace(0.0, t_final, 21), ["t"])

    computed = cudaq.contrib.propagator(hamiltonian, {0: 2}, schedule)

    x_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    expected = scipy_linalg.expm(-1j * x_matrix * t_final)

    np.testing.assert_allclose(computed, expected, atol=1e-4)
