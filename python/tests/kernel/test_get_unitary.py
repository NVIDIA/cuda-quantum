import numpy as np
import cudaq


def test_single_hadamard():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        h(q)

    U = cudaq.get_unitary(k)
    expected = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]],
                                           dtype=np.complex128)
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_two_x_gates_one_qubit():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        x(q)
        x(q)

    U = cudaq.get_unitary(k)
    expected = np.eye(2, dtype=np.complex128)  # X*X = I
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_two_hadamards_two_qubits():

    @cudaq.kernel
    def k():
        q = cudaq.qvector(2)
        h(q[0])
        h(q[1])

    U = cudaq.get_unitary(k)
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    expected = np.kron(H, H)
    np.testing.assert_allclose(U, expected, atol=1e-12)

def test_rotation_h_hadamard_rotation():
    theta1, theta2 = np.pi/5, np.pi/7

    H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]],dtype=np.complex128)
    X = np.array([[0,1],[1,0]],dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    def Rx(theta):
        return np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*X

    @cudaq.kernel
    def k(a: float, b: float):
        q = cudaq.qubit()
        rx(a, q)        
        h(q)
        rx(b, q)

    # pass the two angles into get_unitary:
    U = cudaq.get_unitary(k, theta1, theta2)
    expected = Rx(theta2) @ H @ Rx(theta1)
    np.testing.assert_allclose(U, expected, atol=1e-12)



def test_two_sparse_qubits():
    # acts only on qubit 0 with H and qubit 2 with X; qubit 1 is identity
    H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    @cudaq.kernel
    def k():
        q = cudaq.qvector(3)
        h(q[0])
        x(q[2])

    U = cudaq.get_unitary(k)
    # U = H ⊗ I ⊗ X
    expected = np.kron(np.kron(H, I2), X)
    np.testing.assert_allclose(U, expected, atol=1e-12)