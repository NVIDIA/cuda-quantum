# [Begin Spin]
import cudaq
from cudaq import spin

operator = 2 * spin.x(0) * spin.y(1) * spin.x(2) - 3 * spin.z(0) * spin.z(
    1) * spin.y(2)
# [End Spin]

# [Begin Pauli]
words = ['XYZ', 'IXX']
coefficients = [0.432, 0.324]


@cudaq.kernel
def kernel(coefficients: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(3)

    for i in range(len(coefficients)):
        exp_pauli(coefficients[i], q, words[i])


# [End Pauli]
