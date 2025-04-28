# [Begin Spin]
import cudaq
from cudaq import spin
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
