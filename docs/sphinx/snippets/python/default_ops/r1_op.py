import cudaq
import math

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # R1(λ) = | 1     0    |
    #         | 0  exp(iλ) |
    r1(math.pi, qubit)

if __name__ == "__main__":
    main()