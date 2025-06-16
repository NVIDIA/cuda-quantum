import cudaq
import math

def main():
    qubit = cudaq.qubit()
    # [Begin Rz Op]
    # Apply the unitary transformation
    # Rz(λ) = | exp(-iλ/2)      0     |
    #         |     0       exp(iλ/2) |
    rz(math.pi, qubit)
    # [End Rz Op]
if __name__ == "__main__":
    main()