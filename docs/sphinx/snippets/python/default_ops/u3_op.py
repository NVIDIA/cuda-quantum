import cudaq
import numpy as np

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # U3(θ,φ,λ) = | cos(θ/2)            -exp(iλ) * sin(θ/2)       |
    #             | exp(iφ) * sin(θ/2)   exp(i(λ + φ)) * cos(θ/2) |
    u3(np.pi, np.pi, np.pi / 2, qubit)

if __name__ == "__main__":
    main()