import cudaq
import math

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # Rx(θ) = |  cos(θ/2)  -`isin`(θ/2) |
    #         | -`isin`(θ/2)  cos(θ/2)  |
    rx(math.pi, qubit)

if __name__ == "__main__":
    main()