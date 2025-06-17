import cudaq
import math

def main():
    qubit = cudaq.qubit()
    # [Begin Ry Op]
    # Apply the unitary transformation
    # Ry(θ) = | cos(θ/2)  -sin(θ/2) |
    #         | sin(θ/2)   cos(θ/2) |
    ry(math.pi, qubit)
    # [End Ry Op]
if __name__ == "__main__":
    main()