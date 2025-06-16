import cudaq
import math

def main():
    qubit = cudaq.qubit()
    # [Begin R1 Op]
    # Apply the unitary transformation
    # R1(λ) = | 1     0    |
    #         | 0  exp(iλ) |
    r1(math.pi, qubit)
    # [End R1 Op]
if __name__ == "__main__":
    main()