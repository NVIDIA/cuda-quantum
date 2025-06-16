import cudaq

def main():
    qubit = cudaq.qubit()
    # [Begin Y Op]
    # Apply the unitary transformation
    # Y = | 0  -i |
    #     | i   0 |
    y(qubit)
    # [End Y Op]
if __name__ == "__main__":
    main()