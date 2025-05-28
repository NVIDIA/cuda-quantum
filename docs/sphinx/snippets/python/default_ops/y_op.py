import cudaq

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # Y = | 0  -i |
    #     | i   0 |
    y(qubit)

if __name__ == "__main__":
    main()