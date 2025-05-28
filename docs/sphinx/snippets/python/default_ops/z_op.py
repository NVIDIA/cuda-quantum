import cudaq

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # Z = | 1   0 |
    #     | 0  -1 |
    z(qubit)

if __name__ == "__main__":
    main()