import cudaq

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # X = | 0  1 |
    #     | 1  0 |
    x(qubit)

if __name__ == "__main__":
    main()