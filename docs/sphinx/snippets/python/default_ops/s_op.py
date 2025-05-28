import cudaq

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # S = | 1   0 |
    #     | 0   i |
    s(qubit)

if __name__ == "__main__":
    main()