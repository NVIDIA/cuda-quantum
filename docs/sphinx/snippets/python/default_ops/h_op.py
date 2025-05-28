import cudaq

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # H = (1 / sqrt(2)) * | 1   1 |
    #                     | 1  -1 |
    h(qubit)

if __name__ == "__main__":
    main()