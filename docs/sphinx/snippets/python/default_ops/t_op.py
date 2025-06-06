import cudaq

def main():
    qubit = cudaq.qubit()

    # Apply the unitary transformation
    # T = | 1      0     |
    #     | 0  exp(iπ/4) |
    t(qubit)

if __name__ == "__main__":
    main()