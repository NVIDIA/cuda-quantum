import cudaq

def main():
    qubit = cudaq.qubit()
# [Begin X Op]
    # Apply the unitary transformation
    # X = | 0  1 |
    #     | 1  0 |
    x(qubit)
# [Begin X Op]
if __name__ == "__main__":
    main()