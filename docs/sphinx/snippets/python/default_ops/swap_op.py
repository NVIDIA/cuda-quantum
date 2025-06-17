import cudaq

def main():
    qubit_1, qubit_2 = cudaq.qubit(), cudaq.qubit()
    # [Begin Swap Op]
    # Apply the unitary transformation
    # Swap = | 1 0 0 0 |
    #        | 0 0 1 0 |
    #        | 0 1 0 0 |
    #        | 0 0 0 1 |
    swap(qubit_1, qubit_2)
    # [End Swap Op]
if __name__ == "__main__":
    main()