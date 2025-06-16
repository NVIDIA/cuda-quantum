import cudaq

def main():
    qubit = cudaq.qubit()
    # [Begin H Op]
    # Apply the unitary transformation
    # H = (1 / `sqrt`(2)) * | 1   1 |
    #                     | 1  -1 |
    h(qubit)
    # [End H Op]
if __name__ == "__main__":
    main()