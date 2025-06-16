import cudaq

def main():
    qubit = cudaq.qubit()
    # [Begin S Op]
    # Apply the unitary transformation
    # S = | 1   0 |
    #     | 0   i |
    s(qubit)
    # [End S Op]
if __name__ == "__main__":
    main()