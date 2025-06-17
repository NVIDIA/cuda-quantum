import cudaq

def main():
    qubit = cudaq.qubit()
    # [Begin T Op]
    # Apply the unitary transformation
    # T = | 1      0     |
    #     | 0  exp(iπ/4) |
    t(qubit)
    # [End T Op]
if __name__ == "__main__":
    main()