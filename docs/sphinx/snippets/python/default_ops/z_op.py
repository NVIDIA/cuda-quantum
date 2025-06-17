import cudaq

def main():
    qubit = cudaq.qubit()
    # [Begin Z Op]
    # Apply the unitary transformation
    # Z = | 1   0 |
    #     | 0  -1 |
    z(qubit)
    # [End Z Op]
if __name__ == "__main__":
    main()