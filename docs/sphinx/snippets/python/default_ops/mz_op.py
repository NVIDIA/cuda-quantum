import cudaq

def main():
    qubit = cudaq.qubit()
    # [Begin MZ Op]
    mz(qubit)
    # [End MZ Op]

if __name__ == "__main__":
    main()