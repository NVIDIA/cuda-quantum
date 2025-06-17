import cudaq

def main():
    qubit = cudaq.qubit()
    # [Begin MX Op]
    mx(qubit)
    # [End MX Op]

if __name__ == "__main__":
    main()