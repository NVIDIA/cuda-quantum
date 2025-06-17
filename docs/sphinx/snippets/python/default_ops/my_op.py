import cudaq

def main():
    qubit = cudaq.qubit()
    # [Begin My Op]
    my(qubit)
    # [End My Op]

if __name__ == "__main__":
    main()