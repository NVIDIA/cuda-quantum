import cudaq

def main():
    qumodes = [cudaq.qudit(3) for _ in range(2)]
    # [Begin MZ Op]
    mz(qumodes)
    # [End MZ Op]

if __name__ == "__main__":
    main()