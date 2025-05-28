import cudaq

def main():
    qumodes = [cudaq.qudit(3) for _ in range(2)]
    mz(qumodes)

if __name__ == "__main__":
    main()