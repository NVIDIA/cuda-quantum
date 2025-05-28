import cudaq

def main():
    q = cudaq.qudit(4)
    phase_shift(q, 0.17)

if __name__ == "__main__":
    main()
    