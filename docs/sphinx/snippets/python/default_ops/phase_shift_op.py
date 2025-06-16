import cudaq

def main():
    q = cudaq.qudit(4)
    # [Begin Phase Shift Op]
    phase_shift(q, 0.17)
    # [End Phase Shift Op]

if __name__ == "__main__":
    main()
    