import cudaq

def main():
    q = [cudaq.qudit(3) for _ in range(2)]
    beam_splitter(q[0], q[1], 0.34)

if __name__ == "__main__":
    main()