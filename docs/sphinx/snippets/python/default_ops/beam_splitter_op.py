import cudaq

def main():
    q = [cudaq.qudit(3) for _ in range(2)]
    # [Begin Beam Splitter Op]
    beam_splitter(q[0], q[1], 0.34)
    # [End Beam Splitter Op]
if __name__ == "__main__":
    main()