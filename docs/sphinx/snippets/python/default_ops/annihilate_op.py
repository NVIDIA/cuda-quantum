import cudaq

def main():
    q = cudaq.qudit(3)
    annihilate(q)

if __name__ == "__main__":
    main()