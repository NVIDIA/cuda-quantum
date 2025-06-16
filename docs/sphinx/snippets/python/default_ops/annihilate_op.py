import cudaq

def main():
    q = cudaq.qudit(3)
    # [Begin Annihilate Op]
    annihilate(q)
    # [End Annihilate Op]

if __name__ == "__main__":
    main()