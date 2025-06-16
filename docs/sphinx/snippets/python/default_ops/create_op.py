import cudaq

def main():
    q = cudaq.qudit(3)
    # [Begin Create Op]
    create(q)
    # [End Create Op]

if __name__ == "__main__":
    main()