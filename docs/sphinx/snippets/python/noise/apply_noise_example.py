import cudaq

@cudaq.kernel
def noise_example():
    q, r = cudaq.qubit(), cudaq.qubit()
    cudaq.apply_noise(cudaq.Depolarization2, 0.1, q, r)

def main():
    cudaq.sample(noise_example)

if __name__ == "__main__":
    main()