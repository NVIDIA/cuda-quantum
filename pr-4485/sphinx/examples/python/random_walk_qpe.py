# Run with:
# ```
# python3 random_walk_qpe.py
# ```

import cudaq

# Here we demonstrate an algorithm expressed as a CUDA-Q kernel that
# incorporates non-trivial control flow and conditional quantum instruction
# invocation.


# Define the random walk phase estimation kernel
# Ref: https://arxiv.org/pdf/2208.04526
@cudaq.kernel
def rwpe_kernel(n_iter: int, mu: float, sigma: float) -> float:
    iteration = 0

    # Allocate the qubits
    number_of_qubits = 2
    qubits = cudaq.qvector(number_of_qubits)

    # Alias them
    aux = qubits[0]
    target = qubits[number_of_qubits - 1]

    x(target)

    while iteration < n_iter:
        h(aux)
        rz(1.0 - (mu / sigma), aux)
        rz(.25 / sigma, target)
        x.ctrl(aux, target)
        rz(-.25 / sigma, target)
        x.ctrl(aux, target)
        h(aux)
        if mz(aux):
            mu = mu + sigma * .6065
        else:
            mu = mu - sigma * .6065

        sigma *= .7951
        iteration += 1
        reset(aux)

    return 2.0 * mu


# Main function to execute the kernel
def main():
    cudaq.set_random_seed(123)
    shots = 100
    n_iterations = 24
    mu = 0.7951
    sigma = 0.6065

    phases = cudaq.run(rwpe_kernel, n_iterations, mu, sigma, shots_count=shots)
    for phase in phases:
        print(f"Phase = {phase:.6f}")


if __name__ == "__main__":
    main()
