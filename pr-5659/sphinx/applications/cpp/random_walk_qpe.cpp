// Compile and run with:
// ```
// nvq++ random_walk_qpe.cpp -o qpe.x && ./qpe.x
// ```

#include <cudaq.h>

// Here we demonstrate an algorithm expressed as a CUDA-Q kernel that
// incorporates non-trivial control flow and conditional quantum instruction
// invocation.

// Ref: https://arxiv.org/pdf/2208.04526
struct rwpe {
  double operator()(const int n_iter, double mu, double sigma) __qpu__ {
    int iteration = 0;

    // Allocate the qubits
    cudaq::qvector q(2);

    // Alias them
    auto &aux = q.front();
    auto &target = q.back();

    x(q[1]);

    while (iteration < n_iter) {
      h(aux);
      rz(1.0 - (mu / sigma), aux);
      rz(.25 / sigma, target);
      x<cudaq::ctrl>(aux, target);
      rz(-.25 / sigma, target);
      x<cudaq::ctrl>(aux, target);
      h(aux);
      if (mz(aux)) {
        mu += sigma * .6065;
      } else {
        mu -= sigma * .6065;
      }

      sigma *= .7951;
      iteration += 1;
      // Reset qubit for reuse
      reset(aux);
    }

    return 2. * mu;
  }
};

int main() {

  cudaq::set_random_seed(123);
  const std::size_t shots = 100;
  constexpr int numIters = 24;
  constexpr double mu = 0.7951;
  constexpr double sigma = 0.6065;
  auto phases = cudaq::run(shots, rwpe{}, numIters, mu, sigma);

  for (auto &phase : phases)
    printf("Phase = %lf\n", phase);
}
