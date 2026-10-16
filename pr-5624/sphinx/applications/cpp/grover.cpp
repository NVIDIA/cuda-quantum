// Compile and run with:
// ```
// nvq++ grover.cpp -o grover.x && ./grover.x
// ```

#include <cmath>
#include <cudaq.h>

__qpu__ void reflect_about_uniform(cudaq::qvector<> &qs) {
  auto ctrlQubits = qs.front(qs.size() - 1);
  auto &lastQubit = qs.back();

  // Compute (U) Action (V) produces
  // U V U::Adjoint
  cudaq::compute_action(
      [&]() {
        h(qs);
        x(qs);
      },
      [&]() { z<cudaq::ctrl>(ctrlQubits, lastQubit); });
}

struct run_grover {
  template <typename CallableKernel>
  __qpu__ auto operator()(const int n_qubits, const long target_state,
                          CallableKernel &&oracle) {
    int n_iterations = round(0.25 * M_PI * sqrt(2 ^ n_qubits));

    cudaq::qvector qs(n_qubits);
    h(qs);
    for (int i = 0; i < n_iterations; i++) {
      oracle(target_state, qs);
      reflect_about_uniform(qs);
    }
    mz(qs);
  }
};

struct oracle {
  void operator()(const long target_state, cudaq::qvector<> &qs) __qpu__ {
    cudaq::compute_action(
        [&]() {
          std::size_t n = qs.size();
          for (std::size_t i = 1; i <= n; ++i) {
            if (!((1 << (n - i)) & target_state))
              x(qs[i - 1]);
          }
        },
        [&]() {
          auto ctrlQubits = qs.front(qs.size() - 1);
          z<cudaq::ctrl>(ctrlQubits, qs.back());
        });
  }
};

int main(int argc, char *argv[]) {
  auto secret = 1 < argc ? strtol(argv[1], NULL, 2) : 0b1011;
  auto counts = cudaq::sample(run_grover{}, 4, secret, oracle{});
  printf("Found string %s\n", counts.most_probable().c_str());
}
