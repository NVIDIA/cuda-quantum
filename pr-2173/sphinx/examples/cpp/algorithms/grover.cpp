// Compile and run with:
// ```
// nvq++ grover.cpp -o grover.x && ./grover.x
// ```

#include <cmath>
#include <cudaq.h>
#include <numbers>

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
  __qpu__ auto operator()(const int n_qubits, CallableKernel &&oracle) {
    int n_iterations = round(0.25 * std::numbers::pi * sqrt(2 ^ n_qubits));

    cudaq::qvector qs(n_qubits);
    h(qs);
    for (int i = 0; i < n_iterations; i++) {
      oracle(qs);
      reflect_about_uniform(qs);
    }
    mz(qs);
  }
};

struct oracle {
  const long target_state;

  void operator()(cudaq::qvector<> &qs) __qpu__ {
    cudaq::compute_action(
        [&]() {
          for (int i = 1; i <= qs.size(); ++i) {
            auto target_bit_set = (1 << (qs.size() - i)) & target_state;
            if (!target_bit_set)
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
  oracle compute_oracle{.target_state = secret};
  auto counts = cudaq::sample(run_grover{}, 4, compute_oracle);
  printf("Found string %s\n", counts.most_probable().c_str());
}
