// Compile and run with:
// ```
// nvq++ grover.cpp -o grover.x && ./grover.x
// ```

#include <cudaq.h>

__qpu__ void reflect_about_uniform(cudaq::qspan<> q) {
  auto ctrlQubits = q.front(q.size() - 1);
  auto &lastQubit = q.back();

  h(q);
  x(q);
  z<cudaq::ctrl>(ctrlQubits, lastQubit);
  x(q);
  h(q);
}

struct run_grover {
  template <typename CallableKernel>
  __qpu__ auto operator()(const int n_qubits, const int n_iterations,
                          CallableKernel &&oracle) {
    cudaq::qreg q(n_qubits);
    h(q);
    for (int i = 0; i < n_iterations; i++) {
      oracle(q);
      reflect_about_uniform(q);
    }
    mz(q);
  }
};

struct oracle {
  void operator()(cudaq::qreg<> &q) __qpu__ {
    z<cudaq::ctrl>(q[0], q[2]);
    z<cudaq::ctrl>(q[1], q[2]);
  }
};

int main() {
  auto counts = cudaq::sample(run_grover{}, 3, 1, oracle{});
  counts.dump();
}
