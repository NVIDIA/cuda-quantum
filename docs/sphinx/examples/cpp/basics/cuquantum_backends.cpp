// Compile and run with:
// ```
// nvq++ cuquantum_backends.cpp -o dyn.x --target nvidia && ./dyn.x
// ```

// This example is meant to demonstrate the cuQuantum
// GPU-accelerated backends and their ability to easily handle
// a larger number of qubits compared the CPU-only backend.

// Without the `--target nvidia` flag, this seems to hang, i.e.
// it takes a long time for the CPU-only backend to handle
// this number of qubits.

#include <cudaq.h>

// Define a quantum kernel with a runtime parameter
struct ghz {
  auto operator()(const int N) __qpu__ {

    // Dynamic, vector-like `qreg`
    cudaq::qreg q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(ghz{}, 30);
  counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.data(), count);
  }

  return 0;
}
