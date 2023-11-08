// Compile and run with:
// ```
// nvq++ cutensornet_backends.cpp -o dyn.x --target tensornet
// mpirun -np <N> ./dyn.x
// ```

// This example is meant to demonstrate the `cuTensorNet`
// multi-node/multi-GPU backend.

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
  // Initialize MPI
  cudaq::mpi::initialize();
  if (cudaq::mpi::rank() == 0)
    printf("Number of MPI processes: %d\n", cudaq::mpi::num_ranks());
  auto counts = cudaq::sample(100, ghz{}, 28);
  if (cudaq::mpi::rank() == 0) {
    counts.dump();

    // Fine grain access to the bits and counts
    for (auto &[bits, count] : counts) {
      printf("Observed: %s, %lu\n", bits.data(), count);
    }
  }
  cudaq::mpi::finalize();
  return 0;
}
