// Compile with:
// ```
// nvq++ cutensornet_backends.cpp -o dyn.x --target tensornet
// ```
//
// This example is meant to demonstrate the `cuTensorNet`
// multi-node/multi-GPU backend.
// On a multi-GPU system, we can enable distributed parallelization across MPI
// processes by initializing MPI (see code) and launch the compiled executable
// with MPI.
// ```
// mpirun -np <N> ./dyn.x
// ```

#include <cudaq.h>

// Define a quantum kernel with a runtime parameter
struct ghz {
  auto operator()(const int N) __qpu__ {

    // Dynamically sized vector of qubits
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  // Initialize MPI to enable `cuTensorNet` distributed parallelization (see
  // Simulation Backends/Tensor Network Simulators section of the documentation
  // for more info)
  // ```
  // cudaq::mpi::initialize();
  // if (cudaq::mpi::rank() == 0)
  //   printf("Number of MPI processes: %d\n", cudaq::mpi::num_ranks());
  // ```

  auto counts = cudaq::sample(100, ghz{}, 28);
  counts.dump();

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.data(), count);
  }

  // Finalize MPI if initialized
  // ```
  // cudaq::mpi::finalize();
  // ```
  return 0;
}
