// Compile and run with:
// ```
// nvq++ --target quantum_machines quantum_machines.cpp -o out.x && ./out.x
// ```
// This will submit the job to the Quantum Machines OPX available in the address
// provider by `--quantum-machines-url`. By default, the action runs a on a mock
// executor. To execute or a real QPU please note the executor name by
// `--quantum-machines-executor`.
// ```
// nvq++ --target quantum_machines --quantum-machines-url
// "https://iqcc.qoperator.qm.co" \
//  --quantum-machines-executor iqcc quantum_machines.cpp -o out.x
// ./out.x
// ```
// Assumes a valid set of credentials have been set prior to execution.

#include "math.h"
#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on Quantum Machines OPX.
struct all_h {
  // Maximally entangled state between 5 qubits.
  auto operator()() __qpu__ {
    cudaq::qvector q(5);
    for (int i = 0; i < 4; i++) {
      h(q[i]);
    }
    s(q[0]);
    r1(M_PI / 2, q[1]);
    auto result = mz(q);
  }
};

int main() {
  // Submit asynchronously (e.g., continue executing code in the file until
  // the job has been returned).
  auto future = cudaq::sample_async(all_h{});
  // ... classical code to execute in the meantime ...

  // Get the results of the read in future.
  auto async_counts = future.get();
  async_counts.dump();

  // OR: Submit synchronously (e.g., wait for the job
  // result to be returned before proceeding).
  auto counts = cudaq::sample(all_h{});
  counts.dump();
}
