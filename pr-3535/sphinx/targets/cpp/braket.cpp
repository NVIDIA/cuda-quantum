// Compile and run with:
// ```
// nvq++ --target braket braket.cpp -o out.x && ./out.x
// ```
// This will submit the job to the Amazon Braket state vector simulator
// (default). Alternatively, users can choose any of the available devices by
// specifying its `ARN` with the `--braket-machine`, e.g.,
// ```
// nvq++ --target braket --braket-machine \
// "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet" braket.cpp -o out.x
// ./out.x
// ```
// Assumes a valid set of credentials have been set prior to execution.

#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on Amazon Braket.
struct ghz {
  // Maximally entangled state between 5 qubits.
  auto operator()() __qpu__ {
    cudaq::qvector q(5);
    h(q[0]);
    for (int i = 0; i < 4; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    auto result = mz(q);
  }
};

int main() {
  // Submit asynchronously (e.g., continue executing
  // code in the file until the job has been returned).
  auto future = cudaq::sample_async(ghz{});
  // ... classical code to execute in the meantime ...

  // Get the results of the read in future.
  auto async_counts = future.get();
  async_counts.dump();

  // OR: Submit synchronously (e.g., wait for the job
  // result to be returned before proceeding).
  auto counts = cudaq::sample(ghz{});
  counts.dump();
}
