// Compile and run with:
// ```
// nvq++ --target nvqc nvqc_sample.cpp -o out.x
// ./out.x
// ```
// Assumes a valid NVQC API key has been set in the `NVQC_API_KEY` environment
// variable. Please refer to the documentations for information about how to
// attain NVQC API key.

#include <cudaq.h>
#include <iostream>

// Define a simple quantum kernel to execute on NVQC.
struct ghz {
  // Maximally entangled state between 25 qubits.
  auto operator()() __qpu__ {
    constexpr int NUM_QUBITS = 25;
    cudaq::qvector q(NUM_QUBITS);
    h(q[0]);
    for (int i = 0; i < NUM_QUBITS - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    auto result = mz(q);
  }
};

int main() {
  // Submit to NVQC asynchronously (e.g., continue executing
  // code in the file until the job has been returned).
  auto async_counts_handle = cudaq::sample_async(ghz{});
  // ... classical code to execute in the meantime ...
  std::cout << "Waiting for NVQC result...\n";

  // Calling .get() on the handle to synchronize the result.
  auto async_counts = async_counts_handle.get();
  async_counts.dump();

  // OR: Submit to NVQC synchronously (e.g., wait for the job
  // result to be returned before proceeding).
  auto counts = cudaq::sample(ghz{});
  counts.dump();
}
