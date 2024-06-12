// Compile and run with:
// ```
// nvq++ --target nvqc --nvqc-backend tensornet nvqc_state_amplitude.cpp -o
// out.x
// ./out.x
// ```
// Assumes a valid NVQC API key has been set in the `NVQC_API_KEY` environment
// variable. Please refer to the documentations for information about how to
// attain NVQC API key.

// In this example, we demonstrate amplitude accessor on tensor network state
// data. With a large number of qubits, tensor network allows us to compute
// amplitudes of individual basis states without the need to compute the full
// state vector.
#include "cudaq/algorithms/state.h"
#include <cudaq.h>
#include <iostream>

int main() {
  auto kernel = cudaq::make_kernel();
  const std::size_t NUM_QUBITS = 100;
  auto q = kernel.qalloc(NUM_QUBITS);
  kernel.h(q[0]);
  for (std::size_t qId = 0; qId < NUM_QUBITS - 1; ++qId)
    kernel.x<cudaq::ctrl>(q[qId], q[qId + 1]);
  auto state = cudaq::get_state(kernel);
  const std::vector<int> all0(NUM_QUBITS, 0);
  const std::vector<int> all1(NUM_QUBITS, 1);
  const auto amplitudes = state.amplitudes({all0, all1});
  std::cout << "Number of qubits = " << NUM_QUBITS << "\n";
  std::cout << "Amplitude(00..00) = " << amplitudes[0] << "\n";
  std::cout << "Amplitude(11..11) = " << amplitudes[1] << "\n";
}
