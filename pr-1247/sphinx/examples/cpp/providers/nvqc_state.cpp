// Compile and run with:
// ```
// nvq++ --target nvqc --nvqc-backend tensornet nvqc_state.cpp -o out.x
// ./out.x
// ```
// Assumes a valid NVQC API key and function ID have been stored in environment
// variables or `~/.nvqc_config` file. Alternatively, they can be set in the
// command line like below.
// ```
// nvq++ --target nvqc --nvqc-backend tensornet --nvqc-api-key <YOUR API KEY> \
// --nvqc-function-id <NVQC function Id> nvqc_state.cpp -o out.x
// ./out.x
// ```
// Please refer to the documentations for information about how to attain NVQC
// information.

#include "cudaq/algorithms/state.h"
#include <cudaq.h>
#include <iostream>

int main() {
  auto kernel = cudaq::make_kernel();
  const std::size_t NUM_QUBITS = 20;
  auto q = kernel.qalloc(NUM_QUBITS);
  kernel.h(q[0]);
  for (std::size_t qId = 0; qId < NUM_QUBITS - 1; ++qId)
    kernel.x<cudaq::ctrl>(q[qId], q[qId + 1]);
  auto state = cudaq::get_state(kernel);
  std::cout << "Amplitude(00..00) = " << state[0] << "\n";
  std::cout << "Amplitude(11..11) = " << state[(1ULL << NUM_QUBITS) - 1]
            << "\n";
}
