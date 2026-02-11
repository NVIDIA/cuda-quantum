// Compile with
// ```
// nvq++ teleport.cpp --target qci -o teleport.x
// ```
//
// Make sure to export or otherwise present your user token via the environment,
// e.g., using export:
// ```
// export QCI_AUTH_TOKEN="your token here"
// ```
//
// Then run against the Seeker or AquSim with:
// ```
// ./teleport.x
// ```

#include <cudaq.h>
#include <iostream>

struct teleportation {
  auto operator()() __qpu__ {
    // Initialize a three qubit quantum circuit
    cudaq::qvector qubits(3);

    // Random quantum state on qubit 0.
    rx(3.14, qubits[0]);
    ry(2.71, qubits[0]);
    rz(6.62, qubits[0]);

    // Create a maximally entangled state on qubits 1 and 2.
    h(qubits[1]);
    cx(qubits[1], qubits[2]);

    cx(qubits[0], qubits[1]);
    h(qubits[0]);

    if (mz(qubits[0])) {
      z(qubits[2]);
    }

    if (mz(qubits[1])) {
      x(qubits[2]);
    }

    /// NOTE: If the return statement is changed to `mz(qubits)`, the program
    /// fails. Ref: https://github.com/NVIDIA/cuda-quantum/issues/3708
    return mz(qubits[2]);
  }
};

int main() {
  auto results = cudaq::run(20, teleportation{});
  std::cout << "Measurement results of the teleported qubit:\n[ ";
  for (auto r : results)
    std::cout << r << " ";
  std::cout << "]\n";
  return 0;
}
