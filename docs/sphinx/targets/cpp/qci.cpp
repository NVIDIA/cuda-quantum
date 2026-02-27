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

#include <array>
#include <cudaq.h>
#include <iostream>

struct teleportation {
  auto operator()() __qpu__ {
    std::vector<bool> results(3);

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

    results[0] = mz(qubits[0]);
    results[1] = mz(qubits[1]);

    if (results[0]) {
      z(qubits[2]);
    }

    if (results[1]) {
      x(qubits[2]);
    }

    results[2] = mz(qubits[2]);
    return results;
  }
};

int main() {
  // Note: Increase the number of shots to get closer to expected probabilities.
  constexpr std::size_t num_shots = 25;
  auto results = cudaq::run(num_shots, teleportation{});

  std::array<std::size_t, 3> ones{};
  for (const auto &shot : results)
    for (std::size_t q = 0; q < 3; ++q)
      ones[q] += static_cast<std::size_t>(shot[q]);

  auto freq = [&](std::size_t q) {
    return static_cast<double>(ones[q]) / num_shots;
  };

  // `mz[0]` and `mz[1]` are Bell measurement outcomes, so each is ~50%.
  // Probability of measuring`mz[2]` in 1 is determined by the prepared state,
  // which is ~4.6% for the angles above.
  std::cout << "Results over " << num_shots << " shots:\n";
  for (std::size_t q = 0; q < 3; ++q)
    std::cout << "  mz[" << q << "] = 1:  " << ones[q] << " / " << num_shots
              << "  (" << 100.0 * freq(q) << "%)\n";

  return 0;
}
