// Compile and run with:
// ```
// nvq++ --target fermioniq fermioniq.cpp -o out.x && ./out.x
// ```
// This will submit the job to the fermioniq emulator.
// ```
// nvq++ --target fermioniq
// fermioniq.cpp -o out.x && ./out.x
// ```

#include <cudaq.h>
#include <fstream>

// Define a simple quantum kernel to execute on fermioniq.
struct ghz {
  // Maximally entangled state between 5 qubits.
  auto operator()() __qpu__ {
    cudaq::qvector q(3);
    h(q[0]);
    for (int i = 0; i < 2; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    auto result = mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(1001, ghz());
  counts.dump();
}
