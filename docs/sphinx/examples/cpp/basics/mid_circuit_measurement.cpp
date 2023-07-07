// Compile and run with:
// ```
// nvq++ mid_circuit_measurement.cpp -o teleport.x && ./teleport.x
// ```

#include <cudaq.h>

struct kernel {
  void operator()() __qpu__ {
    cudaq::qreg<3> q;
    // Initial state preparation
    x(q[0]);

    // Create Bell pair
    h(q[1]);
    x<cudaq::ctrl>(q[1], q[2]);

    x<cudaq::ctrl>(q[0], q[1]);
    h(q[0]);

    auto b0 = mz(q[0]);
    auto b1 = mz(q[1]);

    if (b1)
      x(q[2]);
    if (b0)
      z(q[2]);

    mz(q[2]);
  }
};

int main() {

  int nShots = 100;
  // Sample
  auto counts = cudaq::sample(/*shots*/ nShots, kernel{});
  counts.dump();

  // Get the marginal counts on the second qubit
  auto resultsOnZero = counts.get_marginal({0});
  resultsOnZero.dump();

  // Count the "1"
  auto nOnes = resultsOnZero.count("1");

  // Will fail if not equal to number of shots
  assert(nOnes == nShots && "Failure to teleport qubit in |1> state.");
}
