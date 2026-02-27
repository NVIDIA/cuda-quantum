// Compile and run with:
// ```
// nvq++ mid_circuit_measurement.cpp -o teleport.x && ./teleport.x
// ```

#include <cudaq.h>

struct kernel {
  auto operator()() __qpu__ {
    cudaq::qarray<3> q;
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

    return mz(q[2]);
  }
};

int main() {

  int nShots = 100;
  auto results = cudaq::run(/*shots*/ nShots, kernel{});
  std::size_t nOnes = 0;
  // Count the number of times we measured "1"
  for (auto r : results) {
    if (r)
      nOnes++;
  }
  // Print out the results
  printf("Measured '1' on target qubit %zu times out of %d shots.\n", nOnes,
         nShots);
  // Will fail if not equal to number of shots
  assert(nOnes == nShots && "Failure to teleport qubit in |1> state.");
}
