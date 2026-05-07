#include <cudaq.h>
#include <iostream>

/**
 * This example demonstrates the use of the `trace` and `tensor_product`
 * methods on `cudaq::state`. These methods are particularly useful when
 * working with density matrices (mixed states) and for combining subsystem states.
 */

int main() {
  // Define a simple kernel to create a Bell state.
  auto bell = []() __qpu__ {
    cudaq::qubit q, r;
    h(q);
    cx(q, r);
  };

  // 1. Get the state vector (pure state)
  auto stateVector = cudaq::get_state(bell);
  std::cout << "State vector (rank " << stateVector.get_tensor(0).extents.size()
            << "):\n";
  stateVector.dump();

  // 2. Get the density matrix (mixed state) using the 'dm' simulator
  // Note: We can also get this by using the 'dm' target.
  cudaq::set_target("dm");
  auto densityMatrix = cudaq::get_state(bell);
  std::cout << "\nDensity matrix (rank "
            << densityMatrix.get_tensor(0).extents.size() << "):\n";
  densityMatrix.dump();

  // 3. Take the trace of the density matrix.
  // For any valid density matrix, the trace should be 1.0.
  auto tr = densityMatrix.trace();
  std::cout << "\nTrace of density matrix: " << tr << "\n";

  // 4. Compute the tensor product of two states.
  // Combining two 2-qubit Bell states into a 4-qubit joint state.
  auto jointState = densityMatrix.tensor_product(densityMatrix);
  std::cout << "\nJoint state (4 qubits) rank: "
            << jointState.get_tensor(0).extents.size() << "\n";
  std::cout << "Joint state dimensions: " << jointState.get_tensor(0).extents[0]
            << "x" << jointState.get_tensor(0).extents[1] << "\n";

  // The trace of the tensor product of normalized states is also 1.0.
  std::cout << "Trace of joint state: " << jointState.trace() << "\n";

  return 0;
}
