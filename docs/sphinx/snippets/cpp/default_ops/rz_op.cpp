#include <cudaq.h>

int main() {
  cudaq::qubit qubit;

  // Apply the unitary transformation
  // Rz(λ) = | exp(-iλ/2)      0     |
  //         |     0       exp(iλ/2) |
  rz(std::numbers::pi, qubit);

  return 0;
}
