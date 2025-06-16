#include <cudaq.h>

int main() {
  cudaq::qubit qubit;

  // Apply the unitary transformation
  // R1(λ) = | 1     0    |
  //         | 0  exp(iλ) |
  r1(std::numbers::pi, qubit);

  return 0;
}
