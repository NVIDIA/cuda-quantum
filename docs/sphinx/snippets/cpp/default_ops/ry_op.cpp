#include <cudaq.h>

int main() {
  cudaq::qubit qubit;

  // Apply the unitary transformation
  // Ry(θ) = | cos(θ/2)  -sin(θ/2) |
  //         | sin(θ/2)   cos(θ/2) |
  ry(std::numbers::pi, qubit);

  return 0;
}
