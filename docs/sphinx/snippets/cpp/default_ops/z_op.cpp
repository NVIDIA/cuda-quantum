#include <cudaq.h>

int main() {
  cudaq::qubit qubit;

  // Apply the unitary transformation
  // Z = | 1   0 |
  //     | 0  -1 |
  z(qubit);

  return 0;
}
