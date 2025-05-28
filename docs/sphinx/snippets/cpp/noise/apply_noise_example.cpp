#include <cudaq.h>

int main() {
  auto kernel = []() __qpu__ {
    cudaq::qubit q, r;
    cudaq::apply_noise<cudaq::depolarization2>(/*probability=*/0.1, q, r);
  };
  
  return 0;
}