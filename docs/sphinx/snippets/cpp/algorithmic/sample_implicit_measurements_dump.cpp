#include <cudaq.h>

int main() {
  // [Begin Kernel C++]
  auto kernel = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
  };
  // [End Kernel C++]
  // [Begin Sample C++]
  cudaq::sample(kernel).dump();
  // [End Sample C++]
  return 0;
}
