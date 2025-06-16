#include <cudaq.h>
#include <iostream>

struct ghz {
  void operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  // The original RST showed "int main() { ... }"
  // This is a complete main for a runnable example.
  auto counts = cudaq::sample(ghz{}, 3);
  std::cout << "cudaq_ir_simple.cpp: ghz(3) counts:" << std::endl;
  counts.dump();
  return 0;
}
// [End CUDA_IR_Simple_CPP_Content]
