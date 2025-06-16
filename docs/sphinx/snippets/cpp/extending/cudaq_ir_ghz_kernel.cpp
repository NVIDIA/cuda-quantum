#include <cudaq.h>
#include <iostream> // Required for cudaq::sample and counts.dump()

// [Begin CUDA_IR_GHZ_Kernel_Struct_CPP]
// Define a quantum kernel
struct ghz_for_ir_tool_demo { // Renamed from ghz to avoid potential naming
                              // conflicts
  auto operator()() __qpu__ {
    cudaq::qarray<5> q;
    h(q[0]);
    for (int i = 0; i < 4; i++)
      x<cudaq::ctrl>(q[i], q[i + 1]);
    mz(q);
    // The original snippet in RST implies `void` return if mz(q) is the last
    // statement. If a return value was intended from mz, it would be `return
    // mz(q);`
  }
};
// [End CUDA_IR_GHZ_Kernel_Struct_CPP]

int main() {
  auto counts = cudaq::sample(ghz_for_ir_tool_demo{});
  std::cout << "cudaq_ir_ghz_kernel.cpp: ghz_for_ir_tool_demo counts:"
            << std::endl;
  counts.dump();
  return 0;
}
