#include <cudaq.h>
#include <stdio.h>

// [Begin Kernel C++]
// Define a simple kernel for demonstration
struct observe_kernel_demo {
  auto operator()(double theta) __qpu__ {
    cudaq::qarray<2> q;
    x(q[0]);
    ry(theta, q[1]);
    cx(q[0], q[1]);
  }
};
// [End Kernel C++]

int main() {
  // [Begin Observe Result Usage C++]
  // Define a spin_op for demonstration
  cudaq::spin_op spinOp = cudaq::spin_op::x(0) * cudaq::spin_op::x(1) + 0.5 * cudaq::spin_op::z(0);
  double param = 0.23; // Example parameter

  // I only care about the expected value, discard
  // the fine-grain data produced
  double expVal_simple = cudaq::observe(observe_kernel_demo{}, spinOp, param);
  printf("Simple ExpVal: %lf\n", expVal_simple);

  // I require the result with all generated data
  auto result = cudaq::observe(observe_kernel_demo{}, spinOp, param);
  auto expVal_detailed = result.expectation();
  printf("Detailed ExpVal: %lf\n", expVal_detailed);

  // Example for a specific term
  auto x0x1_term = cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
  if (spinOp.has_term(x0x1_term)) {
      auto X0X1Exp = result.expectation(x0x1_term);
      printf("X0X1 ExpVal: %lf\n", X0X1Exp);
      auto X0X1Data = result.counts(x0x1_term);
      printf("X0X1 Counts:\n");
      X0X1Data.dump();
  }
  result.dump(); // Dump all data
  // [End Observe Result Usage C++]
  return 0;
}