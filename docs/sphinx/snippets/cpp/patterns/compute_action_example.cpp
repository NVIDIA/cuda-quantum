#include <cudaq.h>
#include <stdio.h> // For printf

// Define a kernel that uses compute_action
auto kernel_compute_action_cpp = []() __qpu__ {
  cudaq::qarray<1> q;

  // [Begin Compute Action C++ Snippet]
  // Will invoke U V U^dag
  cudaq::compute_action(
      // U_code lambda
      [&]() {
        h(q[0]);
        s(q[0]);
      },
      // V_code lambda
      [&]() {
        x(q[0]);
      });
  // [End Compute Action C++ Snippet]

  mz(q[0]); // Measure for results
};

int main() {
  printf("C++ Compute-Action Example:\n");
  auto counts = cudaq::sample(kernel_compute_action_cpp);
  counts.dump();
  // Expected state after S H X H S_adj on |0>: (1+i)/2 (|0> + |1>)
  // This gives 50% probability for |0> and 50% for |1>.
  // So, counts should show roughly equal numbers for "0" and "1".
  return 0;
}