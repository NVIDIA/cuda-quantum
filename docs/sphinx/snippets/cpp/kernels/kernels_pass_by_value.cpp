#include <cassert> // For assert
#include <cudaq.h>
#include <stdio.h> // For printf
#include <vector>

// [Begin Pass By Value C++]
auto kernel_pbv = [](int i_arg, std::vector<double> v_arg) __qpu__ {
  // i_arg == 2, allocate 2 qubits
  cudaq::qarray q(i_arg);
  // v_arg[1] == 2.0, angle here is 2.0
  if (i_arg > 0 && v_arg.size() > 1) {
    ry(v_arg[1], q[0]);
  }
  mz(q); // Measure all qubits in q

  printf("Inside kernel_pbv: i_arg = %d, v_arg[0] = %f\n", i_arg,
         v_arg.empty() ? -1.0 : v_arg[0]);
  // Change the variables, caller does not see this
  i_arg = 5;
  if (!v_arg.empty()) {
    v_arg[0] = 3.0;
  }
  printf("Inside kernel_pbv (after change): i_arg = %d, v_arg[0] = %f\n", i_arg,
         v_arg.empty() ? -1.0 : v_arg[0]);
};

int main() {
  int k_host = 2;
  std::vector<double> d_host{1.0, 2.0};

  printf("Before kernel_pbv call: k_host = %d, d_host[0] = %f\n", k_host,
         d_host[0]);
  // For __qpu__ kernels, direct call isn't typical; usually via sample/observe.
  // However, the example implies direct invocation logic for pass-by-value
  // illustration. To test this, we'd ideally check host vars after a
  // `cudaq::sample` or similar. Let's use cudaq::sample to ensure the kernel is
  // processed.
  cudaq::sample(kernel_pbv, k_host, d_host);

  // k_host is still 2, pass by value
  // d_host is still {1.0, 2.0}, pass by value
  printf("After kernel_pbv call: k_host = %d, d_host[0] = %f\n", k_host,
         d_host[0]);
  assert(k_host == 2);
  assert(d_host[0] == 1.0);
  assert(d_host[1] == 2.0);
  return 0;
}
// [End Pass By Value C++]
