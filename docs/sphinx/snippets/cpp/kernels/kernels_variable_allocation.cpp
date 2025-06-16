#include <cudaq.h>
#include <stdio.h> // For printf
#include <vector>

// [Begin Variable Allocation C++]
auto kernel_var_alloc = []() __qpu__ {
  // Not Allowed.
  // std::vector<int> i_invalid;
  // i_invalid.push_back(1);

  // Valid variable declarations
  std::vector<int> i(5); // Fixed size
  i[2] = 3;

  std::vector<float> f{1.0f, 2.0f, 3.0f}; // Initializer list implies size

  int k_int = 0;
  k_int = i[2]; // Use variable

  double pi_val = 3.1415926;
  pi_val += f[0]; // Use variable

  // Minimal quantum operation to make it a valid kernel for sampling
  cudaq::qarray<1> q_dummy;
  h(q_dummy[0]);
  mz(q_dummy);
  printf("Kernel with variable allocations executed. i[2]=%d, k_int=%d, "
         "pi_val=%f\n",
         i[2], k_int, pi_val);
};
// [End Variable Allocation C++]

int main() {
  // [Begin Variable Allocation C++ Execution]
  cudaq::sample(kernel_var_alloc);
  // [End Variable Allocation C++ Execution]
  return 0;
}
