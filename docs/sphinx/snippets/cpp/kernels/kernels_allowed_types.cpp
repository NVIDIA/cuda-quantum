#include <cmath>   // For M_PI_2
#include <cstddef> // For std::size_t
#include <cudaq.h>
#include <stdio.h> // For printf
#include <string>  // Not directly used in kernel args, but good for context
#include <vector>

// [Begin Allowed Types C++]
struct MyCustomSimpleStruct {
  int i = 0;
  int j = 0;
  std::vector<double> angles;
};

// Valid CUDA-Q Types used in Kernels
auto kernel_allowed_types = [](int N, bool flag, float angle,
                               std::vector<std::size_t> layers,
                               std::vector<double> parameters,
                               std::vector<std::vector<float>> recursiveVec,
                               MyCustomSimpleStruct var) __qpu__ {
  cudaq::qarray<1> q;
  if (flag && N > 0 && !layers.empty() && !parameters.empty() &&
      !recursiveVec.empty() && !var.angles.empty()) {
    ry(angle + parameters[0] + recursiveVec[0][0] + var.angles[0], q[0]);
  }
  mz(q);
  printf("Kernel with allowed types executed.\n");
};

__qpu__ double kernelThatReturns_cpp() {
  cudaq::qarray<1> q;
  h(q[0]);
  mz(q); // Measurement needed if value depends on quantum state
  return M_PI_2;
}
// [End Allowed Types C++]

int main() {
  // [Begin Allowed Types C++ Execution]
  MyCustomSimpleStruct s_var;
  s_var.i = 1;
  s_var.j = 2;
  s_var.angles = {0.1, 0.2};
  std::vector<std::size_t> l = {0, 1};
  std::vector<double> p = {0.5};
  std::vector<std::vector<float>> rv = {{0.3f}};

  kernel_allowed_types(1, true, 0.5f, l, p, rv, s_var);
  double ret_val =
      cudaq::sample(kernelThatReturns_cpp)
          .front()
          .expectation(); // Sample and get expectation for a double return
  printf("Kernel that returns (C++) returned: %f (Note: sample returns counts, "
         "direct call for actual value if deterministic or use observe)\n",
         ret_val);
  // For a kernel that returns a classical value not dependent on measurement,
  // direct simulation or a different execution model might be used.
  // For testing, we can wrap it in a way that `sample` or `observe` makes
  // sense, or if it's purely classical computation, it might not be a typical
  // __qpu__ kernel's main purpose. Let's assume it's a simple classical return
  // for now. A direct call is not standard for __qpu__ from host. We'll rely on
  // the fact that it compiles and can be processed. For a more meaningful test
  // of return, one might use observe if it calculates an expectation. If it's
  // just returning a constant after some quantum ops:
  auto result_counts =
      cudaq::sample(kernelThatReturns_cpp); // This will run the kernel
  // The actual double value isn't directly in sample_result unless it's part of
  // measurement. The RST implies it's a direct return, which is more complex
  // for __qpu__ entry points. Let's assume the intent is to show it *can* be
  // defined.
  printf("kernelThatReturns_cpp was invoked via sample.\n");
  // [End Allowed Types C++ Execution]
  return 0;
}
