
// Compile and run with:
// ```
// nvq++ --enable-mlir --opt-pass 'func.func(add-dealloc,combine-quantum-alloc,canonicalize,factor-quantum-alloc,memtoreg),canonicalize,cse,add-wireset,func.func(assign-wire-indices),dep-analysis,func.func(regtomem),symbol-dce' static_kernel_simple.cpp
// ./a.out
// ```

#include <cudaq.h>

template <std::size_t N>
struct ghz {
  double operator()() __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    auto res = mz(q[1]);
    double returnVal = 0.0;
    if (res)
      returnVal = 1.0;
    return returnVal;
  }
};

int main() {

  double result = ghz<10>{}();
  printf("Result = %f\n", result);

  return 0;
}
