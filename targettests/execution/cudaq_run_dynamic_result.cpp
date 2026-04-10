// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s -check-prefix=FAIL

#include <cudaq.h>

// A kernel with a dynamic result size
__qpu__ std::vector<bool> branch_vec_test(bool flip) {
  cudaq::qubit ctrl;
  if (flip)
    x(ctrl);
  bool b = mz(ctrl);
  int sz = b ? 2 : 4;
  cudaq::qvector data(sz);
  return mz(data);
}

int main() {
    auto res1 = cudaq::run(1, branch_vec_test, false);
    auto res2 = cudaq::run(1, branch_vec_test, true);
    printf("Result length = %ld\n", res1[0].size());
    printf("Result length = %ld\n", res2[0].size());
}

// FAIL: Could not successfully translate to qir-adaptive
// CHECK: Result length = 4
// CHECK: Result length = 2