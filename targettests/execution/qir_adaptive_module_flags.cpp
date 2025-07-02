/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --target qir-test --target-options --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target qir-test --target-options int_computations --emulate %s -o %t && %t | FileCheck --check-prefix=INT_CHECK %s
// RUN: nvq++ %cpp_std --target qir-test --target-options float_computations --emulate %s -o %t && %t | FileCheck --check-prefix=FLOAT_CHECK %s
// RUN: nvq++ %cpp_std --target qir-test --target-options int_float_computations --emulate %s -o %t && %t | FileCheck --check-prefix=INT_FLOAT_CHECK %s

#include "cudaq.h"

// Test hasMultipleTargetBranching flag
// TODO: add test after the support for switch statements is added.


// Test multiple_return_points flag
// NOTE: we create a common return with a phi node, not currently possible to
// produce a `True` value for the multiple_return_points flag

__qpu__ int kernel_multiple_return_points(int n) {
  cudaq::qvector q(n);
  if (mz(q[0])) {
    x(q[0]);
    return 1;
  }
  return 2;
}

// Test int_computations flag
// TODO: add cudaq.run tests using runtime output functions

__qpu__ int kernel_int_computations(int n, std::int32_t m) {
  cudaq::qvector q(n);
  int j = 0;
  for (auto i = 0; i < n; i++) {
    int k = 0;
    if (i > 5)
      k = 1;
    x(q[k]);
    if (mz(q[k])) {
        j = j + 1;
        m = m + m;
    }
  }

  if (j > 5) {
    x(q[0]);
  }
}

// Test float_computations flags
__qpu__ int kernel_int_float_computations(int n, std::int32_t m) {
  cudaq::qvector q(n);
  int j = 0;
  float jf = 1.2;
  for (auto i = 0; i < n; i++) {
    int k = 0;
    if (i > 5)
      k = 1;
    x(q[k]);
    if (mz(q[k])) {
        j = j + 1;
        m = m + m;
        jf = jf + jf;
    }
  }

  if ((j > 5) && (jf > 3)) {
    x(q[0]);
  }
}

// Test backwards_branching flag
__qpu__ void kernel_iteration_loop(int n) {
  cudaq::qvector q(n);
  int i = 0;
  int j = 0;

  // Use large number of iterations to prevent unrolling.
  while (i < 1025) {
    cx(q[j], q[j + 1]);
    i = i + 1;
    j = j + 1;
    if (j >= n - 1)
        j = 0;
  }
}


__qpu__ void kernel_conditionally_terminating_loops() {
    cudaq::qubit q;
    auto t = false;
    while (!t) {
        x(q);
        t = mz(q);
    }
}

__qpu__ void kernel_iteration_and_conditionally_terminating_loop(int n) {

  cudaq::qvector q(n);
  int i = 0;
  int j = 0;

  // Use large number of iterations to prevent unrolling.
  while (i < 1025) {
    cx(q[j], q[j + 1]);
    i = i + 1;
    j = j + 1;
    if (j >= n - 1)
        j = 0;
  }

  cudaq::qubit q;
  auto t = false;
  while (!t) {
      x(q);
      t = mz(q);
  }
}


int main() {

  // Test multiple_return_points flag
  {
   auto counts = cudaq::sample(kernel_multiple_return_points, 2);
   counts.dump();
  }

// INT_CHECK: ; ModuleID = 'LLVMDialectModule'
// INT_CHECK: {{.*}}
// INT_CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

// INT_CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
// INT_CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
// INT_CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
// INT_CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
// INT_CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
// INT_CHECK: !5 = !{i32 1, !"int_computations", [3 x i8] c"i64"}
// INT_CHECK: !6 = !{i32 1, !"backwards_branching", i2 0}

// Test int_computations flag
{
  auto counts = cudaq::sample(kernel_int_computations, 2, 134);
  counts.dump();
}

// INT_CHECK: {{.*}}
// INT_CHECK: ; ModuleID = 'LLVMDialectModule'
// INT_CHECK: {{.*}}
// INT_CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

// INT_CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
// INT_CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
// INT_CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
// INT_CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
// INT_CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
// INT_CHECK: !5 = !{i32 1, !"int_computations", [6 x i8] c"i1,i64"}
// INT_CHECK: !6 = !{i32 1, !"backwards_branching", i2 0}

// Test float_computations flag
{
  auto counts = cudaq::sample(kernel_int_float_computations, 2, 134);
  counts.dump();
}

// INT_FLOAT_CHECK: {{.*}}
// INT_FLOAT_CHECK: ; ModuleID = 'LLVMDialectModule'
// INT_FLOAT_CHECK: {{.*}}
// INT_FLOAT_CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}

// INT_FLOAT_CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
// INT_FLOAT_CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
// INT_FLOAT_CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
// INT_FLOAT_CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
// INT_FLOAT_CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
// INT_FLOAT_CHECK: !5 = !{i32 1, !"int_computations", [6 x i8] c"i1,i64"}
// INT_FLOAT_CHECK: !6 = !{i32 1, !"float_computations", [3 x i8] c"f64"}
// INT_FLOAT_CHECK: !7 = !{i32 1, !"backwards_branching", i2 0}


// Test backwards_branching flag
{
  auto counts = cudaq::sample(kernel_iteration_loop, 2);
  counts.dump();
}

// INT_CHECK: {{.*}}
// INT_CHECK: ; ModuleID = 'LLVMDialectModule'
// INT_CHECK: {{.*}}
// INT_CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

// INT_CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
// INT_CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
// INT_CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
// INT_CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
// INT_CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
// INT_CHECK: !5 = !{i32 1, !"int_computations", [3 x i8] c"i64"}
// INT_CHECK: !6 = !{i32 1, !"backwards_branching", i2 1}


{
  auto counts = cudaq::sample(kernel_conditionally_terminating_loops);
  counts.dump();
}

// CHECK: {{.*}}
// CHECK: ; ModuleID = 'LLVMDialectModule'
// CHECK: {{.*}}
// CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5}

// CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
// CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
// CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
// CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
// NOTE: -2 is "10" bitstring base 2
// CHECK: !5 = !{i32 1, !"backwards_branching", i2 -2}


{
  auto counts = cudaq::sample(kernel_iteration_and_conditionally_terminating_loop, 2);
  counts.dump();
}


// INT_CHECK: {{.*}}
// INT_CHECK: ; ModuleID = 'LLVMDialectModule'
// INT_CHECK: {{.*}}
// INT_CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

// INT_CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
// INT_CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
// INT_CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
// INT_CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
// INT_CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
// INT_CHECK: !5 = !{i32 1, !"int_computations", [3 x i8] c"i64"}
// NOTE: -1 is "11" bitstring base 2
// INT_CHECK: !6 = !{i32 1, !"backwards_branching", i2 -1}

return 0;

}

