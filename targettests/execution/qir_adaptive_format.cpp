/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std -fenable-cudaq-run --target qir-test--emulate %s -o %t && CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck %s
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithm.h>

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

// Test using runtime output functions
__qpu__ int  kernel_int_output() {
  return 123;
}

__qpu__ float  kernel_float_output() {
  return 123.5;
}

struct MyClass {
  int x;
  float y;
};

__qpu__ MyClass  kernel_struct_output() {
  return MyClass{12,13.5};
}

// Test int_computations flag

__qpu__ void kernel_int_computations(int n, std::int32_t m) {
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
__qpu__ void kernel_int_float_computations(int n, std::int32_t m) {
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

  cudaq::qvector qubits(n);
  int i = 0;
  int j = 0;

  // Use large number of iterations to prevent unrolling.
  while (i < 1025) {
    cx(qubits[j], qubits[j + 1]);
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

  // Test runtime output functions
  { auto results = cudaq::run(1, kernel_int_output); }

  // clang-format off
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK: {{.*}}

  // CHECK: @cstr.69333200 = private constant [4 x i8] c"i32\00"
  // CHECK: define i32 @__nvqpp__mlirgen__function_kernel_int_output._Z17kernel_int_outputv() local_unnamed_addr #0 {
  // CHECK: "0":
  // CHECK:   tail call void @__quantum__rt__int_record_output(i64 123, i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.69333200, i64 0, i64 0))
  // CHECK:   ret i32 123
  // CHECK: }
  // CHECK: declare void @__quantum__rt__int_record_output(i64, i8*) local_unnamed_addr
  // CHECK: attributes #0 = { "entry_point" "output_labeling_schema"="schema_id" "qir_profiles"="adaptive_profile" }
  // clang-format on

  { auto results = cudaq::run(1, kernel_float_output); }

  // clang-format off
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK: {{.*}}
  // CHECK: @cstr.66333200 = private constant [4 x i8] c"f32\00"
  // CHECK: define float @__nvqpp__mlirgen__function_kernel_float_output._Z19kernel_float_outputv() local_unnamed_addr #0 {
  // CHECK: "0":
  // CHECK:   tail call void @__quantum__rt__double_record_output(double 1.235000e+02, i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.66333200, i64 0, i64 0))
  // CHECK:   ret float 1.235000e+02
  // CHECK: }
  // CHECK: declare void @__quantum__rt__double_record_output(double, i8*) local_unnamed_addr
  // CHECK: attributes #0 = { "entry_point" "output_labeling_schema"="schema_id" "qir_profiles"="adaptive_profile" }
  // clang-format on

  { auto results = cudaq::run(1, kernel_struct_output); }

  // clang-format off
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK: {{.*}}
  // CHECK: @cstr.7475706C653C6933322C206633323E00 = private constant [16 x i8] c"tuple<i32, f32>\00"
  // CHECK: @cstr.2E3000 = private constant [3 x i8] c".0\00"
  // CHECK: @cstr.2E3100 = private constant [3 x i8] c".1\00"

  // CHECK: define { i32, float } @__nvqpp__mlirgen__function_kernel_struct_output._Z20kernel_struct_outputv() local_unnamed_addr #0 {
  // CHECK: "0":
  // CHECK:   tail call void @__quantum__rt__tuple_record_output(i64 2, i8* nonnull getelementptr inbounds ([16 x i8], [16 x i8]* @cstr.7475706C653C6933322C206633323E00, i64 0, i64 0))
  // CHECK:   tail call void @__quantum__rt__int_record_output(i64 12, i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.2E3000, i64 0, i64 0))
  // CHECK:   tail call void @__quantum__rt__double_record_output(double 1.350000e+01, i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.2E3100, i64 0, i64 0))
  // CHECK:   ret { i32, float } { i32 12, float 1.350000e+01 }
  // CHECK: }
  // CHECK: declare void @__quantum__rt__int_record_output(i64, i8*) local_unnamed_addr
  // CHECK: declare void @__quantum__rt__double_record_output(double, i8*) local_unnamed_addr
  // CHECK: declare void @__quantum__rt__tuple_record_output(i64, i8*) local_unnamed_addr
  // CHECK: attributes #0 = { "entry_point" "output_labeling_schema"="schema_id" "qir_profiles"="adaptive_profile" }
  // clang-format on

  // Test multiple_return_points flag
  { auto results = cudaq::run(1, kernel_multiple_return_points, 2); }

  // clang-format off
  // CHECK: {{.*}}
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK: {{.*}}
  // CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

  // CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
  // CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
  // CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
  // CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
  // CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
  // CHECK: !5 = !{i32 1, !"int_computations", [7 x i8] c"i32,i64"}
  // CHECK: !6 = !{i32 1, !"backwards_branching", i2 0}
  // clang-format on

  // Test int_computations flag
  {
    auto counts = cudaq::sample(kernel_int_computations, 6, 134);
    counts.dump();
  }

  // clang-format off
  // CHECK: {{.*}}
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK: {{.*}}
  // CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

  // CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
  // CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
  // CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
  // CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
  // CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
  // CHECK: !5 = !{i32 1, !"int_computations", [6 x i8] c"i1,i32"}
  // CHECK: !6 = !{i32 1, !"backwards_branching", i2 0}
  // clang-format on

  // Test float_computations flag
  {
    auto counts = cudaq::sample(kernel_int_float_computations, 6, 134);
    counts.dump();
  }

  // clang-format off
  // CHECK: {{.*}}
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK: {{.*}}
  // CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}

  // CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
  // CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
  // CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
  // CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
  // CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
  // CHECK: !5 = !{i32 1, !"int_computations", [6 x i8] c"i1,i32"}
  // CHECK: !6 = !{i32 1, !"float_computations", [3 x i8] c"f32"}
  // CHECK: !7 = !{i32 1, !"backwards_branching", i2 0}
  // clang-format on

  // Test backwards_branching flag
  {
    auto counts = cudaq::sample(kernel_iteration_loop, 2);
    counts.dump();
  }

  // clang-format off
  // CHECK: {{.*}}
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK: {{.*}}
  // CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

  // CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
  // CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
  // CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
  // CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
  // CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
  // CHECK: !5 = !{i32 1, !"int_computations", [3 x i8] c"i32"}
  // CHECK: !6 = !{i32 1, !"backwards_branching", i2 1}
  // clang-format on

  {
    auto counts = cudaq::sample(kernel_conditionally_terminating_loops);
    counts.dump();
  }

  // clang-format off
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
  // clang-format on

  {
    auto counts =
        cudaq::sample(kernel_iteration_and_conditionally_terminating_loop, 2);
    counts.dump();
  }

  // clang-format off
  // CHECK: {{.*}}
  // CHECK: ; ModuleID = 'LLVMDialectModule'
  // CHECK: {{.*}}
  // CHECK: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

  // CHECK: !0 = !{i32 2, !"Debug Info Version", i32 3}
  // CHECK: !1 = !{i32 1, !"qir_major_version", i32 1}
  // CHECK: !2 = !{i32 7, !"qir_minor_version", i32 0}
  // CHECK: !3 = !{i32 1, !"dynamic_qubit_management", i1 false}
  // CHECK: !4 = !{i32 1, !"dynamic_result_management", i1 false}
  // CHECK: !5 = !{i32 1, !"int_computations", [3 x i8] c"i32"}
  // NOTE: -1 is "11" bitstring base 2
  // CHECK: !6 = !{i32 1, !"backwards_branching", i2 -1}
  // clang-format on

  return 0;
}
