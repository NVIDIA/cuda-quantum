/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// clang-format off
// RUN: nvq++ --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// RUN: nvq++ --enable-mlir --no-aggressive-early-inline --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

// This is a comprehensive set of tests for kernel argument synthesis for remote
// platforms. Note: we use the remote-mqpu platform in MLIR mode as a mock
// environment for NVQC.

#include <cudaq.h>
#include <iostream>

// Macros to define the tests
#define END(...) END_(__VA_ARGS__)
#define END_(...) __VA_ARGS__##_END

#define PARAMS_LOOP_0(type_, name_, val_)                                      \
  PARAMS_LOOP_BODY(type_, name_, val_) PARAMS_LOOP_A
#define PARAMS_LOOP_A(type_, name_, val_)                                      \
  , PARAMS_LOOP_BODY(type_, name_, val_) PARAMS_LOOP_B
#define PARAMS_LOOP_B(type_, name_, val_)                                      \
  , PARAMS_LOOP_BODY(type_, name_, val_) PARAMS_LOOP_A
#define PARAMS_LOOP_0_END
#define PARAMS_LOOP_A_END
#define PARAMS_LOOP_B_END
#define PARAMS_LOOP_BODY(type_, name_, val_) type_ name_

#define VAR_LOOP_0(type_, name_, val_)                                         \
  f(q, VAR_LOOP_BODY(type_, name_, val_));                                     \
  VAR_LOOP_A
#define VAR_LOOP_A(type_, name_, val_)                                         \
  f(q, VAR_LOOP_BODY(type_, name_, val_));                                     \
  VAR_LOOP_B
#define VAR_LOOP_B(type_, name_, val_)                                         \
  f(q, VAR_LOOP_BODY(type_, name_, val_));                                     \
  VAR_LOOP_A
#define VAR_LOOP_0_END
#define VAR_LOOP_A_END
#define VAR_LOOP_B_END
#define VAR_LOOP_BODY(type_, name_, val_) name_

#define INVOKE_LOOP_0(type_, name_, val_)                                      \
  , INVOKE_LOOP_BODY(type_, name_, val_) INVOKE_LOOP_A
#define INVOKE_LOOP_A(type_, name_, val_)                                      \
  , INVOKE_LOOP_BODY(type_, name_, val_) INVOKE_LOOP_B
#define INVOKE_LOOP_B(type_, name_, val_)                                      \
  , INVOKE_LOOP_BODY(type_, name_, val_) INVOKE_LOOP_A
#define INVOKE_LOOP_0_END
#define INVOKE_LOOP_A_END
#define INVOKE_LOOP_B_END
#define INVOKE_LOOP_BODY(type_, name_, val_) val_

// Different overloads for the generic test kernel 'f':
// - It takes a single qubit and another argument of different types.
// - Just perform some random quantum op.
// - The top level kernel (the test kernel), will call these `f` kernels for all
// of its arguments.
template <typename T, typename = typename std::enable_if<
                          std::is_integral<T>::value, T>::type>
void f(cudaq::qubit &q, T k) __qpu__ {
  rx(1.0 * k, q);
}

void f(cudaq::qubit &q, double x) __qpu__ { rx(x, q); }

void f(cudaq::qubit &q, const std::vector<double> &k) __qpu__ {
  for (int i = 0; i < k.size(); ++i)
    rx(k[i], q);
}

void f(cudaq::qubit &q, const std::vector<float> &k) __qpu__ {
  for (int i = 0; i < k.size(); ++i)
    rx(k[i], q);
}

// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// Vector of bool is not fully supported in MLIR mode.
// void f(cudaq::qubit &q, const std::vector<bool>& k) __qpu__ {
//   for (int i = 0; i < k.size(); ++i)
//     if (k[i])
//         x(q);
// }

template <typename T, typename = typename std::enable_if<
                          std::is_integral<T>::value, T>::type>
void f(cudaq::qubit &q, const std::vector<T> &k) __qpu__ {
  for (int i = 0; i < k.size(); ++i)
    rx(1.0 * k[i], q);
}

// List of all the test functors to run
static std::vector<std::function<void()>> ALL_TEST_FUNCTORS;

// Main macro to define a test with different signatures.
// - Dispatch each argument to the test function 'f'.
// - Add an execution functor to ALL_TEST_FUNCTORS.
// For example:
// DEFINE_TEST_KERNEL(test, (int, a, INT_VAL)(std::vector<float>, b,
// VEC_FLOAT_VAL)(bool, c, BOOL_VAL)(std::vector<int>,
// d,VEC_INT_VAL)(std::vector<double>, e, VEC_DOUBLE_VAL));

// will be expanded to (define a kernel with a signature and add a test functor
// to the list)

// void test(int a, std::vector<float> b, bool c, std::vector<int> d,
//                    std::vector<double> e) __qpu__ {
//   cudaq::qubit q;
//   f(q, a);
//   f(q, b);
//   f(q, c);
//   f(q, d);
//   f(q, e);
//   mz(q);
// }
// const bool addedtest = []() {
//   ALL_TEST_FUNCTORS.emplace_back([]() {
//     auto counts = cudaq::sample(test, INT_VAL, VEC_FLOAT_VAL, BOOL_VAL,
//     VEC_INT_VAL, VEC_DOUBLE_VAL); counts.dump();
//   });
//   return true;
// }();

#define DEFINE_TEST_KERNEL(func_, ...)                                         \
  void func_(END(PARAMS_LOOP_0 __VA_ARGS__)) __qpu__ {                         \
    cudaq::qubit q;                                                            \
    END(VAR_LOOP_0 __VA_ARGS__)                                                \
    mz(q);                                                                     \
  }                                                                            \
  const bool added##func_ = []() {                                             \
    ALL_TEST_FUNCTORS.emplace_back([]() {                                      \
      std::cout << "Testing: " << #func_ << "\n";                              \
      auto counts = cudaq::sample(func_ END(INVOKE_LOOP_0 __VA_ARGS__));       \
      counts.dump();                                                           \
    });                                                                        \
    return true;                                                               \
  }();

// Define some dummy argument values to use for testing.
static const bool BOOL_VAL = true;
static const char CHAR_VAL = 'a';
static const int INT_VAL = 1;
static const std::size_t SIZE_T_VAL = 1;
static const int64_t INT64_VAL = 1;
static const double DOUBLE_VAL = 1.0;
static const float FLOAT_VAL = 1.0;
static const std::vector<bool> VEC_BOOL_VAL = {true, false};
static const std::vector<char> VEC_CHAR_VAL = {'a', 'b'};
static const std::vector<int> VEC_INT_VAL = {1, 2};
static const std::vector<std::size_t> VEC_SIZE_T_VAL = {1, 2};
static const std::vector<double> VEC_DOUBLE_VAL = {1.0, 2.0};
static const std::vector<float> VEC_FLOAT_VAL = {1.0, 2.0};

// Define tests
// Note: these are single-qubit tests, so will run very fast.
DEFINE_TEST_KERNEL(test1, (bool, a, BOOL_VAL));
DEFINE_TEST_KERNEL(test2, (char, a, CHAR_VAL));
DEFINE_TEST_KERNEL(test3, (int, a, INT_VAL));
DEFINE_TEST_KERNEL(test4, (std::size_t, a, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test5, (int64_t, a, INT64_VAL));
DEFINE_TEST_KERNEL(test6, (double, a, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test7, (float, a, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test8, (std::vector<bool>, a, VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test9, (std::vector<char>, a, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test10, (std::vector<int>, a, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test11, (std::vector<std::size_t>, a, VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test12, (std::vector<double>, a, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test13, (std::vector<float>, a, VEC_FLOAT_VAL));

// Two-argument tests: all possible combinations
DEFINE_TEST_KERNEL(test2_1, (bool, a, BOOL_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test2_2, (bool, a, BOOL_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test2_3, (bool, a, BOOL_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test2_4, (bool, a, BOOL_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test2_5, (bool, a, BOOL_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test2_6, (bool, a, BOOL_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test2_7, (bool, a, BOOL_VAL) (std::vector<bool>, b,
// VEC_BOOL_VAL)); DEFINE_TEST_KERNEL(test2_8, (bool, a, BOOL_VAL)
// (std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test2_9,
                   (bool, a, BOOL_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test2_10, (bool, a, BOOL_VAL)(std::vector<std::size_t>, b,
                                                 VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test2_11,
                   (bool, a, BOOL_VAL)(std::vector<double>, b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test2_12,
                   (bool, a, BOOL_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));

DEFINE_TEST_KERNEL(test3_1, (char, a, CHAR_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test3_2, (char, a, CHAR_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test3_3, (char, a, CHAR_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test3_4, (char, a, CHAR_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test3_5, (char, a, CHAR_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test3_6, (char, a, CHAR_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test3_7,
//                    (char, a, CHAR_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test3_8,
//                    (char, a, CHAR_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test3_9,
                   (char, a, CHAR_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test3_10, (char, a, CHAR_VAL)(std::vector<std::size_t>, b,
                                                 VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test3_11,
                   (char, a, CHAR_VAL)(std::vector<double>, b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test3_12,
                   (char, a, CHAR_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));

DEFINE_TEST_KERNEL(test4_1, (int, a, INT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test4_2, (int, a, INT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test4_3, (int, a, INT_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test4_4, (int, a, INT_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test4_5, (int, a, INT_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test4_6, (int, a, INT_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test4_7,
//                    (int, a, INT_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test4_8,
//                    (int, a, INT_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test4_9,
                   (int, a, INT_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test4_10, (int, a, INT_VAL)(std::vector<std::size_t>, b,
                                               VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test4_11,
                   (int, a, INT_VAL)(std::vector<double>, b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test4_12,
                   (int, a, INT_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));

DEFINE_TEST_KERNEL(test5_1, (std::size_t, a, SIZE_T_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test5_2, (std::size_t, a, SIZE_T_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test5_3,
                   (std::size_t, a, SIZE_T_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test5_4,
                   (std::size_t, a, SIZE_T_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test5_5,
                   (std::size_t, a, SIZE_T_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test5_6, (std::size_t, a, SIZE_T_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test5_7, (std::size_t, a, SIZE_T_VAL)(std::vector<bool>,
// b,
//                                                          VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test5_8, (std::size_t, a, SIZE_T_VAL)(std::vector<char>,
// b,
//                                                          VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test5_9, (std::size_t, a, SIZE_T_VAL)(std::vector<int>, b,
                                                         VEC_INT_VAL));
DEFINE_TEST_KERNEL(test5_10,
                   (std::size_t, a, SIZE_T_VAL)(std::vector<std::size_t>, b,
                                                VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test5_11, (std::size_t, a, SIZE_T_VAL)(std::vector<double>,
                                                          b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test5_12, (std::size_t, a, SIZE_T_VAL)(std::vector<float>, b,
                                                          VEC_FLOAT_VAL));

DEFINE_TEST_KERNEL(test6_1, (double, a, DOUBLE_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test6_2, (double, a, DOUBLE_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test6_3,
                   (double, a, DOUBLE_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test6_4, (double, a, DOUBLE_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test6_5, (double, a, DOUBLE_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test6_6, (double, a, DOUBLE_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test6_7,
//                    (double, a, DOUBLE_VAL)(std::vector<bool>, b,
//                    VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test6_8,
//                    (double, a, DOUBLE_VAL)(std::vector<char>, b,
//                    VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test6_9,
                   (double, a, DOUBLE_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test6_10, (double, a, DOUBLE_VAL)(std::vector<std::size_t>,
                                                     b, VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test6_11, (double, a, DOUBLE_VAL)(std::vector<double>, b,
                                                     VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test6_12, (double, a, DOUBLE_VAL)(std::vector<float>, b,
                                                     VEC_FLOAT_VAL));

DEFINE_TEST_KERNEL(test7_1, (float, a, FLOAT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test7_2, (float, a, FLOAT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test7_3, (float, a, FLOAT_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test7_4, (float, a, FLOAT_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test7_5, (float, a, FLOAT_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test7_6, (float, a, FLOAT_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test7_7,
//                    (float, a, FLOAT_VAL)(std::vector<bool>, b,
//                    VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test7_8,
//                    (float, a, FLOAT_VAL)(std::vector<char>, b,
//                    VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test7_9,
                   (float, a, FLOAT_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test7_10, (float, a, FLOAT_VAL)(std::vector<std::size_t>, b,
                                                   VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test7_11, (float, a, FLOAT_VAL)(std::vector<double>, b,
                                                   VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test7_12,
                   (float, a, FLOAT_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));

// FIXME: vector<char> is not supported in synthesis
// DEFINE_TEST_KERNEL(test8_1,
//                    (std::vector<char>, a, VEC_CHAR_VAL)(char, b, CHAR_VAL));
// DEFINE_TEST_KERNEL(test8_2,
//                    (std::vector<char>, a, VEC_CHAR_VAL)(int, b, INT_VAL));
// DEFINE_TEST_KERNEL(test8_3, (std::vector<char>, a, VEC_CHAR_VAL)(std::size_t,
// b,
//                                                                  SIZE_T_VAL));
// DEFINE_TEST_KERNEL(test8_4,
//                    (std::vector<char>, a, VEC_CHAR_VAL)(int64_t, b,
//                    INT64_VAL));
// DEFINE_TEST_KERNEL(test8_5,
//                    (std::vector<char>, a, VEC_CHAR_VAL)(double, b,
//                    DOUBLE_VAL));
// DEFINE_TEST_KERNEL(test8_6,
//                    (std::vector<char>, a, VEC_CHAR_VAL)(float, b,
//                    FLOAT_VAL));
// DEFINE_TEST_KERNEL(test8_7, (std::vector<char>, a,
//                              VEC_CHAR_VAL)(std::vector<bool>, b,
//                              VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test8_8, (std::vector<char>, a,
//                              VEC_CHAR_VAL)(std::vector<char>, b,
//                              VEC_CHAR_VAL));
// DEFINE_TEST_KERNEL(test8_9, (std::vector<char>, a,
//                              VEC_CHAR_VAL)(std::vector<int>, b,
//                              VEC_INT_VAL));
// DEFINE_TEST_KERNEL(test8_10,
//                    (std::vector<char>, a,
//                     VEC_CHAR_VAL)(std::vector<std::size_t>, b,
//                     VEC_SIZE_T_VAL));
// DEFINE_TEST_KERNEL(test8_11,
//                    (std::vector<char>, a, VEC_CHAR_VAL)(std::vector<double>,
//                    b,
//                                                         VEC_DOUBLE_VAL));
// DEFINE_TEST_KERNEL(test8_12,
//                    (std::vector<char>, a, VEC_CHAR_VAL)(std::vector<float>,
//                    b,
//                                                         VEC_FLOAT_VAL))

DEFINE_TEST_KERNEL(test9_1,
                   (std::vector<int>, a, VEC_INT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test9_2,
                   (std::vector<int>, a, VEC_INT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test9_3, (std::vector<int>, a, VEC_INT_VAL)(std::size_t, b,
                                                               SIZE_T_VAL));
DEFINE_TEST_KERNEL(test9_4,
                   (std::vector<int>, a, VEC_INT_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test9_5,
                   (std::vector<int>, a, VEC_INT_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test9_6,
                   (std::vector<int>, a, VEC_INT_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test9_7, (std::vector<int>, a,
//                              VEC_INT_VAL)(std::vector<bool>, b,
//                              VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test9_8, (std::vector<int>, a,
//                              VEC_INT_VAL)(std::vector<char>, b,
//                              VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test9_9, (std::vector<int>, a, VEC_INT_VAL)(std::vector<int>,
                                                               b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test9_10,
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<std::size_t>,
                                                      b, VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test9_11,
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<double>, b,
                                                      VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test9_12,
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<float>, b,
                                                      VEC_FLOAT_VAL));

DEFINE_TEST_KERNEL(test10_1, (std::vector<std::size_t>, a,
                              VEC_SIZE_T_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test10_2, (std::vector<std::size_t>, a,
                              VEC_SIZE_T_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test10_3, (std::vector<std::size_t>, a,
                              VEC_SIZE_T_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test10_4, (std::vector<std::size_t>, a,
                              VEC_SIZE_T_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test10_5, (std::vector<std::size_t>, a,
                              VEC_SIZE_T_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test10_6, (std::vector<std::size_t>, a,
                              VEC_SIZE_T_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test10_7,
//                    (std::vector<std::size_t>, a,
//                     VEC_SIZE_T_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test10_8,
//                    (std::vector<std::size_t>, a,
//                     VEC_SIZE_T_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test10_9,
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test10_10, (std::vector<std::size_t>, a,
                               VEC_SIZE_T_VAL)(std::vector<std::size_t>, b,
                                               VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test10_11,
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<double>, b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test10_12,
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));

DEFINE_TEST_KERNEL(test11_1,
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test11_2,
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test11_3, (std::vector<double>, a,
                              VEC_DOUBLE_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test11_4, (std::vector<double>, a,
                              VEC_DOUBLE_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(test11_5, (std::vector<double>, a,
                              VEC_DOUBLE_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test11_6, (std::vector<double>, a,
                              VEC_DOUBLE_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test11_7,
//                    (std::vector<double>, a,
//                    VEC_DOUBLE_VAL)(std::vector<bool>,
//                                                             b,
//                                                             VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test11_8,
//                    (std::vector<double>, a,
//                    VEC_DOUBLE_VAL)(std::vector<char>,
//                                                             b,
//                                                             VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test11_9,
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::vector<int>, b,
                                                            VEC_INT_VAL));
DEFINE_TEST_KERNEL(test11_10, (std::vector<double>, a,
                               VEC_DOUBLE_VAL)(std::vector<std::size_t>, b,
                                               VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test11_11,
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::vector<double>,
                                                            b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test11_12,
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::vector<float>,
                                                            b, VEC_FLOAT_VAL));

DEFINE_TEST_KERNEL(test12_1,
                   (std::vector<float>, a, VEC_FLOAT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(test12_2,
                   (std::vector<float>, a, VEC_FLOAT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(test12_3, (std::vector<float>, a,
                              VEC_FLOAT_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(test12_4, (std::vector<float>, a, VEC_FLOAT_VAL)(int64_t, b,
                                                                    INT64_VAL));
DEFINE_TEST_KERNEL(test12_5, (std::vector<float>, a,
                              VEC_FLOAT_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(test12_6,
                   (std::vector<float>, a, VEC_FLOAT_VAL)(float, b, FLOAT_VAL));
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1383
// DEFINE_TEST_KERNEL(test12_7,
//                    (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<bool>,
//                    b,
//                                                           VEC_BOOL_VAL));
// DEFINE_TEST_KERNEL(test12_8,
//                    (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<char>,
//                    b,
//                                                           VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test12_9, (std::vector<float>, a,
                              VEC_FLOAT_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(test12_10, (std::vector<float>, a,
                               VEC_FLOAT_VAL)(std::vector<std::size_t>, b,
                                              VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(test12_11,
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<double>,
                                                          b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test12_12,
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<float>, b,
                                                          VEC_FLOAT_VAL));

// These tests are just some random tests for more than 2 arguments. An
// exhaustive combinatorial list would be impractical.
DEFINE_TEST_KERNEL(test_mixed_1,
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL)(bool, c, BOOL_VAL));
DEFINE_TEST_KERNEL(test_mixed_2,
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL)(char, c, CHAR_VAL));
DEFINE_TEST_KERNEL(test_mixed_3,
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL)(int, c, INT_VAL));
DEFINE_TEST_KERNEL(test_mixed_4,
                   (int, a, INT_VAL)(char, b, CHAR_VAL)(int, c, INT_VAL));
DEFINE_TEST_KERNEL(test_mixed_5,
                   (std::size_t, a, SIZE_T_VAL)(char, b, CHAR_VAL)(int, c,
                                                                   INT_VAL));
DEFINE_TEST_KERNEL(test_mixed_6,
                   (std::size_t, a, SIZE_T_VAL)(int, b, INT_VAL)(std::size_t, c,
                                                                 SIZE_T_VAL));
// FIXME: vector<char> is not supported in synthesis
// DEFINE_TEST_KERNEL(test_mixed_7,
//                    (int, a, INT_VAL)(std::vector<char>, b,
//                                      VEC_CHAR_VAL)(std::vector<float>, c,
//                                                    VEC_FLOAT_VAL));
// DEFINE_TEST_KERNEL(test_mixed_8,
//                    (int, a, INT_VAL)(std::vector<char>, b,
//                                      VEC_CHAR_VAL)(std::vector<double>, c,
//                                                    VEC_DOUBLE_VAL));
// DEFINE_TEST_KERNEL(test_mixed_9, (std::size_t, a,
//                                   SIZE_T_VAL)(std::vector<double>, b,
//                                               VEC_DOUBLE_VAL)(std::vector<char>,
//                                                               c,
//                                                               VEC_CHAR_VAL));
// DEFINE_TEST_KERNEL(test_mixed_10, (std::size_t, a,
//                                    SIZE_T_VAL)(std::vector<float>, b,
//                                                VEC_FLOAT_VAL)(std::vector<char>,
//                                                               c,
//                                                               VEC_CHAR_VAL));
// DEFINE_TEST_KERNEL(test_mixed_11,
//                    (double, a, DOUBLE_VAL)(std::vector<char>, b,
//                                            VEC_CHAR_VAL)(std::vector<float>,
//                                            c,
//                                                          VEC_FLOAT_VAL));
// DEFINE_TEST_KERNEL(test_mixed_12,
//                    (double, a, DOUBLE_VAL)(std::vector<char>, b,
//                                            VEC_CHAR_VAL)(std::vector<double>,
//                                            c,
//                                                          VEC_DOUBLE_VAL));
// DEFINE_TEST_KERNEL(test_mixed_13,
//                    (float, a, FLOAT_VAL)(std::vector<double>, b,
//                                          VEC_DOUBLE_VAL)(std::vector<char>,
//                                          c,
//                                                          VEC_CHAR_VAL));
// DEFINE_TEST_KERNEL(test_mixed_14,
//                    (float, a, FLOAT_VAL)(std::vector<float>, b,
//                                          VEC_FLOAT_VAL)(std::vector<char>, c,
//                                                         VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(test_mixed_15,
                   (std::vector<float>, a,
                    VEC_FLOAT_VAL)(double, b, DOUBLE_VAL)(std::vector<float>, c,
                                                          VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(test_mixed_16,
                   (std::vector<float>, a,
                    VEC_FLOAT_VAL)(double, b, DOUBLE_VAL)(std::vector<int>, c,
                                                          VEC_INT_VAL));
DEFINE_TEST_KERNEL(test_mixed_17,
                   (std::vector<float>, a,
                    VEC_FLOAT_VAL)(double, b, DOUBLE_VAL)(bool, c, BOOL_VAL)(
                       std::vector<int>, d, VEC_INT_VAL)(std::vector<double>, e,
                                                         VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(test_mixed_18,
                   (int, a, INT_VAL)(std::vector<float>, b, VEC_FLOAT_VAL)(
                       bool, c, BOOL_VAL)(std::vector<int>, d,
                                          VEC_INT_VAL)(std::vector<double>, e,
                                                       VEC_DOUBLE_VAL));

int main() {
  // Run all tests
  for (auto &functor : ALL_TEST_FUNCTORS)
    functor();
  return 0;
}
