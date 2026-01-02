/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// clang-format off
// RUN: nvq++ --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// RUN: nvq++ --enable-mlir -fno-aggressive-inline --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

// This is a comprehensive set of tests for kernel argument synthesis for remote
// platforms.
#include <cudaq.h>
#include <iostream>

// Macros to define the tests
#define CONCAT_IMPL(x, y) x##y
#define MACRO_CONCAT(x, y) CONCAT_IMPL(x, y)
#define STRINGIFY(str) #str

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

void f(cudaq::qubit &q, std::vector<bool> &k) __qpu__ {
  for (int i = 0; i < k.size(); ++i)
    if (k[i])
      x(q);
}

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
  const bool MACRO_CONCAT(added_, func_) = []() {                              \
    ALL_TEST_FUNCTORS.emplace_back([]() {                                      \
      std::cout << "Test case " << STRINGIFY(func_) << ": " << __FILE__ << ":" \
                << __LINE__ << "\n";                                           \
      auto counts = cudaq::sample(func_ END(INVOKE_LOOP_0 __VA_ARGS__));       \
      counts.dump();                                                           \
    });                                                                        \
    return true;                                                               \
  }()

// Define some dummy argument values to use for testing.
static const bool BOOL_VAL = true;
static const char CHAR_VAL = 'a';
static const short SHORT_VAL = 1;
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
static const std::vector<short> VEC_SHORT_VAL = {1, 2};

// Setting these variables to 0 will disable the corresponding test suites.
#define RUN_SINGLE_ARGUMENT_TESTS 1
#define RUN_TWO_POD_ARGUMENTS_TESTS 1
#define RUN_TWO_ARGUMENTS_POD_CONTAINER_TESTS 1
#define RUN_TWO_CONTAINER_ARGUMENTS_TESTS 1
#define RUN_MIXED_ARGUMENTS_TESTS 1

//===----------------------------------------------------------------------===//
// Section 1: Single Argument
//===----------------------------------------------------------------------===//
#if RUN_SINGLE_ARGUMENT_TESTS
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__), (bool, a, BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__), (char, a, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__), (short, a, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__), (int, a, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int64_t, a, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__), (float, a, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<bool>, a, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a, VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL));
#endif

//===----------------------------------------------------------------------===//
// Section 2: Two Arguments of POD type
//===----------------------------------------------------------------------===//
#if RUN_TWO_POD_ARGUMENTS_TESTS
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(bool, b, BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(short, b, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(short, b, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(short, b, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(short, b, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(short, b, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::size_t, b, SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(short, b, SHORT_VAL));
#endif

//===----------------------------------------------------------------------===//
// Section 3: Two Arguments: one POD and one container
//===----------------------------------------------------------------------===//
#if RUN_TWO_ARGUMENTS_POD_CONTAINER_TESTS
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::vector<std::size_t>, b,
                                       VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::vector<double>, b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::vector<short>, b, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(std::vector<std::size_t>, b,
                                       VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(std::vector<double>, b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (char, a, CHAR_VAL)(std::vector<short>, b, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<std::size_t>, b,
                                     VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<double>, b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<short>, b, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(std::vector<bool>, b,
                                                VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(std::vector<char>, b,
                                                VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(std::vector<int>, b,
                                                VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(std::vector<std::size_t>, b,
                                                VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(std::vector<double>, b,
                                                VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(std::vector<float>, b,
                                                VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(std::vector<short>, b,
                                                VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<std::size_t>, b,
                                           VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<double>, b,
                                           VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<float>, b,
                                           VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<short>, b,
                                           VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<std::size_t>, b,
                                         VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<double>, b,
                                         VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<short>, b, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(std::size_t, b,
                                                        SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(short, b, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(std::size_t, b,
                                                      SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(double, b, DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(short, b, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a, VEC_SIZE_T_VAL)(char, b,
                                                                 CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a, VEC_SIZE_T_VAL)(int, b,
                                                                 INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a, VEC_SIZE_T_VAL)(std::size_t, b,
                                                                 SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a, VEC_SIZE_T_VAL)(int64_t, b,
                                                                 INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a, VEC_SIZE_T_VAL)(double, b,
                                                                 DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a, VEC_SIZE_T_VAL)(float, b,
                                                                 FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a, VEC_SIZE_T_VAL)(short, b,
                                                                 SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::size_t, b,
                                                            SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(int64_t, b,
                                                            INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(double, b,
                                                            DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(float, b,
                                                            FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(short, b,
                                                            SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::size_t, b,
                                                          SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(int64_t, b,
                                                          INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(double, b,
                                                          DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(float, b, FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(short, b, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(double, b,
                                                          DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(std::vector<double>, b,
                                         VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(std::vector<short>, b, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(std::size_t, b,
                                                          SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(int64_t, b,
                                                          INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(double, b,
                                                          DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(float, b, FLOAT_VAL));
#endif

//===----------------------------------------------------------------------===//
// Section 4: Two Arguments of Container Type
//===----------------------------------------------------------------------===//
#if RUN_TWO_CONTAINER_ARGUMENTS_TESTS
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(std::vector<bool>, b,
                                                        VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(std::vector<char>, b,
                                                        VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(std::vector<int>, b,
                                                        VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a,
                    VEC_CHAR_VAL)(std::vector<std::size_t>, b, VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(std::vector<double>, b,
                                                        VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(std::vector<float>, b,
                                                        VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<char>, a, VEC_CHAR_VAL)(std::vector<short>, b,
                                                        VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<bool>, b,
                                                      VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<char>, b,
                                                      VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<int>, b,
                                                      VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<std::size_t>,
                                                      b, VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<double>, b,
                                                      VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<float>, b,
                                                      VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<int>, a, VEC_INT_VAL)(std::vector<short>, b,
                                                      VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<bool>, b, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<char>, b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<int>, b, VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<std::size_t>, b,
                                    VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<double>, b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<float>, b, VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<std::size_t>, a,
                    VEC_SIZE_T_VAL)(std::vector<short>, b, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::vector<bool>,
                                                            b, VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::vector<char>,
                                                            b, VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::vector<int>, b,
                                                            VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a,
                    VEC_DOUBLE_VAL)(std::vector<std::size_t>, b,
                                    VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::vector<double>,
                                                            b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<double>, a, VEC_DOUBLE_VAL)(std::vector<float>,
                                                            b, VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<bool>, b,
                                                          VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<char>, b,
                                                          VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<int>, b,
                                                          VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a,
                    VEC_FLOAT_VAL)(std::vector<std::size_t>, b,
                                   VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<double>,
                                                          b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<float>, b,
                                                          VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<short>, b,
                                                          VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(std::vector<bool>, b,
                                                          VEC_BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(std::vector<char>, b,
                                                          VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a, VEC_FLOAT_VAL)(std::vector<int>, b,
                                                          VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a,
                    VEC_FLOAT_VAL)(std::vector<std::size_t>, b,
                                   VEC_SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(std::vector<double>,
                                                          b, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(std::vector<float>, b,
                                                          VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a, VEC_SHORT_VAL)(std::vector<short>, b,
                                                          VEC_SHORT_VAL));
#endif

//===----------------------------------------------------------------------===//
// Section 5: Mixed Arguments Type
//===----------------------------------------------------------------------===//
#if RUN_MIXED_ARGUMENTS_TESTS
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL)(bool, c, BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL)(char, c, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL)(int, c, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(char, b, CHAR_VAL)(int, c, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(char, b, CHAR_VAL)(int, c,
                                                                   INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(int, b, INT_VAL)(std::size_t, c,
                                                                 SIZE_T_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<char>, b,
                                     VEC_CHAR_VAL)(std::vector<float>, c,
                                                   VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<char>, b,
                                     VEC_CHAR_VAL)(std::vector<double>, c,
                                                   VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a,
                    SIZE_T_VAL)(std::vector<double>, b,
                                VEC_DOUBLE_VAL)(std::vector<char>, c,
                                                VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a,
                    SIZE_T_VAL)(std::vector<float>, b,
                                VEC_FLOAT_VAL)(std::vector<char>, c,
                                               VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<char>, b,
                                           VEC_CHAR_VAL)(std::vector<float>, c,
                                                         VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(std::vector<char>, b,
                                           VEC_CHAR_VAL)(std::vector<double>, c,
                                                         VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<double>, b,
                                         VEC_DOUBLE_VAL)(std::vector<char>, c,
                                                         VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (float, a, FLOAT_VAL)(std::vector<float>, b,
                                         VEC_FLOAT_VAL)(std::vector<char>, c,
                                                        VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a,
                    VEC_FLOAT_VAL)(double, b, DOUBLE_VAL)(std::vector<float>, c,
                                                          VEC_FLOAT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a,
                    VEC_FLOAT_VAL)(double, b, DOUBLE_VAL)(std::vector<int>, c,
                                                          VEC_INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<float>, a,
                    VEC_FLOAT_VAL)(double, b, DOUBLE_VAL)(bool, c, BOOL_VAL)(
                       std::vector<int>, d, VEC_INT_VAL)(std::vector<double>, e,
                                                         VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<float>, b, VEC_FLOAT_VAL)(
                       bool, c, BOOL_VAL)(std::vector<int>, d,
                                          VEC_INT_VAL)(std::vector<double>, e,
                                                       VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL)(short, c,
                                                          SHORT_VAL)(bool, d,
                                                                     BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(std::vector<short>, b,
                                       VEC_SHORT_VAL)(short, c, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (bool, a, BOOL_VAL)(char, b, CHAR_VAL)(int, c, INT_VAL)(
                       std::vector<short>, d, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(char, b, CHAR_VAL)(int, c,
                                                        INT_VAL)(short, d,
                                                                 SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a,
                    SIZE_T_VAL)(char, b, CHAR_VAL)(short, c,
                                                   SHORT_VAL)(int, d, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a,
                    SIZE_T_VAL)(int, b, INT_VAL)(std::size_t, c,
                                                 SIZE_T_VAL)(std::vector<short>,
                                                             d, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<char>, b, VEC_CHAR_VAL)(
                       std::vector<float>, c, VEC_FLOAT_VAL)(std::vector<short>,
                                                             d, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (int, a, INT_VAL)(std::vector<char>, b, VEC_CHAR_VAL)(
                       std::vector<short>, c,
                       VEC_SHORT_VAL)(std::vector<double>, d, VEC_DOUBLE_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(short, b, SHORT_VAL)(
                       std::vector<double>, c,
                       VEC_DOUBLE_VAL)(std::vector<short>, d,
                                       VEC_SHORT_VAL)(std::vector<char>, e,
                                                      VEC_CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::size_t, a, SIZE_T_VAL)(short, b, SHORT_VAL)(
                       std::vector<float>, c,
                       VEC_FLOAT_VAL)(std::vector<char>, d,
                                      VEC_CHAR_VAL)(short, e, SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (std::vector<short>, a,
                    VEC_SHORT_VAL)(double, b, DOUBLE_VAL)(std::vector<char>, c,
                                                          VEC_CHAR_VAL)(
                       std::vector<float>, d, VEC_FLOAT_VAL)(std::vector<short>,
                                                             e, VEC_SHORT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (double, a, DOUBLE_VAL)(short, b, SHORT_VAL)(
                       std::vector<char>, c, VEC_CHAR_VAL)(std::vector<short>,
                                                           d, VEC_SHORT_VAL)(
                       std::vector<short>, e,
                       VEC_SHORT_VAL)(std::vector<double>, ff,
                                      VEC_DOUBLE_VAL)(std::size_t, g,
                                                      SIZE_T_VAL));
#endif

int main() {
  // Run all tests
  for (auto &functor : ALL_TEST_FUNCTORS)
    functor();
  return 0;
}
