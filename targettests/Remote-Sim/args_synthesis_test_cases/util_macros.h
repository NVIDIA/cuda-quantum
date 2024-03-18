/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

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
