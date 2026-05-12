/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

#include "args_synthesis.h"

//===----------------------------------------------------------------------===//
// Section 3b: Two Arguments: container first, POD second
//===----------------------------------------------------------------------===//
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

int main() {
  for (auto &functor : ALL_TEST_FUNCTORS)
    functor();
  return 0;
}
