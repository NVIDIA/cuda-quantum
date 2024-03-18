/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "util_macros.h"

//=============================================================================//
//
// Two arguments: one POD and one container
//
//=============================================================================//
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
                   (short, a, SHORT_VAL)(bool, b, BOOL_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(char, b, CHAR_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(int, b, INT_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(int64_t, b, INT64_VAL));
DEFINE_TEST_KERNEL(MACRO_CONCAT(arg_test_, __COUNTER__),
                   (short, a, SHORT_VAL)(float, b, FLOAT_VAL));
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
                   (short, a, SHORT_VAL)(short, b, SHORT_VAL));
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
