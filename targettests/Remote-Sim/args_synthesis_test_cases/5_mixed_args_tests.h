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
// Mixed arguments type tests
// These tests are just some random tests for more than 2 arguments.
//
//=============================================================================//
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
