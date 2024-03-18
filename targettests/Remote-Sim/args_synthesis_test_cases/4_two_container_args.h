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
// Two arguments of container type tests
//
//=============================================================================//

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
