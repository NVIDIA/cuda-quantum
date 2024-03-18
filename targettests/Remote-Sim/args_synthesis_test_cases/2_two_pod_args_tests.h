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
// Two arguments of POD type tests
//
//=============================================================================//
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
