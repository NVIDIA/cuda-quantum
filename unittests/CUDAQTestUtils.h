/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Every test will need at least these headers
#include <cudaq.h>
#include <gtest/gtest.h>

/// This file provides some preprocessor macros that will
/// let us prepend the TEST_SUITE name in TEST(TEST_SUITE, TEST_NAME)
/// with the NVQIR backend name, for all GTEST TEST() macros.
#define LOCAL_NAME NVQIR_BACKEND_NAME
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define CUDAQ_TEST(TEST_SUITE_NAME, TEST_NAME)                                 \
  TEST(CONCAT(CONCAT(LOCAL_NAME, _), TEST_SUITE_NAME), TEST_NAME)
#define CUDAQ_TEST_F(TEST_SUITE_NAME, TEST_NAME)                               \
  TEST_F(TEST_SUITE_NAME, CONCAT(TEST_NAME, CONCAT(_, LOCAL_NAME)))
