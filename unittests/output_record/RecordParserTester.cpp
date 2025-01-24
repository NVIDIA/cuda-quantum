/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/RecordParser.h"
#include <cudaq.h>

CUDAQ_TEST(ParserTester, checkSingleBoolean) {
  const std::string log = "OUTPUT\tBOOL\ttrue";
  cudaq::RecordParser parser;
  auto results = parser.parse(log);
  EXPECT_EQ(1, results.size());
}

CUDAQ_TEST(ParserTester, checkIntegers) {
  const std::string log = "OUTPUT\tINT\t0\n"
                          "OUTPUT\tINT\t1\n"
                          "OUTPUT\tINT\t2\n";
  cudaq::RecordParser parser;
  auto results = parser.parse(log);
  EXPECT_EQ(3, results.size());
}