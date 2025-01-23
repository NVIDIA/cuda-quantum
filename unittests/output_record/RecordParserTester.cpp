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
  EXPECT_EQ(true, *static_cast<bool *>(results[0].buffer));
}

CUDAQ_TEST(ParserTester, checkIntegers) {
  const std::string log = "OUTPUT\tINT\t0\n"
                          "OUTPUT\tINT\t1\n"
                          "OUTPUT\tINT\t2\n";
  cudaq::RecordParser parser;
  auto results = parser.parse(log);
  EXPECT_EQ(3, results.size());
  EXPECT_EQ(2, *static_cast<int *>(results[2].buffer));
}

CUDAQ_TEST(ParserTester, checkArray) {
  const std::string log = "OUTPUT\tARRAY\t4\n"
                          "OUTPUT\tINT\t0\n"
                          "OUTPUT\tINT\t1\n"
                          "OUTPUT\tINT\t1\n"
                          "OUTPUT\tINT\t0";
  cudaq::RecordParser parser;
  auto results = parser.parse(log);
  EXPECT_EQ(1, results.size());
  auto count = results[0].size / sizeof(int);
  EXPECT_EQ(4, count);
  auto *ptr = static_cast<int *>(results[0].buffer);
  std::vector<int> got(count);
  for (int i = 0; i < count; ++i)
    got[i] = ptr[i];
  EXPECT_EQ(1, got[2]);
  EXPECT_EQ(0, got[3]);
}