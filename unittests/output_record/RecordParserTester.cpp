/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/RecordLogDecoder.h"
#include <cudaq.h>

CUDAQ_TEST(ParserTester, checkSingleBoolean) {
  const std::string log = "OUTPUT\tBOOL\ttrue\ti1\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  bool value;
  std::memcpy(&value, origBuffer, sizeof(bool));
  EXPECT_EQ(true, value);
}

CUDAQ_TEST(ParserTester, checkIntegers) {
  const std::string log = "OUTPUT\tINT\t0\ti32\n"
                          "OUTPUT\tINT\t1\ti32\n"
                          "OUTPUT\tINT\t2\ti32\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  EXPECT_EQ(3, bufferSize / sizeof(int));
  int *buffer = static_cast<int *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(i, buffer[i]);
}

CUDAQ_TEST(ParserTester, checkDoubles) {
  const std::string log = "START\n"
                          "OUTPUT\tDOUBLE\t3.14\tf64\n"
                          "OUTPUT\tDOUBLE\t2.717\tf64\n"
                          "END\t0";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  EXPECT_EQ(2, bufferSize / sizeof(double));
  double *buffer = static_cast<double *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  EXPECT_EQ(3.14, buffer[0]);
  EXPECT_EQ(2.717, buffer[1]);
}
