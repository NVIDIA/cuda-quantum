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
  const std::string log = "OUTPUT\tBOOL\ttrue";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  // EXPECT_EQ(1, results.size());
  // EXPECT_EQ(true, *static_cast<bool *>(results[0].buffer));
}

// CUDAQ_TEST(ParserTester, checkIntegers) {
//   const std::string log = "OUTPUT\tINT\t0\n"
//                           "OUTPUT\tINT\t1\n"
//                           "OUTPUT\tINT\t2\n";
//   cudaq::RecordLogDecoder parser;
//   auto results = parser.decode(log);
//   EXPECT_EQ(3, results.size());
//   EXPECT_EQ(2, *static_cast<int *>(results[2].buffer));
// }
