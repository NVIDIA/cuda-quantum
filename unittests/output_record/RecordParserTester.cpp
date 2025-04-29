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
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkMoreBoolean) {
  const std::string log = "OUTPUT\tBOOL\t1\ti1\n"
                          "OUTPUT\tBOOL\t0\ti1\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  EXPECT_EQ(2, bufferSize / sizeof(char));
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  EXPECT_EQ(true, buffer[0]);
  EXPECT_EQ(false, buffer[1]);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
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
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
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
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkArrayOrdered) {
  const std::string log = "OUTPUT\tARRAY\t2\n"
                          "OUTPUT\tINT\t13\ti32\n"
                          "OUTPUT\tINT\t71\ti32\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  EXPECT_EQ(2, bufferSize / sizeof(int));
  int *buffer = static_cast<int *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  EXPECT_EQ(13, buffer[0]);
  EXPECT_EQ(71, buffer[1]);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkArrayLabeled) {
  const std::string log = "OUTPUT\tARRAY\t3\tarray<i32 x 3>\n"
                          "OUTPUT\tINT\t5\t[0]\n"
                          "OUTPUT\tINT\t6\t[1]\n"
                          "OUTPUT\tINT\t7\t[2]\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  std::vector<std::vector<int>> results = {
      reinterpret_cast<std::vector<int> *>(span.data),
      reinterpret_cast<std::vector<int> *>(span.data + span.lengthInBytes)};
  EXPECT_EQ(1, results.size());
  EXPECT_EQ(3, results[0].size());
  EXPECT_EQ(5, results[0][0]);
  EXPECT_EQ(6, results[0][1]);
  EXPECT_EQ(7, results[0][2]);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkArrayIntMultiShot) {
  const std::string log = "OUTPUT\tARRAY\t2\tarray<i32 x 2>\n"
                          "OUTPUT\tINT\t42\t[0]\n"
                          "OUTPUT\tINT\t-13\t[1]\n"
                          "OUTPUT\tARRAY\t2\tarray<i32 x 2>\n"
                          "OUTPUT\tINT\t42\t[0]\n"
                          "OUTPUT\tINT\t-13\t[1]\n"
                          "OUTPUT\tARRAY\t2\tarray<i32 x 2>\n"
                          "OUTPUT\tINT\t42\t[0]\n"
                          "OUTPUT\tINT\t-13\t[1]\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  std::vector<std::vector<int>> results = {
      reinterpret_cast<std::vector<int> *>(span.data),
      reinterpret_cast<std::vector<int> *>(span.data + span.lengthInBytes)};
  EXPECT_EQ(3, results.size());
  EXPECT_EQ(2, results[0].size());
  EXPECT_EQ(2, results[1].size());
  EXPECT_EQ(2, results[2].size());
  EXPECT_EQ(42, results[0][0]);
  EXPECT_EQ(-13, results[0][1]);
  EXPECT_EQ(42, results[1][0]);
  EXPECT_EQ(-13, results[1][1]);
  EXPECT_EQ(42, results[2][0]);
  EXPECT_EQ(-13, results[2][1]);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkArrayDoubleMultiShot) {
  const std::string log = "OUTPUT\tARRAY\t3\tarray<f64 x 3>\n"
                          "OUTPUT\tDOUBLE\t3.14159\t[0]\n"
                          "OUTPUT\tDOUBLE\t2.71828\t[1]\n"
                          "OUTPUT\tDOUBLE\t6.62607\t[2]\n"
                          "OUTPUT\tARRAY\t3\tarray<f64 x 3>\n"
                          "OUTPUT\tDOUBLE\t3.14159\t[0]\n"
                          "OUTPUT\tDOUBLE\t2.71828\t[1]\n"
                          "OUTPUT\tDOUBLE\t6.62607\t[2]\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  std::vector<std::vector<double>> results = {
      reinterpret_cast<std::vector<double> *>(span.data),
      reinterpret_cast<std::vector<double> *>(span.data + span.lengthInBytes)};
  EXPECT_EQ(2, results.size());
  EXPECT_EQ(3, results[0].size());
  EXPECT_EQ(3, results[1].size());
  double tol = 1e-3;
  EXPECT_NEAR(3.14159, results[0][0], tol);
  EXPECT_NEAR(2.71828, results[0][1], tol);
  EXPECT_NEAR(6.62607, results[0][2], tol);
  EXPECT_NEAR(3.14159, results[1][0], tol);
  EXPECT_NEAR(2.71828, results[1][1], tol);
  EXPECT_NEAR(6.62607, results[1][2], tol);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkTupleOrdered) {
  const std::string log = "OUTPUT\tTUPLE\t2\n"
                          "OUTPUT\tINT\t561\ti32\n"
                          "OUTPUT\tBOOL\tfalse\ti1\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  int tuple_0;
  bool tuple_1;
  EXPECT_EQ(bufferSize, sizeof(tuple_0) + sizeof(tuple_1));
  std::memcpy(&tuple_0, buffer, sizeof(tuple_0));
  std::memcpy(&tuple_1, buffer + sizeof(tuple_0), sizeof(tuple_1));
  EXPECT_EQ(561, tuple_0);
  EXPECT_EQ(false, tuple_1);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkTupleLabeled) {
  const std::string log = "OUTPUT\tTUPLE\t3\ttuple<i1, i32, f64>\n"
                          "OUTPUT\tBOOL\ttrue\t.0\n"
                          "OUTPUT\tINT\t37\t.1\n"
                          "OUTPUT\tDOUBLE\t3.1416\t.2\n";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  bool tuple_0;
  int tuple_1;
  double tuple_2;
  EXPECT_EQ(bufferSize, sizeof(tuple_0) + sizeof(tuple_1) + sizeof(tuple_2));
  std::memcpy(&tuple_0, buffer, sizeof(tuple_0));
  std::memcpy(&tuple_1, buffer + sizeof(tuple_0), sizeof(tuple_1));
  std::memcpy(&tuple_2, buffer + sizeof(tuple_0) + sizeof(tuple_1),
              sizeof(tuple_2));
  EXPECT_EQ(true, tuple_0);
  EXPECT_EQ(37, tuple_1);
  EXPECT_EQ(3.1416, tuple_2);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkMultipleShots) {
  const std::string log = "HEADER\tschema_name\tlabeled\n"
                          "START\n"
                          "OUTPUT\tARRAY\t2\tarray<i16 x 2>\n"
                          "OUTPUT\tINT\t2345\t[0]\n"
                          "OUTPUT\tINT\t4567\t[1]\n"
                          "END\t0\n"
                          "START\n"
                          "OUTPUT\tARRAY\t2\tarray<i16 x 2>\n"
                          "OUTPUT\tINT\t7890\t[1]\n"
                          "OUTPUT\tINT\t5678\t[0]\n"
                          "END\t0\n"
                          "START\n"
                          "OUTPUT\tARRAY\t2\tarray<i16 x 2>\n"
                          "OUTPUT\tINT\t1234\t[1]\n"
                          "OUTPUT\tINT\t6789\t[0]\n"
                          "END\t0";
  cudaq::RecordLogDecoder parser;
  parser.decode(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  std::vector<std::vector<std::int16_t>> results = {
      reinterpret_cast<std::vector<std::int16_t> *>(span.data),
      reinterpret_cast<std::vector<std::int16_t> *>(span.data +
                                                    span.lengthInBytes)};
  EXPECT_EQ(3, results.size());
  EXPECT_EQ(2, results[0].size());
  EXPECT_EQ(2, results[1].size());
  EXPECT_EQ(2, results[2].size());
  EXPECT_EQ(2345, results[0][0]);
  EXPECT_EQ(4567, results[0][1]);
  EXPECT_EQ(5678, results[1][0]);
  EXPECT_EQ(7890, results[1][1]);
  EXPECT_EQ(6789, results[2][0]);
  EXPECT_EQ(1234, results[2][1]);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}
