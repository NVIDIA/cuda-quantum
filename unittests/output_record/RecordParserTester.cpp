/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/RecordLogParser.h"
#include <cudaq.h>

CUDAQ_TEST(ParserTester, checkSingleBoolean) {
  {
    const std::string log = "OUTPUT\tBOOL\ttrue\ti1\n";
    cudaq::RecordLogParser parser;
    parser.parse(log);
    auto *origBuffer = parser.getBufferPtr();
    bool value;
    std::memcpy(&value, origBuffer, sizeof(bool));
    EXPECT_EQ(true, value);
    origBuffer = nullptr;
  }
  { // no label
    const std::string log = "OUTPUT\tBOOL\tfalse\n";
    cudaq::RecordLogParser parser;
    parser.parse(log);
    auto *origBuffer = parser.getBufferPtr();
    bool value;
    std::memcpy(&value, origBuffer, sizeof(bool));
    EXPECT_EQ(false, value);
    origBuffer = nullptr;
  }
}

CUDAQ_TEST(ParserTester, checkMoreBoolean) {
  const std::string log = "OUTPUT\tBOOL\t1\ti1\n"
                          "OUTPUT\tBOOL\t0\ti1\n";
  cudaq::RecordLogParser parser;
  parser.parse(log);
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
  {
    const std::string log = "OUTPUT\tINT\t0\ti32\n"
                            "OUTPUT\tINT\t1\ti32\n"
                            "OUTPUT\tINT\t2\ti32\n";
    cudaq::RecordLogParser parser;
    parser.parse(log);
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
  { // no label
    const std::string log = "OUTPUT\tINT\t2147483647\n";
    cudaq::RecordLogParser parser;
    parser.parse(log);
    auto *origBuffer = parser.getBufferPtr();
    std::int32_t value;
    std::memcpy(&value, origBuffer, sizeof(std::int32_t));
    EXPECT_EQ(2147483647, value);
    origBuffer = nullptr;
  }
  {
    const std::string log = "OUTPUT\tINT\t127\ti8\n";
    cudaq::RecordLogParser parser;
    parser.parse(log);
    auto *origBuffer = parser.getBufferPtr();
    std::int8_t value;
    std::memcpy(&value, origBuffer, sizeof(std::int8_t));
    EXPECT_EQ(127, value);
    origBuffer = nullptr;
  }
}

CUDAQ_TEST(ParserTester, checkDoubles) {
  const std::string log = "START\n"
                          "OUTPUT\tDOUBLE\t3.14\tf64\n"
                          "OUTPUT\tDOUBLE\t2.717\tf64\n"
                          "END\t0";
  cudaq::RecordLogParser parser;
  parser.parse(log);
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
  cudaq::RecordLogParser parser;
  parser.parse(log);
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
  cudaq::RecordLogParser parser;
  parser.parse(log);
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
  cudaq::RecordLogParser parser;
  parser.parse(log);
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
  cudaq::RecordLogParser parser;
  parser.parse(log);
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
  cudaq::RecordLogParser parser;
  parser.parse(log);
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
  cudaq::RecordLogParser parser;
  parser.parse(log);
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
                          "METADATA\tqir_profiles\tbase_profile\n"
                          "OUTPUT\tARRAY\t2\tarray<i16 x 2>\n"
                          "OUTPUT\tINT\t2345\t[0]\n"
                          "OUTPUT\tINT\t4567\t[1]\n"
                          "END\t0\n"
                          "START\n"
                          "METADATA\tqir_profiles\tbase_profile\n"
                          "OUTPUT\tARRAY\t2\tarray<i16 x 2>\n"
                          "OUTPUT\tINT\t7890\t[1]\n"
                          "OUTPUT\tINT\t5678\t[0]\n"
                          "END\t0\n"
                          "START\n"
                          "METADATA\tqir_profiles\tbase_profile\n"
                          "OUTPUT\tARRAY\t2\tarray<i16 x 2>\n"
                          "OUTPUT\tINT\t1234\t[1]\n"
                          "OUTPUT\tINT\t6789\t[0]\n"
                          "END\t0";
  cudaq::RecordLogParser parser;
  parser.parse(log);
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

CUDAQ_TEST(ParserTester, checkTupleWithLayoutWithoutBool) {
  const std::string log = "OUTPUT\tTUPLE\t2\ttuple<i64, f64>\n"
                          "OUTPUT\tINT\t37\t.0\n"
                          "OUTPUT\tDOUBLE\t3.1416\t.1\n";
  std::pair<std::size_t, std::vector<std::size_t>> layout;
  layout.first = 16;
  layout.second = {0, 8};
  cudaq::RecordLogParser parser(layout);
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  EXPECT_EQ(16, bufferSize);
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  struct MyTuple {
    std::int64_t i64Val;
    double f64Val;
  };
  MyTuple *tuplePtr = reinterpret_cast<MyTuple *>(buffer);
  EXPECT_EQ(37, tuplePtr->i64Val);
  EXPECT_EQ(3.1416, tuplePtr->f64Val);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;

  // incorrect layout
  layout.first = 24;
  layout.second = {0, 4, 8};
  cudaq::RecordLogParser parser2(layout);
  EXPECT_ANY_THROW(parser2.parse(log));
}

CUDAQ_TEST(ParserTester, checkTupleWithLayoutAndBool) {
  const std::string log = "OUTPUT\tTUPLE\t3\ttuple<i1, i64, f64>\n"
                          "OUTPUT\tBOOL\ttrue\t.0\n"
                          "OUTPUT\tINT\t37\t.1\n"
                          "OUTPUT\tDOUBLE\t3.1416\t.2\n"
                          "OUTPUT\tTUPLE\t3\ttuple<i1, i64, f64>\n"
                          "OUTPUT\tBOOL\tfalse\t.0\n"
                          "OUTPUT\tINT\t42\t.1\n"
                          "OUTPUT\tDOUBLE\t4.14\t.2\n";
  std::pair<std::size_t, std::vector<std::size_t>> layout;
  layout.first = 24;
  layout.second = {0, 8, 16};
  cudaq::RecordLogParser parser(layout);
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  EXPECT_EQ(24 * 2, bufferSize);
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  struct MyTuple {
    bool boolVal;
    std::int64_t i64Val;
    double f64Val;
  };
  std::vector<MyTuple> results = {
      reinterpret_cast<MyTuple *>(span.data),
      reinterpret_cast<MyTuple *>(span.data + span.lengthInBytes)};
  EXPECT_EQ(2, results.size());
  EXPECT_EQ(true, results[0].boolVal);
  EXPECT_EQ(37, results[0].i64Val);
  EXPECT_EQ(3.1416, results[0].f64Val);
  EXPECT_EQ(false, results[1].boolVal);
  EXPECT_EQ(42, results[1].i64Val);
  EXPECT_EQ(4.14, results[1].f64Val);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkFailureCases) {
  cudaq::RecordLogParser parser;
  {
    const std::string emptyLog = "";
    cudaq::RecordLogParser parser;
    parser.parse(emptyLog);
    EXPECT_EQ(nullptr, parser.getBufferPtr());
  }
  {
    const std::string invalidLog = "INVALID\tLOG\n";
    EXPECT_ANY_THROW(parser.parse(invalidLog));
  }
  {
    const std::string invalidLog = "OUTPUT\n";
    EXPECT_ANY_THROW(parser.parse(invalidLog));
  }
  {
    const std::string invalidBool = "OUTPUT\tBOOL\t1.0\ti1\n";
    EXPECT_ANY_THROW(parser.parse(invalidBool));
  }
  {
    const std::string invalidSchema =
        "HEADER\tschema_name\tordered_and_labeled\n";
    EXPECT_ANY_THROW(parser.parse(invalidSchema));
  }
  {
    const std::string missingShotStatus = "START\n"
                                          "OUTPUT\tDOUBLE\t3.14\tf64\n"
                                          "END\n";
    EXPECT_ANY_THROW(parser.parse(missingShotStatus));
  }
  {
    const std::string failedShot = "START\n"
                                   "OUTPUT\tDOUBLE\t0.00\tf64\n"
                                   "END\t1\n";
    EXPECT_ANY_THROW(parser.parse(failedShot));
  }
  {
    const std::string insufficientData = "OUTPUT\tDOUBLE\n";
    EXPECT_ANY_THROW(parser.parse(insufficientData));
  }
  {
    const std::string missingLabel = "OUTPUT\tARRAY\t3\tarray<i32 x 2>\n"
                                     "OUTPUT\tINT\t5\n"
                                     "OUTPUT\tINT\t6\n";
    EXPECT_ANY_THROW(parser.parse(missingLabel));
  }
  {
    const std::string resultLog = "OUTPUT\tRESULT\t1\ti32\n";
    EXPECT_ANY_THROW(parser.parse(resultLog));
  }
  {
    const std::string invalidType = "OUTPUT\tFOO\t123456\ti32\n";
    EXPECT_ANY_THROW(parser.parse(invalidType));
  }
  {
    const std::string invalidType = "OUTPUT\tINT\t123456\ti128\n";
    EXPECT_ANY_THROW(parser.parse(invalidType));
  }
  {
    const std::string sizeMismatch = "OUTPUT\tARRAY\t3\tarray<i32 x 2>\n";
    EXPECT_ANY_THROW(parser.parse(sizeMismatch));
  }
  {
    const std::string invalidArrLabel = "OUTPUT\tARRAY\t3\tarray<3>\n";
    EXPECT_ANY_THROW(parser.parse(invalidArrLabel));
  }
  {
    const std::string invalidIndex = "OUTPUT\tARRAY\t2\tarray<i32 x 2>\n"
                                     "OUTPUT\tINT\t5\t[0]\n"
                                     "OUTPUT\tINT\t6\t[3]\n";
    EXPECT_ANY_THROW(parser.parse(invalidIndex));
  }
  {
    const std::string invalidTupleLabel = "OUTPUT\tTUPLE\t3\ttuple\n";
    EXPECT_ANY_THROW(parser.parse(invalidTupleLabel));
  }
  {
    const std::string sizeMismatch = "OUTPUT\tTUPLE\t3\ttuple<i32, f64>\n";
    EXPECT_ANY_THROW(parser.parse(sizeMismatch));
  }
  {
    const std::string invalidLabel = "OUTPUT\tTUPLE\t2\ttuple<i32, f64>\n"
                                     "OUTPUT\tINT\t5\t.0\n"
                                     "OUTPUT\tDOUBLE\t6.0\t[1]\n";
    EXPECT_ANY_THROW(parser.parse(invalidLabel));
  }
  {
    const std::string missingIndex = "OUTPUT\tTUPLE\t2\ttuple<i32, f64>\n"
                                     "OUTPUT\tINT\t5\n";
    EXPECT_ANY_THROW(parser.parse(missingIndex));
  }
}
