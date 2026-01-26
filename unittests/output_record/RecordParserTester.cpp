/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
  EXPECT_ANY_THROW(parser.parse("log"));
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

CUDAQ_TEST(ParserTester, checkResultType) {
  const std::string log =
      "HEADER\tschema_id\tlabeled\nHEADER\tschema_version\t1."
      "0\nSTART\nMETADATA\tentry_point\nMETADATA\tqir_profiles\tadaptive_"
      "profile\nMETADATA\trequired_num_qubits\t10\nMETADATA\trequired_num_"
      "results\t10\nOUTPUT\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nOUTPU"
      "T\tRESULT\t0\tr00002\nOUTPUT\tRESULT\t0\tr00003\nOUTPUT\tRESULT\t0\tr000"
      "04\nOUTPUT\tRESULT\t0\tr00005\nOUTPUT\tRESULT\t0\tr00006\nOUTPUT\tRESULT"
      "\t0\tr00007\nOUTPUT\tRESULT\t0\tr00008\nOUTPUT\tRESULT\t0\tr00009\nEND\t"
      "0\nSTART\nOUTPUT\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nOUTPUT\t"
      "RESULT\t1\tr00002\nOUTPUT\tRESULT\t1\tr00003\nOUTPUT\tRESULT\t1\tr00004"
      "\nOUTPUT\tRESULT\t1\tr00005\nOUTPUT\tRESULT\t1\tr00006\nOUTPUT\tRESULT\t"
      "1\tr00007\nOUTPUT\tRESULT\t1\tr00008\nOUTPUT\tRESULT\t1\tr00009\nEND\t0"
      "\nSTART\nOUTPUT\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nOUTPUT\tR"
      "ESULT\t1\tr00002\nOUTPUT\tRESULT\t1\tr00003\nOUTPUT\tRESULT\t1\tr00004\n"
      "OUTPUT\tRESULT\t1\tr00005\nOUTPUT\tRESULT\t1\tr00006\nOUTPUT\tRESULT\t1"
      "\tr00007\nOUTPUT\tRESULT\t1\tr00008\nOUTPUT\tRESULT\t1\tr00009\nEND\t0\n"
      "START\nOUTPUT\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nOUTPUT\tRES"
      "ULT\t1\tr00002\nOUTPUT\tRESULT\t1\tr00003\nOUTPUT\tRESULT\t1\tr00004\nOU"
      "TPUT\tRESULT\t1\tr00005\nOUTPUT\tRESULT\t1\tr00006\nOUTPUT\tRESULT\t1\tr"
      "00007\nOUTPUT\tRESULT\t1\tr00008\nOUTPUT\tRESULT\t1\tr00009\nEND\t0\nSTA"
      "RT\nOUTPUT\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nOUTPUT\tRESULT"
      "\t0\tr00002\nOUTPUT\tRESULT\t0\tr00003\nOUTPUT\tRESULT\t0\tr00004\nOUTPU"
      "T\tRESULT\t0\tr00005\nOUTPUT\tRESULT\t0\tr00006\nOUTPUT\tRESULT\t0\tr000"
      "07\nOUTPUT\tRESULT\t0\tr00008\nOUTPUT\tRESULT\t0\tr00009\nEND\t0\nSTART"
      "\nOUTPUT\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nOUTPUT\tRESULT\t"
      "1\tr00002\nOUTPUT\tRESULT\t1\tr00003\nOUTPUT\tRESULT\t1\tr00004\nOUTPUT"
      "\tRESULT\t1\tr00005\nOUTPUT\tRESULT\t1\tr00006\nOUTPUT\tRESULT\t1\tr0000"
      "7\nOUTPUT\tRESULT\t1\tr00008\nOUTPUT\tRESULT\t1\tr00009\nEND\t0\nSTART\n"
      "OUTPUT\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nOUTPUT\tRESULT\t0"
      "\tr00002\nOUTPUT\tRESULT\t0\tr00003\nOUTPUT\tRESULT\t0\tr00004\nOUTPUT\t"
      "RESULT\t0\tr00005\nOUTPUT\tRESULT\t0\tr00006\nOUTPUT\tRESULT\t0\tr00007"
      "\nOUTPUT\tRESULT\t0\tr00008\nOUTPUT\tRESULT\t0\tr00009\nEND\t0\nSTART\nO"
      "UTPUT\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nOUTPUT\tRESULT\t1\t"
      "r00002\nOUTPUT\tRESULT\t1\tr00003\nOUTPUT\tRESULT\t1\tr00004\nOUTPUT\tRE"
      "SULT\t1\tr00005\nOUTPUT\tRESULT\t1\tr00006\nOUTPUT\tRESULT\t1\tr00007\nO"
      "UTPUT\tRESULT\t1\tr00008\nOUTPUT\tRESULT\t1\tr00009\nEND\t0\nSTART\nOUTP"
      "UT\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nOUTPUT\tRESULT\t0\tr00"
      "002\nOUTPUT\tRESULT\t0\tr00003\nOUTPUT\tRESULT\t0\tr00004\nOUTPUT\tRESUL"
      "T\t0\tr00005\nOUTPUT\tRESULT\t0\tr00006\nOUTPUT\tRESULT\t0\tr00007\nOUTP"
      "UT\tRESULT\t0\tr00008\nOUTPUT\tRESULT\t0\tr00009\nEND\t0\nSTART\nOUTPUT"
      "\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nOUTPUT\tRESULT\t1\tr0000"
      "2\nOUTPUT\tRESULT\t1\tr00003\nOUTPUT\tRESULT\t1\tr00004\nOUTPUT\tRESULT"
      "\t1\tr00005\nOUTPUT\tRESULT\t1\tr00006\nOUTPUT\tRESULT\t1\tr00007\nOUTPU"
      "T\tRESULT\t1\tr00008\nOUTPUT\tRESULT\t1\tr00009\nEND\t0\n";

  cudaq::RecordLogParser parser;
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  // This is parsed as a vector of bool vectors
  std::vector<std::vector<bool>> results = {
      reinterpret_cast<std::vector<bool> *>(span.data),
      reinterpret_cast<std::vector<bool> *>(span.data + span.lengthInBytes)};
  // 10 shots
  EXPECT_EQ(10, results.size());
  for (const auto &result : results) {
    // 10 measured bits each
    EXPECT_EQ(10, result.size());
    // This is GHZ result, all bits should be equal
    EXPECT_TRUE(std::all_of(result.begin(), result.end(),
                            [&result](bool bit) { return bit == result[0]; }));
  }
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkResultTypeWithRegisterName) {
  const std::string log =
      "HEADER\tschema_id\tlabeled\nHEADER\tschema_version\t1."
      "0\nSTART\nMETADATA\tentry_point\nMETADATA\tqir_profiles\tadaptive_"
      "profile\nMETADATA\trequired_num_qubits\t1\nMETADATA\trequired_num_"
      "results\t1\nOUTPUT\tRESULT\t1\tresult\nEND\t0\nSTART\nOUTPUT\tRESULT\t1"
      "\tresult\nEND\t0\nSTART\nOUTPUT\tRESULT\t1\tresult\nEND\t0\nSTART\nOUTPU"
      "T\tRESULT\t1\tresult\nEND\t0\nSTART\nOUTPUT\tRESULT\t1\tresult\nEND\t0\n"
      "START\nOUTPUT\tRESULT\t1\tresult\nEND\t0\nSTART\nOUTPUT\tRESULT\t1\tresu"
      "lt\nEND\t0\nSTART\nOUTPUT\tRESULT\t1\tresult\nEND\t0\nSTART\nOUTPUT\tRES"
      "ULT\t1\tresult\nEND\t0\nSTART\nOUTPUT\tRESULT\t1\tresult\nEND\t0\n";
  cudaq::RecordLogParser parser;
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  // This is parsed as a vector of bool vectors
  std::vector<std::vector<bool>> results = {
      reinterpret_cast<std::vector<bool> *>(span.data),
      reinterpret_cast<std::vector<bool> *>(span.data + span.lengthInBytes)};
  // 10 shots
  EXPECT_EQ(10, results.size());
  for (const auto &result : results) {
    // 1 measured bits each
    EXPECT_EQ(1, result.size());
    EXPECT_TRUE(result[0]); // all should be 1
  }
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkFailedShot_0) {
  const std::string log = "START\n"
                          "OUTPUT\tDOUBLE\t0.00\tf64\n"
                          "END\t1\n";
  cudaq::RecordLogParser parser;
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  // No data should be recorded for a failed shot
  EXPECT_EQ(0, bufferSize);
  EXPECT_EQ(nullptr, origBuffer);
}

CUDAQ_TEST(ParserTester, checkFailedShot_1) {
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
                          "END\t255\n"
                          "START\n"
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
  EXPECT_EQ(2, results.size());
  EXPECT_EQ(2, results[0].size());
  EXPECT_EQ(2, results[1].size());
  EXPECT_EQ(2345, results[0][0]);
  EXPECT_EQ(4567, results[0][1]);
  EXPECT_EQ(6789, results[1][0]);
  EXPECT_EQ(1234, results[1][1]);
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkFailedShot_2) {
  std::string log =
      "HEADER\tschema_id\tlabeled\nHEADER\tschema_version\t1."
      "0\nSTART\nMETADATA\tentry_point\nMETADATA\toutput_labeling_"
      "schema\tschema_id\nMETADATA\tqir_profiles\tadaptive_"
      "profile\nMETADATA\trequired_num_qubits\t2\nMETADATA\trequired_num_"
      "results\t2\nOUTPUT\tINT\t0\ti64\nEND\t1\nSTART\nOUTPUT\tINT\t2\ti64\nE"
      "ND\t0\nSTART\nOUTPUT\tINT\t0\ti64\nEND\t0\nSTART\nOUTPUT\tINT\t0\ti64"
      "\nEND\t5\nSTART\nOUTPUT\tINT\t0\ti64\nEND\t0\nSTART\nOUTPUT\tINT\t2\ti"
      "64\nEND\t127\nSTART\nOUTPUT\tINT\t0\ti64\nEND\t0";

  cudaq::RecordLogParser parser;
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  std::vector<std::int64_t> results = {
      reinterpret_cast<std::int64_t *>(span.data),
      reinterpret_cast<std::int64_t *>(span.data + span.lengthInBytes)};
  // Only 4 successful shots
  EXPECT_EQ(4, results.size());
  for (const auto &result : results) {
    // Result should be either 0 or 2
    EXPECT_TRUE(result == 0 || result == 2);
  }
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkFailedShot_3) {
  std::string log =
      "HEADER\tschema_id\tlabeled\nHEADER\tschema_version\t1."
      "0\nSTART\nMETADATA\tentry_point\nMETADATA\toutput_labeling_"
      "schema\tschema_id\nMETADATA\tqir_profiles\tadaptive_"
      "profile\nMETADATA\trequired_num_qubits\t2\nMETADATA\trequired_num_"
      "results\t2\nOUTPUT\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nEND\t0"
      "\nSTART\nOUTPUT\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nEND\t2\nS"
      "TART\nOUTPUT\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nEND\t3\nSTAR"
      "T\nOUTPUT\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nEND\t0\nSTART\n"
      "OUTPUT\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nEND\t0\nSTART\nOUT"
      "PUT\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nEND\t0\nSTART\nOUTPUT"
      "\tRESULT\t0\tr00000\nOUTPUT\tRESULT\t0\tr00001\nEND\t127\nSTART\nOUTPUT"
      "\tRESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nEND\t0\nSTART\nOUTPUT\tR"
      "ESULT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nEND\t0\nSTART\nOUTPUT\tRESU"
      "LT\t1\tr00000\nOUTPUT\tRESULT\t1\tr00001\nEND\t64";

  cudaq::RecordLogParser parser;
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  std::vector<std::vector<bool>> results = {
      reinterpret_cast<std::vector<bool> *>(span.data),
      reinterpret_cast<std::vector<bool> *>(span.data + span.lengthInBytes)};
  // Only 6 successful shots
  EXPECT_EQ(6, results.size());
  for (const auto &result : results) {
    // Result should be either 00 or 11
    EXPECT_TRUE((result[0] == false && result[1] == false) ||
                (result[0] == true && result[1] == true));
  }
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkOrder) {
  const std::string log =
      "HEADER\tschema_id\tlabeled\n"
      "HEADER\tschema_version\t1.0\n"
      "START\n"
      "METADATA\tentry_point\n"
      "METADATA\toutput_labeling_schema\tschema_id\n"
      "METADATA\toutput_names\t[[[0,[0,\"r00000\"]],[1,[1,\"r00001\"]],[2,[2,"
      "\"r00002\"]],[3,[3,\"r00003\"]],[4,[4,\"r00004\"]],[5,[5,\"r00005\"]],["
      "6,[6,\"r00006\"]],[7,[7,\"r00007\"]]]]\n"
      "METADATA\tqir_profiles\tbase_profile\n"
      "METADATA\trequiredQubits\t8\n"
      "METADATA\trequiredResults\t8\n"
      "OUTPUT\tRESULT\t1\tr00003\n"
      "OUTPUT\tRESULT\t0\tr00002\n"
      "OUTPUT\tRESULT\t0\tr00000\n"
      "OUTPUT\tRESULT\t1\tr00004\n"
      "OUTPUT\tRESULT\t1\tr00006\n"
      "OUTPUT\tRESULT\t1\tr00001\n"
      "OUTPUT\tRESULT\t0\tr00007\n"
      "OUTPUT\tRESULT\t0\tr00005\n"
      "END\t0";

  cudaq::RecordLogParser parser;
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  // This is parsed as a vector of bool vectors
  std::vector<std::vector<bool>> results = {
      reinterpret_cast<std::vector<bool> *>(span.data),
      reinterpret_cast<std::vector<bool> *>(span.data + span.lengthInBytes)};
  // 1 shot
  EXPECT_EQ(1, results.size());
  for (const auto &result : results) {
    // 8 measured bits each
    EXPECT_EQ(8, result.size());
    EXPECT_EQ(false, result[0]); // r00000
    EXPECT_EQ(true, result[1]);  // r00001
    EXPECT_EQ(false, result[2]); // r00002
    EXPECT_EQ(true, result[3]);  // r00003
    EXPECT_EQ(true, result[4]);  // r00004
    EXPECT_EQ(false, result[5]); // r00005
    EXPECT_EQ(true, result[6]);  // r00006
    EXPECT_EQ(false, result[7]); // r00007
  }

  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkNamedResults) {
  const std::string log =
      "HEADER\tschema_id\tlabeled\n"
      "HEADER\tschema_version\t1.0\n"
      "START\n"
      "METADATA\tentry_point\n"
      "METADATA\toutput_labeling_schema\tschema_id\n"
      "METADATA\toutput_names\t[[[0,[1,\"result%0\"]],[1,[2,\"result%1\"]],[2,["
      "3,\"result%2\"]],[3,[4,\"result%3\"]]]]\n"
      "METADATA\tqir_profiles\tadaptive_profile\n"
      "METADATA\trequired_num_qubits\t5\n"
      "METADATA\trequired_num_results\t4\n"
      "OUTPUT\tRESULT\t1\tresult%0\n"
      "OUTPUT\tRESULT\t1\tresult%1\n"
      "OUTPUT\tRESULT\t1\tresult%2\n"
      "OUTPUT\tRESULT\t0\tresult%3\n"
      "END\t0\n"
      "START\n"
      "OUTPUT\tRESULT\t1\tresult%0\n"
      "OUTPUT\tRESULT\t1\tresult%1\n"
      "OUTPUT\tRESULT\t0\tresult%3\n"
      "OUTPUT\tRESULT\t1\tresult%2\n"
      "END\t0\n";

  cudaq::RecordLogParser parser;
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  // This is parsed as a vector of bool vectors
  std::vector<std::vector<bool>> results = {
      reinterpret_cast<std::vector<bool> *>(span.data),
      reinterpret_cast<std::vector<bool> *>(span.data + span.lengthInBytes)};
  // 2 shots
  EXPECT_EQ(2, results.size());
  for (const auto &result : results) {
    // 4 measured bits each
    EXPECT_EQ(4, result.size());
    EXPECT_EQ(true, result[0]);  // result%0
    EXPECT_EQ(true, result[1]);  // result%1
    EXPECT_EQ(true, result[2]);  // result%2
    EXPECT_EQ(false, result[3]); // result%3
  }
  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}

CUDAQ_TEST(ParserTester, checkResultTypeWithArray) {
  const std::string log =
      "HEADER\tschema_id\tlabeled\n"
      "HEADER\tschema_version\t1.0\n"
      "START\n"
      "METADATA\tentry_point\n"
      "METADATA\toutput_labeling_schema\tschema_id\n"
      "METADATA\toutput_names\t[[[0,[0,\"r00000\"]],[1,[1,\"r00001\"]]]]\n"
      "METADATA\tqir_profiles\tadaptive_profile\n"
      "METADATA\trequired_num_qubits\t2\n"
      "METADATA\trequired_num_results\t2\n"
      "OUTPUT\tARRAY\t2\tarray<i1 x 2>\n"
      "OUTPUT\tRESULT\t1\tr00000\n"
      "OUTPUT\tRESULT\t1\tr00001\n"
      "END\t0\n"
      "START\n"
      "OUTPUT\tARRAY\t2\tarray<i1 x 2>\n"
      "OUTPUT\tRESULT\t1\tr00000\n"
      "OUTPUT\tRESULT\t1\tr00001\n"
      "END\t0\n"
      "START\n"
      "OUTPUT\tARRAY\t2\tarray<i1 x 2>\n"
      "OUTPUT\tRESULT\t0\tr00000\n"
      "OUTPUT\tRESULT\t0\tr00001\n"
      "END\t0\n"
      "START\n"
      "OUTPUT\tARRAY\t2\tarray<i1 x 2>\n"
      "OUTPUT\tRESULT\t1\tr00000\n"
      "OUTPUT\tRESULT\t1\tr00001\n"
      "END\t0\n";

  cudaq::RecordLogParser parser;
  parser.parse(log);
  auto *origBuffer = parser.getBufferPtr();
  std::size_t bufferSize = parser.getBufferSize();
  char *buffer = static_cast<char *>(malloc(bufferSize));
  std::memcpy(buffer, origBuffer, bufferSize);
  cudaq::details::RunResultSpan span = {buffer, bufferSize};
  // This is parsed as a vector of bool vectors
  std::vector<std::vector<bool>> results = {
      reinterpret_cast<std::vector<bool> *>(span.data),
      reinterpret_cast<std::vector<bool> *>(span.data + span.lengthInBytes)};

  // 4 shots
  EXPECT_EQ(4, results.size());
  for (const auto &result : results) {
    // 2 measured bits each
    EXPECT_EQ(2, result.size());
  }
  // 1st shot: 1, 1
  EXPECT_EQ(1, results[0][0]);
  EXPECT_EQ(1, results[0][1]);
  // 2nd shot: 1, 1
  EXPECT_EQ(1, results[1][0]);
  EXPECT_EQ(1, results[1][1]);
  // 3rd shot: 0, 0
  EXPECT_EQ(0, results[2][0]);
  EXPECT_EQ(0, results[2][1]);
  // 4th shot: 1, 1
  EXPECT_EQ(1, results[3][0]);
  EXPECT_EQ(1, results[3][1]);

  free(buffer);
  buffer = nullptr;
  origBuffer = nullptr;
}
