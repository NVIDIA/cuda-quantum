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

CUDAQ_TEST(ParserTester, checkArrayOrdered) {
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

CUDAQ_TEST(ParserTester, checkArrayLabeled) {
  const std::string log = "HEADER\tschema_name\tlabeled\n"
                          "HEADER\tschema_version\t1.0\n"
                          "START\n"
                          "OUTPUT\tARRAY\t3\tarray<3 x i32>\n"
                          "OUTPUT\tINT\t13\t[0]\n"
                          "OUTPUT\tINT\t42\t[1]\n"
                          "OUTPUT\tINT\t61\t[2]\n"
                          "END\t0";
  cudaq::RecordParser parser;
  auto results = parser.parse(log);
  EXPECT_EQ(1, results.size());
  auto count = results[0].size / sizeof(int);
  EXPECT_EQ(3, count);
  auto *ptr = static_cast<int *>(results[0].buffer);
  std::vector<int> got(count);
  for (int i = 0; i < count; ++i)
    got[i] = ptr[i];
  EXPECT_EQ(13, got[0]);
  EXPECT_EQ(61, got[2]);
}

CUDAQ_TEST(ParserTester, checkMultipleShots) {
  const std::string log = "HEADER\tschema_name\tlabeled\n"
                          "START\n"
                          "OUTPUT\tARRAY\t2\tarray<2 x i1>\n"
                          "OUTPUT\tBOOL\ttrue\t[0]\n"
                          "OUTPUT\tBOOL\ttrue\t[1]\n"
                          "END\t0\n"
                          "START\n"
                          "OUTPUT\tARRAY\t2\tarray<2 x i1>\n"
                          "OUTPUT\tBOOL\tfalse\t[1]\n"
                          "OUTPUT\tBOOL\ttrue\t[0]\n"
                          "END\t0\n"
                          "START\n"
                          "OUTPUT\tARRAY\t2\tarray<2 x i1>\n"
                          "OUTPUT\tBOOL\ttrue\t[0]\n"
                          "OUTPUT\tBOOL\tfalse\t[1]\n"
                          "END\t0";
  cudaq::RecordParser parser;
  auto results = parser.parse(log);
  EXPECT_EQ(3, results.size());
  auto count = results[1].size / sizeof(bool);
  EXPECT_EQ(2, count);
  auto *ptr = static_cast<bool *>(results[1].buffer);
  std::vector<bool> got(count);
  for (int i = 0; i < count; ++i)
    got[i] = ptr[i];
  EXPECT_EQ(true, got[0]);
  EXPECT_EQ(false, got[1]);
}

CUDAQ_TEST(ParserTester, checkTuple) {
  const std::string log = "HEADER\tschema_name\tlabeled\n"
                          "START\n"
                          "OUTPUT\tTUPLE\t3\ttuple<i1, i32, f64>\n"
                          "OUTPUT\tBOOL\ttrue\t.0\n"
                          "OUTPUT\tINT\t37\t.1\n"
                          "OUTPUT\tDOUBLE\t3.1416\t.2\n"
                          "END\t0";
  cudaq::RecordParser parser;
  try {
    parser.parse(log);
    FAIL();
  } catch (std::exception &ex) {
    EXPECT_STREQ("This type is not yet supported", ex.what());
  }
}
