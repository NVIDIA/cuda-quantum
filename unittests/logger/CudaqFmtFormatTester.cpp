/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Tests for cudaq_fmt::format, the entry point shared with the CUDAQ_* logging
// macros. Tests are grouped by FormatArgument appender path:
//   Group A - by-value (default appendArgument<T> template)
//   Group B - by-reference (CUDAQ_INSTANTIATE_FORMAT_REF_ARGUMENT
//   specializations) Group C - C-string (dedicated appendCString via const
//   char*/char* ctors)
// Group D contains cross-cutting behaviour and Group E is a single
// linker-coverage smoke test for every type in the instantiation tables of
// runtime/logger/logger.cpp.

#include "cudaq/runtime/logger/cudaq_fmt.h"

#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <map>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Group A: by-value path (default appendArgument<T> template)
// ---------------------------------------------------------------------------

TEST(CudaqFmtFormat, BoolFormats) {
  EXPECT_EQ(cudaq_fmt::format("{}", true), "true");
  EXPECT_EQ(cudaq_fmt::format("{}", false), "false");
}

TEST(CudaqFmtFormat, IntegerFormats) {
  EXPECT_EQ(cudaq_fmt::format("{}", -7), "-7");
}

TEST(CudaqFmtFormat, FloatingPointFormats) {
  EXPECT_EQ(cudaq_fmt::format("{:.3f}", 3.14159265), "3.142");
}

TEST(CudaqFmtFormat, ComplexFormats) {
  // Custom formatter in common/FmtCore.h: "{real}{+/-}{|imag|}j".
  EXPECT_EQ(cudaq_fmt::format("{}", std::complex<double>{1.5, -2.5}),
            "1.5-2.5j");
}

TEST(CudaqFmtFormat, StringViewFormats) {
  EXPECT_EQ(cudaq_fmt::format("{}", std::string_view{"hello"}), "hello");
}

TEST(CudaqFmtFormat, VoidPointerFormats) {
  int sentinel = 0;
  void *vp = &sentinel;
  const std::string out = cudaq_fmt::format("{:p}", vp);
  // fmt prints pointers as "0x..." (or "(nil)" for null); the address here is
  // non-null so the output must be non-empty.
  EXPECT_FALSE(out.empty());
}
TEST(CudaqFmtFormat, ChronoMillisecondsFormats) {
  EXPECT_EQ(cudaq_fmt::format("{}", std::chrono::milliseconds{1500}), "1500ms");
}

// ---------------------------------------------------------------------------
// Group B: by-reference path (CUDAQ_INSTANTIATE_FORMAT_REF_ARGUMENT)
// ---------------------------------------------------------------------------

TEST(CudaqFmtFormat, StdStringFormats) {
  std::string s = "world";
  EXPECT_EQ(cudaq_fmt::format("{}", s), "world");
  // The by-reference appender uses std::cref, so the source string must not
  // have been moved-from after the format call.
  EXPECT_EQ(s, "world");
}

TEST(CudaqFmtFormat, VectorOfIntFormats) {
  EXPECT_EQ(cudaq_fmt::format("{}", std::vector<int>{1, 2, 3}), "[1, 2, 3]");
}

TEST(CudaqFmtFormat, VectorOfStringFormats) {
  EXPECT_EQ(cudaq_fmt::format("{}", std::vector<std::string>{"a", "b"}),
            "[\"a\", \"b\"]");
}

TEST(CudaqFmtFormat, MapFormats) {
  EXPECT_EQ(
      cudaq_fmt::format("{}", std::map<std::string, std::string>{{"k", "v"}}),
      "{\"k\": \"v\"}");
}

// ---------------------------------------------------------------------------
// Group C: C-string path (dedicated appendCString)
// ---------------------------------------------------------------------------

TEST(CudaqFmtFormat, CStringLiteral) {
  EXPECT_EQ(cudaq_fmt::format("{}", "literal"), "literal");
}

TEST(CudaqFmtFormat, CStringConstPointer) {
  const char *p = "abc";
  EXPECT_EQ(cudaq_fmt::format("{}", p), "abc");
}

TEST(CudaqFmtFormat, CStringMutableArray) {
  // Pass the mutable array directly (no `char *p = buf;` indirection). The
  // array decays to `char *` and selects the non-templated
  // FormatArgument(char *) overload. The templated FormatArgument(const T &)
  // would otherwise deduce T = char[N] and request appendArgument<char *>, a
  // symbol not instantiated in logger.cpp and so would fail to link.
  char buf[] = "xyz";
  EXPECT_EQ(cudaq_fmt::format("{}", buf), "xyz");
}

TEST(CudaqFmtFormat, CStringLiteralDirect) {
  // Same overload-resolution story for a string literal: type const char[N]
  // decays to const char * and selects FormatArgument(const char *) rather
  // than binding to the templated FormatArgument(const T &).
  EXPECT_EQ(cudaq_fmt::format("{}", "xyz"), "xyz");
}

// ---------------------------------------------------------------------------
// Group D: cross-cutting behaviour (not tied to a single type)
// ---------------------------------------------------------------------------

TEST(CudaqFmtFormat, EmptyArgList) {
  // Exercises the sizeof...(Args) == 0 path through std::array<...,0>.
  EXPECT_EQ(cudaq_fmt::format("hello"), "hello");
}

TEST(CudaqFmtFormat, MultipleMixedArgs) {
  EXPECT_EQ(cudaq_fmt::format("{} {} {}", 1, "two", 3.5), "1 two 3.5");
}

TEST(CudaqFmtFormat, PositionalArgs) {
  EXPECT_EQ(cudaq_fmt::format("{1}-{0}", "a", "b"), "b-a");
}

TEST(CudaqFmtFormat, FormatSpecs) {
  EXPECT_EQ(cudaq_fmt::format("{:5d}", 42), "   42");
  EXPECT_EQ(cudaq_fmt::format("{:.2f}", 3.14159), "3.14");
  EXPECT_EQ(cudaq_fmt::format("{:x}", 255), "ff");
}

// ---------------------------------------------------------------------------
// Group E: linker-coverage smoke test
//
// Touches every type in the instantiation tables at runtime/logger/logger.cpp
// lines 210-244. If a CUDAQ_INSTANTIATE_FORMAT_ARGUMENT(...) line is removed,
// the corresponding appendArgument<T> symbol disappears and this test fails to
// link rather than silently dropping support for that type.
// ---------------------------------------------------------------------------

TEST(CudaqFmtFormat, AllInstantiatedTypesLinkAndFormat) {
  // by-value path
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<bool>(true)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<char>('A')).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<signed char>(1)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<unsigned char>(1)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<short>(1)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<unsigned short>(1)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<int>(1)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<unsigned int>(1)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<long>(1)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<unsigned long>(1)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<float>(1.0f)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<double>(1.0)).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", static_cast<long double>(1.0L)).empty());
  EXPECT_FALSE(
      cudaq_fmt::format("{}", std::complex<float>{1.0f, 2.0f}).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", std::complex<double>{1.0, 2.0}).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", std::string_view{"sv"}).empty());

  int sentinel = 0;
  void *vp = &sentinel;
  const void *cvp = &sentinel;
  EXPECT_FALSE(cudaq_fmt::format("{:p}", vp).empty());
  EXPECT_FALSE(cudaq_fmt::format("{:p}", cvp).empty());

  EXPECT_FALSE(cudaq_fmt::format("{}", std::chrono::milliseconds{42}).empty());
  // The chrono formatter for time_point requires a strftime-style spec.
  EXPECT_FALSE(
      cudaq_fmt::format("{:%Y}", std::chrono::system_clock::time_point{})
          .empty());

  // by-reference path
  EXPECT_FALSE(cudaq_fmt::format("{}", std::string{"s"}).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", std::vector<int>{1}).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", std::vector<unsigned int>{1u}).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", std::vector<long>{1L}).empty());
  EXPECT_FALSE(
      cudaq_fmt::format("{}", std::vector<unsigned long>{1UL}).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", std::vector<float>{1.0f}).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", std::vector<double>{1.0}).empty());
  EXPECT_FALSE(cudaq_fmt::format("{}", std::vector<std::string>{"x"}).empty());
  EXPECT_FALSE(
      cudaq_fmt::format("{}", std::map<std::string, std::string>{{"k", "v"}})
          .empty());
  EXPECT_FALSE(
      cudaq_fmt::format("{}", std::vector<std::complex<float>>{{1.0f, 2.0f}})
          .empty());
  EXPECT_FALSE(
      cudaq_fmt::format("{}", std::vector<std::complex<double>>{{1.0, 2.0}})
          .empty());
}
