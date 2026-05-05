/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Tests for the CUDAQ_INFO/WARN/ERROR/DBG macro family. Tests are grouped by
// concern:
//   Group A - level gating (info/warn appear only when their level is enabled)
//   Group B - call-site capture ([file:line] prefix injected by
//   logMessagePacked) Group C - argument formatting (delegates to
//   cudaq_fmt::format) Group D - CUDAQ_ERROR (logs at error level AND throws
//   std::runtime_error) Group E - argument evaluation gating (side-effecting
//   args are skipped when
//             the level is disabled, courtesy of the do/while macro shape)
//   Group F - CUDAQ_DBG no-op when CUDAQ_DEBUG is undefined

#include "cudaq/runtime/logger/logger.h"

#include <gtest/gtest.h>

#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

// spdlog's stdout_color sink emits ANSI escapes when stdout is a TTY (which it
// is when the test binary is launched directly via `docker exec`). Strip them
// before substring-asserting so output looks like "[info] [file.cpp:NN] msg".
std::string stripAnsi(std::string_view s) {
  static const std::regex ansi("\x1b\\[[0-9;]*m");
  return std::regex_replace(std::string(s), ansi, "");
}

template <typename Fn>
std::string captureLogStdout(Fn fn) {
  testing::internal::CaptureStdout();
  fn();
  cudaq::details::flushLogs();
  return stripAnsi(testing::internal::GetCapturedStdout());
}

class LogMacrosTest : public ::testing::Test {
protected:
  void SetUp() override { saved = cudaq::details::getLogLevel(); }
  void TearDown() override { cudaq::details::setLogLevel(saved); }
  cudaq::details::LogLevel saved = cudaq::details::LogLevel::warn;
};

} // namespace

// ---------------------------------------------------------------------------
// Group A - level gating
// ---------------------------------------------------------------------------

TEST_F(LogMacrosTest, InfoEmitsWhenInfoEnabled) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::info);
  const std::string out = captureLogStdout([] { CUDAQ_INFO("hello info"); });
  EXPECT_NE(out.find("hello info"), std::string::npos) << out;
}

TEST_F(LogMacrosTest, InfoSuppressedWhenWarnEnabled) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::warn);
  const std::string out = captureLogStdout([] { CUDAQ_INFO("hello info"); });
  EXPECT_EQ(out.find("hello info"), std::string::npos) << out;
}

TEST_F(LogMacrosTest, WarnEmitsWhenWarnEnabled) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::warn);
  const std::string out = captureLogStdout([] { CUDAQ_WARN("hello warn"); });
  EXPECT_NE(out.find("hello warn"), std::string::npos) << out;
}

TEST_F(LogMacrosTest, WarnSuppressedWhenErrorEnabled) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::error);
  const std::string out = captureLogStdout([] { CUDAQ_WARN("hello warn"); });
  EXPECT_EQ(out.find("hello warn"), std::string::npos) << out;
}

// ---------------------------------------------------------------------------
// Group B - call-site capture ([basename:line] prefix)
// ---------------------------------------------------------------------------

TEST_F(LogMacrosTest, InfoIncludesFileBasename) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::info);
  const std::string out = captureLogStdout([] { CUDAQ_INFO("with location"); });
  EXPECT_NE(out.find("CudaqLogMacrosTester.cpp:"), std::string::npos) << out;
}

TEST_F(LogMacrosTest, InfoIncludesLineNumber) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::info);
  const int expectedLine = __LINE__ + 2;
  const std::string out = captureLogStdout([] { CUDAQ_INFO("with location"); });
  const std::string needle = ":" + std::to_string(expectedLine) + "]";
  EXPECT_NE(out.find(needle), std::string::npos) << out;
}

// ---------------------------------------------------------------------------
// Group C - argument formatting
// ---------------------------------------------------------------------------

TEST_F(LogMacrosTest, InfoFormatsSingleArg) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::info);
  const std::string out = captureLogStdout([] { CUDAQ_INFO("x={}", 42); });
  EXPECT_NE(out.find("x=42"), std::string::npos) << out;
}

TEST_F(LogMacrosTest, InfoFormatsMixedArgs) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::info);
  const std::string out = captureLogStdout(
      [] { CUDAQ_INFO("{} - {} - {}", 1, std::string{"two"}, 3.5); });
  EXPECT_NE(out.find("1 - two - 3.5"), std::string::npos) << out;
}

// ---------------------------------------------------------------------------
// Group D - CUDAQ_ERROR (logs and throws)
// ---------------------------------------------------------------------------

TEST_F(LogMacrosTest, ErrorThrowsRuntimeError) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::error);
  // Capture stdout to swallow the error log line so it doesn't pollute the
  // test output.
  testing::internal::CaptureStdout();
  EXPECT_THROW(CUDAQ_ERROR("oops"), std::runtime_error);
  cudaq::details::flushLogs();
  (void)testing::internal::GetCapturedStdout();
}

TEST_F(LogMacrosTest, ErrorWhatMatchesMessage) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::error);
  testing::internal::CaptureStdout();
  try {
    CUDAQ_ERROR("specific text");
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    EXPECT_STREQ(e.what(), "specific text");
  }
  cudaq::details::flushLogs();
  (void)testing::internal::GetCapturedStdout();
}

TEST_F(LogMacrosTest, ErrorLogsAtErrorLevel) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::error);
  testing::internal::CaptureStdout();
  try {
    CUDAQ_ERROR("boom");
  } catch (const std::runtime_error &) {
    // expected; we're after the side-effect log line
  }
  cudaq::details::flushLogs();
  const std::string out = stripAnsi(testing::internal::GetCapturedStdout());
  EXPECT_NE(out.find("boom"), std::string::npos) << out;
}

// ---------------------------------------------------------------------------
// Group E - argument evaluation gating (do/while + if guard)
// ---------------------------------------------------------------------------

TEST_F(LogMacrosTest, InfoArgsNotEvaluatedWhenSuppressed) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::warn);
  int counter = 0;
  CUDAQ_INFO("{}", [&counter] {
    ++counter;
    return 1;
  }());
  EXPECT_EQ(counter, 0);
}

TEST_F(LogMacrosTest, InfoArgsEvaluatedWhenEnabled) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::info);
  int counter = 0;
  testing::internal::CaptureStdout();
  CUDAQ_INFO("{}", [&counter] {
    ++counter;
    return 1;
  }());
  cudaq::details::flushLogs();
  (void)testing::internal::GetCapturedStdout();
  EXPECT_EQ(counter, 1);
}

// ---------------------------------------------------------------------------
// Group F - CUDAQ_DBG no-op when CUDAQ_DEBUG is undefined
// ---------------------------------------------------------------------------

#ifndef CUDAQ_DEBUG
TEST_F(LogMacrosTest, DbgIsNoopWhenDebugUndefined) {
  cudaq::details::setLogLevel(cudaq::details::LogLevel::trace);
  int counter = 0;
  CUDAQ_DBG("{}", [&counter] {
    ++counter;
    return 1;
  }());
  EXPECT_EQ(counter, 0);
}
#endif
