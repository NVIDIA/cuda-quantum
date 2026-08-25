/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Run under CUDAQ_TIMING_TAGS=1 so the Tracer gate admits tag-1 spans without
// touching the spdlog log level.

#include "cudaq/runtime/logger/chrome_tracer.h"
#include "cudaq/runtime/logger/tracer.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>

namespace {

nlohmann::json loadTrace(const std::string &path) {
  std::ifstream in(path);
  return nlohmann::json::parse(in);
}

std::string tempTracePath(const std::string &suffix) {
  return (std::filesystem::temp_directory_path() /
          ("cudaq-chrome-tracer-" + suffix + ".json"))
      .string();
}

} // namespace

TEST(ChromeTracer, WritesChromeEventFormat) {
  auto path = tempTracePath("format");
  std::error_code ec;
  std::filesystem::remove(path, ec);

  cudaq::Tracer::instance().setBackend(
      std::make_shared<cudaq::ChromeTraceBackend>(path));

  // Include characters that require JSON escaping in the name so a clean
  // parse here also confirms the emitter is escaping them correctly.
  const std::string trickyName = "name with \"quotes\" and \\ backslash";
  cudaq::TraceContext ctx{"chrome_tracer_test", __FILE__, __LINE__};
  auto handle = cudaq::Tracer::instance().beginSpan(ctx, trickyName, 1,
                                                    "detail_here", "scope");
  cudaq::Tracer::instance().endSpan(std::move(handle));

  cudaq::Tracer::instance().setBackend(nullptr);

  ASSERT_TRUE(std::filesystem::exists(path));
  nlohmann::json doc = loadTrace(path);
  ASSERT_TRUE(doc.contains("traceEvents"));
  EXPECT_EQ(doc.value("displayTimeUnit", ""), "ms");

  const auto &events = doc["traceEvents"];
  ASSERT_EQ(events.size(), 1u);
  const auto &e = events[0];
  EXPECT_EQ(e.value("name", ""), trickyName);
  EXPECT_EQ(e.value("cat", ""), "scope");
  EXPECT_EQ(e.value("ph", ""), "X");
  EXPECT_TRUE(e.contains("ts"));
  EXPECT_TRUE(e.contains("dur"));
  EXPECT_TRUE(e.contains("pid"));
  EXPECT_TRUE(e.contains("tid"));
  ASSERT_TRUE(e.contains("args"));
  EXPECT_EQ(e["args"].value("detail", ""), "detail_here");

  std::filesystem::remove(path, ec);
}

TEST(ChromeTracer, InMemoryToJsonWritesNoFile) {
  auto path = tempTracePath("inmemory");
  std::error_code ec;
  std::filesystem::remove(path, ec);

  auto backend = std::make_shared<cudaq::ChromeTraceBackend>();
  cudaq::Tracer::instance().setBackend(backend);

  cudaq::TraceContext ctx{"chrome_tracer_test", __FILE__, __LINE__};
  auto handle =
      cudaq::Tracer::instance().beginSpan(ctx, "memspan", 1, "detail", "scope");
  cudaq::Tracer::instance().endSpan(std::move(handle));

  const std::string json = backend->toJson();
  nlohmann::json doc = nlohmann::json::parse(json);
  ASSERT_TRUE(doc.contains("traceEvents"));
  ASSERT_EQ(doc["traceEvents"].size(), 1u);
  EXPECT_EQ(doc["traceEvents"][0].value("name", ""), "memspan");

  cudaq::Tracer::instance().setBackend(nullptr);
  backend.reset();

  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(ChromeTracer, OmitsArgsWhenEmpty) {
  auto path = tempTracePath("noargs");
  std::error_code ec;
  std::filesystem::remove(path, ec);

  cudaq::Tracer::instance().setBackend(
      std::make_shared<cudaq::ChromeTraceBackend>(path));

  cudaq::TraceContext ctx{"chrome_tracer_test", __FILE__, __LINE__};
  auto handle =
      cudaq::Tracer::instance().beginSpan(ctx, "bare", 1, "", "scope");
  cudaq::Tracer::instance().endSpan(std::move(handle));

  cudaq::Tracer::instance().setBackend(nullptr);

  nlohmann::json doc = loadTrace(path);
  const auto &events = doc["traceEvents"];
  ASSERT_EQ(events.size(), 1u);
  EXPECT_FALSE(events[0].contains("args"));

  std::filesystem::remove(path, ec);
}
