/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: profile_handlers | FileCheck %s

#include "cudaq_internal/compiler/LoweringHandler.h"
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cudaq_internal::compiler;

namespace {

const cudaq::config::Profile unchecked = cudaq::config::UncheckedProfile{};

/// Handler with construction-captured options; "fail" makes getPipeline throw.
class OptionHandler : public LoweringHandler {
  const bool enabled;
  const std::string pipeline;

public:
  mutable int consultations = 0;
  OptionHandler(bool enabled, std::string pipeline)
      : enabled(enabled), pipeline(std::move(pipeline)) {}

  bool supports(const cudaq::config::Profile &,
                const cudaq::config::Profile &) const override {
    ++consultations;
    return enabled;
  }

  std::optional<std::string>
  getPipeline(const cudaq::config::Profile &,
              const cudaq::config::Profile &) const override {
    if (pipeline == "fail")
      throw std::runtime_error("pipeline construction failed");
    return pipeline;
  }
};

using HandlerVector = std::vector<std::unique_ptr<LoweringHandler>>;

} // namespace

static OptionHandler *add(HandlerVector &handlers, bool enabled,
                          const std::string &pipeline) {
  handlers.push_back(std::make_unique<OptionHandler>(enabled, pipeline));
  return static_cast<OptionHandler *>(handlers.back().get());
}

static void printSelected(const HandlerVector &handlers) {
  auto *h = selectLoweringHandler(handlers, unchecked, unchecked);
  std::cout << "selected: "
            << (h ? h->getPipeline(unchecked, unchecked).value_or("none")
                  : "none")
            << "\n";
}

// The first supporting handler wins; the later one is never consulted.
void test_first_match() {
  HandlerVector handlers;
  add(handlers, true, "first");
  auto *second = add(handlers, true, "second");
  printSelected(handlers);
  std::cout << "second handler consultations: " << second->consultations
            << "\n";
}
// CHECK: selected: first
// CHECK: second handler consultations: 0

// Instances of one class act per their captured options: a leading disabled
// instance is skipped in favor of the enabled one.
void test_option_sensitive_instances() {
  HandlerVector handlers;
  add(handlers, false, "off");
  add(handlers, true, "on");
  printSelected(handlers);
}
// CHECK: selected: on

// No supporting handler yields no selection.
void test_no_match() {
  HandlerVector handlers;
  add(handlers, false, "off");
  printSelected(handlers);
}
// CHECK: selected: none

// A selected handler's failure is terminal: no fallback is consulted.
void test_no_fallback_after_failure() {
  HandlerVector handlers;
  add(handlers, true, "fail");
  auto *fallback = add(handlers, true, "fallback");
  auto *handler = selectLoweringHandler(handlers, unchecked, unchecked);
  try {
    (void)handler->getPipeline(unchecked, unchecked);
    std::cout << "unexpected success\n";
  } catch (const std::exception &) {
    std::cout << "pipeline failed\n";
  }
  std::cout << "fallback consultations: " << fallback->consultations << "\n";
}
// CHECK: pipeline failed
// CHECK: fallback consultations: 0

int main() {
  test_first_match();
  test_option_sensitive_instances();
  test_no_match();
  test_no_fallback_after_failure();
  return 0;
}
