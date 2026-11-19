/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: profile_stages | FileCheck %s

// Test how `getPassPipeline` builds a pipeline from a compile target's
// stages.

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace cudaq_internal::compiler;

/// Handler whose returned passes name the profiles it was asked to lower
/// between.
class TestHandler : public LoweringHandler {
  const bool supported;
  const bool forceSupports;

public:
  mutable unsigned calls = 0;

  TestHandler(bool supported_ = true, bool forceSupports_ = false)
      : supported(supported_), forceSupports(forceSupports_) {}

  /// `forceSupports` claims the lowering even when `getStagePasses` will fail.
  bool supports(const cudaq::config::Profile &,
                const cudaq::config::Profile &) const override {
    ++calls;
    return forceSupports || supported;
  }

  std::optional<StagePasses>
  getStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options) const override {
    ++calls;
    if (!supported)
      return std::nullopt;
    return StagePasses{.postDeployPasses = "handler(" + profileKind(input) +
                                           "->" + profileKind(output) + ")"};
  }

  static std::string profileKind(const cudaq::config::Profile &profile) {
    return std::visit(
        [](const auto &profile) { return std::string(profile.name); }, profile);
  }
};

using HandlerVector = std::vector<std::unique_ptr<LoweringHandler>>;

static cudaq::config::CompilationStage customStage(const std::string &high,
                                                   const std::string &mid,
                                                   const std::string &low) {
  cudaq::config::CustomPipelineLowering lowering;
  lowering.highLevelPipeline = high;
  lowering.midLevelPipeline = mid;
  lowering.lowLevelPipeline = low;
  return {.irOutputProfile = cudaq::config::UncheckedProfile{},
          .lowering = std::move(lowering)};
}

static cudaq::config::CompilationStage
inferredStage(cudaq::config::Profile output) {
  return {.irOutputProfile = std::move(output),
          .lowering = cudaq::config::InferredLowering{}};
}

static cudaq::config::CompilationStage
overrideStage(const std::string &pipeline) {
  cudaq::config::CustomPipelineLowering lowering;
  lowering.overridePassPipeline = pipeline;
  return {.irOutputProfile = cudaq::config::UncheckedProfile{},
          .lowering = std::move(lowering)};
}

static std::string planPipeline(const cudaq::CompileTarget &target,
                                const HandlerVector &handlers) {
  std::string diagnostic;
  auto pipeline = getPassPipeline(target, {}, handlers, diagnostic);
  return pipeline ? *pipeline : "FAILED: " + diagnostic;
}

// The stages' passes join in stage order around one deploy and one finalize,
// and each stage's input profile is the previous stage's output profile. The
// first stage adds passes before deploy and the last adds passes after
// finalize.
void test_multiple_stages() {
  cudaq::CompileTarget target;
  auto custom = customStage("high-one", "mid-one", "");
  custom.irOutputProfile = cudaq::config::QuakeReferenceSemanticsProfile{};
  target.stages.push_back(std::move(custom));
  target.stages.push_back(
      inferredStage(cudaq::config::QuakeReferenceSemanticsProfile{}));
  target.stages.push_back(customStage("", "", "low-three"));
  HandlerVector handlers;
  handlers.push_back(std::make_unique<TestHandler>());
  std::cout << planPipeline(target, handlers) << "\n";
}
// clang-format off
// CHECK: canonicalize,hw-jit-prep-pipeline{allow-early-exit=false},high-one,jit-deploy-pipeline,mid-one,handler(quake-reference-semantics->quake-reference-semantics),jit-finalize-pipeline{lower-device-calls=true},low-three{{$}}
// clang-format on

// Only the first stage may add passes before deploy and only the last may add
// passes after finalize. An override pipeline replaces the whole pipeline, so
// it must be the target's only stage.
void test_multiple_stages_rejected() {
  HandlerVector handlers;
  handlers.push_back(std::make_unique<TestHandler>());
  for (auto stages : {
           std::vector{customStage("", "", "low-one"),
                       customStage("", "mid-two", "")},
           std::vector{customStage("", "mid-one", ""),
                       customStage("high-two", "", "")},
           std::vector{
               inferredStage(cudaq::config::QuakeReferenceSemanticsProfile{}),
               overrideStage("override-two")},
       }) {
    cudaq::CompileTarget target;
    target.stages = std::move(stages);
    std::cout << planPipeline(target, handlers) << "\n";
  }
}
// clang-format off
// CHECK: FAILED: compile target stage 1: only the last stage may add passes after jit-finalize-pipeline
// CHECK: FAILED: compile target stage 2: only the first stage may add passes before jit-deploy-pipeline
// CHECK: FAILED: compile target stage 2: an override pipeline must be the target's only stage
// clang-format on

// Unsupporting handlers are skipped and the first supporting handler wins;
// later handlers are never consulted.
void test_first_supporting_handler_wins() {
  cudaq::CompileTarget target;
  target.stages.push_back(
      inferredStage(cudaq::config::QuakeReferenceSemanticsProfile{}));
  HandlerVector handlers;
  auto *first = static_cast<TestHandler *>(
      handlers.emplace_back(std::make_unique<TestHandler>(/*supported=*/false))
          .get());
  auto *second = static_cast<TestHandler *>(
      handlers.emplace_back(std::make_unique<TestHandler>()).get());
  auto *last = static_cast<TestHandler *>(
      handlers.emplace_back(std::make_unique<TestHandler>()).get());
  std::cout << planPipeline(target, handlers) << "\n";
  std::cout << "first handler calls: " << first->calls << "\n";
  std::cout << "second handler calls: " << second->calls << "\n";
  std::cout << "last handler calls: " << last->calls << "\n";
}
// CHECK: handler(unchecked->quake-reference-semantics)
// first handler probed once; unsupported and returns no pipeline
// CHECK: first handler calls: 1{{$}}
// second handler probed once to check support, then again to construct
// CHECK: second handler calls: 2{{$}}
// never consulted since the second handler supports the lowering
// CHECK: last handler calls: 0{{$}}

// No supporting handler for an inferred stage fails the composition with a
// diagnostic identifying the stage and profiles.
void test_no_match_diagnostic() {
  cudaq::CompileTarget target;
  target.stages.push_back(
      inferredStage(cudaq::config::QuakeReferenceSemanticsProfile{}));
  std::cout << planPipeline(target, {}) << "\n";
}
// CHECK: FAILED: compile target stage 1: no lowering handler supports
// CHECK-SAME: lowering from unchecked to quake-reference-semantics

// A selected handler that fails to construct a pipeline is an error, not a
// fallback to the next handler.
void test_selected_failure_diagnostic() {
  cudaq::CompileTarget target;
  target.stages.push_back(
      inferredStage(cudaq::config::QuakeReferenceSemanticsProfile{}));
  HandlerVector handlers;
  handlers.push_back(
      std::make_unique<TestHandler>(false, /*forceSupports=*/true));
  handlers.push_back(std::make_unique<TestHandler>());
  std::cout << planPipeline(target, handlers) << "\n";
}
// clang-format off
// CHECK: FAILED: compile target stage 1: the selected lowering handler failed
// CHECK-SAME: to construct a pipeline from unchecked to quake-reference-semantics
// clang-format on

// `disable_qubit_mapping` sets `device=bypass` on every `qubit-mapping`,
// including one in an override pipeline. It replaces the whole device value
// even when it contains a comma, parentheses, an apostrophe, or a quoted path
// with a space, and keeps the pass's other options.
void test_disable_qubit_mapping() {
  cudaq::CompileTarget target;
  target.disableQubitMapping = true;
  target.stages.push_back(
      customStage("qubit-mapping{device=grid(3,3)}",
                  "qubit-mapping{device=file(/tmp/team's-arch(1).txt) "
                  "extended-layer-size=4}",
                  ""));
  std::cout << "custom: " << planPipeline(target, {}) << "\n";
  target.stages = {overrideStage(
      "qubit-mapping{device=file('/tmp/arch (1).txt') placement=greedy}")};
  std::cout << "override: " << planPipeline(target, {}) << "\n";
}
// clang-format off
// CHECK: custom: {{.*}},qubit-mapping{device=bypass},jit-deploy-pipeline,qubit-mapping{device=bypass extended-layer-size=4},jit-finalize-pipeline
// CHECK: override: qubit-mapping{device=bypass placement=greedy}{{$}}
// clang-format on

// A target with zero stages results in an empty pipeline.
void test_zero_stages() {
  cudaq::CompileTarget target;
  HandlerVector handlers;
  handlers.push_back(std::make_unique<TestHandler>());
  std::string diagnostic;
  const auto pipeline = getPassPipeline(target, {}, handlers, diagnostic);
  if (pipeline)
    std::cout << "zero stages: pipeline=\"" << *pipeline << "\"\n";
  else
    std::cout << "got unexpected diagnostic: " << diagnostic << "\n";
  std::cout << "zero stages handler calls: "
            << static_cast<const TestHandler *>(handlers.front().get())->calls
            << "\n";
}
// CHECK: zero stages: pipeline=""
// CHECK: zero stages handler calls: 0

int main() {
  test_multiple_stages();
  test_multiple_stages_rejected();
  test_first_supporting_handler_wins();
  test_no_match_diagnostic();
  test_selected_failure_diagnostic();
  test_disable_qubit_mapping();
  test_zero_stages();
  return 0;
}
