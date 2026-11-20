/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: profile_stages | FileCheck %s

// Test how `getPassPipeline` builds a pipeline from a compile target's
// stages. Custom pipelines use upstream passes that no JIT pipeline runs as
// markers: `strip-debuginfo`, `sccp`, and `loop-invariant-code-motion`.

#include "common/CompileOptions.h"
#include "cudaq_internal/compiler/JITTargetPipeline.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace cudaq_internal::compiler;

/// Handler whose pass names the profiles it was asked to lower between, as
/// a `decomposition` pattern that is only printed, never run.
class TestHandler : public LoweringHandler {
  const bool supported;
  const bool forceSupports;

public:
  mutable unsigned calls = 0;

  TestHandler(bool supported_ = true, bool forceSupports_ = false)
      : supported(supported_), forceSupports(forceSupports_) {}

  /// `forceSupports` claims the lowering even when `addStagePasses` will fail.
  bool supports(const cudaq::config::Profile &,
                const cudaq::config::Profile &) const override {
    ++calls;
    return forceSupports || supported;
  }

  mlir::LogicalResult addStagePasses(const cudaq::config::Profile &input,
                                     const cudaq::config::Profile &output,
                                     const cudaq::CompileOptions &options,
                                     StagePasses &passes) const override {
    ++calls;
    if (!supported)
      return mlir::failure();
    passes.postDeployPasses =
        [pattern = "handler_" + profileKind(input) + "_to_" +
                   profileKind(output)](mlir::OpPassManager &pm) {
          cudaq::opt::addDecomposition(pm, {pattern});
        };
    return mlir::success();
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

/// The textual form of \p pm without its `builtin.module(...)` anchor.
static std::string pipelineText(const mlir::OpPassManager &pm) {
  std::string text;
  llvm::raw_string_ostream os(text);
  pm.printAsTextualPipeline(os);
  const std::string anchor = "builtin.module(";
  if (text.starts_with(anchor) && text.ends_with(")"))
    text = text.substr(anchor.size(), text.size() - anchor.size() - 1);
  return text;
}

static std::string planPipeline(const cudaq::CompileTarget &target,
                                const HandlerVector &handlers,
                                const cudaq::CompileOptions &options = {}) {
  std::string diagnostic;
  auto pipeline = getPassPipeline(target, options, handlers, diagnostic);
  return pipeline ? pipelineText(pipeline->passes) : "FAILED: " + diagnostic;
}

// The stages' passes join in stage order around one deploy and one finalize,
// and each stage's input profile is the previous stage's output profile. The
// first stage adds passes before deploy and the last adds passes after
// finalize.
void test_multiple_stages() {
  cudaq::CompileTarget target;
  auto custom = customStage("strip-debuginfo", "sccp", "");
  custom.irOutputProfile = cudaq::config::QuakeReferenceSemanticsProfile{};
  target.stages.push_back(std::move(custom));
  target.stages.push_back(
      inferredStage(cudaq::config::QuakeReferenceSemanticsProfile{}));
  target.stages.push_back(customStage("", "", "loop-invariant-code-motion"));
  HandlerVector handlers;
  handlers.push_back(std::make_unique<TestHandler>());
  std::cout << planPipeline(target, handlers) << "\n";
}
// clang-format off
// CHECK: {{^}}canonicalize{{.*}}strip-debuginfo,{{.*}}sccp,decomposition{{.*}}enable-patterns={handler_quake-reference-semantics_to_quake-reference-semantics}{{.*}}symbol-dce,loop-invariant-code-motion{{$}}
// clang-format on

// `getPipelineLayout` reports each custom pipeline at its place and the
// options of prep, deploy, and finalize, and `getPassPipeline` runs
// `canonicalize`, prep, the passes before deploy, deploy, the passes after
// deploy, finalize, and the passes after finalize.
void test_layout_assembly() {
  cudaq::CompileTarget target;
  target.preserveLoops = true;
  target.stages.push_back(
      customStage("strip-debuginfo", "sccp", "loop-invariant-code-motion"));
  cudaq::CompileOptions options;
  options.emulate = true;
  std::string diagnostic;
  auto layout = getPipelineLayout(target, options, {}, diagnostic);
  auto pipeline = getPassPipeline(target, options, {}, diagnostic);
  if (!layout || !pipeline) {
    std::cout << "FAILED: " << diagnostic << "\n";
    return;
  }
  const auto slotText = [&](AddPassesFn StagePasses::*place) {
    mlir::OpPassManager pm("builtin.module");
    layout->addStagePasses(pm, place);
    return pipelineText(pm);
  };
  mlir::OpPassManager expected("builtin.module");
  expected.addPass(mlir::createCanonicalizerPass());
  cudaq::opt::addJITPrepPipeline(expected, layout->prep);
  expected.addPass(mlir::createStripDebugInfoPass());
  cudaq::opt::addJITDeployPipeline(expected, layout->deploy);
  expected.addPass(mlir::createSCCPPass());
  cudaq::opt::addJITFinalizePipeline(expected, layout->finalize);
  expected.addPass(mlir::createLoopInvariantCodeMotionPass());
  std::cout << "layout: pre=" << slotText(&StagePasses::preDeployPasses)
            << " post=" << slotText(&StagePasses::postDeployPasses)
            << " final=" << slotText(&StagePasses::postFinalizePasses)
            << " no-loop-unroll=" << layout->prep.disableLoopUnrolling
            << layout->deploy.disableLoopUnrolling << "\n";
  std::cout << "assembled as layout: "
            << (pipelineText(expected) == pipelineText(pipeline->passes))
            << "\n";
}
// clang-format off
// CHECK: layout: pre=strip-debuginfo post=sccp final=loop-invariant-code-motion no-loop-unroll=11
// CHECK: assembled as layout: 1
// clang-format on

// A target without stages or with an override pipeline has no layout.
void test_no_layout() {
  for (auto stages : {std::vector<cudaq::config::CompilationStage>{},
                      std::vector{overrideStage("sccp")}}) {
    cudaq::CompileTarget target;
    target.stages = std::move(stages);
    std::string diagnostic;
    if (!getPipelineLayout(target, {}, {}, diagnostic))
      std::cout << "no layout: " << diagnostic << "\n";
  }
}
// CHECK: no layout: the target has no stages
// CHECK: no layout: an override pipeline replaces the whole pipeline

// Only the first stage may add passes before deploy and only the last may add
// passes after finalize. An override pipeline replaces the whole pipeline, so
// it must be the target's only stage.
void test_multiple_stages_rejected() {
  HandlerVector handlers;
  handlers.push_back(std::make_unique<TestHandler>());
  for (auto stages : {
           std::vector{customStage("", "", "loop-invariant-code-motion"),
                       customStage("", "sccp", "")},
           std::vector{customStage("", "sccp", ""),
                       customStage("strip-debuginfo", "", "")},
           std::vector{
               inferredStage(cudaq::config::QuakeReferenceSemanticsProfile{}),
               overrideStage("sccp")},
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
// CHECK: enable-patterns={handler_unchecked_to_quake-reference-semantics}
// first handler probed once; unsupported and returns no pipeline
// CHECK: first handler calls: 1{{$}}
// second handler probed to check support and called once to construct
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

// `disable_qubit_mapping` sets `device=bypass` on every `qubit-mapping`
// (which expands to `qubit-mapping-prep` and `qubit-mapping-func`),
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
// CHECK: custom: {{.*}}qubit-mapping-prep{device=bypass {{.*}}device=bypass extended-layer-size=20 {{.*}}qubit-mapping-prep{device=bypass {{.*}}device=bypass extended-layer-size=4 {{.*}}
// CHECK: override: qubit-mapping-prep{device=bypass {{.*}}device=bypass {{.*}}placement=greedy {{.*}}
// clang-format on

// A target with zero stages results in an empty pipeline.
void test_zero_stages() {
  cudaq::CompileTarget target;
  HandlerVector handlers;
  handlers.push_back(std::make_unique<TestHandler>());
  std::string diagnostic;
  const auto pipeline = getPassPipeline(target, {}, handlers, diagnostic);
  if (pipeline)
    std::cout << "zero stages: pipeline=\"" << pipelineText(pipeline->passes)
              << "\"\n";
  else
    std::cout << "got unexpected diagnostic: " << diagnostic << "\n";
  std::cout << "zero stages handler calls: "
            << static_cast<const TestHandler *>(handlers.front().get())->calls
            << "\n";
}
// CHECK: zero stages: pipeline=""
// CHECK: zero stages handler calls: 0

// Custom pipeline text is parsed when the pipeline is built, and a stage whose
// text does not parse fails with its stage number.
void test_invalid_custom_pipeline() {
  cudaq::CompileTarget target;
  target.stages.push_back(
      inferredStage(cudaq::config::QuakeReferenceSemanticsProfile{}));
  target.stages.push_back(customStage("", "no-such-pass", ""));
  HandlerVector handlers;
  handlers.push_back(std::make_unique<TestHandler>());
  std::cout << planPipeline(target, handlers) << "\n";
}
// CHECK: FAILED: compile target stage 2: invalid custom pipeline 'no-such-pass'

// Custom text that requests `combine-measurements` runs it in the pipeline,
// unless the compile emulates or measures an observable: then the pipeline
// leaves it out and defers it until after the JIT code is built. The
// top-level `*-gate-set-mapping` passes are kept for observe.
void test_custom_pipeline_records() {
  cudaq::CompileTarget target;
  target.stages.push_back(customStage(
      "", "iqm-gate-set-mapping,func.func(cse,combine-measurements)", ""));
  cudaq::CompileOptions hardware;
  cudaq::CompileOptions observe;
  observe.measureObservable = cudaq::spin_op::z(0);
  cudaq::CompileOptions emulated;
  emulated.emulate = true;
  for (const auto &[label, options] : {std::pair{"hardware", &hardware},
                                       {"observe", &observe},
                                       {"emulated", &emulated}}) {
    std::string diagnostic;
    auto pipeline = getPassPipeline(target, *options, {}, diagnostic);
    if (!pipeline) {
      std::cout << "FAILED: " << diagnostic << "\n";
      continue;
    }
    std::cout << label << ": defer=" << pipeline->deferCombineMeasurements
              << " gate-set-mapping="
              << pipelineText(pipeline->gateSetMappingPasses)
              << " pipeline=" << pipelineText(pipeline->passes) << "\n";
  }
}
// clang-format off
// CHECK: hardware: defer=0 gate-set-mapping=basis-conversion{basis={phased_rx,z(1)} } pipeline={{.*}}func.func(cse,combine-measurements)
// CHECK: observe: defer=1 gate-set-mapping=basis-conversion{basis={phased_rx,z(1)} } pipeline=
// CHECK-NOT: combine-measurements
// CHECK-SAME: func.func(cse)
// CHECK: emulated: defer=1 gate-set-mapping=basis-conversion{basis={phased_rx,z(1)} } pipeline=
// CHECK-NOT: combine-measurements
// CHECK-SAME: func.func(cse)
// clang-format on

int main() {
  cudaq::registerAllPasses();
  test_multiple_stages();
  test_layout_assembly();
  test_no_layout();
  test_multiple_stages_rejected();
  test_first_supporting_handler_wins();
  test_no_match_diagnostic();
  test_selected_failure_diagnostic();
  test_disable_qubit_mapping();
  test_zero_stages();
  test_invalid_custom_pipeline();
  test_custom_pipeline_records();
  return 0;
}
