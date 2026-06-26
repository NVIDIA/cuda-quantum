/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.  *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/KernelExecution.h"
#include "common/ResultReconstruction.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "nlohmann/json.hpp"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/algorithms/sample/policy.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// The reconstruction reorder/projection behavior reached through a policy's
// resultOutputMap is exercised end-to-end in
// runtime/test/test_kernel_execution_maps.cpp. The cases below cover what that
// integration test does not touch: the value semantics of activeDeviceQubits,
// the policy metadata API that derives the map for mapped executions (and
// leaves it empty for unmapped ones), and validateExecutionMetadata's
// back-compat position fallback.

static cudaq::KernelExecution createKernelExecution(std::string name) {
  return cudaq::KernelExecution(name, "code", std::nullopt, std::nullopt);
}

static std::vector<cudaq::KernelExecution> emitIqmCode(llvm::StringRef source) {
  auto context = cudaq_internal::compiler::getOwningMLIRContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, context.get());
  if (!module)
    throw std::runtime_error("failed to parse test module");

  auto sharedContext = std::shared_ptr<mlir::MLIRContext>(
      context.get(), [](mlir::MLIRContext *) {});
  std::vector<
      cudaq_internal::compiler::CompiledModuleHelper::NamedCompiledArtifact>
      artifacts;
  artifacts.push_back(
      cudaq_internal::compiler::CompiledModuleHelper::createMlirArtifact(
          "mapped", module.get(), sharedContext));
  auto compiled =
      cudaq_internal::compiler::CompiledModuleHelper::createCompiledModule(
          "mapped", cudaq::ResultInfo{}, std::move(artifacts));

  auto target = std::make_unique<cudaq::CompileTarget>();
  target->emitTargetCode = true;
  target->pipelineConfig.codegenTranslation = "iqm";
  cudaq_internal::compiler::Compiler compiler(std::move(target));
  return compiler.emitKernelExecutions(compiled);
}

TEST(KernelExecutionTester, CopiesActiveDeviceQubitsByValue) {
  auto original = createKernelExecution("kernel");
  original.activeDeviceQubits = {3};
  original.targetQubitMapping = {{"QB4", 3}};

  auto copied = original;
  original.activeDeviceQubits[0] = 11;
  original.targetQubitMapping[0].deviceQubit = 11;
  EXPECT_EQ(copied.activeDeviceQubits, (cudaq::ActiveDeviceQubits{3}));
  ASSERT_EQ(copied.targetQubitMapping.size(), 1);
  EXPECT_EQ(copied.targetQubitMapping[0].logicalName, "QB4");
  EXPECT_EQ(copied.targetQubitMapping[0].deviceQubit, 3);
}

// sample_policy and observe_policy share the same setKernelExecutionMetadata
// base, so the sample policy stands in for both. A mapped execution carries
// active device qubits and derives the result map from the enriched
// output_names; an unmapped execution skips local reconstruction and leaves the
// map empty.
TEST(KernelExecutionTester, PoliciesDeriveResultMapFromOutputNames) {
  nlohmann::json outputNames =
      nlohmann::json::parse(R"([[[0,[4,"alpha",0]]]])");

  auto mapped = cudaq::KernelExecution("sample", "code", std::nullopt,
                                       std::nullopt, outputNames);
  mapped.activeDeviceQubits = {4};

  cudaq::sample_policy mappedPolicy;
  mappedPolicy.setKernelExecutionMetadata(mapped);
  ASSERT_EQ(mappedPolicy.resultOutputMap.outputs.size(), 1);
  EXPECT_EQ(mappedPolicy.resultOutputMap.outputs[0].resultIndex, 0);
  EXPECT_EQ(mappedPolicy.resultOutputMap.outputs[0].deviceQubit, 4);
  EXPECT_EQ(mappedPolicy.resultOutputMap.outputs[0].outputName, "alpha");
  EXPECT_EQ(mappedPolicy.resultOutputMap.outputs[0].outputPosition, 0);

  auto unmapped = cudaq::KernelExecution("unmapped", "code", std::nullopt,
                                         std::nullopt, outputNames);
  cudaq::sample_policy unmappedPolicy;
  unmappedPolicy.setKernelExecutionMetadata(unmapped);
  EXPECT_TRUE(unmappedPolicy.resultOutputMap.outputs.empty());
}

TEST(KernelExecutionTester, IqmEmissionDerivesMetadataFromMappedAllocas) {
  auto codes = emitIqmCode(R"mlir(
    module {
      func.func @mapped() attributes {"cudaq-entrypoint", output_names = "[[[0,[7,\"alpha\",0]]]]"} {
        %q = quake.alloca !quake.ref {StartingOffset = 7 : i64}
        return
      }
    }
  )mlir");

  ASSERT_EQ(codes.size(), 1);
  auto resultMap =
      cudaq::makeResultOutputMapFromEnrichedOutputNames(codes[0].output_names);
  ASSERT_EQ(resultMap.outputs.size(), 1);
  EXPECT_EQ(resultMap.outputs[0].resultIndex, 0);
  EXPECT_EQ(resultMap.outputs[0].deviceQubit, 7);
  EXPECT_EQ(resultMap.outputs[0].outputName, "alpha");
  EXPECT_EQ(resultMap.outputs[0].outputPosition, 0);
  EXPECT_EQ(codes[0].activeDeviceQubits, (cudaq::ActiveDeviceQubits{7}));
  ASSERT_EQ(codes[0].targetQubitMapping.size(), 1);
  EXPECT_EQ(codes[0].targetQubitMapping[0].logicalName, "QB8");
  EXPECT_EQ(codes[0].targetQubitMapping[0].deviceQubit, 7);
}

// validateExecutionMetadata accepts legacy two-element output-location tuples:
// with no explicit position, positions fall back to the dense result index, so
// the dense-position invariant still holds.
TEST(KernelExecutionTester, ValidatesLegacyTwoTuplePositionFallback) {
  auto outputNames =
      nlohmann::json::parse(R"([[[0,[2,"alpha"]],[1,[5,"beta"]]]])");
  EXPECT_NO_THROW(cudaq::validateExecutionMetadata({}, outputNames));
}
