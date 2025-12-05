/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "../../lib/Optimizer/Transforms/DecompositionPatterns.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include <gtest/gtest.h>
#include <iterator>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringMap.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>

using namespace mlir;

namespace {

class DecompositionPatternsTest : public ::testing::Test {
protected:
  void SetUp() override {
    context = std::make_unique<MLIRContext>();
    context->loadDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                         func::FuncDialect, quake::QuakeDialect>();
  }

  std::unique_ptr<MLIRContext> context;
};

// Helper to parse control count from gate string like "x(1)" or "z(2)"
std::pair<std::string, size_t> parseGateSpec(StringRef gateSpec) {
  auto pos = gateSpec.find('(');
  if (pos == StringRef::npos) {
    return {gateSpec.str(), 0};
  }

  std::string gateName = gateSpec.substr(0, pos).str();
  StringRef numStr = gateSpec.substr(pos + 1);
  size_t numControls = 0;

  if (numStr.startswith("n")) {
    // Arbitrary number of controls - use a reasonable test value
    numControls = std::numeric_limits<size_t>::max();
  } else {
    numStr.consumeInteger(10, numControls);
  }

  return {gateName, numControls};
}

// Helper function to create a test module with a single gate operation
ModuleOp createTestModule(MLIRContext *context, StringRef gateSpec) {
  auto [gateName, numControls] = parseGateSpec(gateSpec);

  // Limit the number of controls to 2
  numControls = std::min<size_t>(numControls, 2);

  size_t numQubits;
  if (gateName == "swap" || gateName == "exp_pauli") {
    assert(numControls == 0);
    // exp_pauli can have any number of qubits, we hardcode to 2 for the test.
    numQubits = 2;
  } else {
    numQubits = numControls + 1;
  }

  OpBuilder builder(context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Create function type: (qubits...) -> ()
  SmallVector<Type> inputTypes;
  auto refType = quake::RefType::get(context);
  for (size_t i = 0; i < numQubits; ++i) {
    inputTypes.push_back(refType);
  }
  auto funcType = builder.getFunctionType(inputTypes, {});

  // Create function
  auto func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "test_func",
                                           funcType);
  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  // Get operands (controls and target)
  SmallVector<Value> controls;
  for (size_t i = 0; i < numControls; ++i) {
    controls.push_back(entry->getArgument(i));
  }
  Value target = entry->getArgument(numControls);

  // Create the gate operation based on gate name
  Location loc = builder.getUnknownLoc();

  Value pi_2 = cudaq::opt::factory::createFloatConstant(loc, builder, M_PI_2,
                                                        builder.getF64Type());

  if (gateName == "h") {
    builder.create<quake::HOp>(loc, controls, target);
  } else if (gateName == "s") {
    builder.create<quake::SOp>(loc, controls, target);
  } else if (gateName == "t") {
    builder.create<quake::TOp>(loc, controls, target);
  } else if (gateName == "x") {
    builder.create<quake::XOp>(loc, controls, target);
  } else if (gateName == "y") {
    builder.create<quake::YOp>(loc, controls, target);
  } else if (gateName == "z") {
    builder.create<quake::ZOp>(loc, controls, target);
  } else if (gateName == "rx") {
    builder.create<quake::RxOp>(loc, ValueRange{pi_2}, controls, target);
  } else if (gateName == "ry") {
    builder.create<quake::RyOp>(loc, ValueRange{pi_2}, controls, target);
  } else if (gateName == "rz") {
    builder.create<quake::RzOp>(loc, ValueRange{pi_2}, controls, target);
  } else if (gateName == "r1") {
    builder.create<quake::R1Op>(loc, ValueRange{pi_2}, controls, target);
  } else if (gateName == "u3") {
    builder.create<quake::U3Op>(loc, ValueRange{pi_2, pi_2, pi_2}, controls,
                                target);
  } else if (gateName == "phased_rx") {
    builder.create<quake::PhasedRxOp>(loc, ValueRange{{pi_2, pi_2}}, controls,
                                      target);
  } else if (gateName == "swap") {
    // Swap needs 2 targets
    Value target = entry->getArgument(0);
    Value target2 = entry->getArgument(1);
    builder.create<quake::SwapOp>(loc, ValueRange{target, target2});
  } else if (gateName == "exp_pauli") {
    Value target = entry->getArgument(0);
    Value target2 = entry->getArgument(1);
    // Create a veq from the two target qubits using ConcatOp
    SmallVector<Value> targetValues = {target, target2};
    Value qubitsVal = builder.create<quake::ConcatOp>(
        loc, quake::VeqType::get(builder.getContext(), 2), targetValues);

    builder.create<quake::ExpPauliOp>(loc,
                                      /* parameters = */ ValueRange{pi_2},
                                      /* controls = */ ValueRange{},
                                      /* targets = */ qubitsVal,
                                      /* pauliLiteral = */ "XX");
  } else {
    // Unsupported gate for this test
    ADD_FAILURE() << "unknown gate: " << gateName;
  }

  builder.create<func::ReturnOp>(loc);
  return module;
}

// Helper to collect all gate types in a module
llvm::StringSet<> collectGateTypesInModule(ModuleOp module) {
  llvm::StringSet<> gates;

  module.walk([&](Operation *op) {
    if (auto optor = dyn_cast<quake::OperatorInterface>(op)) {
      std::string gateName = optor->getName().stripDialect().str();
      auto numControls = optor.getControls().size();

      if (numControls > 0) {
        gateName += "(" + std::to_string(numControls) + ")";
      }

      gates.insert(gateName);
    }
  });

  return gates;
}

inline std::pair<std::string, size_t>
splitGateAndControls(llvm::StringRef gate) {
  auto parenOpen = gate.find('(');
  std::string gatePrefix;
  size_t gateNum = 0;
  if (parenOpen != llvm::StringRef::npos) {
    gatePrefix = gate.substr(0, parenOpen).str();
    auto parenClose = gate.find(')', parenOpen);
    assert(parenClose != llvm::StringRef::npos);
    std::string numStr =
        gate.substr(parenOpen + 1, parenClose - parenOpen - 1).str();
    if (numStr == "n")
      gateNum = std::numeric_limits<size_t>::max();
    else
      gateNum = static_cast<size_t>(std::stoul(numStr));
  } else {
    gatePrefix = gate.str();
  }
  return {gatePrefix, gateNum};
};

void stripNamespace(std::string &debugName) {
  auto lastColon = debugName.find_last_of(':');
  if (lastColon != llvm::StringRef::npos) {
    debugName = debugName.substr(lastColon + 1);
  }
}

} // namespace

// Test 1: Verify the total number of registered decomposition patterns
TEST_F(DecompositionPatternsTest, TotalPatternCount) {
  auto patternEntries =
      cudaq::DecompositionPatternType::RegistryType::entries();
  unsigned int size =
      std::distance(patternEntries.begin(), patternEntries.end());
  EXPECT_EQ(size, 31) << "Expected 31 decomposition patterns, but found "
                      << size;
}

// Test 2: Verify pattern names match getDebugName()
TEST_F(DecompositionPatternsTest, PatternNamesMatchDebugNames) {
  auto patternEntries =
      cudaq::DecompositionPatternType::RegistryType::entries();

  for (auto &entry : patternEntries) {
    auto patternName = entry.getName();
    // Create the pattern
    auto patternType = cudaq::registry::get<cudaq::DecompositionPatternType>(
        patternName.str());
    ASSERT_NE(patternType, nullptr)
        << "Failed to recover registered pattern type: " << patternName.str();

    auto pattern = patternType->create(context.get());
    ASSERT_NE(pattern, nullptr)
        << "Failed to create pattern: " << patternName.str();

    // Get the debug name
    auto debugName = pattern->getDebugName().str();
    stripNamespace(debugName);

    // Verify they match
    EXPECT_EQ(patternName.str(), debugName)
        << "Pattern name '" << patternName.str()
        << "' does not match debug name '" << debugName << "'";
  }
}

// Test 3: Verify metadata is consistent (source and target gates are valid)
TEST_F(DecompositionPatternsTest, MetadataConsistency) {
  auto patternEntries =
      cudaq::DecompositionPatternType::RegistryType::entries();

  for (auto &entry : patternEntries) {
    std::string patternName = entry.getName().str();
    auto patternType = entry.instantiate();
    std::string sourceGate = patternType->getSourceOp().str();
    auto targetGates = patternType->getTargetOps();

    // Source gate should not be empty
    EXPECT_FALSE(sourceGate.empty())
        << "Pattern '" << patternName << "' has empty source gate";

    // Target gates should not be empty
    EXPECT_FALSE(targetGates.empty())
        << "Pattern '" << patternName << "' has empty target gates";

    // All target gates should be non-empty
    for (auto targetGate : targetGates) {
      EXPECT_FALSE(targetGate.empty())
          << "Pattern '" << patternName << "' has empty target gate in list";
    }
  }
}

// Test 4: Verify pattern decompositions produce only target gates
TEST_F(DecompositionPatternsTest, DecompositionProducesOnlyTargetGates) {
  auto patternEntries =
      cudaq::DecompositionPatternType::RegistryType::entries();

  for (auto &entry : patternEntries) {
    std::string patternName = entry.getName().str();
    auto patternType = entry.instantiate();
    std::string sourceGate = patternType->getSourceOp().str();
    auto targetGates = patternType->getTargetOps();

    // Create a test module with the source gate
    auto module = createTestModule(context.get(), sourceGate);

    // Apply the decomposition pass with only this pattern enabled
    PassManager pm(context.get());
    cudaq::opt::DecompositionPassOptions options;
    std::string ownedEnabledPatterns[]{patternName};
    options.enabledPatterns = ownedEnabledPatterns;
    pm.addPass(cudaq::opt::createDecompositionPass(options));

    // Run the pass
    auto result = pm.run(module);
    ASSERT_TRUE(succeeded(result))
        << "Decomposition pass failed for pattern: " << patternName;

    // Collect all gates in the output
    auto outputGates = collectGateTypesInModule(module);

    // Map from gate prefix to allowed number of controls
    llvm::StringMap<llvm::SmallVector<size_t>> allowedGates;
    for (auto targetGate : targetGates) {
      auto [tPrefix, tNum] = splitGateAndControls(targetGate);
      allowedGates[tPrefix].push_back(tNum);
    }
    auto isAllowedGate = [&](StringRef gate) {
      // Split gate into prefix and number (e.g., "h(1)" -> "h", 1) using
      // utility function
      auto [gatePrefix, gateNum] = splitGateAndControls(gate);

      auto it = allowedGates.find(gatePrefix);
      if (it == allowedGates.end()) {
        return false;
      }
      auto allowedNumControls = it->second;
      // Check if the number of controls is in the allowed list (or if any
      // number is allowed)
      auto isEqOrMax = [gateNum](size_t num) {
        return num == gateNum || num == std::numeric_limits<size_t>::max();
      };
      return std::find_if(allowedNumControls.begin(), allowedNumControls.end(),
                          isEqOrMax) != allowedNumControls.end();
    };

    std::vector<std::string> unexpectedGates;
    for (auto &outputGate : outputGates) {
      if (!isAllowedGate(outputGate.getKey())) {
        unexpectedGates.push_back(outputGate.getKey().str());
      }
    }

    if (!unexpectedGates.empty()) {
      auto expectedGatesStr = llvm::join(targetGates, ", ");
      auto unexpectedGatesStr = llvm::join(unexpectedGates, ", ");

      ADD_FAILURE() << "Pattern '" << patternName
                    << "' produced unexpected gates.\n"
                    << "  Allowed gates: {" << expectedGatesStr << "}\n"
                    << "  Found: {" << unexpectedGatesStr << "}";
    }
  }
}
