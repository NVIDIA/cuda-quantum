/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "SingleQubitChain.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/OptimizationLevel.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include <cmath>
#include <compare>
#include <utility>

namespace cudaq::opt {
#define GEN_PASS_DEF_NORMALIZESINGLEQUBITCLIFFORDT
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

struct Candidate {
  cudaq::opt::SingleQubitChain chain;
  llvm::SmallVector<bool> controlPolarities;
  cudaq::synth::Circuit normalized;
};

struct CircuitCost {
  int tCount;
  std::size_t emittedGateCount;

  auto operator<=>(const CircuitCost &) const = default;
};

class NormalizeSingleQubitCliffordTPass
    : public cudaq::opt::impl::NormalizeSingleQubitCliffordTBase<
          NormalizeSingleQubitCliffordTPass> {
public:
  using NormalizeSingleQubitCliffordTBase::NormalizeSingleQubitCliffordTBase;
  void runOnOperation() override;
};

} // namespace

// The gates with an exact Clifford+T word. The collector separately requires
// that all of a gate's operands thread `!quake.wire` values.
static bool isExactGate(Operation *operation) {
  return isa<cudaq::quake::HOp, cudaq::quake::SOp, cudaq::quake::TOp,
             cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp>(
             operation) &&
         cast<cudaq::quake::OperatorInterface>(operation).getTargets().size() ==
             1;
}

static llvm::SmallVector<bool> getPolarities(Operation *operation) {
  return cudaq::quake::getControlPolarities(
      cast<cudaq::quake::OperatorInterface>(operation));
}

static void appendExactGate(cudaq::synth::Circuit &circuit,
                            Operation *operation) {
  using cudaq::synth::Gate;
  const bool isAdj = cast<cudaq::quake::OperatorInterface>(operation).isAdj();
  llvm::TypeSwitch<Operation *>(operation)
      .Case<cudaq::quake::HOp>([&](auto) { circuit.push_back(Gate::H); })
      .Case<cudaq::quake::SOp>([&](auto) {
        if (isAdj) {
          circuit.push_back(Gate::S);
          circuit.push_back(Gate::S);
        }
        circuit.push_back(Gate::S);
      })
      .Case<cudaq::quake::TOp>([&](auto) {
        circuit.push_back(Gate::T);
        if (isAdj) {
          circuit.push_back(Gate::S);
          circuit.push_back(Gate::S);
          circuit.push_back(Gate::S);
        }
      })
      .Case<cudaq::quake::XOp>([&](auto) { circuit.push_back(Gate::X); })
      .Case<cudaq::quake::YOp>([&](auto) {
        circuit.push_back(Gate::W);
        circuit.push_back(Gate::W);
        circuit.push_back(Gate::X);
        circuit.push_back(Gate::S);
        circuit.push_back(Gate::S);
      })
      .Case<cudaq::quake::ZOp>([&](auto) {
        circuit.push_back(Gate::S);
        circuit.push_back(Gate::S);
      })
      .Default([](Operation *) {
        llvm_unreachable("collected an unsupported exact gate");
      });
}

static cudaq::synth::Circuit
buildMatrixProduct(llvm::ArrayRef<Operation *> operations) {
  cudaq::synth::Circuit circuit;
  for (Operation *operation : llvm::reverse(operations))
    appendExactGate(circuit, operation);
  return circuit;
}

// Prefer lower T-count, then fewer gates in the underlying one-qubit word.
// W is phase bookkeeping for that cost, including when the exact correction
// becomes observable under control. This is not a controlled-decomposition
// cost model.
static CircuitCost emittedCost(const cudaq::synth::Circuit &circuit) {
  std::size_t emittedGateCount = 0;
  for (cudaq::synth::Gate gate : circuit)
    emittedGateCount += gate != cudaq::synth::Gate::W;
  return {circuit.t_count(), emittedGateCount};
}

static CircuitCost inputCost(llvm::ArrayRef<Operation *> chain) {
  return {
      static_cast<int>(llvm::count_if(chain, llvm::IsaPred<cudaq::quake::TOp>)),
      chain.size()};
}

template <typename OpTy>
static void emitGate(OpBuilder &builder, Location location,
                     llvm::SmallVectorImpl<Value> &controls, Value &target,
                     DenseBoolArrayAttr negatedControls) {
  llvm::SmallVector<Value> targets{target};
  auto resultTypes = cudaq::quake::getWireResultTypes(controls, targets);
  auto operation = OpTy::create(
      builder, location, resultTypes, /*is_adj=*/false,
      /*parameters=*/ValueRange{}, controls, targets, negatedControls);
  cudaq::quake::threadWireResults(operation, controls, targets);
  target = targets.front();
}

static llvm::SmallVector<Value>
emitCircuit(OpBuilder &builder, Location location, ValueRange inputs,
            llvm::ArrayRef<bool> controlPolarities,
            const cudaq::synth::Circuit &circuit) {
  ValueRange controlInputs = inputs.drop_back();
  llvm::SmallVector<Value> controls(controlInputs.begin(), controlInputs.end());
  Value target = inputs.back();
  DenseBoolArrayAttr negatedControls =
      cudaq::opt::makeNegatedControlsAttr(builder, controlPolarities);
  for (cudaq::synth::Gate gate : llvm::reverse(circuit)) {
    switch (gate) {
    case cudaq::synth::Gate::H:
      emitGate<cudaq::quake::HOp>(builder, location, controls, target,
                                  negatedControls);
      break;
    case cudaq::synth::Gate::S:
      emitGate<cudaq::quake::SOp>(builder, location, controls, target,
                                  negatedControls);
      break;
    case cudaq::synth::Gate::T:
      emitGate<cudaq::quake::TOp>(builder, location, controls, target,
                                  negatedControls);
      break;
    case cudaq::synth::Gate::X:
      emitGate<cudaq::quake::XOp>(builder, location, controls, target,
                                  negatedControls);
      break;
    case cudaq::synth::Gate::W:
      Value angle =
          cudaq::opt::factory::createF64Constant(location, builder, M_PI_4);
      llvm::SmallVector<Value> targets{target};
      auto resultTypes = cudaq::quake::getWireResultTypes(controls, targets);
      auto phase = cudaq::quake::PhaseOp::create(
          builder, location, resultTypes, /*is_adj=*/false, ValueRange{angle},
          controls, targets, negatedControls);
      cudaq::quake::threadWireResults(phase, controls, targets);
      target = targets.front();
      break;
    }
  }
  controls.push_back(target);
  return controls;
}

static void normalizeBlock(Block &block) {
  llvm::SmallVector<Candidate, 0> candidates;
  auto chains = cudaq::opt::collectSingleQubitChains(block, isExactGate);

  for (cudaq::opt::SingleQubitChain &chain : chains) {
    // A single exact gate cannot improve the T-count or emitted gate count
    // used by this pass, so it does not need normal-form construction.
    if (chain.operations.size() == 1)
      continue;

    cudaq::synth::Circuit inputCircuit = buildMatrixProduct(chain.operations);
    cudaq::synth::Circuit normalized = inputCircuit.normalized();
    if (emittedCost(normalized) >= inputCost(chain.operations))
      continue;

    Candidate candidate;
    candidate.controlPolarities = getPolarities(chain.operations.front());
    candidate.chain = std::move(chain);
    candidate.normalized = std::move(normalized);
    candidates.push_back(std::move(candidate));
  }

  // Later candidates are rewritten first so recorded endpoints for earlier
  // chains remain valid throughout block mutation.
  for (Candidate &candidate : llvm::reverse(candidates)) {
    OpBuilder builder(candidate.chain.operations.front());
    llvm::SmallVector<Value> outputs =
        emitCircuit(builder, candidate.chain.operations.front()->getLoc(),
                    candidate.chain.inputs, candidate.controlPolarities,
                    candidate.normalized);
    cudaq::opt::replaceSingleQubitChain(std::move(candidate.chain), outputs);
  }
}

void NormalizeSingleQubitCliffordTPass::runOnOperation() {
  func::FuncOp function = getOperation();
  if (cudaq::runtime::isQuantumOptimizationDisabled(function))
    return;
  cudaq::opt::forEachNestedBlock(function.getBody(), normalizeBlock);
}
