/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "ScalarWireChain.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <cmath>
#include <compare>
#include <optional>
#include <utility>

namespace cudaq::opt {
#define GEN_PASS_DEF_NORMALIZESINGLEQUBITCLIFFORDT
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

enum class ExactGate { H, S, T, X, Y, Z };

struct ExactWireOp {
  llvm::SmallVector<bool> controlPolarities;
  ExactGate gate;
  bool isAdj;
};

struct Candidate {
  cudaq::opt::ScalarWireChain chain;
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

static std::optional<ExactGate> getExactGate(Operation *operation) {
  if (isa<cudaq::quake::HOp>(operation))
    return ExactGate::H;
  if (isa<cudaq::quake::SOp>(operation))
    return ExactGate::S;
  if (isa<cudaq::quake::TOp>(operation))
    return ExactGate::T;
  if (isa<cudaq::quake::XOp>(operation))
    return ExactGate::X;
  if (isa<cudaq::quake::YOp>(operation))
    return ExactGate::Y;
  if (isa<cudaq::quake::ZOp>(operation))
    return ExactGate::Z;
  return std::nullopt;
}

// Classify one-target exact gates. The shared collector separately requires
// their complete predicates to have scalar linear wire flow.
static std::optional<ExactWireOp> getExactWireOp(Operation *operation) {
  std::optional<ExactGate> gate = getExactGate(operation);
  if (!gate)
    return std::nullopt;

  auto gateInterface = dyn_cast<cudaq::quake::OperatorInterface>(operation);
  if (!gateInterface || gateInterface.getTargets().size() != 1)
    return std::nullopt;

  return ExactWireOp{cudaq::quake::getControlPolarities(gateInterface), *gate,
                     gateInterface.isAdj()};
}

static ExactWireOp requireExactWireOp(Operation *operation) {
  auto exact = getExactWireOp(operation);
  assert(exact && "collected an unsupported exact gate");
  return *exact;
}

static void appendExactGate(cudaq::synth::Circuit &circuit,
                            const ExactWireOp &operation) {
  using cudaq::synth::Gate;
  switch (operation.gate) {
  case ExactGate::H:
    circuit.push_back(Gate::H);
    break;
  case ExactGate::S:
    if (operation.isAdj) {
      circuit.push_back(Gate::S);
      circuit.push_back(Gate::S);
    }
    circuit.push_back(Gate::S);
    break;
  case ExactGate::T:
    circuit.push_back(Gate::T);
    if (operation.isAdj) {
      circuit.push_back(Gate::S);
      circuit.push_back(Gate::S);
      circuit.push_back(Gate::S);
    }
    break;
  case ExactGate::X:
    circuit.push_back(Gate::X);
    break;
  case ExactGate::Y:
    circuit.push_back(Gate::W);
    circuit.push_back(Gate::W);
    circuit.push_back(Gate::X);
    circuit.push_back(Gate::S);
    circuit.push_back(Gate::S);
    break;
  case ExactGate::Z:
    circuit.push_back(Gate::S);
    circuit.push_back(Gate::S);
    break;
  }
}

static cudaq::synth::Circuit
buildMatrixProduct(llvm::ArrayRef<Operation *> operations) {
  cudaq::synth::Circuit circuit;
  for (Operation *operation : llvm::reverse(operations))
    appendExactGate(circuit, requireExactWireOp(operation));
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
  return {static_cast<int>(llvm::count_if(
              chain,
              [](Operation *operation) {
                return requireExactWireOp(operation).gate == ExactGate::T;
              })),
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

static void optimizeBlock(Block &block) {
  llvm::SmallVector<Candidate, 0> candidates;
  auto chains = cudaq::opt::collectScalarWireChains(
      block,
      [](Operation *operation) {
        return getExactWireOp(operation).has_value();
      },
      [](Operation *first, Operation *next) {
        return requireExactWireOp(first).controlPolarities ==
               requireExactWireOp(next).controlPolarities;
      });

  for (cudaq::opt::ScalarWireChain &chain : chains) {
    // A single exact gate cannot improve the T-count or emitted gate count
    // used by this pass, so it does not need normal-form construction.
    if (chain.operations.size() == 1)
      continue;

    cudaq::synth::Circuit inputCircuit = buildMatrixProduct(chain.operations);
    cudaq::synth::Circuit normalized = inputCircuit.normalized();
    if (emittedCost(normalized) >= inputCost(chain.operations))
      continue;

    Candidate candidate;
    candidate.controlPolarities =
        requireExactWireOp(chain.operations.front()).controlPolarities;
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
    cudaq::opt::replaceScalarWireChain(std::move(candidate.chain), outputs);
  }
}

static void optimizeRegion(Region &region) {
  for (Block &block : region) {
    optimizeBlock(block);
    for (Operation &operation : block)
      for (Region &nested : operation.getRegions())
        optimizeRegion(nested);
  }
}

void NormalizeSingleQubitCliffordTPass::runOnOperation() {
  ModuleOp module = getOperation();
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    optimizeRegion(function.getBody());
}
