/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Optimizer/Transforms/ScalarWireTraversal.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <compare>
#include <optional>
#include <utility>

namespace cudaq::opt {
#define GEN_PASS_DEF_OPTIMIZESINGLEQUBITCLIFFORDT
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

enum class ExactGate { H, S, T, X, Y, Z };

struct UnaryWireOp {
  Operation *operation;
  Value input;
  Value output;
  ExactGate gate;
  bool isAdj;
};

struct Candidate {
  llvm::SmallVector<Operation *> operations;
  llvm::SmallVector<cudaq::opt::ScalarWireStep> scopeSteps;
  Value input;
  Value output;
  cudaq::synth::Circuit normalized;
};

struct CircuitCost {
  int tCount;
  std::size_t emittedGateCount;

  auto operator<=>(const CircuitCost &) const = default;
};

class OptimizeSingleQubitCliffordTPass
    : public cudaq::opt::impl::OptimizeSingleQubitCliffordTBase<
          OptimizeSingleQubitCliffordTPass> {
public:
  using OptimizeSingleQubitCliffordTBase::OptimizeSingleQubitCliffordTBase;
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

// Accept only uncontrolled, unary scalar-wire operations. Every other
// operand/result shape is a chain boundary and remains unchanged.
static std::optional<UnaryWireOp> getUnaryWireOp(Operation *operation) {
  std::optional<ExactGate> gate = getExactGate(operation);
  if (!gate)
    return std::nullopt;

  auto gateInterface = dyn_cast<cudaq::quake::OperatorInterface>(operation);
  if (!gateInterface || !gateInterface.getControls().empty() ||
      gateInterface.getTargets().size() != 1 ||
      !isa<cudaq::quake::WireType>(gateInterface.getTargets()[0].getType()) ||
      operation->getNumResults() != 1 ||
      !isa<cudaq::quake::WireType>(operation->getResult(0).getType()))
    return std::nullopt;

  return UnaryWireOp{operation, gateInterface.getTargets()[0],
                     operation->getResult(0), *gate, gateInterface.isAdj()};
}

// Follow a unique scalar-wire use to another unary gate. The shared traversal
// owns direct def-use steps and lexical-scope forwarding; this pass only
// decides whether the reached operation is an exact unary gate.
static std::optional<UnaryWireOp> getNextUnaryWireOp(
    const UnaryWireOp &current,
    llvm::SmallVectorImpl<cudaq::opt::ScalarWireStep> &scopeSteps) {
  std::optional<cudaq::opt::ScalarWireStep> step =
      cudaq::opt::traverseScalarWire(
          current.output, cudaq::opt::ScalarWireTraversalDirection::Forward);
  while (step && step->continueOperand) {
    scopeSteps.push_back(*step);
    step = cudaq::opt::traverseScalarWire(
        step->wire, cudaq::opt::ScalarWireTraversalDirection::Forward);
  }
  return step ? getUnaryWireOp(step->operation) : std::nullopt;
}

// Collect a maximal chain through exactly-once scalar-wire values. Unsupported
// operations and non-linear wire flow terminate the chain.
static llvm::SmallVector<UnaryWireOp> collectLinearChain(
    Operation *operation, llvm::SmallDenseSet<Operation *> &collected,
    llvm::SmallVectorImpl<cudaq::opt::ScalarWireStep> &scopeSteps) {
  if (collected.contains(operation))
    return {};

  std::optional<UnaryWireOp> first = getUnaryWireOp(operation);
  if (!first || !first->input.hasOneUse())
    return {};

  llvm::SmallVector<UnaryWireOp> chain;
  std::optional<UnaryWireOp> current = first;
  while (current) {
    chain.push_back(*current);
    collected.insert(current->operation);
    if (!current->output.hasOneUse())
      break;
    std::optional<UnaryWireOp> nextGate =
        getNextUnaryWireOp(*current, scopeSteps);
    if (!nextGate)
      break;
    current = nextGate;
  }
  return chain;
}

static void appendExactGate(cudaq::synth::Circuit &circuit,
                            const UnaryWireOp &operation) {
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
buildMatrixProduct(llvm::ArrayRef<UnaryWireOp> operations) {
  cudaq::synth::Circuit circuit;
  for (const UnaryWireOp &operation : llvm::reverse(operations))
    appendExactGate(circuit, operation);
  return circuit;
}

// Prefer lower T-count, then fewer emitted Clifford gates. Scalar W phases
// are not emitted as Quake operations.
static CircuitCost emittedCost(const cudaq::synth::Circuit &circuit) {
  std::size_t emittedGateCount = 0;
  for (cudaq::synth::Gate gate : circuit)
    emittedGateCount += gate != cudaq::synth::Gate::W;
  return {circuit.t_count(), emittedGateCount};
}

static CircuitCost inputCost(llvm::ArrayRef<UnaryWireOp> chain) {
  return {static_cast<int>(llvm::count_if(chain,
                                          [](const UnaryWireOp &gate) {
                                            return gate.gate == ExactGate::T;
                                          })),
          chain.size()};
}

template <typename OpTy>
static Value emitGate(OpBuilder &builder, Location location, Value input) {
  auto operation =
      OpTy::create(builder, location, TypeRange{input.getType()},
                   /*is_adj=*/false, /*parameters=*/ValueRange{},
                   /*controls=*/ValueRange{}, /*targets=*/ValueRange{input},
                   /*negated_qubit_controls=*/DenseBoolArrayAttr{});
  return operation.getWires()[0];
}

static Value emitCircuit(OpBuilder &builder, Location location, Value input,
                         const cudaq::synth::Circuit &circuit) {
  Value current = input;
  for (cudaq::synth::Gate gate : llvm::reverse(circuit)) {
    switch (gate) {
    case cudaq::synth::Gate::H:
      current = emitGate<cudaq::quake::HOp>(builder, location, current);
      break;
    case cudaq::synth::Gate::S:
      current = emitGate<cudaq::quake::SOp>(builder, location, current);
      break;
    case cudaq::synth::Gate::T:
      current = emitGate<cudaq::quake::TOp>(builder, location, current);
      break;
    case cudaq::synth::Gate::X:
      current = emitGate<cudaq::quake::XOp>(builder, location, current);
      break;
    case cudaq::synth::Gate::W:
      // TODO: emit an anchored `quake.phase(pi/4)` once Quake supports it.
      break;
    }
  }
  return current;
}

static void optimizeBlock(Block &block) {
  llvm::SmallVector<Candidate> candidates;
  llvm::SmallDenseSet<Operation *> collected;

  for (Operation &operation : block) {
    llvm::SmallVector<cudaq::opt::ScalarWireStep> scopeSteps;
    llvm::SmallVector<UnaryWireOp> chain =
        collectLinearChain(&operation, collected, scopeSteps);
    if (chain.empty())
      continue;

    cudaq::synth::Circuit inputCircuit = buildMatrixProduct(chain);
    cudaq::synth::Circuit normalized = inputCircuit.normalized();
    if (emittedCost(normalized) >= inputCost(chain))
      continue;

    Candidate candidate;
    candidate.input = chain.front().input;
    candidate.output = chain.back().output;
    candidate.scopeSteps = std::move(scopeSteps);
    candidate.normalized = std::move(normalized);
    for (const UnaryWireOp &gate : chain)
      candidate.operations.push_back(gate.operation);
    candidates.push_back(std::move(candidate));
  }

  // Later candidates are rewritten first so recorded endpoints for earlier
  // chains remain valid throughout block mutation.
  for (Candidate &candidate : llvm::reverse(candidates)) {
    OpBuilder builder(candidate.operations.front());
    Value output = emitCircuit(builder, candidate.operations.front()->getLoc(),
                               candidate.input, candidate.normalized);
    // Thread the normalized output through lexical-scope forwarding in
    // traversal order, then replace the original chain's final value with the
    // last scope result visible at that point.
    Value replacement = output;
    for (cudaq::opt::ScalarWireStep &scopeStep : candidate.scopeSteps) {
      scopeStep.continueOperand->set(replacement);
      replacement = scopeStep.wire;
    }
    candidate.output.replaceAllUsesWith(replacement);
    for (Operation *operation : llvm::reverse(candidate.operations))
      operation->erase();
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

void OptimizeSingleQubitCliffordTPass::runOnOperation() {
  // TODO: Support controlled quake.apply after Quake can represent scalar W
  // phase factors.
  ModuleOp module = getOperation();
  WalkResult result = module.walk([](cudaq::quake::ApplyOp apply) {
    return apply.getControls().empty() ? WalkResult::advance()
                                       : WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return;

  for (func::FuncOp function : module.getOps<func::FuncOp>())
    optimizeRegion(function.getBody());
}
