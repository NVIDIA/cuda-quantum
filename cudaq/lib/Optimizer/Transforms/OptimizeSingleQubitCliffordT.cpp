/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
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
#include "mlir/Interfaces/CallInterfaces.h"
#include <cmath>
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

struct ExactWireOp {
  Operation *operation;
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> outputs;
  llvm::SmallVector<bool> controlPolarities;
  ExactGate gate;
  bool isAdj;
};

struct ScopeStep {
  Value wire;
  OpOperand *continueOperand;
};

struct ScalarWireStep {
  Operation *operation;
  std::optional<ScopeStep> scopeStep;
};

struct Candidate {
  llvm::SmallVector<Operation *> operations;
  llvm::SmallVector<llvm::SmallVector<ScopeStep>> scopeSteps;
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> outputs;
  llvm::SmallVector<bool> controlPolarities;
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

// Accept one-target exact gates whose complete predicate is in scalar linear
// form. Every other operand or result shape is a chain boundary.
static std::optional<ExactWireOp> getExactWireOp(Operation *operation) {
  std::optional<ExactGate> gate = getExactGate(operation);
  if (!gate)
    return std::nullopt;

  auto gateInterface = dyn_cast<cudaq::quake::OperatorInterface>(operation);
  auto flow = cudaq::quake::detail::getScalarWireFlow(operation);
  if (!gateInterface || !flow || gateInterface.getTargets().size() != 1)
    return std::nullopt;

  return ExactWireOp{operation,
                     std::move(flow->inputs),
                     std::move(flow->results),
                     cudaq::quake::getControlPolarities(gateInterface),
                     *gate,
                     gateInterface.isAdj()};
}

// Returns whether `nested` is inside `outer` through only single-block
// `cc.scope` operations. Any other enclosing region prevents traversal.
static bool entersSingleBlockLexicalScopesOnly(Block *nested, Block *outer) {
  while (nested != outer) {
    if (!nested)
      return false;
    auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(nested->getParentOp());
    if (!scope || scope.getAtomicQuantumRegionAttr() ||
        !scope.getInitRegion().hasOneBlock())
      return false;
    nested = scope->getBlock();
  }
  return true;
}

/// Return whether an operation can be followed as a direct scalar-wire step.
/// Calls, region operations, and terminators require control-flow semantics
/// that this pass deliberately does not model.
static bool isDirectScalarWireStep(Operation *operation) {
  return !isa<CallOpInterface>(operation) && operation->getNumRegions() == 0 &&
         !operation->hasTrait<OpTrait::IsTerminator>();
}

// Follow the unique scalar-wire use forward. A direct use reaches its user;
// a `cc.continue` use reaches the matching result of its enclosing scope.
static std::optional<ScalarWireStep> traverseScalarWire(Value wire) {
  if (!isa<cudaq::quake::WireType>(wire.getType()) || !wire.hasOneUse())
    return std::nullopt;

  OpOperand *use = &*wire.getUses().begin();
  Operation *user = use->getOwner();
  if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(user)) {
    auto scope = dyn_cast<cudaq::cc::ScopeOp>(cont->getParentOp());
    if (!scope || scope.getAtomicQuantumRegionAttr() ||
        !scope.getInitRegion().hasOneBlock() ||
        scope.getInitRegion().front().getTerminator() != user ||
        cont.getNumOperands() != scope->getNumResults())
      return std::nullopt;
    unsigned index = use->getOperandNumber();
    if (index >= scope->getNumResults() ||
        !isa<cudaq::quake::WireType>(scope->getResult(index).getType()))
      return std::nullopt;
    Value result = scope->getResult(index);
    if (!result.hasOneUse())
      return std::nullopt;
    return ScalarWireStep{scope, ScopeStep{result, use}};
  }
  if (!isDirectScalarWireStep(user) ||
      !entersSingleBlockLexicalScopesOnly(user->getBlock(),
                                          wire.getParentBlock()))
    return std::nullopt;
  return ScalarWireStep{user, std::nullopt};
}

struct WirePathEnd {
  Operation *operation;
  Value wire;
};

// Follow one tuple lane through transparent scopes to its next direct user.
static std::optional<WirePathEnd>
traceWire(Value wire, llvm::SmallVectorImpl<ScopeStep> &scopeSteps) {
  auto step = traverseScalarWire(wire);
  while (step && step->scopeStep) {
    scopeSteps.push_back(*step->scopeStep);
    wire = step->scopeStep->wire;
    step = traverseScalarWire(wire);
  }
  return step ? std::optional<WirePathEnd>{WirePathEnd{step->operation, wire}}
              : std::nullopt;
}

// A controlled chain continues only when every output lane reaches the same
// gate at its corresponding input position with the same ordered predicate.
static std::optional<ExactWireOp> matchNextExactGate(
    const ExactWireOp &current,
    llvm::MutableArrayRef<llvm::SmallVector<ScopeStep>> scopeSteps) {
  llvm::SmallVector<std::optional<WirePathEnd>> pathEnds;
  pathEnds.reserve(current.outputs.size());
  for (auto [output, steps] : llvm::zip(current.outputs, scopeSteps))
    pathEnds.push_back(traceWire(output, steps));

  if (llvm::any_of(pathEnds, [](const auto &path) { return !path; }))
    return std::nullopt;

  Operation *nextOperation = pathEnds.front()->operation;
  if (llvm::any_of(pathEnds, [&](const auto &path) {
        return path->operation != nextOperation;
      }))
    return std::nullopt;

  std::optional<ExactWireOp> next = getExactWireOp(nextOperation);
  if (!next || next->inputs.size() != current.outputs.size() ||
      next->controlPolarities != current.controlPolarities)
    return std::nullopt;

  for (auto [index, path] : llvm::enumerate(pathEnds))
    if (next->inputs[index] != path->wire)
      return std::nullopt;
  return next;
}

// Collect a maximal chain through exactly-once scalar-wire values. Unsupported
// operations and non-linear wire flow terminate the chain.
static llvm::SmallVector<ExactWireOp> collectLinearChain(
    Operation *operation, llvm::SmallDenseSet<Operation *> &collected,
    llvm::SmallVectorImpl<llvm::SmallVector<ScopeStep>> &scopeSteps) {
  if (collected.contains(operation))
    return {};

  std::optional<ExactWireOp> first = getExactWireOp(operation);
  if (!first || llvm::any_of(first->inputs,
                             [](Value input) { return !input.hasOneUse(); }))
    return {};

  scopeSteps.resize(first->inputs.size());
  llvm::SmallVector<ExactWireOp> chain;
  std::optional<ExactWireOp> current = std::move(first);
  while (current) {
    chain.push_back(*current);
    collected.insert(current->operation);
    if (llvm::any_of(current->outputs,
                     [](Value output) { return !output.hasOneUse(); }))
      break;
    std::optional<ExactWireOp> nextGate =
        matchNextExactGate(*current, scopeSteps);
    if (!nextGate)
      break;
    current = std::move(nextGate);
  }
  return chain;
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
buildMatrixProduct(llvm::ArrayRef<ExactWireOp> operations) {
  cudaq::synth::Circuit circuit;
  for (const ExactWireOp &operation : llvm::reverse(operations))
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

static CircuitCost inputCost(llvm::ArrayRef<ExactWireOp> chain) {
  return {static_cast<int>(llvm::count_if(chain,
                                          [](const ExactWireOp &gate) {
                                            return gate.gate == ExactGate::T;
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
  llvm::SmallDenseSet<Operation *> collected;

  for (Operation &operation : block) {
    llvm::SmallVector<llvm::SmallVector<ScopeStep>> scopeSteps;
    llvm::SmallVector<ExactWireOp> chain =
        collectLinearChain(&operation, collected, scopeSteps);
    if (chain.empty())
      continue;

    // A single exact gate cannot improve the T-count or emitted gate count
    // used by this pass, so it does not need normal-form construction.
    if (chain.size() == 1)
      continue;

    cudaq::synth::Circuit inputCircuit = buildMatrixProduct(chain);
    cudaq::synth::Circuit normalized = inputCircuit.normalized();
    if (emittedCost(normalized) >= inputCost(chain))
      continue;

    Candidate candidate;
    candidate.inputs = chain.front().inputs;
    candidate.outputs = chain.back().outputs;
    candidate.controlPolarities = chain.front().controlPolarities;
    candidate.scopeSteps = std::move(scopeSteps);
    candidate.normalized = std::move(normalized);
    for (const ExactWireOp &gate : chain)
      candidate.operations.push_back(gate.operation);
    candidates.push_back(std::move(candidate));
  }

  // Later candidates are rewritten first so recorded endpoints for earlier
  // chains remain valid throughout block mutation.
  for (Candidate &candidate : llvm::reverse(candidates)) {
    OpBuilder builder(candidate.operations.front());
    llvm::SmallVector<Value> outputs = emitCircuit(
        builder, candidate.operations.front()->getLoc(), candidate.inputs,
        candidate.controlPolarities, candidate.normalized);
    for (auto [output, original, steps] :
         llvm::zip(outputs, candidate.outputs, candidate.scopeSteps)) {
      Value replacement = output;
      for (ScopeStep &scopeStep : steps) {
        scopeStep.continueOperand->set(replacement);
        replacement = scopeStep.wire;
      }
      original.replaceAllUsesWith(replacement);
    }
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
  ModuleOp module = getOperation();
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    optimizeRegion(function.getBody());
}
