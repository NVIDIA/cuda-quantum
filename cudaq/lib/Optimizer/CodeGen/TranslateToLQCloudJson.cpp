/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nlohmann/json.hpp"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/CodeGen/Emitter.h"
#include "cudaq/Optimizer/CodeGen/LQCloudJsonEmitter.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <set>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

struct CircuitState {
  cudaq::quake::AllocaOp allocation;
  // Number of qubits in the combined allocation. After the qubit-mapping
  // pass the circuit operates directly on physical lines (the combined
  // allocation is re-indexed accordingly), so instruction indices are
  // emitted as-is; the runtime restores the user's virtual ordering from
  // the mapping_reorder_idx attribute.
  std::size_t numberOfQubits = 0;
  std::set<std::size_t> measuredQubits;
  bool isRun = false;
  std::size_t numberOfMeasurements = 0;
  llvm::DenseMap<Value, std::size_t> measurementEvents;
  std::vector<std::size_t> returnedEvents;
  std::set<std::size_t> synchronizedQubits;
  bool resetPrefixOpen = true;
  std::vector<nlohmann::json> instructions;
};

nlohmann::json instruction(StringRef name, std::vector<std::size_t> qubits,
                           std::vector<std::size_t> clbits = {},
                           std::vector<double> parameters = {}) {
  return {{"name", name.str()},
          {"qubits", std::move(qubits)},
          {"clbits", std::move(clbits)},
          {"params", std::move(parameters)}};
}

std::optional<std::size_t> getQubitIndex(Value value, CircuitState &state) {
  auto extract = value.getDefiningOp<cudaq::quake::ExtractRefOp>();
  if (!extract || extract.getVeq() != state.allocation.getRefOrVec())
    return std::nullopt;

  std::optional<int64_t> index;
  if (extract.hasConstantIndex())
    index = extract.getConstantIndex();
  else
    index = cudaq::getIndexValueAsInt(extract.getIndex());
  if (!index || *index < 0 ||
      static_cast<std::size_t>(*index) >= state.numberOfQubits)
    return std::nullopt;
  return static_cast<std::size_t>(*index);
}

std::optional<std::vector<std::size_t>>
getNativeQubitIndices(CircuitState &state, cudaq::quake::OperatorInterface op,
                      StringRef gateName, std::size_t expectedControls,
                      std::size_t expectedTargets,
                      std::size_t expectedParameters) {
  if (op.getControls().size() != expectedControls ||
      op.getTargets().size() != expectedTargets ||
      op.getParameters().size() != expectedParameters) {
    op.emitOpError() << "LQCloud " << gateName << " expects "
                     << expectedControls << " control(s), " << expectedTargets
                     << " target(s), and " << expectedParameters
                     << " parameter(s)";
    return std::nullopt;
  }

  std::vector<std::size_t> indices;
  indices.reserve(expectedControls + expectedTargets);
  for (auto values : {op.getControls(), op.getTargets()}) {
    for (Value value : values) {
      auto index = getQubitIndex(value, state);
      if (!index) {
        op.emitOpError() << "LQCloud " << gateName
                         << " requires constant in-range qubit indices";
        return std::nullopt;
      }
      indices.push_back(*index);
    }
  }
  return indices;
}

LogicalResult emitReset(CircuitState &state, cudaq::quake::ResetOp reset) {
  if (!state.isRun && !state.measuredQubits.empty())
    return reset.emitOpError("LQCloud resets after measurement require run");
  auto index = getQubitIndex(reset.getTargets(), state);
  if (!index)
    return reset.emitOpError(
        "LQCloud reset requires a constant in-range qubit index");
  if (!state.resetPrefixOpen) {
    state.instructions.push_back(instruction("reset", {*index}));
    if (state.isRun)
      state.synchronizedQubits.erase(*index);
    else
      state.synchronizedQubits.clear();
  }
  return success();
}

LogicalResult emitBarrier(CircuitState &state,
                          cudaq::quake::BarrierOp barrier) {
  std::vector<std::size_t> qubits;
  for (Value target : barrier.getTargets()) {
    auto index = getQubitIndex(target, state);
    if (!index)
      return barrier.emitOpError(
          "LQCloud barrier requires constant in-range qubit indices");
    qubits.push_back(*index);
  }
  std::sort(qubits.begin(), qubits.end());
  if (std::adjacent_find(qubits.begin(), qubits.end()) != qubits.end())
    return barrier.emitOpError("LQCloud barrier requires distinct qubits");

  // In particular, the first full barrier can also synchronize initial resets.
  // Never merge across a gate or measurement, or join disjoint target sets.
  if (state.instructions.back().at("name") != "barrier" ||
      state.instructions.back().at("qubits") != qubits)
    state.instructions.push_back(instruction("barrier", qubits));
  state.resetPrefixOpen = false;
  if (!state.isRun)
    state.synchronizedQubits.clear();
  state.synchronizedQubits.insert(qubits.begin(), qubits.end());
  return success();
}

LogicalResult emitOperator(CircuitState &state,
                           cudaq::quake::OperatorInterface op) {
  if (!state.isRun && !state.measuredQubits.empty())
    return op.emitOpError(
        "LQCloud does not support mid-circuit measurement or quantum "
        "operations after measurement");
  if (op.getNegatedControls())
    return op.emitOpError("LQCloud does not support negated controls");

  if (isa<cudaq::quake::HOp>(op.getOperation())) {
    auto qubits = getNativeQubitIndices(state, op, "H", 0, 1, 0);
    if (!qubits)
      return failure();
    state.instructions.push_back(instruction("h", std::move(*qubits)));
  } else if (auto x = dyn_cast<cudaq::quake::XOp>(op.getOperation());
             x && !x.isAdj()) {
    auto qubits = getNativeQubitIndices(state, op, "X", 0, 1, 0);
    if (!qubits)
      return failure();
    state.instructions.push_back(instruction("x", std::move(*qubits)));
  } else if (auto y = dyn_cast<cudaq::quake::YOp>(op.getOperation());
             y && !y.isAdj()) {
    auto qubits = getNativeQubitIndices(state, op, "Y", 0, 1, 0);
    if (!qubits)
      return failure();
    state.instructions.push_back(instruction("y", std::move(*qubits)));
  } else if (auto rz = dyn_cast<cudaq::quake::RzOp>(op.getOperation())) {
    auto qubits = getNativeQubitIndices(state, op, "Rz", 0, 1, 1);
    if (!qubits)
      return failure();
    auto angle = cudaq::getParameterValueAsDouble(rz.getParameter());
    if (!angle || !std::isfinite(*angle))
      return rz.emitOpError("LQCloud requires a finite constant Rz angle");
    if (rz.isAdj())
      *angle = -*angle;
    state.instructions.push_back(
        instruction("rz", std::move(*qubits), {}, {*angle}));
  } else if (auto z = dyn_cast<cudaq::quake::ZOp>(op.getOperation())) {
    auto qubits = getNativeQubitIndices(state, op, "CZ", 1, 1, 0);
    if (!qubits)
      return failure();
    if ((*qubits)[0] == (*qubits)[1])
      return z.emitOpError(
          "LQCloud CZ control and target must be different qubits");
    state.instructions.push_back(instruction("cz", std::move(*qubits)));
  } else {
    return op.emitOpError()
           << "LQCloud does not support operation '" << op->getName() << "'";
  }

  state.resetPrefixOpen = false;
  if (state.isRun) {
    for (auto values : {op.getControls(), op.getTargets()})
      for (Value value : values)
        state.synchronizedQubits.erase(*getQubitIndex(value, state));
  } else {
    state.synchronizedQubits.clear();
  }
  return success();
}

LogicalResult emitMeasurement(CircuitState &state,
                              cudaq::quake::MzOp measurement) {
  if (measurement.getTargets().size() != 1 ||
      !isa<cudaq::quake::RefType>(measurement.getTargets().front().getType()) ||
      !isa<cudaq::quake::MeasureType, cudaq::cc::MeasureHandleType>(
          measurement.getMeasOut().getType()))
    return measurement.emitOpError("LQCloud requires scalar Z measurements");
  if (!state.isRun && measurement.getRegisterName())
    return measurement.emitOpError(
        "LQCloud does not support named measurement registers");

  const auto event = state.numberOfMeasurements;
  if (state.isRun) {
    state.measurementEvents[measurement.getMeasOut()] = event;
    ++state.numberOfMeasurements;
  }

  auto index = getQubitIndex(measurement.getTargets().front(), state);
  if (!index)
    return measurement.emitOpError(
        "LQCloud measurement requires a constant in-range qubit index");
  if (!state.measuredQubits.insert(*index).second && !state.isRun)
    return measurement.emitOpError(
        "LQCloud does not support repeated measurements");

  if (state.isRun
          ? !state.synchronizedQubits.count(*index)
          : state.measuredQubits.size() == 1 &&
                state.synchronizedQubits.size() != state.numberOfQubits) {
    std::vector<std::size_t> allQubits(state.numberOfQubits);
    std::iota(allQubits.begin(), allQubits.end(), 0);
    state.instructions.push_back(instruction("barrier", allQubits));
    state.synchronizedQubits.insert(allQubits.begin(), allQubits.end());
  }
  state.resetPrefixOpen = false;
  state.synchronizedQubits.erase(*index);
  state.instructions.push_back(
      instruction("measure", {*index}, {state.isRun ? event : *index}));
  return success();
}

Value stripReturnCasts(Value value) {
  while (auto cast = value.getDefiningOp<cudaq::cc::CastOp>())
    value = cast.getValue();
  return value;
}

std::optional<std::pair<Value, std::size_t>> returnSlot(Value pointer) {
  pointer = stripReturnCasts(pointer);
  if (auto address = pointer.getDefiningOp<cudaq::cc::ComputePtrOp>()) {
    if (address.getNumIndices() != 1)
      return std::nullopt;
    auto index = address.getConstantIndex(0);
    if (!index || *index < 0)
      return std::nullopt;
    return std::make_pair(stripReturnCasts(address.getBase()), *index);
  }
  return std::make_pair(pointer, 0);
}

LogicalResult collectRunReturn(func::FuncOp func,
                               ArrayRef<Operation *> operations,
                               CircuitState &state) {
  cudaq::quake::EvinceOp output;
  for (Operation *op : operations)
    if (auto evince = dyn_cast<cudaq::quake::EvinceOp>(op)) {
      if (output || evince.getArgs().size() != 1)
        return evince.emitOpError(
            "LQCloud run requires one returned measurement bool list");
      output = evince;
    }
  if (!output)
    return func.emitOpError(
        "LQCloud run requires a returned measurement bool list");
  Value value = output.getArgs().front();
  // An inlined run entry can retain a scope around the quantum kernel.
  while (auto scope = value.getDefiningOp<cudaq::cc::ScopeOp>()) {
    auto terminator = dyn_cast<cudaq::cc::ContinueOp>(
        scope.getInitRegion().front().getTerminator());
    auto result = dyn_cast<OpResult>(value);
    if (!terminator || !result ||
        result.getResultNumber() >= terminator.getNumOperands())
      return output.emitOpError(
          "LQCloud cannot resolve the returned scope value");
    value = terminator.getOperand(result.getResultNumber());
  }
  auto type = dyn_cast<cudaq::cc::SequenceType>(value.getType());
  auto list = value.getDefiningOp<cudaq::cc::SequenceInitOp>();
  if (!type || !type.getElementType().isInteger(1) || !list)
    return output.emitOpError(
        "LQCloud run supports only a static measurement bool list");
  auto size = [&]() -> std::optional<int64_t> {
    if (Value length = list.getLength())
      return cudaq::getIndexValueAsInt(length);
    if (auto pointer =
            dyn_cast<cudaq::cc::PointerType>(list.getBuffer().getType()))
      if (auto array = dyn_cast<cudaq::cc::ArrayType>(pointer.getElementType());
          array && !array.isUnknownSize())
        return array.getSize();
    return std::nullopt;
  }();
  if (!size || *size <= 0 ||
      static_cast<std::size_t>(*size) != state.numberOfMeasurements)
    return output.emitOpError("LQCloud run must return every measurement once");
  Value buffer = stripReturnCasts(list.getBuffer());
  std::vector<std::optional<std::size_t>> events(*size);
  for (Operation *op : operations) {
    auto store = dyn_cast<cudaq::cc::StoreOp>(op);
    if (!store)
      continue;
    auto slot = returnSlot(store.getPtrvalue());
    if (!slot || slot->first != buffer)
      continue;
    Value bit = stripReturnCasts(store.getValue());
    if (auto extend = bit.getDefiningOp<arith::ExtUIOp>())
      bit = extend.getIn();
    auto discriminate = bit.getDefiningOp<cudaq::quake::DiscriminateOp>();
    if (slot->second >= events.size())
      return store.emitOpError("LQCloud run return slot exceeds the bool list");
    // A frontend list comprehension can initialize slots before the unrolled
    // measurement stores. Only the final direct measurement is observable.
    if (!discriminate) {
      if (!bit.getDefiningOp<arith::ConstantOp>())
        return store.emitOpError(
            "LQCloud run return slots must be direct measurements");
      events[slot->second].reset();
      continue;
    }
    auto event = state.measurementEvents.find(discriminate.getMeasurement());
    if (event == state.measurementEvents.end())
      return store.emitOpError(
          "LQCloud run return references an unknown measurement");
    events[slot->second] = event->second;
  }
  std::set<std::size_t> seen;
  for (const auto &event : events) {
    if (!event || !seen.insert(*event).second)
      return output.emitOpError("LQCloud run requires each measurement exactly "
                                "once in the return list");
    state.returnedEvents.push_back(*event);
  }
  return success();
}

LogicalResult emitEntryPoint(nlohmann::json &json, func::FuncOp func) {
  if (func.getBody().getBlocks().size() != 1)
    return func.emitOpError("LQCloud requires a single straight-line block");
  if (func.getNumArguments() != 0 || func.getNumResults() != 0)
    return func.emitOpError(
        "LQCloud requires a fully specialized void entrypoint");
  for (NamedAttribute attribute : func->getAttrs()) {
    StringRef name = attribute.getName().strref();
    // mapping_v2p and mapping_reorder_idx are attached by the qubit-mapping
    // pass: the placement is consumed below and the runtime restores the
    // user's qubit ordering from the reorder indices. Any other mapping
    // attribute carries layout information we cannot interpret.
    if (name.starts_with("mapping") && name != "mapping_v2p" &&
        name != "mapping_reorder_idx")
      return func.emitOpError()
             << "LQCloud does not support mapping attribute '" << name << "'";
  }

  SmallVector<cudaq::quake::AllocaOp> allocations;
  SmallVector<Operation *> operations;
  auto walked = func.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (operation == func.getOperation())
      return WalkResult::advance();
    if (operation->getNumRegions() && !isa<cudaq::cc::ScopeOp>(operation)) {
      operation->emitOpError("LQCloud requires a static straight-line circuit");
      return WalkResult::interrupt();
    }
    if (auto scope = dyn_cast<cudaq::cc::ScopeOp>(operation);
        scope && !scope.getInitRegion().hasOneBlock()) {
      scope.emitOpError("LQCloud requires single-block scopes");
      return WalkResult::interrupt();
    }
    operations.push_back(operation);
    if (auto allocation = dyn_cast<cudaq::quake::AllocaOp>(operation))
      allocations.push_back(allocation);
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();
  if (allocations.size() != 1)
    return func.emitOpError(
        "LQCloud requires exactly one combined quantum allocation");
  auto allocation = allocations.front();
  auto allocationType = dyn_cast<cudaq::quake::VeqType>(allocation.getType());
  if (!allocationType || !allocationType.hasSpecifiedSize() ||
      allocationType.getSize() == 0)
    return allocation.emitOpError(
        "LQCloud requires a non-empty, statically sized quantum allocation");

  CircuitState state;
  state.allocation = allocation;
  state.numberOfQubits = allocationType.getSize();
  state.isRun = func->hasAttr(cudaq::runtime::enableCudaqRun);
  for (std::size_t index = 0; index < state.numberOfQubits; ++index)
    state.instructions.push_back(instruction("reset", {index}));
  std::vector<std::size_t> allQubits(state.numberOfQubits);
  std::iota(allQubits.begin(), allQubits.end(), 0);
  state.synchronizedQubits.insert(allQubits.begin(), allQubits.end());
  state.instructions.push_back(instruction("barrier", std::move(allQubits)));
  for (Operation *op : operations) {
    Operation &operation = *op;
    if (isa<func::ReturnOp, cudaq::cc::ContinueOp, cudaq::cc::ScopeOp,
            cudaq::quake::AllocaOp, cudaq::quake::DiscriminateOp>(operation))
      continue;
    if (auto deallocation = dyn_cast<cudaq::quake::DeallocOp>(operation)) {
      if (deallocation.getReference() != state.allocation.getRefOrVec())
        return deallocation.emitOpError(
            "LQCloud deallocation does not match the combined allocation");
      state.resetPrefixOpen = false;
      continue;
    }
    if (auto extract = dyn_cast<cudaq::quake::ExtractRefOp>(operation)) {
      if (!getQubitIndex(extract.getRef(), state))
        return extract.emitOpError(
            "LQCloud requires every qubit reference to use a constant index "
            "into the combined allocation");
      continue;
    }
    if (auto reset = dyn_cast<cudaq::quake::ResetOp>(operation)) {
      if (failed(emitReset(state, reset)))
        return failure();
      continue;
    }
    if (auto barrier = dyn_cast<cudaq::quake::BarrierOp>(operation)) {
      if (failed(emitBarrier(state, barrier)))
        return failure();
      continue;
    }
    if (auto measurement = dyn_cast<cudaq::quake::MzOp>(operation)) {
      if (failed(emitMeasurement(state, measurement)))
        return failure();
      continue;
    }
    if (auto quantumOperator =
            dyn_cast<cudaq::quake::OperatorInterface>(operation)) {
      if (failed(emitOperator(state, quantumOperator)))
        return failure();
      continue;
    }
    if (state.isRun && isa<cudaq::quake::EvinceOp>(operation))
      continue; // Checked against the measurement values after emission.
    auto dialect = operation.getName().getDialectNamespace();
    if (state.isRun) {
      if (isa<arith::ConstantOp, arith::ExtUIOp, arith::TruncIOp,
              cudaq::cc::AllocaOp, cudaq::cc::ComputePtrOp, cudaq::cc::CastOp,
              cudaq::cc::StoreOp, cudaq::cc::SequenceInitOp>(operation))
        continue;
    } else if (dialect == "cc" || dialect == "arith" || dialect == "llvm")
      continue;
    return operation.emitOpError() << "LQCloud does not support operation '"
                                   << operation.getName() << "'";
  }

  if (state.measuredQubits.empty())
    return func.emitOpError("LQCloud requires at least one measurement");

  if (state.isRun && failed(collectRunReturn(func, operations, state)))
    return failure();

  // Mapping's reorder indices refer to the measured physical qubits in
  // ascending order. Compact only these bits, excluding unmeasured ancillas.
  if (!state.isRun)
    for (auto &op : state.instructions)
      if (op.at("name") == "measure") {
        auto qubit = op.at("qubits").at(0).get<std::size_t>();
        op["clbits"] = {std::distance(state.measuredQubits.begin(),
                                      state.measuredQubits.find(qubit))};
      }

  json["n_qubits"] = state.numberOfQubits;
  json["n_clbits"] =
      state.isRun ? state.numberOfMeasurements : state.measuredQubits.size();
  if (state.isRun)
    json["run_return_indices"] = state.returnedEvents;
  json["instructions"] = std::move(state.instructions);
  return success();
}

LogicalResult emitModule(nlohmann::json &json, ModuleOp module) {
  func::FuncOp entryPoint;
  for (Operation &operation : module.getBody()->getOperations()) {
    auto func = dyn_cast<func::FuncOp>(operation);
    if (!func)
      return operation.emitOpError()
             << "LQCloud does not support top-level operation '"
             << operation.getName() << "'";
    if (!func->hasAttr(cudaq::entryPointAttrName))
      return func.emitOpError(
          "LQCloud does not support non-entrypoint functions");
    if (entryPoint)
      return module.emitError("LQCloud module has multiple entrypoints");
    entryPoint = func;
  }
  if (!entryPoint)
    return module.emitError("LQCloud module does not contain an entrypoint");
  return emitEntryPoint(json, entryPoint);
}

} // namespace

LogicalResult cudaq::translateToLQCloudJson(Operation *op,
                                            llvm::raw_ostream &os) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("LQCloud translation requires a builtin.module");

  nlohmann::json json;
  if (failed(emitModule(json, module)))
    return failure();
  os << json.dump(2) << '\n';
  return success();
}
