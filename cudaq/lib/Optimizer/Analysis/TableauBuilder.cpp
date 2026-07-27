/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "stim.h"
#include "cudaq/Optimizer/Analysis/CircuitValidation.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <cmath>
#include <numeric>
#include <optional>

using namespace mlir;

namespace cudaq::opt {
namespace {

constexpr std::size_t W = stim::MAX_BITWORD_WIDTH;

/// The constant value of a floating-point angle operand, if it is defined by an
/// `arith.constant`.
std::optional<double> constantAngle(Value value) {
  if (auto cst =
          dyn_cast_if_present<arith::ConstantFloatOp>(value.getDefiningOp()))
    return cast<FloatAttr>(cst.getValue()).getValueAsDouble();
  return std::nullopt;
}

/// Compiles one straight-line Clifford kernel into a stabilizer tableau. The
/// tableau produced is the inverse of the kernel's operation.
class TableauBuilder {
public:
  using Qubit = uint32_t;

  LogicalResult build(func::FuncOp func, stim::Tableau<W> &tableau) {
    for (BlockArgument arg : func.getArguments())
      if (isa<quake::RefType, quake::VeqType>(arg.getType()))
        if (allocateQubits(arg).wasInterrupted())
          return failure();

    auto result = func.walk([&](Operation *op) -> WalkResult {
      if (auto nullWire = dyn_cast<quake::NullWireOp>(op))
        return allocateQubits(nullWire.getResult());
      if (auto borrow = dyn_cast<quake::BorrowWireOp>(op))
        return allocateQubits(borrow.getResult());
      if (auto alloc = dyn_cast<quake::AllocaOp>(op))
        return allocateQubits(alloc.getResult());
      if (auto extract = dyn_cast<quake::ExtractRefOp>(op))
        return visitExtractOp(extract);
      if (auto unwrap = dyn_cast<quake::UnwrapOp>(op))
        return visitUnwrapOp(unwrap);
      if (auto optor = dyn_cast<quake::OperatorInterface>(op)) {
        for (auto &&[result, operand] : llvm::zip(
                 quake::getQuantumResults(op), quake::getQuantumOperands(op)))
          qubitMap[result] = qubitMap[operand];
        if (failed(emitOperator(optor)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    // The inverse tableau is what the simulator accumulates. Asking for the
    // forward one costs an O(n^3) inversion that dominates the build. Since
    // both kernels are built the same way and equality is symmetric under
    // inversion (T == T' iff T^-1 == T'^-1), comparing the inverses is
    // equivalent and skips that cost.
    tableau = stim::circuit_to_tableau<W>(circuit, /*ignore_noise=*/false,
                                          /*ignore_measurement=*/false,
                                          /*ignore_reset=*/false,
                                          /*inverse=*/true);
    if (tableau.num_qubits < numQubits)
      tableau.expand(numQubits, /*resize_pad_factor=*/1.0);
    return success();
  }

private:
  WalkResult allocateQubits(Value value) {
    auto [entry, success] = qubitMap.try_emplace(value);
    if (!success) {
      value.getDefiningOp()->emitError("Qubit already allocated.");
      return WalkResult::interrupt();
    }
    auto &qubits = entry->second;
    if (auto veq = dyn_cast<quake::VeqType>(value.getType())) {
      if (!veq.hasSpecifiedSize()) {
        value.getDefiningOp()->emitError("Veq doesn't have a specified size.");
        return WalkResult::interrupt();
      }
      qubits.resize(veq.getSize());
      std::iota(qubits.begin(), qubits.end(), numQubits);
      numQubits += veq.getSize();
    } else {
      qubits.push_back(numQubits);
      numQubits += 1;
    }
    return WalkResult::advance();
  }

  WalkResult visitExtractOp(quake::ExtractRefOp op) {
    ArrayRef<Qubit> qubits = qubitMap[op.getVeq()];
    std::size_t index = 0;
    if (op.hasConstantIndex())
      index = op.getRawIndex();
    else if (failed(getValueAsInt(op.getIndex(), index))) {
      op.emitError("Failed to get index as an integer.");
      return WalkResult::interrupt();
    }
    qubitMap.try_emplace(op.getResult()).first->second.push_back(qubits[index]);
    return WalkResult::advance();
  }

  WalkResult visitUnwrapOp(quake::UnwrapOp op) {
    ArrayRef<Qubit> qubits = qubitMap[op.getOperand()];
    qubitMap.try_emplace(op.getResult())
        .first->second.push_back(qubits.front());
    return WalkResult::advance();
  }

  LogicalResult getValueAsInt(Value value, std::size_t &result) {
    if (auto op =
            dyn_cast_if_present<arith::ConstantIntOp>(value.getDefiningOp()))
      if (auto index = dyn_cast<IntegerAttr>(op.getValue())) {
        result = index.getInt();
        return mlir::success();
      }
    return failure();
  }

  LogicalResult getQubits(ValueRange values, SmallVectorImpl<Qubit> &qubits) {
    for (Value value : values) {
      if (auto veq = dyn_cast<quake::VeqType>(value.getType())) {
        if (!veq.hasSpecifiedSize())
          return failure();
        llvm::append_range(qubits, qubitMap[value]);
      } else {
        qubits.push_back(qubitMap[value][0]);
      }
    }
    return mlir::success();
  }

  void emit(const std::string &name, ArrayRef<Qubit> targets) {
    circuit.safe_append_u(
        name, std::vector<uint32_t>(targets.begin(), targets.end()));
  }

  LogicalResult emitOperator(quake::OperatorInterface optor) {
    Operation *op = optor.getOperation();
    SmallVector<Qubit, 4> controls, targets;
    if (failed(getQubits(optor.getControls(), controls)) ||
        failed(getQubits(optor.getTargets(), targets)))
      return failure();
    bool adj = optor.isAdj();

    auto emitControlledPauli = [&](const std::string &name) -> LogicalResult {
      if (controls.size() != 1)
        return failure();
      for (Qubit t : targets)
        emit(name, {controls[0], t});
      return mlir::success();
    };

    if (isa<quake::HOp>(op)) {
      if (!controls.empty())
        return failure();
      emit("H", targets);
      return mlir::success();
    }
    if (isa<quake::SOp>(op)) {
      if (!controls.empty())
        return failure();
      emit(adj ? "S_DAG" : "S", targets);
      return mlir::success();
    }
    if (isa<quake::XOp>(op)) {
      if (controls.empty()) {
        emit("X", targets);
        return mlir::success();
      }
      return emitControlledPauli("CX");
    }
    if (isa<quake::YOp>(op)) {
      if (controls.empty()) {
        emit("Y", targets);
        return mlir::success();
      }
      return emitControlledPauli("CY");
    }
    if (isa<quake::ZOp>(op)) {
      if (controls.empty()) {
        emit("Z", targets);
        return mlir::success();
      }
      return emitControlledPauli("CZ");
    }
    if (isa<quake::SwapOp>(op)) {
      if (!controls.empty() || targets.size() != 2)
        return failure();
      emit("SWAP", targets);
      return mlir::success();
    }
    if (isa<quake::RxOp, quake::RyOp, quake::RzOp, quake::R1Op>(op))
      return emitRotation(optor, controls, targets, adj);

    return failure();
  }

  LogicalResult emitRotation(quake::OperatorInterface optor,
                             ArrayRef<Qubit> controls, ArrayRef<Qubit> targets,
                             bool adj) {
    if (!controls.empty())
      return failure();
    auto params = optor.getParameters();
    if (params.size() != 1)
      return failure();
    auto angle = constantAngle(params[0]);
    if (!angle)
      return failure();
    double steps = (adj ? -*angle : *angle) / M_PI_2;
    double rounded = std::round(steps);
    if (std::abs(steps - rounded) > 1e-9)
      return failure();
    int k = ((static_cast<long long>(rounded) % 4) + 4) % 4;
    if (k == 0)
      return mlir::success(); // identity

    Operation *op = optor.getOperation();
    const char *rzGates[4] = {nullptr, "S", "Z", "S_DAG"};
    const char *rxGates[4] = {nullptr, "SQRT_X", "X", "SQRT_X_DAG"};
    const char *ryGates[4] = {nullptr, "SQRT_Y", "Y", "SQRT_Y_DAG"};
    const char *name = isa<quake::RxOp>(op)   ? rxGates[k]
                       : isa<quake::RyOp>(op) ? ryGates[k]
                                              : rzGates[k];
    emit(name, targets);
    return mlir::success();
  }

  std::size_t numQubits = 0;
  DenseMap<Value, SmallVector<Qubit, 4>> qubitMap;
  stim::Circuit circuit;
};

} // namespace

CliffordComparisonResult compareTableaux(func::FuncOp baseline,
                                         func::FuncOp candidate) {
  CliffordComparisonResult result;

  stim::Tableau<W> baselineT(0), candidateT(0);
  if (failed(TableauBuilder().build(baseline, baselineT))) {
    result.error = "failed to build baseline tableau (non-Clifford op?)";
    return result;
  }
  if (failed(TableauBuilder().build(candidate, candidateT))) {
    result.error = "failed to build candidate tableau (non-Clifford op?)";
    return result;
  }
  if (baselineT.num_qubits != candidateT.num_qubits) {
    result.error = "tableau qubit count mismatch (different qubit counts)";
    return result;
  }

  result.computed = true;
  result.equivalent = (baselineT == candidateT);
  return result;
}

} // namespace cudaq::opt
