/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include <span>

namespace cudaq::details {

/// @brief Converts angles of a uniformly controlled rotation to angles of
/// non-controlled rotations.
std::vector<double> convertAngles(const std::span<double> alphas);

/// @brief Return the control indices dictated by the gray code implementation.
///
/// Here, numBits is the number of controls.
std::vector<std::size_t> getControlIndices(std::size_t numBits);

/// @brief Return angles required to implement a uniformly controlled z-rotation
/// on the `kth` qubit.
std::vector<double> getAlphaZ(const std::span<double> data,
                              std::size_t numQubits, std::size_t k);

/// @brief Return angles required to implement a uniformly controlled y-rotation
/// on the `kth` qubit.
std::vector<double> getAlphaY(const std::span<double> data,
                              std::size_t numQubits, std::size_t k);
} // namespace cudaq::details

class StateGateBuilder {
public:
  StateGateBuilder(mlir::OpBuilder &b, mlir::Location &l, mlir::Value &q)
      : builder(b), loc(l), qubits(q) {}

  template <typename Op>
  void applyRotationOp(double theta, std::size_t target) {
    auto qubit = createQubitRef(target);
    auto thetaValue = createAngleValue(theta);
    builder.create<Op>(loc, thetaValue, mlir::ValueRange{}, qubit);
  };

  void applyX(std::size_t control, std::size_t target) {
    auto qubitC = createQubitRef(control);
    auto qubitT = createQubitRef(target);
    builder.create<quake::XOp>(loc, qubitC, qubitT);
  };

private:
  mlir::Value createQubitRef(std::size_t index) {
    if (qubitRefs.contains(index)) {
      return qubitRefs[index];
    }

    auto indexValue = builder.create<mlir::arith::ConstantIntOp>(
        loc, index, builder.getIntegerType(64));
    auto ref = builder.create<quake::ExtractRefOp>(loc, qubits, indexValue);
    qubitRefs[index] = ref;
    return ref;
  }

  mlir::Value createAngleValue(double angle) {
    return builder.create<mlir::arith::ConstantFloatOp>(
        loc, llvm::APFloat{angle}, builder.getF64Type());
  }

  mlir::OpBuilder &builder;
  mlir::Location &loc;
  mlir::Value &qubits;

  std::unordered_map<std::size_t, mlir::Value> qubitRefs =
      std::unordered_map<std::size_t, mlir::Value>();
};

class StateDecomposer {
public:
  StateDecomposer(StateGateBuilder &b, std::span<std::complex<double>> a)
      : builder(b), amplitudes(a), numQubits(log2(a.size())) {}

  /// @brief Decompose the input state vector data to a set of controlled
  /// operations and rotations. This function takes as input a `OpBuilder`
  /// and appends the operations of the decomposition to its internal
  /// representation. This implementation follows the algorithm defined in
  /// `https://arxiv.org/pdf/quant-ph/0407010.pdf`.
  void decompose() {

    // Decompose the state into phases and magnitudes.
    bool needsPhaseEqualization = false;
    std::vector<double> phases;
    std::vector<double> magnitudes;
    for (const auto &a : amplitudes) {
      phases.push_back(std::arg(a));
      magnitudes.push_back(std::abs(a));
      // FIXME: remove magic number.
      needsPhaseEqualization |= std::abs(phases.back()) > 1e-10;
    }

    // N.B: The algorithm, as described in the paper, creates a circuit that
    // begins with a target state and brings it to the all zero state. Hence,
    // this implementation do the two steps described in Section III in reverse
    // order.

    // Apply uniformly controlled y-rotations, the construction in Eq. (4).
    for (std::size_t j = 1; j <= numQubits; ++j) {
      auto k = numQubits - j + 1;
      auto numControls = j - 1;
      auto target = j - 1;
      auto alphaYk = cudaq::details::getAlphaY(magnitudes, numQubits, k);
      applyRotation<quake::RyOp>(alphaYk, numControls, target);
    }

    if (!needsPhaseEqualization)
      return;

    // Apply uniformly controlled z-rotations, the construction in Eq. (4).
    for (std::size_t j = 1; j <= numQubits; ++j) {
      auto k = numQubits - j + 1;
      auto numControls = j - 1;
      auto target = j - 1;
      auto alphaZk = cudaq::details::getAlphaZ(phases, numQubits, k);
      if (alphaZk.empty())
        continue;
      applyRotation<quake::RzOp>(alphaZk, numControls, target);
    }
  }

private:
  /// @brief Apply a uniformly controlled rotation on the target qubit.
  template <typename Op>
  void applyRotation(const std::span<double> alphas, std::size_t numControls,
                     std::size_t target) {

    // In our model the index 1 (i.e. |01>) in quantum state data
    // corresponds to qubits[0] = 1 and qubits[1] = 0.
    // Revert the order of qubits as the state preparation algorithm
    // we use assumes the opposite.
    auto qubitIndex = [&](std::size_t i) { return numQubits - i - 1; };

    auto thetas = cudaq::details::convertAngles(alphas);
    if (numControls == 0) {
      builder.applyRotationOp<Op>(thetas[0], qubitIndex(target));
      return;
    }

    auto controlIndices = cudaq::details::getControlIndices(numControls);
    assert(thetas.size() == controlIndices.size());
    for (auto [i, c] : llvm::enumerate(controlIndices)) {
      builder.applyRotationOp<Op>(thetas[i], qubitIndex(target));
      builder.applyX(qubitIndex(c), qubitIndex(target));
    }
  }

  StateGateBuilder &builder;
  std::span<std::complex<double>> amplitudes;
  std::size_t numQubits;
};
