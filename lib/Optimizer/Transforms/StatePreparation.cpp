/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <span>

namespace cudaq::opt {
#define GEN_PASS_DEF_STATEPREPARATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "state-preparation"

using namespace mlir;

namespace cudaq::details {

std::vector<std::size_t> grayCode(std::size_t numBits) {
  std::vector<std::size_t> result(1ULL << numBits);
  for (std::size_t i = 0; i < (1ULL << numBits); ++i)
    result[i] = ((i >> 1) ^ i);
  return result;
}

std::vector<std::size_t> getControlIndices(std::size_t numBits) {
  auto code = grayCode(numBits);
  std::vector<std::size_t> indices;
  for (auto i = 0u; i < code.size(); ++i) {
    // The position of the control in the lth CNOT gate is set to match
    // the position where the lth and (l + 1)th bit strings g[l] and g[l+1] of
    // the binary reflected Gray code differ.
    auto position = std::log2(code[i] ^ code[(i + 1) % code.size()]);
    // N.B: The algorithm expects the least significant bit (LSb) on the left
    //
    //  lsb -v
    //       001
    //         ^- msb
    //
    // Meaning that the bitstring 001 represents the number four instead of one.
    // The above position calculation uses the 'normal' convention of writing
    // numbers with the LSb on the left.
    //
    // Now, what we need to find out is the position of the 1 in the bitstring.
    // If we take LSB as being position 0, then for the normal convention its
    // position will be 0. Using the algorithm's convention it will be 2. Hence,
    // we need to convert the position we find using:
    //
    // numBits - position - 1
    //
    // The extra -1 is to account for indices starting at 0. Using the above
    // examples:
    //
    // bitstring: 001
    // numBits: 3
    // position: 0
    //
    // We have the converted position: 2, which is what we need.
    indices.emplace_back(numBits - position - 1);
  }
  return indices;
}

std::vector<double> convertAngles(const std::span<double> alphas) {
  // Implements Eq. (3) from https://arxiv.org/pdf/quant-ph/0407010.pdf
  //
  // N.B: The paper does fails to explicitly define what is the dot operator in
  // the exponent of -1. Ref. 3 solves the mystery: its the bitwise inner
  // product.
  auto bitwiseInnerProduct = [](std::size_t a, std::size_t b) {
    auto product = a & b;
    auto sumOfProducts = 0;
    while (product) {
      sumOfProducts += product & 0b1 ? 1 : 0;
      product = product >> 1;
    }
    return sumOfProducts;
  };
  std::vector<double> thetas(alphas.size(), 0);
  for (std::size_t i = 0u; i < alphas.size(); ++i) {
    for (std::size_t j = 0u; j < alphas.size(); ++j)
      thetas[i] +=
          bitwiseInnerProduct(j, ((i >> 1) ^ i)) & 0b1 ? -alphas[j] : alphas[j];
    thetas[i] /= alphas.size();
  }
  return thetas;
}

std::vector<double> getAlphaZ(const std::span<double> data,
                              std::size_t numQubits, std::size_t k) {
  // Implements Eq. (5) from https://arxiv.org/pdf/quant-ph/0407010.pdf
  std::vector<double> angles;
  double divisor = static_cast<double>(1ULL << (k - 1));
  for (std::size_t j = 1; j <= (1ULL << (numQubits - k)); ++j) {
    double angle = 0.0;
    for (std::size_t l = 1; l <= (1ULL << (k - 1)); ++l)
      // N.B: There is an extra '-1' on these indices computations to account
      // for the fact that our indices start at 0.
      angle += data[(2 * j - 1) * (1 << (k - 1)) + l - 1] -
               data[(2 * j - 2) * (1 << (k - 1)) + l - 1];
    angles.push_back(angle / divisor);
  }
  return angles;
}

std::vector<double> getAlphaY(const std::span<double> data,
                              std::size_t numQubits, std::size_t k) {
  // Implements Eq. (8) from https://arxiv.org/pdf/quant-ph/0407010.pdf
  // N.B: There is an extra '-1' on these indices computations to account for
  // the fact that our indices start at 0.
  std::vector<double> angles;
  for (std::size_t j = 1; j <= (1ULL << (numQubits - k)); ++j) {
    double numerator = 0;
    for (std::size_t l = 1; l <= (1ULL << (k - 1)); ++l) {
      numerator +=
          std::pow(std::abs(data[(2 * j - 1) * (1 << (k - 1)) + l - 1]), 2);
    }

    double denominator = 0;
    for (std::size_t l = 1; l <= (1ULL << k); ++l) {
      denominator += std::pow(std::abs(data[(j - 1) * (1 << k) + l - 1]), 2);
    }

    if (denominator == 0.0) {
      assert(numerator == 0.0 &&
             "If the denominator is zero, the numerator must also be zero.");
      angles.push_back(0.0);
      continue;
    }
    angles.push_back(2.0 * std::asin(std::sqrt(numerator / denominator)));
  }
  return angles;
}
} // namespace cudaq::details

class StateGateBuilder {
public:
  StateGateBuilder(PatternRewriter &r, mlir::Location &l, mlir::Value &q)
      : rewriter(r), loc(l), qubits(q) {}

  template <typename Op>
  void applyRotationOp(double theta, std::size_t target) {
    auto qubit = createQubitRef(target);
    auto thetaValue = createAngleValue(theta);
    rewriter.create<Op>(loc, thetaValue, mlir::ValueRange{}, qubit);
  };

  void applyX(std::size_t control, std::size_t target) {
    auto qubitC = createQubitRef(control);
    auto qubitT = createQubitRef(target);
    rewriter.create<quake::XOp>(loc, qubitC, qubitT);
  };

private:
  mlir::Value createQubitRef(std::size_t index) {
    if (qubitRefs.contains(index))
      return qubitRefs[index];

    auto ref = rewriter.create<quake::ExtractRefOp>(loc, qubits, index);
    qubitRefs[index] = ref;
    return ref;
  }

  mlir::Value createAngleValue(double angle) {
    return rewriter.create<mlir::arith::ConstantFloatOp>(
        loc, llvm::APFloat{angle}, rewriter.getF64Type());
  }

  PatternRewriter &rewriter;
  mlir::Location &loc;
  mlir::Value &qubits;

  std::unordered_map<std::size_t, mlir::Value> qubitRefs =
      std::unordered_map<std::size_t, mlir::Value>();
};

class StateDecomposer {
public:
  StateDecomposer(StateGateBuilder &b, std::span<std::complex<double>> a,
                  double t)
      : builder(b), amplitudes(a), numQubits(log2(a.size())),
        phaseThreshold(t) {}

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
      needsPhaseEqualization |= std::abs(phases.back()) > phaseThreshold;
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
  double phaseThreshold;
};

namespace {

/// Replace a qubit initialization from vectors with quantum gates.
/// For example:
///
/// Before StatePreparation (state-prep):
/// ```
/// module {
///   func.func @foo() attributes {
///     %0 = cc.address_of @foo.rodata_0 : !cc.ptr<!cc.array<complex<f32> x 4>>
///     %1 = quake.alloca !quake.veq<2>
///     %2 = quake.init_state %1, %0 : (!quake.veq<2>,
///       !cc.ptr<!cc.array<complex<f32> x 4>>) -> !quake.veq<2>
///     return
///  }
///  cc.global constant private @foo.rodata_0
///  (dense<[(0.707106769,0.000000e+00),
///      (0.707106769,0.000000e+00), (0.000000e+00,0.000000e+00),
///      (0.000000e+00,0.000000e+00)]> : tensor<4xcomplex<f32>>) :
///    !cc.array<complex<f32> x 4>
/// }
/// ```
///
/// After StatePreparation (state-prep):
/// ```
/// module {
///   func.func @foo() attributes {
///     %0 = quake.alloca !quake.veq<2>
///     %1 = quake.extract_ref %0[1] : (!quake.veq<2>, i64) -> !quake.ref
///     %cst = arith.constant 0.000000e+00 : f64
///     quake.ry (%cst) %1 : (f64, !quake.ref) -> ()
///     %2 = quake.extract_ref %0[0] : (!quake.veq<2>, i64) -> !quake.ref
///     %cst_0 = arith.constant 0.78539816339744839 : f64
///     quake.ry (%cst_0) %2 : (f64, !quake.ref) -> ()
///     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
///     %cst_1 = arith.constant 0.78539816339744839 : f64
///     quake.ry (%cst_1) %2 : (f64, !quake.ref) -> ()
///     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
///     return
///   }
/// }
/// ```
class StatePrepPattern : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  explicit StatePrepPattern(MLIRContext *ctx, double phaseThreshold)
      : OpRewritePattern(ctx), phaseThreshold(phaseThreshold) {}

  LogicalResult matchAndRewrite(quake::InitializeStateOp init,
                                PatternRewriter &rewriter) const override {
    auto loc = init.getLoc();
    auto qubits = init.getOperand(0);
    if (auto alloc = qubits.getDefiningOp<quake::AllocaOp>()) {

      // Find vector data.
      auto data = init.getOperand(1);
      auto cast = data.getDefiningOp<cudaq::cc::CastOp>();
      if (cast)
        data = cast.getOperand();

      if (auto addr = data.getDefiningOp<cudaq::cc::AddressOfOp>()) {
        auto globalName = addr.getGlobalName();
        auto module = init->getParentOfType<mlir::ModuleOp>();
        auto symbol = module.lookupSymbol(globalName);
        if (auto global = dyn_cast<cudaq::cc::GlobalOp>(symbol)) {

          // Read state initialization data from the global array.
          auto vec = cudaq::opt::factory::readGlobalConstantArray(global);

          // Prepare state from vector data.
          auto gateBuilder = StateGateBuilder(rewriter, loc, qubits);
          auto decomposer = StateDecomposer(gateBuilder, vec, phaseThreshold);
          decomposer.decompose();

          // Use prepared qubits instead of the initialized state.
          init.replaceAllUsesWith(qubits);

          // Erase the init so we don't try to replace it again.
          rewriter.eraseOp(init);
          return success();
        }
      }
    }
    return init.emitError(
        "StatePreparation failed to replace quake.state_init");
  }

private:
  double phaseThreshold;
};

class StatePreparationPass
    : public cudaq::opt::impl::StatePreparationBase<StatePreparationPass> {
protected:
public:
  using StatePreparationBase::StatePreparationBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Function before state preparation:\n"
                            << func << "\n\n");

    RewritePatternSet patterns(ctx);
    patterns.insert<StatePrepPattern>(ctx, phaseThreshold);

    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns)))) {
      func.emitOpError("State preparation failed");
      signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Function after state preparation:\n"
                            << func << "\n\n");
  }
};

} // namespace
