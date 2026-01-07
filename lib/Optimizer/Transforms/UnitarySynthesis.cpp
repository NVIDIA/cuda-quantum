/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "common/EigenDense.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>

namespace cudaq::opt {
#define GEN_PASS_DEF_UNITARYSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "unitary-synthesis"

using namespace mlir;
using namespace std::complex_literals;

namespace {

constexpr double TOL = 1e-7;

/// Base class for unitary synthesis, i.e. decomposing an arbitrary unitary
/// matrix into native gate set. The native gate set here includes all the
/// quantum operations supported by CUDA-Q. Additional passes may be required to
/// convert CUDA-Q gate set to hardware specific gate set.
class Decomposer {
private:
  Eigen::MatrixXcd targetMatrix;

public:
  /// Function which implements the unitary synthesis algorithm. The result of
  /// decomposition which depends on the algorithm must be convertible to
  /// quantum operations. For example, result is saved into class member(s) as
  /// the parameters to be applied to  `Rx`, `Ry`, and `Rz` gates.
  virtual void decompose() = 0;
  /// Create the replacement function which invokes native quantum operations.
  /// The original `quake.custom_op` is replaced by `quake.apply` operation that
  /// calls the new replacement function with the same operands as the original
  /// operation. The 'control' and 'adjoint' variations are handled by
  /// `ApplySpecialization` pass.
  virtual void emitDecomposedFuncOp(quake::CustomUnitarySymbolOp customOp,
                                    PatternRewriter &rewriter,
                                    std::string funcName) = 0;
  bool isAboveThreshold(double value) { return std::abs(value) > TOL; };
  virtual ~Decomposer() = default;
};

/// Result structure for 1-q Euler decomposition in ZYZ basis
struct EulerAngles {
  double alpha;
  double beta;
  double gamma;
};

struct OneQubitOpZYZ : public Decomposer {
  Eigen::Matrix2cd targetMatrix;
  EulerAngles angles;
  /// Updates to the global phase
  double phase;

  /// This logic is based on https://arxiv.org/pdf/quant-ph/9503016 and its
  /// corresponding explanation in https://threeplusone.com/pubs/on_gates.pdf,
  /// Section 4.
  void decompose() override {
    /// Rescale the input unitary matrix, `u`, to be special unitary.
    /// Extract a phase factor, `phase`, so that
    /// `determinant(inverse_phase * unitary) = 1`
    auto det = targetMatrix.determinant();
    phase = 0.5 * std::arg(det);
    Eigen::Matrix2cd specialUnitary = std::exp(-1i * phase) * targetMatrix;
    auto abs00 = std::abs(specialUnitary(0, 0));
    auto abs01 = std::abs(specialUnitary(0, 1));
    if (abs00 >= abs01)
      angles.beta = 2.0 * std::acos(abs00);
    else
      angles.beta = 2.0 * std::asin(abs01);
    auto sum =
        std::atan2(specialUnitary(1, 1).imag(), specialUnitary(1, 1).real());
    auto diff =
        std::atan2(specialUnitary(1, 0).imag(), specialUnitary(1, 0).real());
    angles.alpha = sum + diff;
    angles.gamma = sum - diff;
  }

  void emitDecomposedFuncOp(quake::CustomUnitarySymbolOp customOp,
                            PatternRewriter &rewriter,
                            std::string funcName) override {
    auto parentModule = customOp->getParentOfType<ModuleOp>();
    Location loc = customOp->getLoc();
    auto targets = customOp.getTargets();
    auto funcTy =
        FunctionType::get(parentModule.getContext(), targets[0].getType(), {});
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(parentModule.getBody());
    auto func =
        rewriter.create<func::FuncOp>(parentModule->getLoc(), funcName, funcTy);
    func.setPrivate();
    auto *block = func.addEntryBlock();
    rewriter.setInsertionPointToStart(block);
    auto arguments = func.getArguments();
    FloatType floatTy = rewriter.getF64Type();
    /// NOTE: Operator notation is right-to-left, whereas circuit notation
    /// is left-to-right. Hence, angles are applied as:
    /// Rz(gamma)Ry(beta)Rz(alpha)
    if (isAboveThreshold(angles.gamma)) {
      auto gamma = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, angles.gamma, floatTy);
      rewriter.create<quake::RzOp>(loc, gamma, ValueRange{}, arguments);
    }
    if (isAboveThreshold(angles.beta)) {
      auto beta = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, angles.beta, floatTy);
      rewriter.create<quake::RyOp>(loc, beta, ValueRange{}, arguments);
    }
    if (isAboveThreshold(angles.alpha)) {
      auto alpha = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, angles.alpha, floatTy);
      rewriter.create<quake::RzOp>(loc, alpha, ValueRange{}, arguments);
    }
    /// NOTE: Typically global phase can be ignored but, if this decomposition
    /// is applied in a kernel that is called with `cudaq::control`, the global
    /// phase will become a local phase and give a wrong result if we don't keep
    /// track of that.
    /// NOTE: R1-Rz pair results in a half the applied global phase angle,
    /// hence, we need to multiply the angle by 2
    auto globalPhase = 2.0 * phase;
    if (isAboveThreshold(globalPhase)) {
      auto phase = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, globalPhase, floatTy);
      Value negPhase = rewriter.create<arith::NegFOp>(loc, phase);
      rewriter.create<quake::R1Op>(loc, phase, ValueRange{}, arguments[0]);
      rewriter.create<quake::RzOp>(loc, negPhase, ValueRange{}, arguments[0]);
    }
    rewriter.create<func::ReturnOp>(loc);
    rewriter.restoreInsertionPoint(insPt);
  }

  OneQubitOpZYZ(const Eigen::Matrix2cd &vec) {
    targetMatrix = vec;
    decompose();
  }
};

/// Result for 2-q KAK decomposition
struct KAKComponents {
  // KAK decomposition allows to express arbitrary 2-qubit unitary (U) in the
  // form: U = (a1 ⊗ a0) x exp(i(xXX + yYY + zZZ)) x (b1 ⊗ b0) where, a0, a1,
  // b0, b1 are single qubit operations, and the exponential is specified by the
  // 3 coefficients of the canonical class vector - x, y, z
  Eigen::Matrix2cd a0;
  Eigen::Matrix2cd a1;
  Eigen::Matrix2cd b0;
  Eigen::Matrix2cd b1;
  double x;
  double y;
  double z;
};

/// Helper function to convert a matrix into 'magic' basis
/// M = 1 / sqrt(2) *  1  0  0  i
///                    0  i  1  0
///                    0  i −1  0
///                    1  0  0 −i
const Eigen::Matrix4cd &MagicBasisMatrix() {
  static Eigen::Matrix4cd MagicBasisMatrix;
  MagicBasisMatrix << 1.0, 0.0, 0.0, 1i, 0.0, 1i, 1.0, 0, 0, 1i, -1.0, 0, 1.0,
      0, 0, -1i;
  MagicBasisMatrix = MagicBasisMatrix * M_SQRT1_2;
  return MagicBasisMatrix;
}

/// Helper function to convert a matrix into 'magic' basis
const Eigen::Matrix4cd &MagicBasisMatrixAdj() {
  static Eigen::Matrix4cd MagicBasisMatrixAdj = MagicBasisMatrix().adjoint();
  return MagicBasisMatrixAdj;
}

/// Helper function to extract the coefficients of canonical vector
/// Gamma matrix = +1 +1 −1 +1
///                +1 +1 +1 −1
///                +1 −1 −1 −1
///                +1 −1 +1 +1
const Eigen::Matrix4cd &GammaFactor() {

  static Eigen::Matrix4cd GammaT;
  GammaT << 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1;
  GammaT /= 4;
  return GammaT;
}

/// Given an input matrix which is unitary, find two orthogonal matrices, 'left'
/// and 'right', and a diagonal unitary matrix, 'diagonal', such that
/// `input_matrix = left * diagonal * right.transpose()`. This function uses QZ
/// decomposition for this purpose.
/// NOTE: This function may not generate accurate diagonal matrix in some corner
/// cases like degenerate matrices.
std::tuple<Eigen::Matrix4d, Eigen::Matrix4cd, Eigen::Matrix4d>
bidiagonalize(const Eigen::Matrix4cd &matrix) {
  Eigen::Matrix4d real = matrix.real();
  Eigen::Matrix4d imag = matrix.imag();
  Eigen::RealQZ<Eigen::Matrix4d> qz(4);
  qz.compute(real, imag);
  Eigen::Matrix4d left = qz.matrixQ();
  Eigen::Matrix4d right = qz.matrixZ();
  if (left.determinant() < 0.0)
    left.col(0) *= -1.0;
  if (right.determinant() < 0.0)
    right.row(0) *= -1.0;
  Eigen::Matrix4cd diagonal = left.transpose() * matrix * right.transpose();
  assert(diagonal.isDiagonal(TOL));
  return std::make_tuple(left, diagonal, right);
}

/// Separate input matrix into local operations. The input matrix must be
/// special orthogonal. Given a map, SU(2) × SU(2) -> SO(4),
/// map(A, B) = M.adjoint() (A ⊗ B∗) M, find A and B.
std::tuple<Eigen::Matrix2cd, Eigen::Matrix2cd, std::complex<double>>
extractSU2FromSO4(const Eigen::Matrix4cd &matrix) {
  /// Verify input matrix is special orthogonal
  assert(std::abs(std::abs(matrix.determinant()) - 1.0) < TOL);
  assert((matrix * matrix.transpose() - Eigen::Matrix4cd::Identity()).norm() <
         TOL);
  Eigen::Matrix4cd mb = MagicBasisMatrix() * matrix * MagicBasisMatrixAdj();
  /// Use Kronecker factorization
  size_t r = 0;
  size_t c = 0;
  double largest = std::abs(mb(r, c));
  for (size_t i = 0; i < 4; i++)
    for (size_t j = 0; j < 4; j++) {
      if (std::abs(mb(i, j)) >= largest) {
        largest = std::abs(mb(i, j));
        r = i;
        c = j;
      }
    }
  Eigen::Matrix2cd part1 = Eigen::Matrix2cd::Zero();
  Eigen::Matrix2cd part2 = Eigen::Matrix2cd::Zero();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      part1((r >> 1) ^ i, (c >> 1) ^ j) = mb(r ^ (i << 1), c ^ (j << 1));
      part2((r & 1) ^ i, (c & 1) ^ j) = mb(r ^ i, c ^ j);
    }
  }
  auto det1 = part1.determinant();
  if (std::abs(det1) > TOL)
    part1 /= (std::sqrt(det1));
  auto det2 = part2.determinant();
  if (std::abs(det2) > TOL)
    part2 /= (std::sqrt(det2));
  std::complex<double> phase =
      mb(r, c) / (part1(r >> 1, c >> 1) * part2(r & 1, c & 1));
  if (phase.real() < 0.0) {
    part1 *= -1;
    phase = -phase;
  }
  assert(mb.isApprox(phase * Eigen::kroneckerProduct(part1, part2), TOL));
  assert(part1.isUnitary(TOL) && part2.isUnitary(TOL));
  return std::make_tuple(part1, part2, phase);
}

/// Compute exp(i(x XX + y YY + z ZZ)) matrix for verification
Eigen::Matrix4cd canonicalVecToMatrix(double x, double y, double z) {
  Eigen::Matrix2cd X{Eigen::Matrix2cd::Zero()};
  Eigen::Matrix2cd Y{Eigen::Matrix2cd::Zero()};
  Eigen::Matrix2cd Z{Eigen::Matrix2cd::Zero()};
  X << 0, 1, 1, 0;
  Y << 0, -1i, 1i, 0;
  Z << 1, 0, 0, -1;
  auto XX = Eigen::kroneckerProduct(X, X);
  auto YY = Eigen::kroneckerProduct(Y, Y);
  auto ZZ = Eigen::kroneckerProduct(Z, Z);
  return (1i * (x * XX + y * YY + z * ZZ)).exp();
}

struct TwoQubitOpKAK : public Decomposer {
  Eigen::Matrix4cd targetMatrix;
  KAKComponents components;
  /// Updates to the global phase
  std::complex<double> phase;

  /// This logic is based on the Cartan's KAK decomposition.
  /// Ref: https://arxiv.org/pdf/quant-ph/0507171
  /// Ref: https://arxiv.org/pdf/0806.4015
  void decompose() override {
    /// Step0: Convert to special unitary
    phase = std::pow(targetMatrix.determinant(), 0.25);
    auto specialUnitary = targetMatrix / phase;
    /// Step1: Convert into magic basis
    Eigen::Matrix4cd matrixMagicBasis =
        MagicBasisMatrixAdj() * specialUnitary * MagicBasisMatrix();
    /// Step2: Diagonalize
    auto [left, diagonal, right] = bidiagonalize(matrixMagicBasis);
    /// Step3: Get the KAK components
    auto [a1, a0, aPh] = extractSU2FromSO4(left);
    components.a0 = a0;
    components.a1 = a1;
    phase *= aPh;
    auto [b1, b0, bPh] = extractSU2FromSO4(right);
    components.b0 = b0;
    components.b1 = b1;
    phase *= bPh;
    /// Step4: Get the coefficients of canonical class vector
    if (diagonal.determinant().real() < 0.0)
      diagonal(0, 0) *= 1.0;
    Eigen::Vector4cd diagonalPhases;
    for (size_t i = 0; i < 4; i++)
      diagonalPhases(i) = std::arg(diagonal(i, i));
    auto coefficients = GammaFactor() * diagonalPhases;
    components.x = coefficients(1).real();
    components.y = coefficients(2).real();
    components.z = coefficients(3).real();
    phase *= std::exp(1i * coefficients(0));
    /// Final check to verify results
    auto canVecToMat =
        canonicalVecToMatrix(components.x, components.y, components.z);
    assert(targetMatrix.isApprox(phase * Eigen::kroneckerProduct(a1, a0) *
                                     canVecToMat *
                                     Eigen::kroneckerProduct(b1, b0),
                                 TOL));
  }

  void emitDecomposedFuncOp(quake::CustomUnitarySymbolOp customOp,
                            PatternRewriter &rewriter,
                            std::string funcName) override {
    auto a0 = OneQubitOpZYZ(components.a0);
    a0.emitDecomposedFuncOp(customOp, rewriter, funcName + "a0");
    auto a1 = OneQubitOpZYZ(components.a1);
    a1.emitDecomposedFuncOp(customOp, rewriter, funcName + "a1");
    auto b0 = OneQubitOpZYZ(components.b0);
    b0.emitDecomposedFuncOp(customOp, rewriter, funcName + "b0");
    auto b1 = OneQubitOpZYZ(components.b1);
    b1.emitDecomposedFuncOp(customOp, rewriter, funcName + "b1");
    auto parentModule = customOp->getParentOfType<ModuleOp>();
    Location loc = customOp->getLoc();
    auto targets = customOp.getTargets();
    auto funcTy =
        FunctionType::get(parentModule.getContext(), targets.getTypes(), {});
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(parentModule.getBody());
    auto func =
        rewriter.create<func::FuncOp>(parentModule->getLoc(), funcName, funcTy);
    func.setPrivate();
    auto *block = func.addEntryBlock();
    rewriter.setInsertionPointToStart(block);
    auto arguments = func.getArguments();
    FloatType floatTy = rewriter.getF64Type();
    /// NOTE: Operator notation is right-to-left, whereas circuit notation is
    /// left-to-right. Hence, operations are applied in reverse order.
    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "b0"), false,
        ValueRange{}, ValueRange{arguments[1]});
    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "b1"), false,
        ValueRange{}, ValueRange{arguments[0]});
    /// TODO: Refactor to use a transformation pass for `quake.exp_pauli`
    /// XX
    if (isAboveThreshold(components.x)) {
      rewriter.create<quake::HOp>(loc, arguments[0]);
      rewriter.create<quake::HOp>(loc, arguments[1]);
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
      auto xAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.x, floatTy);
      rewriter.create<quake::RzOp>(loc, xAngle, ValueRange{}, arguments[0]);
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
      rewriter.create<quake::HOp>(loc, arguments[1]);
      rewriter.create<quake::HOp>(loc, arguments[0]);
    }
    /// YY
    if (isAboveThreshold(components.y)) {
      auto piBy2 = cudaq::opt::factory::createFloatConstant(loc, rewriter,
                                                            M_PI_2, floatTy);
      rewriter.create<quake::RxOp>(loc, piBy2, ValueRange{}, arguments[0]);
      rewriter.create<quake::RxOp>(loc, piBy2, ValueRange{}, arguments[1]);
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
      auto yAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.y, floatTy);
      rewriter.create<quake::RzOp>(loc, yAngle, ValueRange{}, arguments[0]);
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
      Value negPiBy2 = rewriter.create<arith::NegFOp>(loc, piBy2);
      rewriter.create<quake::RxOp>(loc, negPiBy2, ValueRange{}, arguments[1]);
      rewriter.create<quake::RxOp>(loc, negPiBy2, ValueRange{}, arguments[0]);
    }
    /// ZZ
    if (isAboveThreshold(components.z)) {
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
      auto zAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.z, floatTy);
      rewriter.create<quake::RzOp>(loc, zAngle, ValueRange{}, arguments[0]);
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
    }
    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "a0"), false,
        ValueRange{}, ValueRange{arguments[1]});
    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "a1"), false,
        ValueRange{}, ValueRange{arguments[0]});
    auto globalPhase = 2.0 * std::arg(phase);
    if (isAboveThreshold(globalPhase)) {
      auto phase = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, globalPhase, floatTy);
      Value negPhase = rewriter.create<arith::NegFOp>(loc, phase);
      rewriter.create<quake::R1Op>(loc, phase, ValueRange{}, arguments[0]);
      rewriter.create<quake::RzOp>(loc, negPhase, ValueRange{}, arguments[0]);
    }
    rewriter.create<func::ReturnOp>(loc);
    rewriter.restoreInsertionPoint(insPt);
  }

  TwoQubitOpKAK(const Eigen::MatrixXcd &vec) {
    targetMatrix = vec;
    decompose();
  }
};

class CustomUnitaryPattern
    : public OpRewritePattern<quake::CustomUnitarySymbolOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::CustomUnitarySymbolOp customOp,
                                PatternRewriter &rewriter) const override {
    auto parentModule = customOp->getParentOfType<ModuleOp>();
    /// Get the global constant holding the concrete matrix corresponding to
    /// this custom operation invocation
    StringRef generatorName = customOp.getGenerator().getRootReference();
    auto globalOp =
        parentModule.lookupSymbol<cudaq::cc::GlobalOp>(generatorName);
    /// The decomposed sequence of quantum operations are in a function
    auto pair = generatorName.split(".rodata");
    std::string funcName = pair.first.str() + ".kernel" + pair.second.str();
    /// If the replacement function doesn't exist, create it here
    if (!parentModule.lookupSymbol<func::FuncOp>(funcName)) {
      auto matrix = cudaq::opt::factory::readGlobalConstantArray(globalOp);
      size_t dimension = std::sqrt(matrix.size());
      auto unitary =
          Eigen::Map<Eigen::MatrixXcd>(matrix.data(), dimension, dimension);
      unitary.transposeInPlace();
      if (!unitary.isUnitary(TOL)) {
        customOp.emitWarning("The custom operation matrix must be unitary.");
        return failure();
      }
      switch (dimension) {
      case 2: {
        auto zyz = OneQubitOpZYZ(unitary);
        zyz.emitDecomposedFuncOp(customOp, rewriter, funcName);
      } break;
      case 4: {
        auto kak = TwoQubitOpKAK(unitary);
        kak.emitDecomposedFuncOp(customOp, rewriter, funcName);
      } break;
      default:
        customOp.emitWarning(
            "Decomposition of only 1 and 2 qubit custom operations supported.");
        return failure();
      }
    }
    rewriter.replaceOpWithNewOp<quake::ApplyOp>(
        customOp, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName), customOp.isAdj(),
        customOp.getControls(), customOp.getTargets());
    return success();
  }
};

class UnitarySynthesisPass
    : public cudaq::opt::impl::UnitarySynthesisBase<UnitarySynthesisPass> {
public:
  using UnitarySynthesisBase::UnitarySynthesisBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();
    for (Operation &op : *module.getBody()) {
      auto func = dyn_cast<func::FuncOp>(op);
      if (!func)
        continue;
      RewritePatternSet patterns(ctx);
      patterns.insert<CustomUnitaryPattern>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "Before unitary synthesis: " << func << '\n');
      if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                              std::move(patterns))))
        signalPassFailure();
      LLVM_DEBUG(llvm::dbgs() << "After unitary synthesis: " << func << '\n');
    }
  }
};

} // namespace
