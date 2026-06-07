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
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
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
  virtual void
  emitDecomposedFuncOp(cudaq::quake::CustomUnitaryConstantOp customOp,
                       PatternRewriter &rewriter, std::string funcName) = 0;
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
    using namespace std::complex_literals;
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

  void emitDecomposedFuncOp(cudaq::quake::CustomUnitaryConstantOp customOp,
                            PatternRewriter &rewriter,
                            std::string funcName) override {
    auto parentModule = customOp->getParentOfType<ModuleOp>();
    Location loc = customOp->getLoc();
    auto targets = customOp.getTargets();
    auto funcTy =
        FunctionType::get(parentModule.getContext(), targets[0].getType(), {});
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(parentModule.getBody());
    auto func = func::FuncOp::create(rewriter, parentModule->getLoc(), funcName,
                                     funcTy);
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
      cudaq::quake::RzOp::create(rewriter, loc, gamma, ValueRange{}, arguments);
    }
    if (isAboveThreshold(angles.beta)) {
      auto beta = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, angles.beta, floatTy);
      cudaq::quake::RyOp::create(rewriter, loc, beta, ValueRange{}, arguments);
    }
    if (isAboveThreshold(angles.alpha)) {
      auto alpha = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, angles.alpha, floatTy);
      cudaq::quake::RzOp::create(rewriter, loc, alpha, ValueRange{}, arguments);
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
      Value negPhase = arith::NegFOp::create(rewriter, loc, phase);
      cudaq::quake::R1Op::create(rewriter, loc, phase, ValueRange{},
                                 arguments[0]);
      cudaq::quake::RzOp::create(rewriter, loc, negPhase, ValueRange{},
                                 arguments[0]);
    }
    func::ReturnOp::create(rewriter, loc);
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
  using namespace std::complex_literals;
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
  using namespace std::complex_literals;
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
    using namespace std::complex_literals;
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

  void emitDecomposedFuncOp(cudaq::quake::CustomUnitaryConstantOp customOp,
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
    /// This 2-qubit decomposer always emits a 2-qubit replacement function. The
    /// arity is fixed to 2 (not taken from `customOp`) so that it stays correct
    /// when invoked as a base case of the recursive QSD on a larger operation.
    SmallVector<Type, 2> argTys(2, targets[0].getType());
    auto funcTy = FunctionType::get(parentModule.getContext(), argTys, {});
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(parentModule.getBody());
    auto func = func::FuncOp::create(rewriter, parentModule->getLoc(), funcName,
                                     funcTy);
    func.setPrivate();
    auto *block = func.addEntryBlock();
    rewriter.setInsertionPointToStart(block);
    auto arguments = func.getArguments();
    FloatType floatTy = rewriter.getF64Type();
    /// NOTE: Operator notation is right-to-left, whereas circuit notation is
    /// left-to-right. Hence, operations are applied in reverse order.
    cudaq::quake::ApplyOp::create(
        rewriter, loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "b0"), false,
        ValueRange{}, ValueRange{arguments[1]});
    cudaq::quake::ApplyOp::create(
        rewriter, loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "b1"), false,
        ValueRange{}, ValueRange{arguments[0]});
    /// TODO: Refactor to use a transformation pass for `quake.exp_pauli`
    /// XX
    if (isAboveThreshold(components.x)) {
      cudaq::quake::HOp::create(rewriter, loc, arguments[0]);
      cudaq::quake::HOp::create(rewriter, loc, arguments[1]);
      cudaq::quake::XOp::create(rewriter, loc, arguments[1], arguments[0]);
      auto xAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.x, floatTy);
      cudaq::quake::RzOp::create(rewriter, loc, xAngle, ValueRange{},
                                 arguments[0]);
      cudaq::quake::XOp::create(rewriter, loc, arguments[1], arguments[0]);
      cudaq::quake::HOp::create(rewriter, loc, arguments[1]);
      cudaq::quake::HOp::create(rewriter, loc, arguments[0]);
    }
    /// YY
    if (isAboveThreshold(components.y)) {
      auto piBy2 = cudaq::opt::factory::createFloatConstant(loc, rewriter,
                                                            M_PI_2, floatTy);
      cudaq::quake::RxOp::create(rewriter, loc, piBy2, ValueRange{},
                                 arguments[0]);
      cudaq::quake::RxOp::create(rewriter, loc, piBy2, ValueRange{},
                                 arguments[1]);
      cudaq::quake::XOp::create(rewriter, loc, arguments[1], arguments[0]);
      auto yAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.y, floatTy);
      cudaq::quake::RzOp::create(rewriter, loc, yAngle, ValueRange{},
                                 arguments[0]);
      cudaq::quake::XOp::create(rewriter, loc, arguments[1], arguments[0]);
      Value negPiBy2 = arith::NegFOp::create(rewriter, loc, piBy2);
      cudaq::quake::RxOp::create(rewriter, loc, negPiBy2, ValueRange{},
                                 arguments[1]);
      cudaq::quake::RxOp::create(rewriter, loc, negPiBy2, ValueRange{},
                                 arguments[0]);
    }
    /// ZZ
    if (isAboveThreshold(components.z)) {
      cudaq::quake::XOp::create(rewriter, loc, arguments[1], arguments[0]);
      auto zAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.z, floatTy);
      cudaq::quake::RzOp::create(rewriter, loc, zAngle, ValueRange{},
                                 arguments[0]);
      cudaq::quake::XOp::create(rewriter, loc, arguments[1], arguments[0]);
    }
    cudaq::quake::ApplyOp::create(
        rewriter, loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "a0"), false,
        ValueRange{}, ValueRange{arguments[1]});
    cudaq::quake::ApplyOp::create(
        rewriter, loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "a1"), false,
        ValueRange{}, ValueRange{arguments[0]});
    auto globalPhase = 2.0 * std::arg(phase);
    if (isAboveThreshold(globalPhase)) {
      auto phase = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, globalPhase, floatTy);
      Value negPhase = arith::NegFOp::create(rewriter, loc, phase);
      cudaq::quake::R1Op::create(rewriter, loc, phase, ValueRange{},
                                 arguments[0]);
      cudaq::quake::RzOp::create(rewriter, loc, negPhase, ValueRange{},
                                 arguments[0]);
    }
    func::ReturnOp::create(rewriter, loc);
    rewriter.restoreInsertionPoint(insPt);
  }

  TwoQubitOpKAK(const Eigen::MatrixXcd &vec) {
    targetMatrix = vec;
    decompose();
  }
};

/// Result for the Cosine-Sine decomposition of a `2m x 2m` unitary.
/// `U = blockDiag(l0, l1) * [[C, -S], [S, C]] * blockDiag(r0, r1)` where the
/// `l*`, `r*` are `m x m` unitaries and `C = diag(c)`, `S = diag(s)` satisfy
/// `c^2 + s^2 = 1` element-wise.
struct CSDComponents {
  Eigen::MatrixXcd l0, l1, r0, r1;
  Eigen::VectorXd c, s;
};

/// Deterministically extend a set of orthonormal columns to a full unitary by
/// projecting the standard basis vectors onto the orthogonal complement. Used
/// to fill the columns of a singular-vector basis that are left undetermined by
/// degenerate (repeated) singular values, e.g. for permutation-like operators
/// such as Toffoli.
Eigen::MatrixXcd completeUnitaryBasis(const Eigen::MatrixXcd &defined, int n) {
  Eigen::MatrixXcd basis = defined;
  int col = static_cast<int>(basis.cols());
  for (int i = 0; i < n && col < n; ++i) {
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(n);
    v(i) = 1.0;
    if (col > 0)
      v -= basis.leftCols(col) * (basis.leftCols(col).adjoint() * v);
    double nv = v.norm();
    if (nv > TOL) {
      basis.conservativeResize(n, col + 1);
      basis.col(col) = v / nv;
      ++col;
    }
  }
  return basis;
}

/// Compute the Cosine-Sine decomposition of a `2m x 2m` unitary.
/// Eigen has no built-in CSD, so it is assembled from the SVD of the top-left
/// block. The right singular vectors are shared by both block rows, which lets
/// us recover the remaining factors. Degenerate singular values are handled via
/// `completeUnitaryBasis`. Ref: arXiv quant-ph/0404089.
CSDComponents cosineSineDecomposition(const Eigen::MatrixXcd &u) {
  int m = static_cast<int>(u.rows()) / 2;
  Eigen::MatrixXcd u00 = u.topLeftCorner(m, m);
  Eigen::MatrixXcd u01 = u.topRightCorner(m, m);
  Eigen::MatrixXcd u11 = u.bottomRightCorner(m, m);
  Eigen::MatrixXcd u10 = u.bottomLeftCorner(m, m);
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(u00, Eigen::ComputeFullU |
                                                  Eigen::ComputeFullV);
  Eigen::MatrixXcd l0 = svd.matrixU();
  Eigen::MatrixXcd v = svd.matrixV();
  Eigen::MatrixXcd r0 = v.adjoint();
  Eigen::VectorXd c = svd.singularValues().cwiseMin(1.0).cwiseMax(0.0);
  Eigen::VectorXd s(m);
  for (int k = 0; k < m; ++k)
    s(k) = std::sqrt(std::max(0.0, 1.0 - c(k) * c(k)));
  /// `u10 * v = l1 * diag(s)`; recover `l1` column-by-column, completing the
  /// columns belonging to (near-)zero sines.
  Eigen::MatrixXcd mCols = u10 * v;
  Eigen::MatrixXcd l1 = Eigen::MatrixXcd::Zero(m, m);
  std::vector<int> defined;
  for (int k = 0; k < m; ++k)
    if (s(k) > TOL) {
      l1.col(k) = mCols.col(k) / s(k);
      defined.push_back(k);
    }
  if (static_cast<int>(defined.size()) < m) {
    Eigen::MatrixXcd known(m, defined.size());
    for (size_t i = 0; i < defined.size(); ++i)
      known.col(i) = l1.col(defined[i]);
    Eigen::MatrixXcd full = completeUnitaryBasis(known, m);
    std::vector<bool> isDefined(m, false);
    for (int d : defined)
      isDefined[d] = true;
    int next = static_cast<int>(defined.size());
    for (int k = 0; k < m; ++k)
      if (!isDefined[k])
        l1.col(k) = full.col(next++);
  }
  /// `r1` from `u11 = l1 diag(c) r1` where cosine dominates, otherwise from
  /// `u01 = -l0 diag(s) r1`.
  Eigen::MatrixXcd fromC = l1.adjoint() * u11;
  Eigen::MatrixXcd fromS = -(l0.adjoint() * u01);
  Eigen::MatrixXcd r1(m, m);
  for (int k = 0; k < m; ++k) {
    if (c(k) >= s(k))
      r1.row(k) = fromC.row(k) / c(k);
    else
      r1.row(k) = fromS.row(k) / s(k);
  }
#ifndef NDEBUG
  Eigen::MatrixXcd cMat = c.cast<std::complex<double>>().asDiagonal();
  Eigen::MatrixXcd sMat = s.cast<std::complex<double>>().asDiagonal();
  Eigen::MatrixXcd ld = Eigen::MatrixXcd::Zero(2 * m, 2 * m);
  ld.topLeftCorner(m, m) = l0;
  ld.bottomRightCorner(m, m) = l1;
  Eigen::MatrixXcd rd = Eigen::MatrixXcd::Zero(2 * m, 2 * m);
  rd.topLeftCorner(m, m) = r0;
  rd.bottomRightCorner(m, m) = r1;
  Eigen::MatrixXcd cs(2 * m, 2 * m);
  cs.topLeftCorner(m, m) = cMat;
  cs.topRightCorner(m, m) = -sMat;
  cs.bottomLeftCorner(m, m) = sMat;
  cs.bottomRightCorner(m, m) = cMat;
  assert((ld * cs * rd).isApprox(u, TOL) && "CSD reconstruction failed");
#endif
  return {l0, l1, r0, r1, c, s};
}

/// Demultiplex a quantum multiplexor `blockDiag(a, b)` (two `m x m` unitaries
/// selected by the most-significant qubit) into
/// `(I2 (x) v) * blockDiag(d, d^dagger) * (I2 (x) w)`, where `blockDiag(d,
/// d^dagger)` is a uniformly-controlled Rz on the multiplexor qubit. The
/// returned `angles` are the corresponding Rz rotation angles.
/// Ref: arXiv quant-ph/0406176.
void demultiplex(const Eigen::MatrixXcd &a, const Eigen::MatrixXcd &b,
                 Eigen::MatrixXcd &vOut, Eigen::MatrixXcd &wOut,
                 std::vector<double> &angles) {
  int m = static_cast<int>(a.rows());
  /// `a * b^dagger` is unitary (hence normal); its complex Schur form is
  /// diagonal with unitary Schur vectors, giving an orthonormal eigenbasis.
  Eigen::MatrixXcd x = a * b.adjoint();
  Eigen::ComplexSchur<Eigen::MatrixXcd> schur(x);
  vOut = schur.matrixU();
  Eigen::VectorXcd d(m);
  angles.resize(m);
  for (int k = 0; k < m; ++k) {
    double phi = std::arg(schur.matrixT()(k, k));
    d(k) = std::exp(std::complex<double>(0.0, phi / 2.0));
    /// quake `Rz(theta)` applies `exp(-i theta/2)` on the |0> branch; matching
    /// the |0> branch phase `exp(i phi/2)` of `d` requires `theta = -phi`.
    angles[k] = -phi;
  }
  Eigen::MatrixXcd dMat = d.asDiagonal();
  wOut = dMat * vOut.adjoint() * b;
  assert((vOut * dMat * wOut).isApprox(a, TOL) &&
         (vOut * dMat.adjoint() * wOut).isApprox(b, TOL) &&
         "demultiplexing failed");
}

/// Recursive Quantum Shannon Decomposition (QSD) for an arbitrary `n`-qubit
/// (`2^n x 2^n`) unitary with `n >= 3`. One level of QSD splits the unitary,
/// via a Cosine-Sine decomposition and two multiplexor demultiplexings, into
/// four `(n-1)`-qubit unitaries and three uniformly-controlled rotations
/// (Rz, Ry, Rz) acting on the most-significant qubit. The four sub-unitaries
/// are synthesized recursively, with `TwoQubitOpKAK` and `OneQubitOpZYZ` as the
/// base cases. Each factorization step is exact, so no global phase correction
/// is needed at this level (the base cases track their own global phase).
/// Refs: arXiv quant-ph/0404089, arXiv quant-ph/0406176.
struct NQubitOpQSD : public Decomposer {
  Eigen::MatrixXcd targetMatrix;
  size_t numQubits;
  /// `targetMatrix = (I (x) vL) * muxRz(muxRzLeft) * (I (x) wL) * muxRy(muxRy)
  ///               * (I (x) vR) * muxRz(muxRzRight) * (I (x) wR)`
  Eigen::MatrixXcd vL, wL, vR, wR;
  std::vector<double> muxRzLeft, muxRy, muxRzRight;

  void decompose() override {
    auto csd = cosineSineDecomposition(targetMatrix);
    demultiplex(csd.r0, csd.r1, vR, wR, muxRzRight);
    demultiplex(csd.l0, csd.l1, vL, wL, muxRzLeft);
    int m = static_cast<int>(targetMatrix.rows()) / 2;
    muxRy.resize(m);
    for (int k = 0; k < m; ++k)
      muxRy[k] = 2.0 * std::atan2(csd.s(k), csd.c(k));
  }

  /// Emit a uniformly-controlled rotation (multiplexed Rz or Ry) on `target`
  /// controlled by `controls` (most-significant first), using the optimal
  /// gray-code construction. For `k` controls this emits `2^k` rotations and
  /// exactly `2^k` CNOTs -- the minimum for a uniformly-controlled rotation --
  /// rather than the `2^{k+1}-2` CNOTs produced by a naive recursive split
  /// (which inserts a redundant CNOT pair at every recursion boundary). The
  /// per-state angles are mapped to the gray-code rotation angles through the
  /// Walsh-Hadamard-like transform, and the CNOT after rotation `i` targets the
  /// control flipped between gray(i) and gray(i+1). Refs: Möttönen et al.
  /// (arXiv quant-ph/0407010), Shende-Bullock-Markov (arXiv quant-ph/0406176).
  void emitMux(const std::vector<double> &angles, ArrayRef<Value> controls,
               Value target, bool isRy, PatternRewriter &rewriter, Location loc,
               FloatType floatTy) {
    auto emitRot = [&](double angle) {
      if (!isAboveThreshold(angle))
        return;
      auto a =
          cudaq::opt::factory::createFloatConstant(loc, rewriter, angle, floatTy);
      if (isRy)
        cudaq::quake::RyOp::create(rewriter, loc, a, ValueRange{}, target);
      else
        cudaq::quake::RzOp::create(rewriter, loc, a, ValueRange{}, target);
    };
    size_t k = controls.size();
    // Base case: a plain (uncontrolled) rotation.
    if (k == 0) {
      emitRot(angles[0]);
      return;
    }
    size_t n = angles.size(); // == 2^k
    // Transform the desired per-state angles into the gray-code rotation
    // angles: theta'_i = 2^{-k} * sum_j (-1)^{<gray(i), j>} * angles[j].
    std::vector<double> theta(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      size_t gi = i ^ (i >> 1); // binary-reflected gray code of i
      double acc = 0.0;
      for (size_t j = 0; j < n; ++j) {
        int sign = (__builtin_popcountll(gi & j) & 1) ? -1 : 1;
        acc += sign * angles[j];
      }
      theta[i] = acc / static_cast<double>(n);
    }
    // Emit `R(theta'_i)` followed by a single CNOT whose control is the qubit
    // flipped between gray(i) and gray(i+1). `controls` is most-significant
    // first, so bit position `p` (0 == least significant) maps to
    // `controls[k-1-p]`. The closing CNOT (i == n-1) flips the most-significant
    // control, returning the accumulated control parity to the identity.
    for (size_t i = 0; i < n; ++i) {
      emitRot(theta[i]);
      size_t flipBit = (i == n - 1) ? (k - 1) : __builtin_ctzll(i + 1);
      Value ctrl = controls[k - 1 - flipBit];
      cudaq::quake::XOp::create(rewriter, loc, ctrl, target);
    }
  }

  /// Synthesize an `(n-1)`-qubit sub-unitary into its own replacement function,
  /// dispatching to the appropriate decomposer for the base cases.
  void emitChild(const Eigen::MatrixXcd &matrix,
                 cudaq::quake::CustomUnitaryConstantOp customOp,
                 PatternRewriter &rewriter, std::string funcName) {
    size_t childQubits = numQubits - 1;
    if (childQubits == 1) {
      Eigen::Matrix2cd m2 = matrix;
      OneQubitOpZYZ(m2).emitDecomposedFuncOp(customOp, rewriter, funcName);
    } else if (childQubits == 2) {
      TwoQubitOpKAK(matrix).emitDecomposedFuncOp(customOp, rewriter, funcName);
    } else {
      NQubitOpQSD(matrix).emitDecomposedFuncOp(customOp, rewriter, funcName);
    }
  }

  void emitDecomposedFuncOp(cudaq::quake::CustomUnitaryConstantOp customOp,
                            PatternRewriter &rewriter,
                            std::string funcName) override {
    emitChild(wR, customOp, rewriter, funcName + "wr");
    emitChild(vR, customOp, rewriter, funcName + "vr");
    emitChild(wL, customOp, rewriter, funcName + "wl");
    emitChild(vL, customOp, rewriter, funcName + "vl");
    auto parentModule = customOp->getParentOfType<ModuleOp>();
    Location loc = customOp->getLoc();
    auto targets = customOp.getTargets();
    /// This decomposer emits a `numQubits`-qubit replacement function. The arity
    /// is taken from `numQubits` (not from `customOp`) so that it stays correct
    /// when QSD is invoked recursively on an `(n-1)`-qubit sub-unitary of a
    /// larger parent operation.
    SmallVector<Type> argTys(numQubits, targets[0].getType());
    auto funcTy = FunctionType::get(parentModule.getContext(), argTys, {});
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(parentModule.getBody());
    auto func = func::FuncOp::create(rewriter, parentModule->getLoc(), funcName,
                                     funcTy);
    func.setPrivate();
    auto *block = func.addEntryBlock();
    rewriter.setInsertionPointToStart(block);
    auto arguments = func.getArguments();
    FloatType floatTy = rewriter.getF64Type();
    /// `arguments[0]` is the most-significant qubit (the multiplexor target),
    /// `arguments[1..n-1]` carry the `(n-1)`-qubit sub-unitaries and act as the
    /// controls for the multiplexed rotations.
    Value top = arguments[0];
    SmallVector<Value> lower(arguments.begin() + 1, arguments.end());
    auto applyChild = [&](std::string name) {
      cudaq::quake::ApplyOp::create(
          rewriter, loc, TypeRange{},
          SymbolRefAttr::get(rewriter.getContext(), name), false, ValueRange{},
          ValueRange(lower));
    };
    /// Operator notation is right-to-left, so the right-most factor is emitted
    /// first.
    applyChild(funcName + "wr");
    emitMux(muxRzRight, lower, top, false, rewriter, loc, floatTy);
    applyChild(funcName + "vr");
    emitMux(muxRy, lower, top, true, rewriter, loc, floatTy);
    applyChild(funcName + "wl");
    emitMux(muxRzLeft, lower, top, false, rewriter, loc, floatTy);
    applyChild(funcName + "vl");
    func::ReturnOp::create(rewriter, loc);
    rewriter.restoreInsertionPoint(insPt);
  }

  NQubitOpQSD(const Eigen::MatrixXcd &vec) {
    targetMatrix = vec;
    numQubits = static_cast<size_t>(std::llround(std::log2(vec.rows())));
    decompose();
  }
};

class CustomUnitaryPattern
    : public OpRewritePattern<cudaq::quake::CustomUnitaryConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::CustomUnitaryConstantOp customOp,
                                PatternRewriter &rewriter) const override {
    auto parentModule = customOp->getParentOfType<ModuleOp>();
    /// Get the global constant holding the concrete matrix corresponding to
    /// this custom operation invocation
    StringRef generatorName = customOp.getMatrix().getRootReference();
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
      if (dimension == 2) {
        auto zyz = OneQubitOpZYZ(unitary);
        zyz.emitDecomposedFuncOp(customOp, rewriter, funcName);
      } else if (dimension == 4) {
        auto kak = TwoQubitOpKAK(unitary);
        kak.emitDecomposedFuncOp(customOp, rewriter, funcName);
      } else if (dimension >= 8 && (dimension & (dimension - 1)) == 0) {
        /// Recursive Quantum Shannon Decomposition for 3+ qubit operations.
        auto qsd = NQubitOpQSD(unitary);
        qsd.emitDecomposedFuncOp(customOp, rewriter, funcName);
      } else {
        customOp.emitWarning("Decomposition supports custom operations on a "
                             "power-of-two matrix dimension only.");
        return failure();
      }
    }
    rewriter.replaceOpWithNewOp<cudaq::quake::ApplyOp>(
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
      if (failed(
              applyPatternsGreedily(func.getOperation(), std::move(patterns))))
        signalPassFailure();
      LLVM_DEBUG(llvm::dbgs() << "After unitary synthesis: " << func << '\n');
    }
  }
};

} // namespace
