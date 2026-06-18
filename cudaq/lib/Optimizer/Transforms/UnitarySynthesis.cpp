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
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <algorithm>
#include <memory>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>

namespace cudaq::opt {
#define GEN_PASS_DEF_UNITARYSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "unitary-synthesis"

using namespace mlir;

namespace {

/// Tight tolerance for the input-unitarity *contract* and the pure
/// numerical-safety guards (division by a near-zero determinant, Gram-Schmidt
/// linear-independence). A custom operation's matrix must be unitary to this
/// bound before we attempt to synthesize it.
constexpr double TOL = 1e-7;

/// Separate, calibrated tolerance for the always-on synthesis *reconstruction*
/// self-checks (verifying that a computed factorization reproduces its target
/// block). Eigen's `isApprox(B, p)` is a *relative* test
/// (`||A - B|| <= p * min(||A||, ||B||)`), so this is a relative bound.
///
/// This is deliberately looser than `TOL` and is calibrated to measured
/// behavior rather than chosen arbitrarily. The robust `atan2` angle formula in
/// `OneQubitOpZYZ` (see below) removes the dominant numerical failure (a
/// near-degenerate controlled single-qubit sub-block returning `NaN`). After
/// that fix, a *correctly* synthesized unitary reconstructs to ~1e-10 when
/// well-conditioned; however, rare near-degenerate sub-blocks that arise deep
/// in the 5-qubit recursion (in the cosine-sine decomposition and multiplexor
/// demultiplexing) push the reconstruction residual up to ~8e-7 while the
/// emitted circuit still reproduces the target to <4e-7 end-to-end (verified by
/// forced emission and state reconstruction over Haar-random 5-qubit inputs).
/// A genuinely *wrong* decomposition, by contrast, reconstructs to O(0.1).
/// `RECON_TOL` therefore sits ~1 order above the measured correct-synthesis
/// residual and ~4 orders below the failure regime, so the advertised 3-5 qubit
/// range synthesizes predictably (no spurious bail on a valid 5-qubit unitary)
/// while the composable `emitWarning`/bail path still rejects genuinely broken
/// results. The input-unitarity contract above stays at the tight `TOL`.
constexpr double RECON_TOL = 1e-5;

/// Base class for unitary synthesis, i.e. decomposing an arbitrary unitary
/// matrix into native gate set. The native gate set here includes all the
/// quantum operations supported by CUDA-Q. Additional passes may be required to
/// convert CUDA-Q gate set to hardware specific gate set.
class Decomposer {
private:
  Eigen::MatrixXcd targetMatrix;

protected:
  /// Set to `false` by a decomposer when its synthesis cannot reproduce the
  /// target matrix within `TOL` (e.g. accumulated numerical error). The pass
  /// uses this to stay composable: on an invalid result it leaves the original
  /// IR unchanged instead of emitting a wrong circuit or crashing the compiler.
  bool valid = true;

public:
  /// Whether the decomposition reproduces the target matrix within `TOL`.
  bool isValid() const { return valid; }
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
                       OpBuilder &rewriter, std::string funcName) = 0;
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
    /// The Y-rotation angle is `beta = 2 * atan2(|u01|, |u00|)`. Using `atan2`
    /// of the two magnitudes (rather than `acos(|u00|)` or `asin(|u01|)`) is
    /// numerically robust: `acos`/`asin` are ill-conditioned near their +/-1
    /// endpoints (derivative diverges, amplifying round-off to ~1e-8) and,
    /// worse, return NaN when their argument exceeds 1 by even a rounding ULP
    /// -- which happens for the near-degenerate sub-blocks produced deep in a
    /// recursive Quantum Shannon Decomposition (e.g. a controlled single-qubit
    /// gate, where |u00| ~ 0 and |u01| ~ 1). `atan2` is well-conditioned over
    /// the whole domain and needs no clamping, so the same gate is emitted
    /// reliably across the advertised 3-5 qubit range.
    angles.beta = 2.0 * std::atan2(abs01, abs00);
    auto sum =
        std::atan2(specialUnitary(1, 1).imag(), specialUnitary(1, 1).real());
    auto diff =
        std::atan2(specialUnitary(1, 0).imag(), specialUnitary(1, 0).real());
    angles.alpha = sum + diff;
    angles.gamma = sum - diff;
  }

  void emitDecomposedFuncOp(cudaq::quake::CustomUnitaryConstantOp customOp,
                            OpBuilder &rewriter,
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
/// cases like degenerate matrices. The QZ step can lose accuracy on the
/// numerically marginal sub-blocks produced deep in a recursive Quantum Shannon
/// Decomposition; instead of asserting (which would abort the whole
/// compilation), `ok` is cleared when the result is not diagonal within
/// `RECON_TOL` so the caller can bail out gracefully and leave the IR
/// unchanged. The final KAK reconstruction check is the authoritative
/// correctness backstop.
std::tuple<Eigen::Matrix4d, Eigen::Matrix4cd, Eigen::Matrix4d>
bidiagonalize(const Eigen::Matrix4cd &matrix, bool &ok) {
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
  if (!diagonal.isDiagonal(RECON_TOL))
    ok = false;
  return std::make_tuple(left, diagonal, right);
}

/// Separate input matrix into local operations. The input matrix must be
/// special orthogonal. Given a map, SU(2) × SU(2) -> SO(4),
/// map(A, B) = M.adjoint() (A ⊗ B∗) M, find A and B.
std::tuple<Eigen::Matrix2cd, Eigen::Matrix2cd, std::complex<double>>
extractSU2FromSO4(const Eigen::Matrix4cd &matrix, bool &ok) {
  /// Verify input matrix is special orthogonal. On a marginal sub-block from a
  /// deep recursion this can drift; clear `ok` instead of asserting so the
  /// caller bails out gracefully rather than aborting compilation.
  if (!(std::abs(std::abs(matrix.determinant()) - 1.0) < RECON_TOL))
    ok = false;
  if (!((matrix * matrix.transpose() - Eigen::Matrix4cd::Identity()).norm() <
        RECON_TOL))
    ok = false;
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
  if (!mb.isApprox(phase * Eigen::kroneckerProduct(part1, part2), RECON_TOL))
    ok = false;
  if (!(part1.isUnitary(RECON_TOL) && part2.isUnitary(RECON_TOL)))
    ok = false;
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
    /// Step2: Diagonalize. The intermediate sanity checks below are composable:
    /// they clear `ok` (and hence `valid`) instead of asserting, so a marginal
    /// sub-block from a deep recursion makes the pass leave the IR unchanged
    /// rather than aborting the compiler.
    bool ok = true;
    auto [left, diagonal, right] = bidiagonalize(matrixMagicBasis, ok);
    /// Step3: Get the KAK components
    auto [a1, a0, aPh] = extractSU2FromSO4(left, ok);
    components.a0 = a0;
    components.a1 = a1;
    phase *= aPh;
    auto [b1, b0, bPh] = extractSU2FromSO4(right, ok);
    components.b0 = b0;
    components.b1 = b1;
    phase *= bPh;
    valid = valid && ok;
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
    /// Final check to verify results. This runs unconditionally (not just in
    /// debug builds): on failure we mark the decomposition invalid so the pass
    /// can leave the input IR untouched rather than emit a wrong circuit.
    auto canVecToMat =
        canonicalVecToMatrix(components.x, components.y, components.z);
    if (!targetMatrix.isApprox(phase * Eigen::kroneckerProduct(a1, a0) *
                                   canVecToMat *
                                   Eigen::kroneckerProduct(b1, b0),
                               RECON_TOL))
      valid = false;
  }

  void emitDecomposedFuncOp(cudaq::quake::CustomUnitaryConstantOp customOp,
                            OpBuilder &rewriter,
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
Eigen::MatrixXcd completeUnitaryBasis(const Eigen::MatrixXcd &definedCols,
                                      Eigen::Index n) {
  Eigen::MatrixXcd basis = definedCols;
  /// `Eigen::Index` is the natural index type of Eigen containers, so taking
  /// `.cols()` directly avoids a truncating cast and any signed/unsigned mix
  /// with the Eigen indexing API below.
  Eigen::Index col = basis.cols();
  for (Eigen::Index i = 0; i < n && col < n; ++i) {
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
CSDComponents cosineSineDecomposition(const Eigen::MatrixXcd &u, bool &ok) {
  Eigen::Index m = u.rows() / 2;
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
  for (Eigen::Index k = 0; k < m; ++k)
    s(k) = std::sqrt(std::max(0.0, 1.0 - c(k) * c(k)));
  /// `u10 * v = l1 * diag(s)`; recover `l1` column-by-column, completing the
  /// columns belonging to (near-)zero sines.
  Eigen::MatrixXcd mCols = u10 * v;
  Eigen::MatrixXcd l1 = Eigen::MatrixXcd::Zero(m, m);
  std::vector<Eigen::Index> definedCols;
  for (Eigen::Index k = 0; k < m; ++k)
    if (s(k) > TOL) {
      l1.col(k) = mCols.col(k) / s(k);
      definedCols.push_back(k);
    }
  if (static_cast<Eigen::Index>(definedCols.size()) < m) {
    Eigen::MatrixXcd known(m, definedCols.size());
    for (size_t i = 0; i < definedCols.size(); ++i)
      known.col(i) = l1.col(definedCols[i]);
    Eigen::MatrixXcd full = completeUnitaryBasis(known, m);
    std::vector<bool> isDefined(m, false);
    for (Eigen::Index d : definedCols)
      isDefined[d] = true;
    Eigen::Index next = static_cast<Eigen::Index>(definedCols.size());
    for (Eigen::Index k = 0; k < m; ++k)
      if (!isDefined[k])
        l1.col(k) = full.col(next++);
  }
  /// `r1` from `u11 = l1 diag(c) r1` where cosine dominates, otherwise from
  /// `u01 = -l0 diag(s) r1`.
  Eigen::MatrixXcd fromC = l1.adjoint() * u11;
  Eigen::MatrixXcd fromS = -(l0.adjoint() * u01);
  Eigen::MatrixXcd r1(m, m);
  for (Eigen::Index k = 0; k < m; ++k) {
    if (c(k) >= s(k))
      r1.row(k) = fromC.row(k) / c(k);
    else
      r1.row(k) = fromS.row(k) / s(k);
  }
  /// Verify the reconstruction `blockDiag(l0,l1) * CS * blockDiag(r0,r1) == u`.
  /// This runs unconditionally (host `double` precision, independent of the
  /// backend's execution precision): `ok` is cleared on failure so the caller
  /// can abandon synthesis and leave the input IR unchanged.
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
  if (!(ld * cs * rd).isApprox(u, RECON_TOL))
    ok = false;
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
                 std::vector<double> &angles, bool &ok) {
  Eigen::Index m = a.rows();
  /// `a * b^dagger` is unitary (hence normal); its complex Schur form is
  /// diagonal with unitary Schur vectors, giving an orthonormal eigenbasis.
  Eigen::MatrixXcd x = a * b.adjoint();
  Eigen::ComplexSchur<Eigen::MatrixXcd> schur(x);
  vOut = schur.matrixU();
  Eigen::VectorXcd d(m);
  angles.resize(m);
  for (Eigen::Index k = 0; k < m; ++k) {
    double phi = std::arg(schur.matrixT()(k, k));
    d(k) = std::exp(std::complex<double>(0.0, phi / 2.0));
    /// quake `Rz(theta)` applies `exp(-i theta/2)` on the |0> branch; matching
    /// the |0> branch phase `exp(i phi/2)` of `d` requires `theta = -phi`.
    angles[k] = -phi;
  }
  Eigen::MatrixXcd dMat = d.asDiagonal();
  wOut = dMat * vOut.adjoint() * b;
  /// Verify `a = v d w` and `b = v d^dagger w` (always-on); clear `ok` on
  /// failure so the caller can leave the input IR unchanged.
  if (!((vOut * dMat * wOut).isApprox(a, RECON_TOL) &&
        (vOut * dMat.adjoint() * wOut).isApprox(b, RECON_TOL)))
    ok = false;
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
  /// The four `(n-1)`-qubit sub-unitary decomposers, built during `decompose()`
  /// so the whole recursion tree is validated before any IR is emitted.
  std::unique_ptr<Decomposer> childWr, childVr, childWl, childVl;

  /// Build the decomposer for an `(n-1)`-qubit sub-unitary, dispatching to the
  /// base cases (ZYZ for 1 qubit, KAK for 2) or recursing with QSD otherwise.
  std::unique_ptr<Decomposer> makeChild(const Eigen::MatrixXcd &matrix) {
    size_t childQubits = numQubits - 1;
    if (childQubits == 1) {
      Eigen::Matrix2cd m2 = matrix;
      return std::make_unique<OneQubitOpZYZ>(m2);
    }
    if (childQubits == 2)
      return std::make_unique<TwoQubitOpKAK>(matrix);
    return std::make_unique<NQubitOpQSD>(matrix);
  }

  void decompose() override {
    bool ok = true;
    auto csd = cosineSineDecomposition(targetMatrix, ok);
    demultiplex(csd.r0, csd.r1, vR, wR, muxRzRight, ok);
    demultiplex(csd.l0, csd.l1, vL, wL, muxRzLeft, ok);
    valid = valid && ok;
    int m = static_cast<int>(targetMatrix.rows()) / 2;
    muxRy.resize(m);
    for (int k = 0; k < m; ++k)
      muxRy[k] = 2.0 * std::atan2(csd.s(k), csd.c(k));
    /// Only recurse once this level reconstructs cleanly: feeding the marginal
    /// sub-blocks of a failed split into the children could trip their own
    /// numerical checks. Validate the whole tree before emitting any IR.
    if (valid) {
      childWr = makeChild(wR);
      childVr = makeChild(vR);
      childWl = makeChild(wL);
      childVl = makeChild(vL);
      valid = childWr->isValid() && childVr->isValid() && childWl->isValid() &&
              childVl->isValid();
    }
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
               Value target, bool isRy, OpBuilder &rewriter, Location loc,
               FloatType floatTy) {
    auto emitRot = [&](double angle) {
      if (!isAboveThreshold(angle))
        return;
      auto a = cudaq::opt::factory::createFloatConstant(loc, rewriter, angle,
                                                        floatTy);
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
        int sign = (llvm::popcount(gi & j) & 1) ? -1 : 1;
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
      size_t flipBit = (i == n - 1) ? (k - 1) : llvm::countr_zero(i + 1);
      Value ctrl = controls[k - 1 - flipBit];
      cudaq::quake::XOp::create(rewriter, loc, ctrl, target);
    }
  }

  void emitDecomposedFuncOp(cudaq::quake::CustomUnitaryConstantOp customOp,
                            OpBuilder &rewriter,
                            std::string funcName) override {
    /// Emit using the children built (and validated) in `decompose()`.
    childWr->emitDecomposedFuncOp(customOp, rewriter, funcName + "wr");
    childVr->emitDecomposedFuncOp(customOp, rewriter, funcName + "vr");
    childWl->emitDecomposedFuncOp(customOp, rewriter, funcName + "wl");
    childVl->emitDecomposedFuncOp(customOp, rewriter, funcName + "vl");
    auto parentModule = customOp->getParentOfType<ModuleOp>();
    Location loc = customOp->getLoc();
    auto targets = customOp.getTargets();
    /// This decomposer emits a `numQubits`-qubit replacement function. The
    /// arity is taken from `numQubits` (not from `customOp`) so that it stays
    /// correct when QSD is invoked recursively on an `(n-1)`-qubit sub-unitary
    /// of a larger parent operation.
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

/// Build the decomposer for a `dimension x dimension` custom-operation matrix,
/// or return null when the dimension is outside the supported synthesis range
/// (a power of two with `2 <= dimension <= 32`, i.e. 1-5 qubits). The returned
/// decomposer has already run its factorization; callers consult `isValid()` to
/// learn whether the reconstruction stayed within tolerance.
static std::shared_ptr<Decomposer>
makeDecomposer(const Eigen::MatrixXcd &unitary, size_t dimension) {
  if (dimension == 2) {
    Eigen::Matrix2cd m2 = unitary;
    return std::make_shared<OneQubitOpZYZ>(m2);
  }
  if (dimension == 4)
    return std::make_shared<TwoQubitOpKAK>(unitary);
  if (dimension >= 8 && dimension <= 32 && (dimension & (dimension - 1)) == 0)
    return std::make_shared<NQubitOpQSD>(unitary);
  return nullptr;
}

/// Cached classification of a custom operation's generator matrix. The
/// decomposition (SVD/Schur/CSD) is computed once and reused by both the
/// conversion-target legality check and the rewrite pattern.
struct DecompInfo {
  enum Status {
    Convertible,         ///< supported dimension, unitary, reconstruction valid
    NonUnitary,          ///< matrix is not unitary to `TOL`
    UnsupportedDim,      ///< not a power of two, or a power of two > 32
    ReconstructionFailed ///< supported dimension but reconstruction missed
  };
  std::shared_ptr<Decomposer> decomposer; ///< non-null only when `Convertible`
  Status status = UnsupportedDim;
  size_t dimension = 0;
  bool powerOfTwo = false;
};

/// Classify (and cache) a custom operation by its generator symbol so each
/// distinct matrix is factored exactly once.
static const DecompInfo &
getOrBuildDecompInfo(cudaq::quake::CustomUnitaryConstantOp customOp,
                     llvm::StringMap<DecompInfo> &cache) {
  StringRef generatorName = customOp.getMatrix().getRootReference();
  auto it = cache.find(generatorName);
  if (it != cache.end())
    return it->second;
  auto parentModule = customOp->getParentOfType<ModuleOp>();
  auto globalOp = parentModule.lookupSymbol<cudaq::cc::GlobalOp>(generatorName);
  auto matrix = cudaq::opt::factory::readGlobalConstantArray(globalOp);
  size_t dimension = std::sqrt(matrix.size());
  auto unitary =
      Eigen::Map<Eigen::MatrixXcd>(matrix.data(), dimension, dimension);
  unitary.transposeInPlace();
  DecompInfo info;
  info.dimension = dimension;
  info.powerOfTwo = dimension != 0 && (dimension & (dimension - 1)) == 0;
  if (!unitary.isUnitary(TOL)) {
    info.status = DecompInfo::NonUnitary;
  } else if (auto decomp = makeDecomposer(unitary, dimension)) {
    info.status = decomp->isValid() ? DecompInfo::Convertible
                                    : DecompInfo::ReconstructionFailed;
    info.decomposer = std::move(decomp);
  } else {
    info.status = DecompInfo::UnsupportedDim;
  }
  return cache.try_emplace(generatorName, std::move(info)).first->second;
}

/// Replacement-function name for a custom operation's generator symbol.
static std::string replacementFuncName(StringRef generatorName) {
  auto pair = generatorName.split(".rodata");
  return pair.first.str() + ".kernel" + pair.second.str();
}

/// Lowers a synthesizable `quake.custom_unitary_constant` into a `quake.apply`
/// of its decomposed kernel. The kernels are generated up-front by the pass
/// below and the conversion target marks a custom op illegal exactly when such
/// a kernel exists, so this pattern is a pure structural replacement -- the "do
/// not commit unless legal" guarantee is enforced by the legality predicate.
class CustomUnitaryPattern
    : public OpConversionPattern<cudaq::quake::CustomUnitaryConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::CustomUnitaryConstantOp customOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string funcName =
        replacementFuncName(customOp.getMatrix().getRootReference());
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

    /// Classification cache so every generator matrix is decomposed at most
    /// once, shared by the kernel pre-generation and the post-pass diagnostics.
    llvm::StringMap<DecompInfo> cache;

    /// Pre-generate the decomposed replacement kernel for every synthesizable
    /// custom operation. We walk the custom ops in reverse and insert each new
    /// kernel at the start of the module, so the kernels end up in the original
    /// (forward) custom-op order regardless of the conversion driver's
    /// traversal order. A plain `OpBuilder` is used so these sibling functions
    /// are created outside the conversion process.
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(module.getBody());
    SmallVector<cudaq::quake::CustomUnitaryConstantOp> customOps;
    module.walk([&](cudaq::quake::CustomUnitaryConstantOp op) {
      customOps.push_back(op);
    });
    for (auto op : llvm::reverse(customOps)) {
      std::string funcName =
          replacementFuncName(op.getMatrix().getRootReference());
      if (module.lookupSymbol<func::FuncOp>(funcName))
        continue; // a previous invocation of this generator already built it
      const DecompInfo &info = getOrBuildDecompInfo(op, cache);
      if (info.status == DecompInfo::Convertible)
        info.decomposer->emitDecomposedFuncOp(op, builder, funcName);
    }

    /// Use the dialect-conversion framework to replace each synthesizable
    /// custom operation with a `quake.apply` of its decomposed kernel. A custom
    /// op is *illegal* (must be rewritten) exactly when such a kernel exists;
    /// everything else is *legal* and left untouched, so the pass stays
    /// composable (no `signalPassFailure`, no crash) -- the rewrite is never
    /// committed unless the legality constraint is met.
    ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalOp<cudaq::quake::CustomUnitaryConstantOp>(
        [](cudaq::quake::CustomUnitaryConstantOp op) {
          return !op->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
              replacementFuncName(op.getMatrix().getRootReference()));
        });

    RewritePatternSet patterns(ctx);
    patterns.insert<CustomUnitaryPattern>(ctx);

    LLVM_DEBUG(llvm::dbgs() << "Before unitary synthesis: " << module << '\n');
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    /// Keep the user-visible diagnostics for custom operations we intentionally
    /// left unchanged. These are warnings/notes, never errors: another pass or
    /// back-end may still legalize the op.
    module.walk([&](cudaq::quake::CustomUnitaryConstantOp op) {
      auto it = cache.find(op.getMatrix().getRootReference());
      if (it == cache.end())
        return;
      switch (it->second.status) {
      case DecompInfo::NonUnitary:
        op.emitWarning("The custom operation matrix must be unitary.");
        break;
      case DecompInfo::ReconstructionFailed:
        /// In the advertised 3-5 qubit range but the reconstruction self-check
        /// failed. Stay composable -- leave the op unchanged -- but make the
        /// outcome visible instead of a silent no-op in Release builds.
        op.emitWarning(
            "unitary-synthesis: could not synthesize this custom operation "
            "within the reconstruction tolerance; leaving it unchanged.");
        break;
      case DecompInfo::UnsupportedDim:
        if (it->second.powerOfTwo)
          LLVM_DEBUG(llvm::dbgs()
                     << "unitary-synthesis: matrix dimension "
                     << it->second.dimension
                     << " exceeds the supported 5-qubit cap (32); leaving "
                        "custom op unchanged\n");
        else
          op.emitWarning("Decomposition supports custom operations on a "
                         "power-of-two matrix dimension only.");
        break;
      case DecompInfo::Convertible:
        break;
      }
    });
    LLVM_DEBUG(llvm::dbgs() << "After unitary synthesis: " << module << '\n');
  }
};

} // namespace
