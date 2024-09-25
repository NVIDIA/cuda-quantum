/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

namespace {

constexpr double TOL = 1e-8;

class Decomposer {
public:
  virtual void decompose() = 0;
  virtual void emitDecomposedFuncOp(quake::CustomUnitarySymbolOp customOp,
                                    PatternRewriter &rewriter,
                                    std::string funcName) = 0;
  bool isAboveThreshold(double value) { return std::abs(value) > TOL; };
  virtual ~Decomposer() = default;
};

struct EulerAngles {
  double alpha;
  double beta;
  double gamma;
};

struct OneQubitOpZYZ : public Decomposer {
  /// TODO: Update to use Eigen matrix
  std::array<std::complex<double>, 4> matrix;
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
    auto det = (matrix[0] * matrix[3]) - (matrix[1] * matrix[2]);
    phase = 0.5 * std::arg(det);
    std::array<std::complex<double>, 4> specialUnitary;
    std::transform(
        matrix.begin(), matrix.end(), specialUnitary.begin(),
        [&](auto element) { return element * std::exp(-1i * phase); });
    auto abs00 = std::abs(specialUnitary[0]);
    auto abs01 = std::abs(specialUnitary[1]);
    if (abs00 >= abs01)
      angles.beta = 2 * std::acos(abs00);
    else
      angles.beta = 2 * std::asin(abs01);
    auto sum = std::atan2(specialUnitary[3].imag(), specialUnitary[3].real());
    auto diff = std::atan2(specialUnitary[2].imag(), specialUnitary[2].real());
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
    /// is left-to-right. Hence, angles are applied as
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
    auto globalPhase = 2 * phase;
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

  OneQubitOpZYZ(const std::vector<std::complex<double>> &vec) {
    std::copy(vec.begin(), vec.begin() + 4, matrix.begin());
    decompose();
  }

  OneQubitOpZYZ(const Eigen::Matrix2cd &vec) {
    for (size_t r = 0; r < 2; r++)
      for (size_t c = 0; c < 2; c++)
        matrix[r * 2 + c] = vec(r, c);
    decompose();
  }
};

/// Temporary helper code
void display4x4Matrix(const Eigen::Matrix4cd &matrix, std::string msg = "") {
  llvm::outs() << msg << "\n";
  for (size_t r = 0; r < 4; r++) {
    for (size_t c = 0; c < 4; c++) {
      double real =
          std::abs(matrix(r, c).real()) > TOL ? matrix(r, c).real() : 0.0;
      double imag =
          std::abs(matrix(r, c).imag()) > TOL ? matrix(r, c).imag() : 0.0;
      llvm::outs() << "(" << real << ", " << imag << ")\t";
    }
    llvm::outs() << "\n";
  }
}

// Compute exp(i(x XX + y YY + z ZZ)) matrix
Eigen::Matrix4cd interactionMatrixExp(double x, double y, double z) {
  constexpr std::complex<double> I{0.0, 1.0};
  Eigen::MatrixXcd X{Eigen::MatrixXcd::Zero(2, 2)};
  Eigen::MatrixXcd Y{Eigen::MatrixXcd::Zero(2, 2)};
  Eigen::MatrixXcd Z{Eigen::MatrixXcd::Zero(2, 2)};
  X << 0, 1, 1, 0;
  Y << 0, -I, I, 0;
  Z << 1, 0, 0, -1;
  auto XX = Eigen::kroneckerProduct(X, X);
  auto YY = Eigen::kroneckerProduct(Y, Y);
  auto ZZ = Eigen::kroneckerProduct(Z, Z);
  Eigen::MatrixXcd herm = x * XX + y * YY + z * ZZ;
  herm = I * herm;
  Eigen::MatrixXcd unitary = herm.exp();
  return unitary;
}

inline bool isSquare(const Eigen::MatrixXcd &in_mat) {
  return in_mat.rows() == in_mat.cols();
}

// If the matrix is finite: no NaN elements
template <typename Derived>
inline bool isFinite(const Eigen::MatrixBase<Derived> &x) {
  return ((x - x).array() == (x - x).array()).all();
}

bool isDiagonal(const Eigen::MatrixXcd &in_mat, double in_tol = TOL) {
  if (!isFinite(in_mat)) {
    return false;
  }

  for (int i = 0; i < in_mat.rows(); ++i) {
    for (int j = 0; j < in_mat.cols(); ++j) {
      if (i != j) {
        if (std::abs(in_mat(i, j)) > in_tol) {
          return false;
        }
      }
    }
  }

  return true;
}

bool allClose(const Eigen::MatrixXcd &in_mat1, const Eigen::MatrixXcd &in_mat2,
              double in_tol = TOL) {
  if (!isFinite(in_mat1) || !isFinite(in_mat2)) {
    return false;
  }

  if (in_mat1.rows() == in_mat2.rows() && in_mat1.cols() == in_mat2.cols()) {
    for (int i = 0; i < in_mat1.rows(); ++i) {
      for (int j = 0; j < in_mat1.cols(); ++j) {
        if (std::abs(in_mat1(i, j) - in_mat2(i, j)) > in_tol) {
          return false;
        }
      }
    }

    return true;
  }
  return false;
}

bool isHermitian(const Eigen::MatrixXcd &in_mat) {
  if (!isSquare(in_mat) || !isFinite(in_mat)) {
    return false;
  }
  return allClose(in_mat, in_mat.adjoint());
}

// Eigen::MatrixXd constructBlockDiagonalMatrix(const Eigen::MatrixXd &top,
//                                              const Eigen::MatrixXd &bottom) {
//   Eigen::MatrixXd blockDiagMat = Eigen::MatrixXd::Zero(
//       top.rows() + bottom.rows(), top.cols() + bottom.cols());
//   blockDiagMat.block(0, 0, top.rows(), top.cols()) = top;
//   blockDiagMat.block(top.rows(), top.cols(), bottom.rows(), bottom.cols()) =
//       bottom;
//   return blockDiagMat;
// }

void diagonalize(const Eigen::Matrix4cd &matrix){
  // Split into two real matrices with the real and imaginary parts
  Eigen::Matrix4d A;
  Eigen::Matrix4d B;
  for (int r = 0; r < matrix.rows(); r++) {
    for (int c = 0; c < matrix.cols(); c++) {
      A(r, c) = matrix(r, c).real();
      B(r, c) = matrix(r, c).imag();
    }
  }
  assert(isHermitian(A * B.transpose()));
  assert(isHermitian(A.transpose() * B));

  /// Using Eckart-Young

  // Get the orthogonal matrices
  Eigen::JacobiSVD<Eigen::Matrix4d> svdA(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  assert(svdA.matrixU().isUnitary(TOL));
  assert(svdA.matrixV().isUnitary(TOL));
  display4x4Matrix(svdA.singularValues().asDiagonal(), "Diagonal matrix from SVD"); 

  // D is a square diagonal matrix whose diagonal elements are strictly positive
  Eigen::Matrix4d D = svdA.singularValues().asDiagonal().toDenseMatrix();

  // D and G are square matrices of the same dimension, rank(A)
  // Find the rank
  size_t rank = 4;
  while (rank > 0 && std::abs(A(rank - 1, rank - 1) < TOL)) 
    rank--;
  // DG† = GD , DG = G†D
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(rank, rank);
  for (size_t r = 0; r < rank; r++) 
    for (size_t c = 0; c < rank; c++) 
      G(r, c) = D(r, c);
    
  // Trying something
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(D);
  // Eigen::MatrixXd Dg = solver.eigenvectors().adjoint() * G * solver.eigenvectors();  

  Eigen::MatrixXd P = solver.eigenvectors(); 

  auto UPrime = P.adjoint() * svdA.matrixU().adjoint();
  auto V = svdA.matrixV() * P;
  
  display4x4Matrix(UPrime.adjoint(), "left");
  display4x4Matrix(V, "right");
  // auto bPrime = svdA.matrixU().transpose() * B * svdA.matrixV().transpose();
  // Eigen::MatrixXd overlap = Eigen::MatrixXd::Zero(rank, rank);
  // for (size_t r = 0; r < rank;r++) 
  //   for (size_t c = 0; c < rank; c++) 
  //     overlap(r, c) = bPrime(r, c);
    
  llvm::outs() << isDiagonal(UPrime.adjoint() * matrix * V) << "\n";  

}

struct KAKComponents {
  // KAK decomposition allows to express arbitrary 2-qubit unitary (U) in the
  // form: U = (a1 ⊗ a0) x exp(i(xXX + yYY + zZZ)) x (b1 ⊗ b0) where, a0, a1
  // (after), b0, b1 (before) are single qubit operations, and the exponential
  // is specified by the 3 elements of canonical class vector x, y, z
  Eigen::Matrix2cd a0;
  Eigen::Matrix2cd a1;
  Eigen::Matrix2cd b0;
  Eigen::Matrix2cd b1;
  double x;
  double y;
  double z;
};

const Eigen::Matrix4cd &MagicBasisMatrix() {
  static Eigen::Matrix4cd MagicBasisMatrix;
  constexpr std::complex<double> i{0.0, 1.0};
  MagicBasisMatrix << 1.0, 0.0, 0.0, i, 0.0, i, 1.0, 0, 0, i, -1.0, 0, 1.0, 0,
      0, -i;
  MagicBasisMatrix = MagicBasisMatrix * M_SQRT1_2;
  return MagicBasisMatrix;
}

const Eigen::Matrix4cd &MagicBasisMatrixAdj() {
  static Eigen::Matrix4cd MagicBasisMatrixAdj = MagicBasisMatrix().adjoint();
  return MagicBasisMatrixAdj;
}

const Eigen::Matrix4cd &GammaFactor() {
  /// Gamma matrix = +1 +1 −1 +1
  ///                +1 +1 +1 −1
  ///                +1 −1 −1 −1
  ///                +1 −1 +1 +1
  static Eigen::Matrix4cd GammaT;
  GammaT << 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1;
  GammaT /= 4;
  return GammaT;
}

std::tuple<Eigen::Matrix2cd, Eigen::Matrix2cd, std::complex<double>>
extractSU2FromSO4(const Eigen::Matrix4cd &matrix) {
  /// Use Kronecker factorization
  Eigen::Matrix2cd part1 = Eigen::Matrix2cd::Zero();
  Eigen::Matrix2cd part2 = Eigen::Matrix2cd::Zero();
  int r = 0, c = 0;
  double largest = std::abs(matrix(r, c));
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      if (largest < std::abs(matrix(i, j))) {
        largest = std::abs(matrix(i, j));
        r = i;
        c = j;
      }
    }
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      part1((r >> 1) ^ i, (c >> 1) ^ j) = matrix(r ^ (i << 1), c ^ (j << 1));
      part2((r & 1) ^ i, (c & 1) ^ j) = matrix(r ^ i, c ^ j);
    }
  }
  auto det1 = part1.determinant();
  if (std::abs(det1) > TOL)
    part1 /= (std::sqrt(det1));
  auto det2 = part2.determinant();
  if (std::abs(det2) > TOL)
    part2 /= (std::sqrt(det2));
  std::complex<double> phase =
      matrix(r, c) / (part1(r >> 1, c >> 1) * part2(r & 1, c & 1));
  if (phase.real() < 0.0) {
    part1 *= -1;
    phase = -phase;
  }

  display4x4Matrix(matrix, "input matrix");
  auto checkResult = phase * Eigen::kroneckerProduct(part1, part2);
  display4x4Matrix(checkResult, "phase * part1 * part2");
  // assert(matrix.isApprox(checkResult, TOL));

  return std::make_tuple(part1, part2, phase);
}

struct TwoQubitOpKAK : public Decomposer {
  Eigen::Matrix4cd matrix;
  KAKComponents components;
  /// Updates to the global phase
  std::complex<double> phase;

  /// This logic is based on the Cartan's KAK decomposition.
  /// Ref: https://arxiv.org/pdf/quant-ph/0507171
  /// Ref: https://arxiv.org/pdf/0806.4015
  void decompose() override {
    using namespace std::complex_literals;
    display4x4Matrix(matrix, "Input matrix");

    /// Step0: Convert to special unitary
    phase = std::pow(matrix.determinant(), 0.25);
    auto specialUnitary = matrix / phase;

    display4x4Matrix(specialUnitary, "Special unitary target matrix");

    /// Step1: Convert the target matrix into magic basis
    Eigen::Matrix4cd matrixMagicBasis =
        MagicBasisMatrixAdj() * specialUnitary * MagicBasisMatrix();
    display4x4Matrix(matrixMagicBasis, "matrixMagicBasis");

    diagonalize(matrixMagicBasis);

    /// Step2: Diagonalize the exponentiation of magic matrix
    Eigen::Matrix4cd mSquare = matrixMagicBasis.transpose() * matrixMagicBasis;
    assert(mSquare.isApprox(mSquare.transpose(), TOL));



    Eigen::ComplexEigenSolver<Eigen::Matrix4cd> solver(mSquare);
    Eigen::Vector4cd eigenVal = solver.eigenvalues();
    Eigen::Matrix4cd eigenVec = solver.eigenvectors();

    assert(mSquare.isApprox(
        eigenVec * eigenVal.asDiagonal() * eigenVec.adjoint(), TOL));

    assert(std::abs(std::abs(eigenVec.determinant()) - 1.0) < TOL);
    if (eigenVec.determinant().real() < 0)
      for (int c = 0; c < eigenVec.cols(); c++)
        eigenVec(0, c) = -eigenVec(0, c);

    display4x4Matrix(eigenVec, "Eigen vector matrix");
    assert(std::abs(std::abs(eigenVec.determinant()) - 1.0) < TOL);
    display4x4Matrix(eigenVec.transpose() * eigenVec, "Is this identity??");

    Eigen::Matrix4cd diagonalMat = eigenVal.asDiagonal().toDenseMatrix();
    display4x4Matrix(diagonalMat, "Diagonal matrix");
    assert(std::abs(std::abs(diagonalMat.determinant()) - 1.0) < TOL);

    Eigen::Matrix4cd diagMatSqRoot = diagonalMat.unaryExpr(
        [](std::complex<double> x) { return std::sqrt(x); });

    display4x4Matrix(diagMatSqRoot, "Square root of diagonal matrix");
    assert(std::abs(std::abs(diagMatSqRoot.determinant()) - 1.0) < TOL);

    if (diagMatSqRoot.determinant().real() < 0)
      diagMatSqRoot(0, 0) = -diagMatSqRoot(0, 0);
    assert(std::abs(std::abs(diagMatSqRoot.determinant()) - 1.0) < TOL);

    /// Step3: Get the KAK components
    Eigen::Matrix4cd before = MagicBasisMatrix() * matrixMagicBasis * eigenVec *
                              diagMatSqRoot.adjoint() * MagicBasisMatrixAdj();
    assert(std::abs(std::abs(before.determinant()) - 1.0) < TOL);
    display4x4Matrix(before, "1st component K1 i.e. before");

    Eigen::Matrix4cd after =
        MagicBasisMatrix() * eigenVec.adjoint() * MagicBasisMatrixAdj();
    assert(std::abs(std::abs(after.determinant()) - 1.0) < TOL);
    display4x4Matrix(after, "2nd component K2 i.e. after");

    Eigen::Matrix4cd canonicalVec =
        MagicBasisMatrix() * diagMatSqRoot * MagicBasisMatrixAdj();
    assert(std::abs(std::abs(canonicalVec.determinant()) - 1.0) < TOL);
    display4x4Matrix(canonicalVec, "Middle component A i.e. canonical vector");

    Eigen::Matrix4cd checkInterim = before * canonicalVec * after;
    display4x4Matrix(checkInterim, "Intermediate output matrix");
    assert(specialUnitary.isApprox(checkInterim, TOL));

    /// Step4: Get the coefficients of canonical class vector
    Eigen::Vector4cd diagonalPhases;
    for (size_t i = 0; i < 4; i++)
      diagonalPhases(i) = std::arg(diagMatSqRoot(i, i));

    auto coefficients = GammaFactor() * diagonalPhases;
    components.x = coefficients(1).real();
    components.y = coefficients(2).real();
    components.z = coefficients(3).real();
    phase *= std::exp(1i * coefficients(0));

    llvm::outs() << "x = " << components.x << "\ty = " << components.y
                 << "\tz = " << components.z << "\n";

    Eigen::Matrix4cd checkResult =
        before *
        interactionMatrixExp(components.x, components.y, components.z) * after;
    assert(specialUnitary.isApprox(std::exp(1i * coefficients(0)) * checkResult,
                                   TOL));

    auto beforeRes = extractSU2FromSO4(before);
    components.b0 = std::get<0>(beforeRes);
    components.b1 = std::get<1>(beforeRes);
    phase *= std::get<2>(beforeRes);

    auto afterRes = extractSU2FromSO4(after);
    components.a0 = std::get<0>(afterRes);
    components.a1 = std::get<1>(afterRes);
    phase *= std::get<2>(afterRes);

    /// Final check
    Eigen::Matrix4cd checkFinal =
        (std::get<2>(beforeRes) *
         Eigen::kroneckerProduct(components.b0, components.b1)) *
        interactionMatrixExp(components.x, components.y, components.z) *
        (std::get<2>(afterRes) *
         Eigen::kroneckerProduct(components.a0, components.a1));
    display4x4Matrix(checkFinal, "Final result");
    // assert(specialUnitary.isApprox(std::exp(1i * coefficients(0)) *
    // checkFinal,
    //                                TOL));
  }

  void emitDecomposedFuncOp(quake::CustomUnitarySymbolOp customOp,
                            PatternRewriter &rewriter,
                            std::string funcName) override {
    auto b0 = OneQubitOpZYZ(components.b0);
    b0.emitDecomposedFuncOp(customOp, rewriter, funcName + "b0");
    auto b1 = OneQubitOpZYZ(components.b1);
    b1.emitDecomposedFuncOp(customOp, rewriter, funcName + "b1");
    auto a0 = OneQubitOpZYZ(components.a0);
    a0.emitDecomposedFuncOp(customOp, rewriter, funcName + "a0");
    auto a1 = OneQubitOpZYZ(components.a1);
    a1.emitDecomposedFuncOp(customOp, rewriter, funcName + "a1");

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
        SymbolRefAttr::get(rewriter.getContext(), funcName + "a1"), false,
        ValueRange{}, ValueRange{arguments[1]});
    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "a0"), false,
        ValueRange{}, ValueRange{arguments[0]});

    /// TODO: Refactor to use a transformation pass for `quake.exp_pauli`
    /// XX
    if (isAboveThreshold(components.x)) {
      rewriter.create<quake::HOp>(loc, arguments[0]);
      rewriter.create<quake::HOp>(loc, arguments[1]);
      rewriter.create<quake::XOp>(loc, arguments[0], arguments[1]);
      auto xAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.x, floatTy);
      rewriter.create<quake::RzOp>(loc, xAngle, ValueRange{}, arguments[1]);
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
      rewriter.create<quake::HOp>(loc, arguments[0]);
      rewriter.create<quake::HOp>(loc, arguments[1]);
    }
    /// YY
    if (isAboveThreshold(components.y)) {
      auto piBy2 = cudaq::opt::factory::createFloatConstant(loc, rewriter,
                                                            M_PI_2, floatTy);
      rewriter.create<quake::RxOp>(loc, piBy2, ValueRange{}, arguments[0]);
      rewriter.create<quake::RxOp>(loc, piBy2, ValueRange{}, arguments[1]);
      rewriter.create<quake::XOp>(loc, arguments[0], arguments[1]);
      auto yAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.y, floatTy);
      rewriter.create<quake::RzOp>(loc, yAngle, ValueRange{}, arguments[1]);
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
      Value negPiBy2 = rewriter.create<arith::NegFOp>(loc, piBy2);
      rewriter.create<quake::RxOp>(loc, negPiBy2, ValueRange{}, arguments[0]);
      rewriter.create<quake::RxOp>(loc, negPiBy2, ValueRange{}, arguments[1]);
    }
    /// ZZ
    if (isAboveThreshold(components.z)) {
      rewriter.create<quake::XOp>(loc, arguments[0], arguments[1]);
      auto zAngle = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, -2.0 * components.z, floatTy);
      rewriter.create<quake::RzOp>(loc, zAngle, ValueRange{}, arguments[1]);
      rewriter.create<quake::XOp>(loc, arguments[1], arguments[0]);
    }
    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "b1"), false,
        ValueRange{}, ValueRange{arguments[1]});
    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{},
        SymbolRefAttr::get(rewriter.getContext(), funcName + "b0"), false,
        ValueRange{}, ValueRange{arguments[0]});

    auto globalPhase = 2 * std::arg(phase);
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
    matrix = vec;
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
      switch (matrix.size()) {
      case 4: {
        auto zyz = OneQubitOpZYZ(matrix);
        zyz.emitDecomposedFuncOp(customOp, rewriter, funcName);
      } break;
      case 16: {
        auto unitary = Eigen::Map<Eigen::Matrix4cd>(matrix.data());
        unitary.transposeInPlace();
        if (!unitary.isUnitary(TOL)) {
          customOp.emitWarning("The custom operation matrix must be unitary.");
          return failure();
        }
        auto kak = TwoQubitOpKAK(unitary);
        kak.emitDecomposedFuncOp(customOp, rewriter, funcName);
      } break;
      default:
        customOp.emitWarning(
            "Decomposition of only single qubit custom operations supported.");
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
