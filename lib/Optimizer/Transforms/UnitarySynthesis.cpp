/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#include "cudaq/Optimizer/UnitaryDecomposition.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

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
  virtual void emitDecomposedFuncOp(quake::CustomUnitarySymbolOp customOp,
                                    PatternRewriter &rewriter,
                                    std::string funcName) = 0;
  bool isAboveThreshold(double value) { return std::abs(value) > TOL; };
  virtual ~Decomposer() = default;
};

struct OneQubitOpZYZ : public Decomposer {
  Eigen::Matrix2cd targetMatrix;
  cudaq::detail::EulerAngles angles;
  /// Updates to the global phase
  double phase;

  void decompose() override {
    auto result = cudaq::detail::decomposeZYZ(targetMatrix);
    angles.alpha = result.alpha;
    angles.beta = result.beta;
    angles.gamma = result.gamma;
    phase = result.phase;
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

struct TwoQubitOpKAK : public Decomposer {
  Eigen::Matrix4cd targetMatrix;
  cudaq::detail::KAKComponents components;
  /// Updates to the global phase
  std::complex<double> phase;

  void decompose() override {
    auto result = cudaq::detail::decomposeKAK(targetMatrix);
    components.a0 = result.a0;
    components.a1 = result.a1;
    components.b0 = result.b0;
    components.b1 = result.b1;
    components.x = result.x;
    components.y = result.y;
    components.z = result.z;
    phase = result.phase;
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
