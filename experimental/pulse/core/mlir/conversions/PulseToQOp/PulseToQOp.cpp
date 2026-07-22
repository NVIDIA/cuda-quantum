// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PulseToQOp conversion pass: lower pulse dialect ops to qop dialect ops
// for Hamiltonian/Lindblad construction.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

#include <cmath>

using namespace mlir;

namespace {

// ---- Utility: extract integer attribute value ----
static int64_t getI64Attr(Operation *op, StringRef name, int64_t fallback = 0) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

static double getF64Attr(Operation *op, StringRef name, double fallback = 0.0) {
  if (auto attr = op->getAttrOfType<FloatAttr>(name))
    return attr.getValueAsDouble();
  return fallback;
}

// ---- Build a qop.spin + qop.const_scalar + qop.make_product term ----
static Value buildStaticTerm(OpBuilder &b, Location loc, Value target,
                             StringRef spinKind, double coeffReal,
                             double coeffImag) {
  auto handlerTy = qop::HandlerType::get(b.getContext());
  auto scalarTy = qop::ScalarType::get(b.getContext());
  auto productTy = qop::ProductType::get(b.getContext());

  // qop.spin
  auto kindAttr = qop::symbolizeHandlerKind(spinKind);
  if (!kindAttr)
    return {};
  auto spin = b.create<qop::SpinOp>(
      loc, handlerTy, target,
      qop::HandlerKindAttr::get(b.getContext(), *kindAttr));

  // qop.const_scalar
  auto scalar =
      b.create<qop::ConstScalarOp>(loc, scalarTy, b.getF64FloatAttr(coeffReal),
                                   b.getF64FloatAttr(coeffImag));

  // qop.make_product
  auto product =
      b.create<qop::MakeProductOp>(loc, productTy, scalar, ValueRange{spin});

  return product;
}

// ---- Build a time-dependent drive term with callback ----
static Value buildDriveTerm(OpBuilder &b, Location loc, Value target,
                            StringRef spinKind, StringRef callbackName) {
  auto handlerTy = qop::HandlerType::get(b.getContext());
  auto scalarTy = qop::ScalarType::get(b.getContext());
  auto productTy = qop::ProductType::get(b.getContext());

  auto kindAttr = qop::symbolizeHandlerKind(spinKind);
  if (!kindAttr)
    return {};
  auto spin = b.create<qop::SpinOp>(
      loc, handlerTy, target,
      qop::HandlerKindAttr::get(b.getContext(), *kindAttr));
  auto cbScalar = b.create<qop::CallbackScalarOp>(
      loc, scalarTy, FlatSymbolRefAttr::get(b.getContext(), callbackName));
  auto product =
      b.create<qop::MakeProductOp>(loc, productTy, cbScalar, ValueRange{spin});
  return product;
}

struct PulseToQOpPass
    : public PassWrapper<PulseToQOpPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseToQOpPass)

  StringRef getArgument() const final { return "pulse-to-qop"; }
  StringRef getDescription() const final {
    return "Lower pulse dialect ops to qop Hamiltonian/Lindblad construction";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<qop::QOpDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getContext());

    // Collect drive ops, qubit info, and dissipator metadata
    SmallVector<Operation *> driveOps;
    DenseMap<int64_t, double> qubitFreqHz;
    SmallVector<std::pair<Value, int64_t>> qubitTargets;

    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "pulse.get_drive_line") {
        if (auto qubitAttr = op->getAttrOfType<IntegerAttr>("qubit")) {
          int64_t qi = qubitAttr.getInt();
          if (auto freqAttr = op->getAttrOfType<FloatAttr>("frequency_hz"))
            qubitFreqHz[qi] = freqAttr.getValueAsDouble();
        }
      }
      if (op->getName().getStringRef() == "pulse.drive")
        driveOps.push_back(op);
    });

    // Find the func.func @main and insert QOp construction at the end
    func::FuncOp mainFunc;
    module.walk([&](func::FuncOp fn) {
      if (fn.getName() == "main")
        mainFunc = fn;
    });
    if (!mainFunc) {
      module.emitError("no @main function found");
      return signalPassFailure();
    }

    // Insert before the return op
    auto &block = mainFunc.getBody().front();
    Operation *returnOp = block.getTerminator();
    b.setInsertionPoint(returnOp);
    auto loc = returnOp->getLoc();

    auto productTy = qop::ProductType::get(b.getContext());
    auto opTy = qop::OpType::get(b.getContext());
    auto superOpTy = qop::SuperOpType::get(b.getContext());

    SmallVector<Value> allProducts;

    // 1. Static Hamiltonian terms: omega_q * sigma_z / 2 for each qubit
    for (auto &[qi, freqHz] : qubitFreqHz) {
      auto target = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(qi));
      double omega = freqHz * 2.0 * M_PI * 1e-9; // Convert Hz to rad/ns
      auto term = buildStaticTerm(b, loc, target, "spin_z", omega / 2.0, 0.0);
      if (term)
        allProducts.push_back(term);
    }

    // 2. Time-dependent drive terms
    int driveIdx = 0;
    for (auto *op : driveOps) {
      std::string cbName = "drive_envelope_" + std::to_string(driveIdx);
      int64_t qubitIdx = -1;

      // Trace back to find the qubit index from the drive line
      if (auto lineOp = op->getOperand(0).getDefiningOp()) {
        if (lineOp->getName().getStringRef() == "pulse.get_drive_line") {
          if (auto qa = lineOp->getAttrOfType<IntegerAttr>("qubit"))
            qubitIdx = qa.getInt();
        }
        // Follow SSA chain for updated drive lines
        if (qubitIdx < 0) {
          Operation *cur = lineOp;
          while (cur) {
            if (cur->getName().getStringRef() == "pulse.get_drive_line") {
              if (auto qa = cur->getAttrOfType<IntegerAttr>("qubit"))
                qubitIdx = qa.getInt();
              break;
            }
            if (cur->getNumOperands() > 0)
              cur = cur->getOperand(0).getDefiningOp();
            else
              break;
          }
        }
      }
      if (qubitIdx < 0)
        qubitIdx = driveIdx;

      auto target =
          b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(qubitIdx));

      // X-component
      auto termX = buildDriveTerm(b, loc, target, "spin_x", cbName + "_x");
      if (termX)
        allProducts.push_back(termX);

      // Y-component (for DRAG or nonzero phase)
      auto termY = buildDriveTerm(b, loc, target, "spin_y", cbName + "_y");
      if (termY)
        allProducts.push_back(termY);

      driveIdx++;
    }

    // 3. Assemble Hamiltonian: qop.make_sum
    Value hamiltonian;
    if (!allProducts.empty()) {
      hamiltonian = b.create<qop::MakeSumOp>(loc, opTy, allProducts);
    } else {
      // Trivial Hamiltonian: identity
      auto target = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(0));
      auto term = buildStaticTerm(b, loc, target, "spin_i", 0.0, 0.0);
      hamiltonian = b.create<qop::MakeSumOp>(loc, opTy, ValueRange{term});
    }

    // 4. Dissipators from module attributes (T1, T2)
    SmallVector<Value> collapseOps;

    auto t1AttrRaw = module->getAttrOfType<ArrayAttr>("pulse.t1_times");
    if (!t1AttrRaw)
      t1AttrRaw = module->getAttrOfType<ArrayAttr>("t1_times");
    if (auto t1Attr = t1AttrRaw) {
      for (int64_t qi = 0; qi < (int64_t)t1Attr.size(); qi++) {
        double t1 = cast<FloatAttr>(t1Attr[qi]).getValueAsDouble();
        if (t1 > 0) {
          double gamma = 1.0 / t1;
          auto target =
              b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(qi));
          auto lowering = buildStaticTerm(b, loc, target, "spin_lowering",
                                          std::sqrt(gamma), 0.0);
          if (lowering) {
            auto collapseOp =
                b.create<qop::MakeSumOp>(loc, opTy, ValueRange{lowering});
            collapseOps.push_back(collapseOp);
          }
        }
      }
    }

    auto t2AttrRaw = module->getAttrOfType<ArrayAttr>("pulse.t2_times");
    if (!t2AttrRaw)
      t2AttrRaw = module->getAttrOfType<ArrayAttr>("t2_times");
    if (auto t2Attr = t2AttrRaw) {
      for (int64_t qi = 0; qi < (int64_t)t2Attr.size(); qi++) {
        double t2 = cast<FloatAttr>(t2Attr[qi]).getValueAsDouble();
        double t1 = 0.0;
        auto t1a = module->getAttrOfType<ArrayAttr>("pulse.t1_times");
        if (!t1a)
          t1a = module->getAttrOfType<ArrayAttr>("t1_times");
        if (t1a)
          if (qi < (int64_t)t1a.size())
            t1 = cast<FloatAttr>(t1a[qi]).getValueAsDouble();
        double gammaPhi = 0.0;
        if (t2 > 0) {
          gammaPhi = 1.0 / t2;
          if (t1 > 0)
            gammaPhi -= 1.0 / (2.0 * t1);
          if (gammaPhi < 0)
            gammaPhi = 0;
        }
        if (gammaPhi > 0) {
          auto target =
              b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(qi));
          auto dephase = buildStaticTerm(b, loc, target, "spin_z",
                                         std::sqrt(2.0 * gammaPhi), 0.0);
          if (dephase) {
            auto collapseOp =
                b.create<qop::MakeSumOp>(loc, opTy, ValueRange{dephase});
            collapseOps.push_back(collapseOp);
          }
        }
      }
    }

    // 5. Construct Lindblad super-operator
    b.create<qop::LindbladOp>(loc, superOpTy, hamiltonian, collapseOps);
  }
};

} // namespace

namespace pulse {

std::unique_ptr<mlir::Pass> createPulseToQOpPass() {
  return std::make_unique<PulseToQOpPass>();
}

} // namespace pulse
