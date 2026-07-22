// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// QOpToCuDensityMat conversion pass: lower qop dialect ops to cudm dialect ops
// for GPU-accelerated quantum state evolution.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.h.inc"

using namespace mlir;

namespace {

struct QOpToCuDensityMatPass
    : public PassWrapper<QOpToCuDensityMatPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QOpToCuDensityMatPass)

  StringRef getArgument() const final { return "qop-to-cudm"; }
  StringRef getDescription() const final {
    return "Lower qop dialect Hamiltonian/Lindblad ops to cudm operator "
           "construction and evolve";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cudm::CuDensityMatDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getContext());

    // Find @main
    func::FuncOp mainFunc;
    module.walk([&](func::FuncOp fn) {
      if (fn.getName() == "main")
        mainFunc = fn;
    });
    if (!mainFunc) {
      module.emitError("no @main function found");
      return signalPassFailure();
    }

    auto &block = mainFunc.getBody().front();
    Operation *returnOp = block.getTerminator();
    b.setInsertionPoint(returnOp);
    auto loc = returnOp->getLoc();

    // --- Determine n_qubits from module attributes or from qop.spin ops ---
    int64_t nQubits = 1;
    auto nqAttr = module->getAttrOfType<IntegerAttr>("qop.n_qubits");
    if (!nqAttr)
      nqAttr = module->getAttrOfType<IntegerAttr>("n_qubits");
    if (nqAttr)
      nQubits = nqAttr.getInt();
    else {
      // Count unique qubit targets from qop.spin ops
      DenseSet<int64_t> targets;
      module.walk([&](qop::SpinOp spin) {
        if (auto cst = spin.getTarget().getDefiningOp<arith::ConstantOp>()) {
          if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
            targets.insert(ia.getInt());
        }
      });
      if (!targets.empty())
        nQubits = *std::max_element(targets.begin(), targets.end()) + 1;
    }

    SmallVector<int64_t> modeExtents(nQubits, 2);

    auto handleTy = cudm::HandleType::get(b.getContext());
    auto stateTy = cudm::StateType::get(b.getContext());
    auto wsTy = cudm::WorkspaceType::get(b.getContext());
    auto elemOpTy = cudm::ElementaryOpType::get(b.getContext());
    auto opTermTy = cudm::OpTermType::get(b.getContext());
    auto operatorTy = cudm::OperatorType::get(b.getContext());

    // 1. cudm.init_handle
    auto handle = b.create<cudm::InitHandleOp>(loc, handleTy);

    // 2. cudm.create_state (|0...0>)
    auto purityAttr =
        cudm::StatePurityAttr::get(b.getContext(), cudm::StatePurity::Pure);
    auto dtypeAttr =
        cudm::ComputeTypeAttr::get(b.getContext(), cudm::ComputeType::F64);
    auto modeExtentsAttr = b.getDenseI64ArrayAttr(modeExtents);
    auto stateIn = b.create<cudm::CreateStateOp>(
        loc, stateTy, handle, purityAttr, dtypeAttr, modeExtentsAttr,
        b.getI64IntegerAttr(0), b.getBoolAttr(false));
    auto stateOut = b.create<cudm::CreateStateOp>(
        loc, stateTy, handle, purityAttr, dtypeAttr, modeExtentsAttr,
        b.getI64IntegerAttr(0), b.getBoolAttr(false));

    // 3. cudm.create_workspace
    auto workspace = b.create<cudm::CreateWorkspaceOp>(loc, wsTy, handle);

    // 4. cudm.create_operator (the composite Hamiltonian operator)
    auto compositeOp = b.create<cudm::CreateOperatorOp>(loc, operatorTy, handle,
                                                        modeExtentsAttr);

    // 5. Walk qop.spin -> cudm.create_elementary_op for each Pauli leaf
    DenseMap<Operation *, Value> spinToCudmElem;
    module.walk([&](qop::SpinOp spin) {
      b.setInsertionPoint(returnOp);
      auto kind = spin.getKind();

      // Map spin kind to 2x2 Pauli matrix data (dense, f64, real+imag
      // interleaved)
      SmallVector<double, 8> pauliData;
      switch (kind) {
      case qop::HandlerKind::SpinX:
        pauliData = {0, 0, 1, 0, 1, 0, 0, 0}; // [[0,1],[1,0]]
        break;
      case qop::HandlerKind::SpinY:
        pauliData = {0, 0, 0, -1, 0, 1, 0, 0}; // [[0,-i],[i,0]]
        break;
      case qop::HandlerKind::SpinZ:
        pauliData = {1, 0, 0, 0, 0, 0, -1, 0}; // [[1,0],[0,-1]]
        break;
      case qop::HandlerKind::SpinI:
        pauliData = {1, 0, 0, 0, 0, 0, 1, 0}; // [[1,0],[0,1]]
        break;
      case qop::HandlerKind::SpinLowering:
        pauliData = {0, 0, 0, 0, 1, 0, 0, 0}; // [[0,0],[1,0]]
        break;
      case qop::HandlerKind::SpinRaising:
        pauliData = {0, 0, 1, 0, 0, 0, 0, 0}; // [[0,1],[0,0]]
        break;
      default:
        spin.emitError("unsupported spin kind for cudm lowering");
        return;
      }

      auto dataType = RankedTensorType::get({2, 2, 2}, b.getF64Type());
      auto dataAttr = DenseFPElementsAttr::get(dataType, pauliData);
      auto tensorVal = b.create<arith::ConstantOp>(loc, dataAttr);

      auto sparsityAttr =
          cudm::SparsityAttr::get(b.getContext(), cudm::Sparsity::None);
      SmallVector<int64_t> elemExtents = {2};

      auto elemOp = b.create<cudm::CreateElementaryOpOp>(
          loc, elemOpTy, handle, tensorVal, sparsityAttr, dtypeAttr,
          b.getDenseI64ArrayAttr(elemExtents), FlatSymbolRefAttr());

      spinToCudmElem[spin.getOperation()] = elemOp;
    });

    // 6. Walk qop.make_product -> cudm.create_op_term +
    // append_elementary_product
    DenseMap<Operation *, Value> productToCudmTerm;
    module.walk([&](qop::MakeProductOp product) {
      b.setInsertionPoint(returnOp);

      auto opTerm = b.create<cudm::CreateOpTermOp>(loc, opTermTy, handle,
                                                   modeExtentsAttr);

      double coeffReal = 1.0, coeffImag = 0.0;
      if (auto constScalar =
              product.getCoefficient().getDefiningOp<qop::ConstScalarOp>()) {
        coeffReal = constScalar.getReal().convertToDouble();
        coeffImag = constScalar.getImag().convertToDouble();
      }

      SmallVector<Value> elemOps;
      SmallVector<int32_t> modesActedOn;
      SmallVector<int32_t> duality;

      for (auto factor : product.getFactors()) {
        if (auto spinOp = factor.getDefiningOp<qop::SpinOp>()) {
          auto it = spinToCudmElem.find(spinOp.getOperation());
          if (it != spinToCudmElem.end())
            elemOps.push_back(it->second);

          int32_t targetMode = 0;
          if (auto cst =
                  spinOp.getTarget().getDefiningOp<arith::ConstantOp>()) {
            if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
              targetMode = (int32_t)ia.getInt();
          }
          modesActedOn.push_back(targetMode);
          duality.push_back(0);
        }
      }

      FlatSymbolRefAttr callbackAttr;
      if (auto cbScalar =
              product.getCoefficient().getDefiningOp<qop::CallbackScalarOp>()) {
        callbackAttr = cbScalar.getCallbackAttr();
        coeffReal = 1.0;
        coeffImag = 0.0;
      }

      b.create<cudm::AppendElementaryProductOp>(
          loc, handle, opTerm, elemOps, b.getDenseI32ArrayAttr(modesActedOn),
          b.getDenseI32ArrayAttr(duality), b.getF64FloatAttr(coeffReal),
          b.getF64FloatAttr(coeffImag), callbackAttr);

      productToCudmTerm[product.getOperation()] = opTerm;
    });

    // 7. Walk qop.make_sum -> cudm.operator_append_term for each term
    module.walk([&](qop::MakeSumOp sum) {
      b.setInsertionPoint(returnOp);
      for (auto term : sum.getTerms()) {
        auto it = productToCudmTerm.find(term.getDefiningOp());
        if (it == productToCudmTerm.end())
          continue;
        b.create<cudm::OperatorAppendTermOp>(
            loc, handle, compositeOp, it->second, b.getI32IntegerAttr(0),
            b.getF64FloatAttr(1.0), b.getF64FloatAttr(0.0),
            FlatSymbolRefAttr());
      }
    });

    // 8. Walk qop.lindblad -> append collapse operators with duality=1
    module.walk([&](qop::LindbladOp lindblad) {
      b.setInsertionPoint(returnOp);
      for (auto collapseOp : lindblad.getCollapseOps()) {
        if (auto sumOp = collapseOp.getDefiningOp<qop::MakeSumOp>()) {
          for (auto term : sumOp.getTerms()) {
            auto it = productToCudmTerm.find(term.getDefiningOp());
            if (it == productToCudmTerm.end())
              continue;
            b.create<cudm::OperatorAppendTermOp>(
                loc, handle, compositeOp, it->second, b.getI32IntegerAttr(1),
                b.getF64FloatAttr(1.0), b.getF64FloatAttr(0.0),
                FlatSymbolRefAttr());
          }
        }
      }
    });

    // 9. cudm.evolve
    double tStart = 0.0, tEnd = 100.0;
    int64_t numSteps = 100;
    auto tsAttr = module->getAttrOfType<FloatAttr>("qop.t_start");
    if (!tsAttr)
      tsAttr = module->getAttrOfType<FloatAttr>("t_start");
    if (tsAttr)
      tStart = tsAttr.getValueAsDouble();
    auto teAttr = module->getAttrOfType<FloatAttr>("qop.t_end");
    if (!teAttr)
      teAttr = module->getAttrOfType<FloatAttr>("t_end");
    if (teAttr)
      tEnd = teAttr.getValueAsDouble();
    auto nsAttr = module->getAttrOfType<IntegerAttr>("qop.num_steps");
    if (!nsAttr)
      nsAttr = module->getAttrOfType<IntegerAttr>("num_steps");
    if (nsAttr)
      numSteps = nsAttr.getInt();

    auto integratorAttr = cudm::IntegratorKindAttr::get(
        b.getContext(), cudm::IntegratorKind::MagnusCF4);

    b.create<cudm::EvolveOp>(
        loc, stateTy, handle, compositeOp, stateIn, stateOut, workspace,
        integratorAttr, b.getF64FloatAttr(tStart), b.getF64FloatAttr(tEnd),
        b.getI64IntegerAttr(numSteps), cudm::ComputeTypeAttr());

    // 10. Cleanup (destroy in reverse order)
    b.create<cudm::DestroyOperatorOp>(loc, compositeOp);
    b.create<cudm::DestroyWorkspaceOp>(loc, workspace);
    b.create<cudm::DestroyStateOp>(loc, stateOut);
    b.create<cudm::DestroyStateOp>(loc, stateIn);
    b.create<cudm::DestroyHandleOp>(loc, handle);
  }
};

} // namespace

namespace qop {

std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass() {
  return std::make_unique<QOpToCuDensityMatPass>();
}

} // namespace qop
