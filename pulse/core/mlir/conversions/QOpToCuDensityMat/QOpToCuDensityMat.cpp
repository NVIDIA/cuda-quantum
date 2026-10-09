/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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
    auto handle = cudm::InitHandleOp::create(b, loc, handleTy);

    SmallVector<qop::LindbladOp> lindbladOps;
    module.walk([&](qop::LindbladOp op) { lindbladOps.push_back(op); });
    if (lindbladOps.size() != 1) {
      module.emitError("qop-to-cudm requires exactly one qop.lindblad op");
      return signalPassFailure();
    }
    const bool hasDissipation = !lindbladOps.front().getCollapseOps().empty();

    // 2. cudm.create_state (|0...0>). Open-system evolution requires a
    // density matrix; unitary evolution can retain the cheaper state vector.
    auto purityAttr = cudm::StatePurityAttr::get(
        b.getContext(),
        hasDissipation ? cudm::StatePurity::Mixed : cudm::StatePurity::Pure);
    auto dtypeAttr =
        cudm::ComputeTypeAttr::get(b.getContext(), cudm::ComputeType::F64);
    auto modeExtentsAttr = b.getDenseI64ArrayAttr(modeExtents);
    auto stateIn = cudm::CreateStateOp::create(
        b, loc, stateTy, handle, purityAttr, dtypeAttr, modeExtentsAttr,
        b.getI64IntegerAttr(0), b.getBoolAttr(false));
    auto stateOut = cudm::CreateStateOp::create(
        b, loc, stateTy, handle, purityAttr, dtypeAttr, modeExtentsAttr,
        b.getI64IntegerAttr(0), b.getBoolAttr(false));

    // 3. cudm.create_workspace
    auto workspace = cudm::CreateWorkspaceOp::create(b, loc, wsTy, handle);

    // 4. cudm.create_operator (the composite Hamiltonian operator)
    auto compositeOp = cudm::CreateOperatorOp::create(b, loc, operatorTy,
                                                      handle, modeExtentsAttr);

    // 5. Walk qop.spin -> cudm.create_elementary_op for each Pauli leaf
    DenseMap<Operation *, Value> spinToCudmElem;
    DenseMap<Operation *, Value> spinToDaggerCudmElem;
    SmallVector<Value> allElementaryOps;
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
        // cuDensityMat uses column-major storage: [[0,-i],[i,0]].
        pauliData = {0, 0, 0, 1, 0, -1, 0, 0};
        break;
      case qop::HandlerKind::SpinZ:
        pauliData = {1, 0, 0, 0, 0, 0, -1, 0}; // [[1,0],[0,-1]]
        break;
      case qop::HandlerKind::SpinI:
        pauliData = {1, 0, 0, 0, 0, 0, 1, 0}; // [[1,0],[0,1]]
        break;
      case qop::HandlerKind::SpinLowering:
        pauliData = {0, 0, 0, 0, 1, 0, 0, 0}; // [[0,1],[0,0]]
        break;
      case qop::HandlerKind::SpinRaising:
        pauliData = {0, 0, 1, 0, 0, 0, 0, 0}; // [[0,0],[1,0]]
        break;
      default:
        spin.emitError("unsupported spin kind for cudm lowering");
        return;
      }

      auto dataType = RankedTensorType::get({2, 2, 2}, b.getF64Type());
      auto dataAttr = DenseFPElementsAttr::get(dataType, pauliData);
      auto tensorVal = arith::ConstantOp::create(b, loc, dataAttr);

      auto sparsityAttr =
          cudm::SparsityAttr::get(b.getContext(), cudm::Sparsity::None);
      SmallVector<int64_t> elemExtents = {2};

      auto elemOp = cudm::CreateElementaryOpOp::create(
          b, loc, elemOpTy, handle, tensorVal, sparsityAttr, dtypeAttr,
          b.getDenseI64ArrayAttr(elemExtents), FlatSymbolRefAttr());

      spinToCudmElem[spin.getOperation()] = elemOp;
      allElementaryOps.push_back(elemOp);

      // Pauli operators are self-adjoint. Raising and lowering are each
      // other's adjoints, so materialize the counterpart for C^dagger C in
      // Lindblad anticommutator terms.
      if (kind != qop::HandlerKind::SpinLowering &&
          kind != qop::HandlerKind::SpinRaising) {
        spinToDaggerCudmElem[spin.getOperation()] = elemOp;
      } else {
        SmallVector<double, 8> daggerData;
        if (kind == qop::HandlerKind::SpinLowering)
          daggerData = {0, 0, 1, 0, 0, 0, 0, 0};
        else
          daggerData = {0, 0, 0, 0, 1, 0, 0, 0};
        auto daggerAttr = DenseFPElementsAttr::get(dataType, daggerData);
        auto daggerTensor = arith::ConstantOp::create(b, loc, daggerAttr);
        auto daggerElem = cudm::CreateElementaryOpOp::create(
            b, loc, elemOpTy, handle, daggerTensor, sparsityAttr, dtypeAttr,
            b.getDenseI64ArrayAttr(elemExtents), FlatSymbolRefAttr());
        spinToDaggerCudmElem[spin.getOperation()] = daggerElem;
        allElementaryOps.push_back(daggerElem);
      }
    });

    // 6. Lower qop products to reusable cuDensityMat operator terms.
    struct ProductData {
      SmallVector<Operation *> factors;
      SmallVector<Value> elementaryOps;
      SmallVector<int32_t> modes;
      double coefficientReal = 1.0;
      double coefficientImag = 0.0;
      bool hasCallback = false;
    };
    DenseMap<Operation *, ProductData> productData;
    DenseMap<Operation *, Value> productToCudmTerm;
    SmallVector<Value> allOperatorTerms;
    bool conversionFailed = false;

    auto createTerm = [&](ArrayRef<Value> elementaryOps,
                          ArrayRef<int32_t> modes, ArrayRef<int32_t> dualities,
                          double coefficientReal, double coefficientImag,
                          qop::CallbackScalarOp callback = {}) -> Value {
      auto term = cudm::CreateOpTermOp::create(b, loc, opTermTy, handle,
                                               modeExtentsAttr);
      FlatSymbolRefAttr callbackAttr;
      if (callback)
        callbackAttr = callback.getCallbackAttr();
      auto append = cudm::AppendElementaryProductOp::create(
          b, loc, handle, term, elementaryOps, b.getDenseI32ArrayAttr(modes),
          b.getDenseI32ArrayAttr(dualities), b.getF64FloatAttr(coefficientReal),
          b.getF64FloatAttr(coefficientImag), callbackAttr);
      if (callback) {
        if (auto kind = callback->getAttr("cudm.callback_kind"))
          append->setAttr("cudm.callback_kind", kind);
        if (auto parameters = callback->getAttr("cudm.callback_params"))
          append->setAttr("cudm.callback_params", parameters);
      }
      allOperatorTerms.push_back(term);
      return term;
    };

    module.walk([&](qop::MakeProductOp product) {
      b.setInsertionPoint(returnOp);
      ProductData data;
      qop::CallbackScalarOp callback;
      if (auto scalar =
              product.getCoefficient().getDefiningOp<qop::ConstScalarOp>()) {
        data.coefficientReal = scalar.getReal().convertToDouble();
        data.coefficientImag = scalar.getImag().convertToDouble();
      } else if (auto scalar = product.getCoefficient()
                                   .getDefiningOp<qop::CallbackScalarOp>()) {
        callback = scalar;
        data.hasCallback = true;
      } else {
        product.emitError("coefficient must be a qop constant or callback");
        conversionFailed = true;
        return;
      }

      for (Value factor : product.getFactors()) {
        auto spin = factor.getDefiningOp<qop::SpinOp>();
        if (!spin) {
          product.emitError("only qop.spin factors are currently supported");
          conversionFailed = true;
          return;
        }
        auto elementary = spinToCudmElem.find(spin.getOperation());
        auto target = spin.getTarget().getDefiningOp<arith::ConstantOp>();
        auto targetAttr =
            target ? dyn_cast<IntegerAttr>(target.getValue()) : IntegerAttr{};
        if (elementary == spinToCudmElem.end() || !targetAttr ||
            targetAttr.getInt() < 0 || targetAttr.getInt() >= nQubits) {
          product.emitError("spin target must be a valid constant mode index");
          conversionFailed = true;
          return;
        }
        data.factors.push_back(spin.getOperation());
        data.elementaryOps.push_back(elementary->second);
        data.modes.push_back(static_cast<int32_t>(targetAttr.getInt()));
      }
      if (data.elementaryOps.empty()) {
        product.emitError("operator product must contain at least one factor");
        conversionFailed = true;
        return;
      }

      SmallVector<int32_t> ketDualities(data.elementaryOps.size(), 0);
      auto term =
          createTerm(data.elementaryOps, data.modes, ketDualities,
                     data.coefficientReal, data.coefficientImag, callback);
      productData[product.getOperation()] = std::move(data);
      productToCudmTerm[product.getOperation()] = term;
    });
    if (conversionFailed)
      return signalPassFailure();

    // 7. Build -i[H,rho]. A pure-state evolution only needs the ket action;
    // a mixed-state evolution also needs the opposite-sign bra action.
    auto hamiltonian =
        lindbladOps.front().getHamiltonian().getDefiningOp<qop::MakeSumOp>();
    if (!hamiltonian) {
      lindbladOps.front().emitError(
          "Hamiltonian must be represented by qop.make_sum");
      return signalPassFailure();
    }
    for (Value product : hamiltonian.getTerms()) {
      auto term = productToCudmTerm.find(product.getDefiningOp());
      if (term == productToCudmTerm.end()) {
        hamiltonian.emitError("contains an unsupported product term");
        return signalPassFailure();
      }
      cudm::OperatorAppendTermOp::create(
          b, loc, handle, compositeOp, term->second, b.getI32IntegerAttr(0),
          b.getF64FloatAttr(0.0), b.getF64FloatAttr(-1.0), FlatSymbolRefAttr());
      if (hasDissipation)
        cudm::OperatorAppendTermOp::create(
            b, loc, handle, compositeOp, term->second, b.getI32IntegerAttr(1),
            b.getF64FloatAttr(0.0), b.getF64FloatAttr(1.0),
            FlatSymbolRefAttr());
    }

    // 8. Build D[C](rho) = C rho C^dagger
    //                         - 1/2 {C^dagger C, rho}.
    // Pulse-generated collapse operators contain one product each. Reject a
    // sum here rather than silently dropping the required cross terms.
    for (Value collapse : lindbladOps.front().getCollapseOps()) {
      auto sum = collapse.getDefiningOp<qop::MakeSumOp>();
      if (!sum || sum.getTerms().size() != 1) {
        lindbladOps.front().emitError(
            "each collapse operator must contain exactly one product");
        return signalPassFailure();
      }
      auto product = sum.getTerms().front().getDefiningOp();
      auto found = productData.find(product);
      if (found == productData.end() || found->second.hasCallback) {
        lindbladOps.front().emitError(
            "collapse products must have constant coefficients");
        return signalPassFailure();
      }
      const ProductData &data = found->second;
      const double coefficientNorm =
          data.coefficientReal * data.coefficientReal +
          data.coefficientImag * data.coefficientImag;

      SmallVector<Value> jumpOps(data.elementaryOps.begin(),
                                 data.elementaryOps.end());
      SmallVector<int32_t> jumpModes(data.modes.begin(), data.modes.end());
      // C rho C^dagger: the bra-side product is the reversed sequence of
      // adjoint factors, not a second copy of C. This distinction is essential
      // for non-Hermitian collapse operators such as sigma-minus.
      for (auto [factor, mode] :
           llvm::reverse(llvm::zip(data.factors, data.modes))) {
        auto dagger = spinToDaggerCudmElem.find(factor);
        if (dagger == spinToDaggerCudmElem.end()) {
          lindbladOps.front().emitError(
              "could not form the adjoint of a collapse factor");
          return signalPassFailure();
        }
        jumpOps.push_back(dagger->second);
        jumpModes.push_back(mode);
      }
      SmallVector<int32_t> jumpDualities(data.elementaryOps.size(), 0);
      jumpDualities.append(data.elementaryOps.size(), 1);
      auto jumpTerm = createTerm(jumpOps, jumpModes, jumpDualities, 1.0, 0.0);
      cudm::OperatorAppendTermOp::create(
          b, loc, handle, compositeOp, jumpTerm, b.getI32IntegerAttr(0),
          b.getF64FloatAttr(coefficientNorm), b.getF64FloatAttr(0.0),
          FlatSymbolRefAttr());

      // cuDensityMat composes an elementary product in application order. Its
      // documented C * C.dag() construction is therefore represented by the
      // same factor sequence as the two-sided term, with every factor acting
      // on the ket side.
      SmallVector<int32_t> normDualities(jumpOps.size(), 0);
      auto normTerm = createTerm(jumpOps, jumpModes, normDualities, 1.0, 0.0);
      for (int32_t duality : {0, 1})
        cudm::OperatorAppendTermOp::create(
            b, loc, handle, compositeOp, normTerm, b.getI32IntegerAttr(duality),
            b.getF64FloatAttr(-0.5 * coefficientNorm), b.getF64FloatAttr(0.0),
            FlatSymbolRefAttr());
    }

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

    auto integrator = cudm::IntegratorKind::RungeKutta4;
    if (auto value = module->getAttrOfType<StringAttr>("qop.integrator")) {
      auto parsed = cudm::symbolizeIntegratorKind(value.getValue());
      if (!parsed) {
        module.emitError("unsupported qop.integrator: ") << value.getValue();
        return signalPassFailure();
      }
      integrator = *parsed;
    }
    auto integratorAttr =
        cudm::IntegratorKindAttr::get(b.getContext(), integrator);

    cudm::EvolveOp::create(
        b, loc, stateTy, handle, compositeOp, stateIn, stateOut, workspace,
        integratorAttr, b.getF64FloatAttr(tStart), b.getF64FloatAttr(tEnd),
        b.getI64IntegerAttr(numSteps), cudm::ComputeTypeAttr());

    // 10. Cleanup (destroy in reverse order)
    cudm::DestroyOperatorOp::create(b, loc, compositeOp);
    for (Value term : llvm::reverse(allOperatorTerms))
      cudm::DestroyOpTermOp::create(b, loc, term);
    for (Value elementaryOp : llvm::reverse(allElementaryOps))
      cudm::DestroyElementaryOpOp::create(b, loc, elementaryOp);
    cudm::DestroyWorkspaceOp::create(b, loc, workspace);
    cudm::DestroyStateOp::create(b, loc, stateOut);
    cudm::DestroyStateOp::create(b, loc, stateIn);
    cudm::DestroyHandleOp::create(b, loc, handle);

    // The cuDensityMat graph no longer depends on the source QOp graph.
    SmallVector<Operation *> qopOps;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "qop")
        qopOps.push_back(op);
    });
    for (auto iter = qopOps.rbegin(); iter != qopOps.rend(); ++iter)
      (*iter)->erase();
  }
};

} // namespace

namespace qop {

std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass() {
  return std::make_unique<QOpToCuDensityMatPass>();
}

} // namespace qop
