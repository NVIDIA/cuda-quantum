/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOWERCCMEASUREHANDLE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "lower-cc-measure-handle"

using namespace mlir;

namespace {

// Lower `!cc.measure_handle` (the IR alias of `cudaq::measure_handle`) to
// `i64` everywhere it appears: SSA result types, block argument types, and
// function signatures, as well as recursively inside aggregate CC types and
// function types. The substitution preserves bit-width and never moves
// operations, so an imperative walk is sufficient -- there is no need for the
// partial-conversion framework or source/target materializations.
class LowerCCMeasureHandlePass
    : public cudaq::opt::impl::LowerCCMeasureHandleBase<
          LowerCCMeasureHandlePass> {
public:
  using LowerCCMeasureHandleBase::LowerCCMeasureHandleBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto i64Ty = IntegerType::get(&getContext(), 64);

    // Convert `t` recursively, returning a new type if any
    // `!cc.measure_handle` was replaced by `i64`, otherwise the original.
    std::function<Type(Type)> convert = [&](Type t) -> Type {
      if (isa<cudaq::cc::MeasureHandleType>(t))
        return i64Ty;
      if (auto sv = dyn_cast<cudaq::cc::StdvecType>(t)) {
        auto inner = convert(sv.getElementType());
        if (inner != sv.getElementType())
          return cudaq::cc::StdvecType::get(inner);
        return t;
      }
      if (auto p = dyn_cast<cudaq::cc::PointerType>(t)) {
        auto inner = convert(p.getElementType());
        if (inner != p.getElementType())
          return cudaq::cc::PointerType::get(inner);
        return t;
      }
      if (auto a = dyn_cast<cudaq::cc::ArrayType>(t)) {
        auto inner = convert(a.getElementType());
        if (inner != a.getElementType())
          return a.isUnknownSize() ? cudaq::cc::ArrayType::get(inner)
                                   : cudaq::cc::ArrayType::get(
                                         &getContext(), inner, a.getSize());
        return t;
      }
      if (auto fn = dyn_cast<FunctionType>(t)) {
        SmallVector<Type> ins;
        ins.reserve(fn.getNumInputs());
        SmallVector<Type> outs;
        outs.reserve(fn.getNumResults());
        bool changed = false;
        for (auto in : fn.getInputs()) {
          auto c = convert(in);
          ins.push_back(c);
          if (c != in)
            changed = true;
        }
        for (auto out : fn.getResults()) {
          auto c = convert(out);
          outs.push_back(c);
          if (c != out)
            changed = true;
        }
        if (changed)
          return FunctionType::get(&getContext(), ins, outs);
        return t;
      }
      // `cc.callable` and `cc.indirect_callable` wrap a `FunctionType` whose
      // inputs or results may name `!cc.measure_handle`. Without recursing
      // into the carried signature, the callable's declared type would
      // diverge from its post-rewrite call-site result type and
      // `cc.call_indirect_callable`'s verifier would reject the module.
      // CUDA-QX's `memory_circuit.cpp` round-trips a `qkernel` typed
      // `... -> std::vector<measure_handle>`, which is the load-bearing case.
      if (auto c = dyn_cast<cudaq::cc::CallableType>(t)) {
        auto inner = convert(c.getSignature());
        if (inner != c.getSignature())
          return cudaq::cc::CallableType::get(cast<FunctionType>(inner));
        return t;
      }
      if (auto ic = dyn_cast<cudaq::cc::IndirectCallableType>(t)) {
        auto inner = convert(ic.getSignature());
        if (inner != ic.getSignature())
          return cudaq::cc::IndirectCallableType::get(
              cast<FunctionType>(inner));
        return t;
      }
      return t;
    };

    // Rewrite SSA result types and block-argument types in place.
    module.walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        auto newTy = convert(result.getType());
        if (newTy != result.getType())
          result.setType(newTy);
      }
      for (auto &region : op->getRegions())
        for (auto &block : region.getBlocks())
          for (auto arg : block.getArguments()) {
            auto newTy = convert(arg.getType());
            if (newTy != arg.getType())
              arg.setType(newTy);
          }
    });

    // Rewrite type-carrying attributes that aren't reachable through SSA
    // values. `cc.alloca`'s `elementType` is the only one that is realistic
    // for handle code; `cc.sizeof`/`cc.offsetof`/`cc.global` only matter if a
    // user introspects a handle's layout, which the spec does not endorse.
    module.walk([&](cudaq::cc::AllocaOp alloca) {
      auto eltTy = alloca.getElementType();
      auto newEltTy = convert(eltTy);
      if (newEltTy != eltTy)
        alloca.setElementTypeAttr(TypeAttr::get(newEltTy));
    });

    // Function signatures need to be updated separately; `setType` on the
    // entry block arguments above only fixes the block side, not the
    // FunctionType attribute the verifier will compare against.
    module.walk([&](func::FuncOp fn) {
      auto fnTy = fn.getFunctionType();
      auto newTy = convert(fnTy);
      if (newTy != fnTy)
        fn.setType(cast<FunctionType>(newTy));
    });
  }
};

} // namespace
