/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include <unordered_set>

namespace cudaq::opt {
#define GEN_PASS_DEF_REPAIRLINEARTYPE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "repair-linear-type"

using namespace mlir;

namespace {
class RepairLinearTypePass
    : public cudaq::opt::impl::RepairLinearTypeBase<RepairLinearTypePass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!funcOp || funcOp.empty())
      return;

    auto *ctx = &getContext();
    funcOp.walk([&](Operation *op) {
      if (!hasLinearTypeResult(op))
        return;
      for (auto result : op->getOpResults()) {
        // Perform the same checks as the verifier (from QuakeOps.cpp).
        if (!cudaq::quake::isLinearType(result.getType()))
          continue;
        if (result.hasOneUse() || result.use_empty())
          continue;

        // Check if this is a terminator like cf.cond_br, which is allowed to be
        // a σ (linear typed value split node)
        std::unordered_set<Operation *> uniqs;
        for (auto *u : result.getUsers())
          uniqs.insert(u);
        if (uniqs.size() == 1 &&
            (*uniqs.begin())->hasTrait<OpTrait::IsTerminator>())
          continue;

        LLVM_DEBUG(
            llvm::dbgs()
            << "Operation needs to have the linear type property restored.\n"
            << *op << '\n');
        // At this point, we know `result` has a linear type and is used more
        // than once. We want to repair the linear type semantics by using an
        // intermediate ref type value, which is properly SSA. The reason for
        // this precise correction is so that the IR isn't dribbled back and
        // forth between valid and invalid linear forms by the MLIR
        // infrastructure.
        //
        //   %34 = quake.foo %33 : (!quake.wire) -> !quake.wire
        //   ...
        //  ^bb8:
        //   %54 = quake.bar %34 : (!quake.wire) -> !quake.wire
        //   ...
        //  ^bb14:
        //   %86 = quake.baz %34 : (!quake.wire) -> !quake.wire
        //   ──────────────────────────────────────────────────
        //   %34 = quake.foo %33 : (!quake.wire) -> !quake.wire
        //   %35 = quake.wrap_new %34 : (!quake.wire) -> !quake.ref
        //   ...
        //  ^bb8:
        //   %53 = quake.unwrap %35 : (!quake.ref) -> !quake.wire
        //   %54 = quake.bar %53 : (!quake.wire) -> !quake.wire
        //   ...
        //  ^bb14:
        //   %85 = quake.unwrap %35 : (!quake.ref) -> !quake.wire
        //   %86 = quake.baz %85 : (!quake.wire) -> !quake.wire

        SmallVector<Operation *> origUsers{result.getUsers().begin(),
                                           result.getUsers().end()};
        OpBuilder builder(ctx);
        builder.setInsertionPointAfter(op);
        Type resTy = convertToRefTy(result.getType());
        auto loc = op->getLoc();
        // Create a new ref.
        auto wrap =
            cudaq::quake::WrapNewOp::create(builder, loc, resTy, result);

        // Update all the original users with an unwrap of the new ref.
        auto wireTy = cudaq::quake::WireType::get(ctx);
        for (auto *user : origUsers) {
          builder.setInsertionPoint(user);
          auto unwrap =
              cudaq::quake::UnwrapOp::create(builder, loc, wireTy, wrap);
          result.replaceUsesWithIf(
              unwrap, [&](OpOperand &use) { return use.getOwner() == user; });
        }

        if (!result.hasOneUse()) {
          op->emitOpError("Failed to be restored to a linear-type");
          signalPassFailure();
        }
      }
    });
  }

  static Type convertToRefTy(Type ty) {
    auto *ctx = ty.getContext();
    if (ty == cudaq::quake::WireType::get(ctx))
      return cudaq::quake::RefType::get(ctx);
    auto cableTy = cast<cudaq::quake::CableType>(ty);
    return cudaq::quake::VeqType::get(ctx, cableTy.getSize());
  }

  static bool hasLinearTypeResult(Operation *op) {
    for (auto result : op->getOpResults())
      if (cudaq::quake::isLinearType(result.getType()))
        return true;
    return false;
  }
};
} // namespace
