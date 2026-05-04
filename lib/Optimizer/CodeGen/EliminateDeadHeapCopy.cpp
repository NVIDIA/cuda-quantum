/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ELIMINATEDEADHEAPCOPY
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "eliminate-dead-heap-copy"

using namespace mlir;

namespace {

/// When a kernel returns a vector, the frontend copies the stack data to the
/// heap via malloc+memcpy (from __nvqpp_vectorCopyCtor) so the data outlives
/// the callee's stack frame. After inlining and ReturnToOutputLog, the output
/// logging reads from the heap buffer through cc.cast ops, and the
/// cc.stdvec_init that wrapped the malloc becomes dead. This pass redirects
/// those cc.cast reads to the memcpy source (the original stack buffer) and
/// erases the now-dead malloc, memcpy, and cc.stdvec_init.
struct EliminateDeadHeapCopyPass
    : public cudaq::opt::impl::EliminateDeadHeapCopyBase<
          EliminateDeadHeapCopyPass> {
  using EliminateDeadHeapCopyBase::EliminateDeadHeapCopyBase;

  void runOnOperation() override {
    auto func = getOperation();
    SmallVector<func::CallOp> mallocCalls;
    func.walk([&](func::CallOp callOp) {
      if (callOp.getCallee() == "malloc")
        mallocCalls.push_back(callOp);
    });

    for (auto mallocCall : mallocCalls) {
      // malloc should return exactly one result (the allocated pointer).
      if (mallocCall->getNumResults() != 1)
        continue;
      Value mallocResult = mallocCall.getResult(0);

      // Classify users of the malloc result.
      func::CallOp memcpyCall;
      SmallVector<cudaq::cc::StdvecInitOp> deadVecInits;
      SmallVector<cudaq::cc::CastOp> castUsers;
      bool hasUnsafeUser = false;

      for (auto *user : mallocResult.getUsers()) {
        if (auto userCall = dyn_cast<func::CallOp>(user)) {
          if (userCall.getCallee().starts_with("llvm.memcpy") &&
              userCall.getOperand(0) == mallocResult) {
            if (memcpyCall) {
              // Multiple memcpys to the same malloc dest — bail out.
              hasUnsafeUser = true;
              break;
            }
            memcpyCall = userCall;
            continue;
          }
        }
        // A dead stdvec_init (no remaining users) can be safely erased.
        // One with live users is treated as unsafe.
        if (auto vecInit = dyn_cast<cudaq::cc::StdvecInitOp>(user)) {
          if (vecInit->use_empty()) {
            deadVecInits.push_back(vecInit);
            continue;
          }
        }
        // A cc.cast is safe to redirect: since the memcpy copies from
        // source to the malloc buffer, reading through either pointer
        // yields the same data.
        if (auto castOp = dyn_cast<cudaq::cc::CastOp>(user)) {
          castUsers.push_back(castOp);
          continue;
        }
        // Any other user prevents elimination.
        hasUnsafeUser = true;
        break;
      }

      if (!memcpyCall || hasUnsafeUser)
        continue;

      Value memcpySrc = memcpyCall.getOperand(1);

      // Redirect cc.cast users from the malloc result to the memcpy source.
      for (auto castOp : castUsers)
        castOp->replaceUsesOfWith(mallocResult, memcpySrc);

      // Erase dead stdvec_inits.
      for (auto vecInit : deadVecInits)
        vecInit->erase();

      // Erase memcpy and malloc.
      memcpyCall->erase();
      mallocCall->erase();
    }
  }
};

} // namespace
