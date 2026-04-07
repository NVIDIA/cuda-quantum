/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "resource-count-preprocess"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_RESOURCECOUNTPREPROCESS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

struct ResourceCountPreprocessPass
    : public cudaq::opt::impl::ResourceCountPreprocessBase<
          ResourceCountPreprocessPass> {
  using ResourceCountPreprocessBase::ResourceCountPreprocessBase;
  SetVector<Operation *> to_erase;
  DenseMap<Value, std::size_t> qubitIndexMap;
  std::size_t nextQubitIndex = 0;

  /// Resolve a quake value to a concrete qubit index.
  std::optional<std::size_t> resolveQubitIndex(Value v) {
    if (auto extractRef = v.getDefiningOp<quake::ExtractRefOp>())
      if (extractRef.hasConstantIndex())
        return extractRef.getConstantIndex();
    if (auto borrow = v.getDefiningOp<quake::BorrowWireOp>())
      return static_cast<std::size_t>(borrow.getIdentity());
    // Single-qubit alloca: assign a unique index by declaration order.
    if (v.getDefiningOp<quake::AllocaOp>() &&
        isa<quake::RefType>(v.getType())) {
      auto it = qubitIndexMap.find(v);
      if (it != qubitIndexMap.end())
        return it->second;
      auto idx = nextQubitIndex++;
      qubitIndexMap[v] = idx;
      return idx;
    }
    return std::nullopt;
  }

  bool preCount(Operation *op, size_t to_add) {
    if (!isQuakeOperation(op))
      return false;

    auto opi = dyn_cast<quake::OperatorInterface>(op);

    if (!opi)
      return false;

    // Measures may affect control flow, don't remove for now
    if (isa<quake::MeasurementInterface>(op))
      return false;

    auto name = op->getName().stripDialect();

    std::vector<std::size_t> controlIndices, targetIndices;
    for (auto ctrl : opi.getControls())
      if (auto idx = resolveQubitIndex(ctrl))
        controlIndices.push_back(*idx);
    for (auto tgt : opi.getTargets())
      if (auto idx = resolveQubitIndex(tgt))
        targetIndices.push_back(*idx);

    if (dumpPreprocessed)
      llvm::outs() << "Preprocessing " << name << "(" << controlIndices.size()
                   << ")"
                   << " for " << to_add << " counts\n";

    countGate(name.str(), controlIndices, targetIndices, to_add);
    to_erase.insert(op);
    return true;
  }

  void preprocessOp(Operation *op, size_t to_add = 1) {
    if (preCount(op, to_add))
      return;

    if (auto loop = dyn_cast<cudaq::cc::LoopOp>(op)) {
      cudaq::opt::LoopComponents comp;
      if (cudaq::opt::isaInvariantLoop(loop, true, false, &comp)) {
        auto loopSize = comp.getIterationsConstant();
        if (!loopSize.has_value())
          return;
        auto iterations = loopSize.value();
        for (auto &b : loop.getBodyRegion().getBlocks())
          for (auto &op : b.getOperations())
            preprocessOp(&op, to_add * iterations);
      }
    } else if (auto ifop = dyn_cast<cudaq::cc::IfOp>(op)) {
      auto cond = ifop.getCondition();
      auto defop = cond.getDefiningOp();
      if (auto cop = dyn_cast<mlir::arith::ConstantOp>(defop)) {
        if (auto value = dyn_cast<BoolAttr>(cop.getValue())) {
          auto &region = value ? ifop.getThenRegion() : ifop.getElseRegion();
          for (auto &b : region.getBlocks())
            for (auto &op : b.getOperations())
              preprocessOp(&op, to_add);
        }
      }
    }
  }

  void runOnOperation() override {
    auto func = getOperation();

    for (auto &b : func.getBody()) {
      // We only pre-process the main block as the other blocks may be
      // conditional when the IR is lowered to CFG.
      if (&b != &func.getBody().front())
        continue;
      for (auto &op : b.getOperations())
        preprocessOp(&op);
    }
    for (auto op : to_erase)
      op->erase();

    to_erase.clear();
    qubitIndexMap.clear();
    nextQubitIndex = 0;
  }
};
