/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_QIRINSERTARRAYRECORD
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "qir-insert-array-record"

using namespace mlir;

namespace {

// Trace a pointer to back to its corresponding `AllocaOp`
static cudaq::cc::AllocaOp tracePointerToAlloca(Value ptr) {
  llvm::DenseSet<Value> visited;
  while (ptr) {
    if (!visited.insert(ptr).second)
      return {};
    Operation *defOp = ptr.getDefiningOp();
    if (!defOp)
      return {};
    if (auto allocaOp = dyn_cast<cudaq::cc::AllocaOp>(defOp))
      return allocaOp;
    if (auto castOp = dyn_cast<cudaq::cc::CastOp>(defOp)) {
      ptr = castOp.getValue();
      continue;
    }
    if (auto computePtrOp = dyn_cast<cudaq::cc::ComputePtrOp>(defOp)) {
      ptr = computePtrOp.getBase();
      continue;
    }
    return {};
  }
  return {};
}

// Walk a function to identify all the measure-discriminate-store patterns and
// collect the associated `AllocaOp` when the measurement results are stored.
// Collect only unique AllocaOps - since each may correspond to multiple
// measurement operations. When there are no explicit stores, track the first
// measurement operation and the get the total number of measurements.
struct AllocaMeasureStoreAnalysis {
  AllocaMeasureStoreAnalysis() = default;

  explicit AllocaMeasureStoreAnalysis(func::FuncOp funcOp) {
    size_t totalMeasurementCount = 0;
    Operation *firstMeasureOp = nullptr;
    DenseMap<Value, Operation *> valueToMeasurement;
    llvm::SetVector<cudaq::cc::AllocaOp> uniqueAllocaOps;

    // First pass: identify measurements and propagate through uses
    funcOp.walk([&](Operation *op) {
      if (op->hasTrait<cudaq::QuantumMeasure>()) {
        if (op->hasAttr(cudaq::opt::ResultIndexAttrName)) {
          totalMeasurementCount++;
          if (!firstMeasureOp)
            firstMeasureOp = op;
        }
        for (auto result : op->getResults())
          valueToMeasurement[result] = op;
        return WalkResult::advance();
      }

      // TODO: Check if more operations need to be added here.
      if (!isa<quake::DiscriminateOp, cudaq::cc::CastOp>(op)) {
        return WalkResult::advance();
      }

      // Find the operands derived from measurements
      for (auto operand : op->getOperands()) {
        if (valueToMeasurement.count(operand)) {
          for (auto result : op->getResults())
            valueToMeasurement[result] = valueToMeasurement[operand];
        }
        break; // Checking one operand is enough
      }
      return WalkResult::advance();
    });

    // Second pass: find stores of measurement values and trace to `alloca` ops
    funcOp.walk([&](cudaq::cc::StoreOp storeOp) {
      if (valueToMeasurement.count(storeOp.getValue())) {
        Value ptr = storeOp.getPtrvalue();
        auto allocaOp = tracePointerToAlloca(ptr);
        if (allocaOp)
          uniqueAllocaOps.insert(allocaOp);
      }
    });

    if (!uniqueAllocaOps.empty()) {
      // Use array sizes when explicit storage exists
      for (auto allocaOp : uniqueAllocaOps) {
        if (auto arrType =
                allocaOp.getElementType().dyn_cast<cudaq::cc::ArrayType>()) {
          arraySize += arrType.getSize();
        } else {
          arraySize += 1;
        }
      }
      allocaOps.append(uniqueAllocaOps.begin(), uniqueAllocaOps.end());
    } else if (totalMeasurementCount > 0) {
      // This could be individual qubit(s)
      arraySize = totalMeasurementCount;
      firstMeasurementOp = firstMeasureOp;
    }
  }

  SmallVector<cudaq::cc::AllocaOp> allocaOps;
  size_t arraySize = 0;
  Operation *firstMeasurementOp = nullptr;
};

// Inserts a QIR array record output call to declare measurement result storage.
// QIR requires `__quantum__rt__array_record_output()` be called before multiple
// measurements to declare the output array size and type label. This is
// required in `sample` API since it always returns a vector of measurement
// results. Following logic is used to determine the insertion point:
//   1. After first alloca (if explicit array storage exists)
//   2. Before first measurement (if no explicit storage)
// The label string is created as "array<i8 x N>" where N is the total number of
// measurement results. The array record output call is created as:
// `__quantum__rt__array_record_output(N, label);`
LogicalResult
insertArrayRecordingCalls(func::FuncOp funcOp, size_t resultCount,
                          const SmallVector<cudaq::cc::AllocaOp> &allocaOps,
                          Operation *firstMeasureOp) {
  if (resultCount == 0)
    return success();

  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);
  mlir::Location loc = funcOp.getLoc();
  // We insert only one array record call
  if (!allocaOps.empty())
    builder.setInsertionPointAfter(allocaOps[0]);
  else if (firstMeasureOp)
    builder.setInsertionPoint(firstMeasureOp);
  else
    return failure();

  // Create the label string: "array<i8 x N>"
  std::string labelStr = "array<i8 x " + std::to_string(resultCount) + ">";
  auto strLitTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(
      builder.getContext(), builder.getI8Type(), labelStr.size() + 1));
  Value lit = builder.create<cudaq::cc::CreateStringLiteralOp>(
      loc, strLitTy, builder.getStringAttr(labelStr));
  auto i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());
  Value label = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, lit);
  Value size = builder.create<arith::ConstantIntOp>(loc, resultCount, 64);
  builder.create<func::CallOp>(loc, TypeRange{},
                               cudaq::opt::QIRArrayRecordOutput,
                               ArrayRef<Value>{size, label});

  // Add the declaration to the module if it doesn't already exist
  auto module = funcOp->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol(cudaq::opt::QIRArrayRecordOutput)) {
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::opt::QIRArrayRecordOutput))) {
      return failure();
    }
  }
  return success();
}

struct QirInsertArrayRecordPass
    : public cudaq::opt::impl::QirInsertArrayRecordBase<
          QirInsertArrayRecordPass> {

  using QirInsertArrayRecordBase::QirInsertArrayRecordBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp || funcOp.empty() ||
          !funcOp->hasAttr(cudaq::entryPointAttrName) ||
          funcOp->hasAttr(cudaq::runtime::enableCudaqRun))
        return WalkResult::advance();

      AllocaMeasureStoreAnalysis analysis(funcOp);
      if (analysis.arraySize == 0)
        return WalkResult::advance();

      LLVM_DEBUG(llvm::dbgs() << "Before adding array recording call:\n"
                              << *funcOp);
      if (failed(insertArrayRecordingCalls(funcOp, analysis.arraySize,
                                           analysis.allocaOps,
                                           analysis.firstMeasurementOp)))
        return WalkResult::interrupt();
      LLVM_DEBUG(llvm::dbgs() << "After adding array recording call:\n"
                              << *funcOp);

      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createQirInsertArrayRecord() {
  return std::make_unique<QirInsertArrayRecordPass>();
}
