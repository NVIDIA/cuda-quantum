/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "LoopAnalysis.h"

#define DEBUG_TYPE "resource-count-preprocess"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_RESOURCECOUNTPREPROCESS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

//   void countOpsSmarter(Operation *op, size_t to_add = 1) {
//     count(op, to_add);

//     if (auto loop = dyn_cast<cudaq::cc::LoopOp>(op)) {
//         cudaq::opt::LoopComponents comp;
//         if (cudaq::opt::isaInvariantLoop(loop, true, true, &comp)) {
//             auto iterations = comp.getIterationsConstant().value();

//             for (auto &b : loop.getBodyRegion().getBlocks())
//                 for (auto &op : b.getOperations())
//                     countOpsSmarter(&op, to_add * iterations);
//         } else {
//             //llvm::outs() << "???\n";
//         }
//     } else if (auto ifop = dyn_cast<cudaq::cc::IfOp>(op)) {
//         auto cond = ifop.getCondition();
//         cond.dump();
//         auto defop = cond.getDefiningOp();
//         if (auto cop = dyn_cast<mlir::arith::ConstantOp>(defop)) {
//             if (auto value = dyn_cast<BoolAttr>(cop.getValue())) {
//                 auto &region = value ? ifop.getThenRegion() : ifop.getElseRegion();
//                 for (auto &b : region.getBlocks())
//                     for (auto &op : b.getOperations())
//                         countOpsSmarter(&op, to_add);
//             }
//         } else {
//             //llvm::outs() << "Condition is not constant, not sure what to do here.\n";
//         }
//     } else {
//         for (auto &r : op->getRegions())
//             for (auto &b : r.getBlocks())
//                 for (auto &op : b.getOperations())
//                     countOpsSmarter(&op, to_add);
//     }
//   }

//   void countOpsSmarter() {
//     auto mod = getOperation();
//     mod.dump();

//     for (auto &op : *mod.getBody())
//         countOpsSmarter(&op);

//     // hcount = counts["h"];
//     // rzcount = counts["rz"];
//     // xcount = counts["x"];
//     // cxcount = counts["cx"];
//     // ccxcount = counts["ccx"];
//   }

// //   void countOpsDumb() {
// //     auto mod = getOperation();

// //     mod.walk([&](quake::HOp hop) { hcount++; });
// //     mod.walk([&](quake::RzOp rzop) { rzcount++; });
// //     mod.walk([&](quake::XOp xop) {
// //         switch (xop.getControls().size()) {
// //         case 0:
// //             xcount++;
// //             break;
// //         case 1:
// //             cxcount++;
// //             break;
// //         case 2:
// //             ccxcount++;
// //             break;
// //         default:
// //             break;
// //         }
// //     });
// //   }

//   void runOnOperation() override {
//     // countOpsDumb()    
//   }
// };

struct ResourceCountPreprocessPass
    : public cudaq::opt::impl::ResourceCountPreprocessBase<ResourceCountPreprocessPass> {
  using ResourceCountPreprocessBase::ResourceCountPreprocessBase;
    SetVector<Operation *> to_erase;

    bool preCount(Operation *op, size_t to_add) {
        if (!isQuakeOperation(op))
            return false;

        auto opi = dyn_cast<quake::OperatorInterface>(op);

        if (!opi)
            return false;

        // Measures may affect control flow, don't remove for now
        if (isa<quake::MzOp>(op) || isa<quake::MyOp>(op) || isa<quake::MxOp>(op))
            return false;

        auto name = op->getName().stripDialect();

        size_t controls = opi.getControls().size();;
        
        std::string gatestr(controls, 'c');
        gatestr += name;

        // llvm::outs() << "Precounting for " << to_add << " times: ";
        // op->dump();

        countGate(gatestr, to_add);
        to_erase.insert(op);
        return true;
    }

    void preprocessOp(Operation *op, size_t to_add = 1) {
        // llvm::outs() << "Processing ";
        // op->dump();

        if (preCount(op, to_add))
            return;

        if (auto loop = dyn_cast<cudaq::cc::LoopOp>(op)) {
            cudaq::opt::LoopComponents comp;
            if (cudaq::opt::isaInvariantLoop(loop, true, false, &comp)) {
                auto iterations = comp.getIterationsConstant().value();

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
        // func.dump();

        for (auto &b : func.getBody())
            for (auto &op : b.getOperations())
                preprocessOp(&op);
        
        for (auto op : to_erase) {
            // llvm::outs() << "Erasing ";
            // op->dump();
            op->erase();
        }
    }
};