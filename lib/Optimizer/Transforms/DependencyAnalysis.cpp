/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_DEPENDENCYANALYSIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
class DependencyNode {
protected:
  SmallVector<DependencyNode *> successors;
  SmallVector<DependencyNode *> dependencies;
  // TODO: should probably be a set of qids here
  size_t qindex;
  Operation *associated;

  // Print with tab index to should depth in graph
  void printSub(int tabIndex) {
    for (int i = 0; i < tabIndex; i++) {
      llvm::outs() << "\t";
    }
    llvm::outs() << "qid: " << qindex << " -- ";
    if (isJoin())
      llvm::outs() << "join\n";
    else if (associated != nullptr)
      associated->dump();

    for (auto dependency : dependencies) {
      dependency->printSub(tabIndex + 1);
    }
  }

public:
  DependencyNode(size_t index, Operation *op, DependencyNode *from)
      : successors(), dependencies(), qindex(index), associated(op) {
    if (from) {
      addSuccessor(from);
    }
  };

  DependencyNode(DependencyNode *from)
      : successors({from}), dependencies(), qindex(from->qindex),
        associated(nullptr) {
    addSuccessor(from);
  };

  inline void addSuccessor(DependencyNode *other) {
    successors.push_back(other);
    other->dependencies.push_back(this);
  }

  inline bool isJoin() {
    // TODO: Is second part ever not true?
    return associated == nullptr && dependencies.size() > 0;
  }

  void print() { printSub(0); }
};

inline bool isMeasureOp(Operation *op) {
  return dyn_cast<quake::MxOp>(*op) || dyn_cast<quake::MyOp>(*op) ||
         dyn_cast<quake::MzOp>(*op);
}

inline bool isBeginOp(Operation *op) {
  return dyn_cast<quake::UnwrapOp>(*op) || dyn_cast<quake::ExtractRefOp>(*op) ||
         dyn_cast<quake::NullWireOp>(*op);
}

class DependencyAnalysis {
private:
  SmallVector<DependencyNode *> perOp;
  DenseMap<BlockArgument, DependencyNode *> map;

  inline DependencyNode *getDNodeId(Operation *op, int res_index) {
    if (op->hasAttr("dnodeids")) {
      auto ids = op->getAttr("dnodeids").cast<DenseI32ArrayAttr>();
      if (ids[res_index] != -1) {
        auto dnode = perOp[ids[res_index]];
        return dnode;
      }
    }

    return nullptr;
  }

public:
  DependencyAnalysis() : perOp(), map(){};

  DependencyNode *handleDependencyOp(Operation *op, DependencyNode *next,
                                     int res_index) {
    // Reached end of graph (beginning of circuit)
    if (isBeginOp(op))
      return nullptr;

    // If we've already visited this operation, return memoized dnode
    auto dnodeid = getDNodeId(op, res_index);
    if (dnodeid) {
      dnodeid->addSuccessor(next);
      return dnodeid;
    }

    // Lookup qid for result
    auto qid = op->getAttrOfType<DenseI32ArrayAttr>("qids")[res_index];

    // Construct new dnode
    DependencyNode *newNode = new DependencyNode(qid, op, next);

    // TODO: Only one dnodeid per op with qid sets?
    // Dnodeid for the relevant result is the next slot of the dnode vector
    SmallVector<int32_t> ids(op->getNumResults(), -1);
    ids[res_index] = perOp.size();

    // Add dnodeid attribute
    OpBuilder builder(op);
    op->setAttr("dnodeids",
                builder.getDenseI32ArrayAttr({ids.begin(), ids.end()}));
    perOp.push_back(newNode);

    // Recursively visit children
    for (auto operand : op->getOperands()) {
      handleDependencyValue(operand, newNode);
    }

    return newNode;
  }

  DependencyNode *handleDependencyArg(BlockArgument arg, DependencyNode *next) {
    // If we've already handled this block argument, return memoized value
    if (auto prev = map.lookup(arg)) {
      prev->addSuccessor(next);
      return prev;
    }

    auto block = arg.getParentBlock();
    DependencyNode *newNode = next;
    // TODO: better way to check for multiple predecessors?
    // TODO: get single or get unique?
    // If join point, insert join node
    if (!block->getSinglePredecessor()) {
      newNode = new DependencyNode(next);
      map.insert({arg, newNode});
    }

    // Look up operands from all branch instructions that can jump
    // to the parent block and recursively visit them
    for (auto predecessor : block->getPredecessors()) {
      if (auto branch =
              dyn_cast<BranchOpInterface>(predecessor->getTerminator())) {
        unsigned numSuccs = branch->getNumSuccessors();
        for (unsigned i = 0; i < numSuccs; ++i) {
          if (block && branch->getSuccessor(i) != block)
            continue;
          auto brArgs = branch.getSuccessorOperands(i).getForwardedOperands();
          auto operand = brArgs[arg.getArgNumber()];
          handleDependencyValue(operand, newNode);
        }
      }
    }

    return newNode;
  }

  DependencyNode *handleDependencyValue(Value v, DependencyNode *next) {
    // Block arguments do not have associated operations,
    // but may require inserting joins, so they are handled specially
    if (auto arg = dyn_cast<BlockArgument>(v))
      return handleDependencyArg(arg, next);

    auto defOp = v.getDefiningOp();
    if (defOp) {
      // Find the value amongst those returned by the operation
      int i = 0;
      for (auto res : defOp->getResults()) {
        if (res == v)
          break;
        i++;
      }

      return handleDependencyOp(defOp, next, i);
    }

    // TODO: FAIL
    llvm::outs() << "UNKNOWN VALUE\n";
    v.dump();
    return nullptr;
  }
};

struct DependencyAnalysisPass
    : public cudaq::opt::impl::DependencyAnalysisBase<DependencyAnalysisPass> {
  using DependencyAnalysisBase::DependencyAnalysisBase;

  void runOnOperation() override {
    auto func = getOperation();

    DependencyAnalysis engine;

    func.walk([&](quake::MzOp mop) {
      // TODO: Making assumption wire is second measure result
      auto graph = engine.handleDependencyValue(mop.getResult(1), nullptr);
      if (graph)
        graph->print();
    });
  }
};

} // namespace
