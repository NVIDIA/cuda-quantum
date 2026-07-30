/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/GreedyOpPartitioner.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include <list>

using namespace mlir;

static bool isQubitValue(Value v) {
  return isa<cudaq::quake::WireType, cudaq::quake::RefType>(v.getType());
}

cudaq::opt::GreedyOpPartitioner::GreedyOpPartitioner(Operation *op,
                                                     unsigned maxQubits) {
  auto func = dyn_cast<func::FuncOp>(op);
  if (!func)
    return;
  for (Block &block : func.getBody())
    partitionBlock(block, maxQubits);
}

void cudaq::opt::GreedyOpPartitioner::partitionBlock(Block &block,
                                                     unsigned maxQubits) {
  // An open partition: the set of ops collected so far, the currently live
  // output wire values for each qubit in the partition, and the qubit count.
  struct Part {
    SmallVector<Operation *> ops;
    DenseSet<Value> liveWires;
    unsigned qubitCount = 0;
  };

  // std::list gives stable addresses, which wireOwner relies on.
  std::list<Part> open;
  // Maps each currently-live wire value to the open partition that owns it.
  DenseMap<Value, Part *> wireOwner;

  // Saves a partition to results and removes it from the open list.
  auto flush = [&](Part *P) {
    for (Value w : P->liveWires)
      wireOwner.erase(w);
    if (!P->ops.empty())
      partitions.push_back(std::move(P->ops));
    open.erase(std::find_if(open.begin(), open.end(),
                            [P](const Part &x) { return &x == P; }));
  };

  // Adds op to partition P, updating qubit tracking for both semantics:
  //   wire operands  — consumed; erase from wireOwner so the value is dead
  //   ref  operands  — persistent; register on first encounter, never erase
  //   qubit results  — new timeline; insert into wireOwner (wire sources and
  //                    ref AllocaOps both produce qubit results)
  auto addOp = [&](Part &P, Operation *op, unsigned newQubits) {
    for (Value w : op->getOperands()) {
      if (isa<cudaq::quake::WireType>(w.getType())) {
        P.liveWires.erase(w);
        wireOwner.erase(w);
      } else if (isa<cudaq::quake::RefType>(w.getType())) {
        if (!wireOwner.count(w)) {
          P.liveWires.insert(w);
          wireOwner[w] = &P;
        }
      }
    }
    for (Value w : op->getResults())
      if (isQubitValue(w)) {
        P.liveWires.insert(w);
        wireOwner[w] = &P;
      }
    P.qubitCount += newQubits;
    P.ops.push_back(op);
  };

  auto qubitInputCount = [](Operation *op) {
    unsigned n = 0;
    for (Value w : op->getOperands())
      if (isQubitValue(w))
        ++n;
    return n;
  };

  for (Operation &op : block) {
    bool hasQubits = llvm::any_of(op.getOperands(), isQubitValue) ||
                     llvm::any_of(op.getResults(), isQubitValue);
    if (!hasQubits)
      continue;

    // Reject ops that require more qubit timelines than the partition limit.
    if (qubitInputCount(&op) > maxQubits)
      continue;

    // Source ops introduce new qubit timelines with no qubit inputs:
    // null_wire/borrow_wire for wire semantics, AllocaOp(!quake.ref) for ref.
    bool isSource =
        isa<quake::NullWireOp, quake::BorrowWireOp>(op) ||
        (isa<quake::AllocaOp>(op) && llvm::any_of(op.getResults(), [](Value v) {
           return isa<cudaq::quake::RefType>(v.getType());
         }));

    SmallPtrSet<Part *, 4> touched;
    unsigned extQubits =
        0; // qubit timelines entering from outside any open partition
    if (isSource) {
      for (Value w : op.getResults())
        if (isQubitValue(w))
          ++extQubits;
    } else {
      for (Value w : op.getOperands()) {
        if (!isQubitValue(w))
          continue;
        auto it = wireOwner.find(w);
        if (it != wireOwner.end())
          touched.insert(it->second);
        else
          ++extQubits; // wire from a closed partition or uninitialized source
      }
    }

    if (touched.empty()) {
      // Op is disjoint from all open partitions. Reuse an existing one with
      // room, or open a fresh one.
      Part *target = nullptr;
      for (Part &P : open)
        if (P.qubitCount + extQubits <= maxQubits) {
          target = &P;
          break;
        }
      if (!target) {
        open.emplace_back();
        target = &open.back();
      }
      addOp(*target, &op, extQubits);

    } else if (touched.size() == 1) {
      Part *P = *touched.begin();
      if (P->qubitCount + extQubits > maxQubits) {
        // This partition is full; commit it and restart for this op alone.
        flush(P);
        open.emplace_back();
        // After flush, all of op's wire inputs are external to the new part.
        addOp(open.back(), &op, qubitInputCount(&op));
      } else {
        addOp(*P, &op, extQubits);
      }

    } else {
      // Op bridges multiple open partitions.
      unsigned mergedCount = extQubits;
      for (Part *P : touched)
        mergedCount += P->qubitCount;

      if (mergedCount <= maxQubits) {
        // Merge all touched partitions into one.
        Part *target = *touched.begin();
        for (Part *other : llvm::drop_begin(touched)) {
          for (Operation *o : other->ops)
            target->ops.push_back(o);
          for (Value w : other->liveWires) {
            target->liveWires.insert(w);
            wireOwner[w] = target;
          }
          target->qubitCount += other->qubitCount;
          open.erase(
              std::find_if(open.begin(), open.end(),
                           [other](const Part &x) { return &x == other; }));
        }
        addOp(*target, &op, extQubits);
      } else {
        // Close only the partitions causing the overflow — largest first —
        // leaving survivors to absorb this op.
        //
        // Track how many of this op's wire inputs each touched partition owns,
        // so we know how many extra external qubits closing it introduces.
        DenseMap<Part *, unsigned> ownedInputs;
        for (Value w : op.getOperands()) {
          if (!isWire(w))
            continue;
          auto it = wireOwner.find(w);
          if (it != wireOwner.end() && touched.count(it->second))
            ownedInputs[it->second]++;
        }

        SmallVector<Part *> sortedTouched(touched.begin(), touched.end());
        llvm::sort(sortedTouched, [](Part *a, Part *b) {
          return a->qubitCount > b->qubitCount;
        });

        // mergedCount tracks survivors + current extQubits. Close partitions
        // until it fits, updating extQubits as each partition's owned inputs
        // become external.
        unsigned mergedCount2 = mergedCount;
        SmallVector<Part *> toFlush;
        for (Part *P : sortedTouched) {
          if (mergedCount2 <= maxQubits)
            break;
          unsigned owned = ownedInputs.lookup(P);
          mergedCount2 = mergedCount2 - P->qubitCount + owned;
          extQubits += owned;
          toFlush.push_back(P);
        }
        for (Part *P : toFlush)
          flush(P);

        // Collect surviving touched partitions and merge them.
        SmallPtrSet<Part *, 4> flushedSet(toFlush.begin(), toFlush.end());
        SmallVector<Part *> survivors;
        for (Part *P : touched)
          if (!flushedSet.count(P))
            survivors.push_back(P);

        if (survivors.empty()) {
          open.emplace_back();
          addOp(open.back(), &op, qubitInputCount(&op));
        } else {
          Part *target = survivors[0];
          for (Part *other : llvm::drop_begin(survivors)) {
            for (Operation *o : other->ops)
              target->ops.push_back(o);
            for (Value w : other->liveWires) {
              target->liveWires.insert(w);
              wireOwner[w] = target;
            }
            target->qubitCount += other->qubitCount;
            open.erase(
                std::find_if(open.begin(), open.end(),
                             [other](const Part &x) { return &x == other; }));
          }
          addOp(*target, &op, extQubits);
        }
      }
    }
  }

  // Commit all remaining open partitions.
  SmallVector<Part *> remaining;
  for (Part &P : open)
    remaining.push_back(&P);
  for (Part *P : remaining)
    flush(P);
}
