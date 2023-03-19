/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXInterfaces.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

/// Not all arrays are splitable
namespace {

/// This analysis checks whether we can split an array, which is not always the
/// case:  For example, an array that is indexed by a runtime parameter cannot
/// be split at compilation time.
///
/// To do this analysis, we define to kinds of arrays:
///
///  * Base: an array created either by an `qtx.alloca` or an
///  `qtx.array_create`.
///
///  * Derived: an array created by operations that return a new array, e.g.,
///  a borrow operation.
///
struct SplitAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitAnalysis)

  SplitAnalysis(Operation *op) {
    auto circuit = cast<qtx::CircuitOp>(op);
    circuit.walk([&](Operation *op) {
      if (auto allocaOp = dyn_cast<qtx::AllocaOp>(op)) {
        Value result = allocaOp.getResult();
        if (!result.getType().isa<qtx::WireArrayType>())
          return;
        toSplit.push_back(result);
        baseArray[result] = result;
      } else if (auto createOp = dyn_cast<qtx::ArrayCreateOp>(op)) {
        Value result = createOp.getResult();
        toSplit.push_back(result);
        baseArray[result] = result;
      } else if (auto splitOp = dyn_cast<qtx::ArraySplitOp>(op)) {
        Value array = splitOp.getArray();
        auto defOp = array.getDefiningOp();
        if (isa<qtx::AllocaOp>(defOp) || isa<qtx::ArrayCreateOp>(defOp))
          toSplit.erase(std::remove(toSplit.begin(), toSplit.end(), array));
      } else if (auto borrowOp = dyn_cast<qtx::ArrayBorrowOp>(op)) {
        Value array = borrowOp.getArray();
        Value new_array = borrowOp.getNewArray();
        for (auto index : borrowOp.getIndices()) {
          // FIXME: This might be too much of a constraint.  What about the
          // result of arithmetic operations that take constants as input?
          auto indexDefOp = index.getDefiningOp<arith::ConstantOp>();
          if (!indexDefOp) {
            partialSplit = true;
            toSplit.erase(
                std::remove(toSplit.begin(), toSplit.end(), baseArray[array]));
          }
        }
        baseArray[new_array] = baseArray[array];
      } else if (auto yieldOp = dyn_cast<qtx::ArrayYieldOp>(op)) {
        Value array = yieldOp.getArray();
        Value new_array = yieldOp.getNewArray();
        baseArray[new_array] = baseArray[array];
      }
    });
  }

  // A flag indicating whether a this will be a partial split
  bool partialSplit = false;
  // List of base arrays that we can split
  SmallVector<Value> toSplit;
  // Maps array values to an base array
  DenseMap<Value, Value> baseArray;
};
} // namespace

// TODO: Handle arrays that are circuit's arguments.
struct SplitArrays : public cudaq::opt::SplitArraysBase<SplitArrays> {

  void runOnOperation() override {
    SplitAnalysis analysis = getAnalysis<SplitAnalysis>();
    if (!allowPartialSplit && analysis.partialSplit) {
      getOperation()->emitError("Cannot split all arrays");
      return signalPassFailure();
    }
    auto circuit = cast<qtx::CircuitOp>(getOperation());
    auto builder = circuit.getBodyBuilder();
    // Starting from a base array in `analysis.toSplit`, we handle it and
    // traverse the chain of derived arrays while maintaining the state of
    // which are the current individual wires correspond to the array,
    // `arrayWires`. Depending on the instruction, there might be a need to
    // update these individual wires.  The chain ends until a sink operations
    // is found, i.e. an operation that takes the array and do not returns a
    // new derived one.
    SmallVector<Operation *> toErase;
    SmallVector<Value> arrayWires;
    SmallVector<Value> newOperands;
    for (Value array : analysis.toSplit) {
      // An array must have only one user, so we take it.
      Operation *userOp = *(array.getUsers().begin());
      // Split the array
      auto defOp = array.getDefiningOp();
      builder.setInsertionPointAfter(defOp);
      auto splitOp = builder.create<qtx::ArraySplitOp>(defOp->getLoc(), array);
      arrayWires.assign(splitOp.getResults().begin(),
                        splitOp.getResults().end());
      // Traverse the chain
      while (userOp) {
        toErase.push_back(userOp);
        if (auto deallocOp = dyn_cast<qtx::DeallocOp>(userOp)) {
          getNewOperands(array, arrayWires, deallocOp.getTargets(),
                         newOperands);
          builder.setInsertionPointAfter(deallocOp);
          builder.create<qtx::DeallocOp>(deallocOp.getLoc(), newOperands);
          break;
        } else if (auto splitOp = dyn_cast<qtx::ArraySplitOp>(userOp)) {
          for (auto [i, wire] : llvm::enumerate(splitOp.getResults()))
            wire.replaceAllUsesWith(arrayWires[i]);
          break;
        } else if (auto borrowOp = dyn_cast<qtx::ArrayBorrowOp>(userOp)) {
          auto indices = getIndicesAsInts(borrowOp);
          for (auto [i, wire] : llvm::enumerate(borrowOp.getWires()))
            wire.replaceAllUsesWith(arrayWires[indices[i]]);
          array = borrowOp.getNewArray();
        } else if (auto yieldOp = dyn_cast<qtx::ArrayYieldOp>(userOp)) {
          auto indices = getBaseIndicesAsInts(yieldOp);
          for (auto [i, wire] : llvm::enumerate(yieldOp.getWires()))
            arrayWires[indices[i]] = wire;
          array = yieldOp.getNewArray();
        } else if (auto resetOp = dyn_cast<qtx::ResetOp>(userOp)) {
          handleNonSink(builder, resetOp, array, arrayWires, newOperands);
        } else if (auto mxOp = dyn_cast<qtx::MxOp>(userOp)) {
          handleNonSink(builder, mxOp, array, arrayWires, newOperands,
                        /*offset=*/1);
        } else if (auto myOp = dyn_cast<qtx::MyOp>(userOp)) {
          handleNonSink(builder, myOp, array, arrayWires, newOperands,
                        /*offset=*/1);
        } else if (auto mzOp = dyn_cast<qtx::MzOp>(userOp)) {
          handleNonSink(builder, mzOp, array, arrayWires, newOperands,
                        /*offset=*/1);
        }
        userOp = *(array.getUsers().begin());
        assert(userOp);
      }
    }

    for (auto op : toErase) {
      op->dropAllUses();
      op->erase();
    }
  }

  template <class T>
  void handleNonSink(OpBuilder &builder, T op, Value &array,
                     SmallVectorImpl<Value> &arrayWires,
                     SmallVectorImpl<Value> &newOperands, unsigned offset = 0) {
    unsigned arrayPos =
        getNewOperands(array, arrayWires, op.getTargets(), newOperands);
    builder.setInsertionPointAfter(op);
    auto newOp = builder.create<T>(op.getLoc(), newOperands);
    auto newArrayWires =
        newOp.getResults().slice(arrayPos + offset, arrayWires.size());
    arrayWires.assign(newArrayWires.begin(), newArrayWires.end());
    array = op.getResult(arrayPos + offset);
    updateWires(array, arrayWires.size(), op.getResults(), newOp.getResults());
  }

  /// Creates a new vector of operands in which the `array` operand is
  /// replaced by the wires operands that currently represent the array.
  /// Returns the position (index) of the array in the old operands vector.
  unsigned getNewOperands(Value array, const SmallVectorImpl<Value> &arrayWires,
                          ValueRange operands,
                          SmallVectorImpl<Value> &newOperands) {
    newOperands.clear();
    size_t arrayOperandIndex = 0;
    for (auto [i, target] : llvm::enumerate(operands)) {
      if (target == array) {
        newOperands.append(arrayWires.begin(), arrayWires.end());
        arrayOperandIndex = i;
        continue;
      }
      newOperands.push_back(target);
    }
    return arrayOperandIndex;
  }

  void updateWires(Value array, unsigned offset, ResultRange oldResults,
                   ResultRange newResults) {
    for (unsigned i = 0; auto wire : oldResults) {
      if (wire == array) {
        i += offset;
        continue;
      }
      wire.replaceAllUsesWith(newResults[i++]);
    }
  }

  SmallVector<unsigned> getIndicesAsInts(qtx::ArrayBorrowOp op) {
    SmallVector<unsigned> indices;
    for (auto indexValue : op.getIndices()) {
      indices.push_back(getIndexAsInt(indexValue));
    }
    return indices;
  }

  unsigned getIndexAsInt(Value indexValue) {
    auto constOp = dyn_cast<arith::ConstantOp>(indexValue.getDefiningOp());
    return dyn_cast<IntegerAttr>(constOp.getValue()).getInt();
  }

  /// The `qtx.array_yield` operation take as operands only the wires that are
  /// being given back to the array.  We need to figure out the index from the
  /// base array where these wires come from.  For example:
  /// ```
  ///   %a0 = alloca : !qtx.wire_array<3>
  ///   %w0, %a1 = array_borrow %i from %a0
  ///    ^_______.
  ///            |
  ///   %w1 = h %w0
  ///    ^_________________.
  ///                      |
  ///   %a2 = array_yield %w1 to %a1
  ///   dealloc %a2
  /// ```
  ///
  /// In this case, `%w1` comes from index `%i`
  ///
  SmallVector<unsigned> getBaseIndicesAsInts(qtx::ArrayYieldOp op) {
    SmallVector<unsigned> indices;
    for (auto wire : op.getWires()) {
      auto defOp = wire.getDefiningOp();
      while (1) {
        if (auto splitOp = dyn_cast<qtx::ArraySplitOp>(defOp)) {
          auto wires = splitOp.getWires();
          auto it = std::find(wires.begin(), wires.end(), wire);
          indices.push_back(std::distance(wires.begin(), it));
          break;
        } else if (auto borrowOp = dyn_cast<qtx::ArrayBorrowOp>(defOp)) {
          auto wires = borrowOp.getWires();
          auto it = std::find(wires.begin(), wires.end(), wire);
          auto i = std::distance(wires.begin(), it);
          indices.push_back(getIndexAsInt(borrowOp.getIndices()[i]));
          break;
        } else if (auto optor = dyn_cast<qtx::OperatorInterface>(defOp)) {
          auto wires = optor.getNewTargets();
          auto it = std::find(wires.begin(), wires.end(), wire);
          if (it != wires.end()) {
            wire = optor.getTarget(std::distance(wires.begin(), it));
            defOp = wire.getDefiningOp();
            continue;
          }
        }
        assert(0);
      }
    }
    return indices;
  }
};

std::unique_ptr<Pass> cudaq::opt::createSplitArraysPass() {
  return std::make_unique<SplitArrays>();
}
