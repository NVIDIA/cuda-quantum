/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/SetVector.h>
#include <memory>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// ConversionTarget and OperatorInfo, parsed from target basis strings such as
// ["x", "x(1)", "z"]
//===----------------------------------------------------------------------===//

struct OperatorInfo {
  StringRef name;
  size_t numControls;

  OperatorInfo(StringRef infoStr) : name(), numControls(0) {
    auto nameEnd = infoStr.find_first_of('(');
    name = infoStr.take_front(nameEnd);
    if (nameEnd < infoStr.size())
      infoStr = infoStr.drop_front(nameEnd);

    if (infoStr.consume_front("(")) {
      infoStr = infoStr.ltrim();
      if (infoStr.consume_front("n"))
        numControls = std::numeric_limits<size_t>::max();
      else
        infoStr.consumeInteger(10, numControls);
      assert(infoStr.trim().consume_front(")"));
    }
  }
};

struct BasisTarget : public ConversionTarget {

  BasisTarget(MLIRContext &context, ArrayRef<std::string> targetBasis)
      : ConversionTarget(context) {
    constexpr size_t unbounded = std::numeric_limits<size_t>::max();

    // Parse the list of target operations and build a set of legal operations
    for (const std::string &targetInfo : targetBasis) {
      legalOperatorSet.emplace_back(targetInfo);
    }

    addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                    cudaq::cc::CCDialect, func::FuncDialect,
                    math::MathDialect>();
    addDynamicallyLegalDialect<quake::QuakeDialect>([&](Operation *op) {
      if (auto optor = dyn_cast<quake::OperatorInterface>(op)) {
        auto name = optor->getName().stripDialect();
        for (auto info : legalOperatorSet) {
          if (info.name != name)
            continue;
          if (info.numControls == unbounded ||
              optor.getControls().size() == info.numControls)
            return info.numControls == optor.getControls().size();
        }
        return false;
      }

      // Handle quake.exp_pauli.
      if (isa<quake::ExpPauliOp>(op)) {
        // If the target defines it as a legal op, return true, else false.
        return std::find_if(legalOperatorSet.begin(), legalOperatorSet.end(),
                            [](auto &&el) { return el.name == "exp_pauli"; }) !=
               legalOperatorSet.end();
      }

      return true;
    });
  }

  SmallVector<OperatorInfo, 8> legalOperatorSet;
};

} // namespace

std::unique_ptr<ConversionTarget>
cudaq::createBasisTarget(MLIRContext &context,
                         ArrayRef<std::string> targetBasis) {
  return std::make_unique<BasisTarget>(context, targetBasis);
}
