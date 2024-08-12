/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ARGUMENTSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "argument-synthesis"

using namespace mlir;

namespace {
class ArgumentSynthesisPass
    : public cudaq::opt::impl::ArgumentSynthesisBase<ArgumentSynthesisPass> {
public:
  using ArgumentSynthesisBase::ArgumentSynthesisBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    StringRef funcName = func.getName();
    std::string text;
    if (std::find_if(funcList.begin(), funcList.end(),
                     [&](const std::string &item) {
                       auto pos = item.find(':');
                       if (pos == std::string::npos)
                         return false;
                       std::string itemName = item.substr(0, pos);
                       bool result = itemName == funcName;
                       if (result)
                         text = item.substr(pos + 1);
                       return result;
                     }) == funcList.end()) {
      // If the function isn't on the list, do nothing.
      LLVM_DEBUG(llvm::dbgs() << funcName << " not in list.\n");
      return;
    }

    // If there are no substitutions, we're done.
    if (text.empty()) {
      LLVM_DEBUG(llvm::dbgs() << funcName << " has no substitutions.");
      return;
    }

    // If we're here, we have a FuncOp and we have substitutions that can be
    // applied.
    //
    // 1. Create a Module with the substitutions that we'll be making.
    auto *ctx = func.getContext();
    LLVM_DEBUG(llvm::dbgs() << "substitution pattern: '" << text << "'\n");
    auto substMod = [&]() -> OwningOpRef<ModuleOp> {
      if (text.front() == '*') {
        // Substitutions are a raw string after the '*' character.
        return parseSourceString<ModuleOp>(text.substr(1), ctx);
      }
      // Substitutions are in a text file (command-line usage).
      return parseSourceFile<ModuleOp>(text, ctx);
    }();
    assert(*substMod && "module must have been created");

    // 2. Go through the Module and process each substitution.
    std::vector<bool> processedArgs(func.getFunctionType().getNumInputs());
    std::vector<std::tuple<unsigned, Value, Value>> replacements;
    for (auto &op : *substMod) {
      auto subst = dyn_cast<cudaq::cc::ArgumentSubstitutionOp>(op);
      if (!subst) {
        if (auto symInterface = dyn_cast<SymbolOpInterface>(op)) {
          auto name = symInterface.getName();
          auto srcMod = func->getParentOfType<ModuleOp>();
          auto obj = srcMod.lookupSymbol(name);
          if (!obj)
            srcMod.getBody()->push_back(op.clone());
        }
        continue;
      }
      auto pos = subst.getPosition();
      if (pos >= processedArgs.size()) {
        func.emitError("Argument " + std::to_string(pos) + " is invalid.");
        signalPassFailure();
        return;
      }
      if (processedArgs[pos]) {
        func.emitError("Argument " + std::to_string(pos) +
                       " was already substituted.");
        signalPassFailure();
        return;
      }

      // OK, substitute the code for the argument.
      Block &entry = func.getRegion().front();
      processedArgs[pos] = true;
      OpBuilder builder{ctx};
      Block *splitBlock = entry.splitBlock(entry.begin());
      builder.setInsertionPointToEnd(&entry);
      builder.create<cf::BranchOp>(func.getLoc(), &subst.getBody().front());
      Operation *lastOp = &subst.getBody().front().back();
      builder.setInsertionPointToEnd(&subst.getBody().front());
      builder.create<cf::BranchOp>(func.getLoc(), splitBlock);
      func.getBlocks().splice(Region::iterator{splitBlock},
                              subst.getBody().getBlocks());
      if (lastOp &&
          lastOp->getResult(0).getType() == entry.getArgument(pos).getType()) {
        LLVM_DEBUG(llvm::dbgs()
                   << funcName << " argument " << std::to_string(pos)
                   << " was substituted.\n");
        replacements.emplace_back(pos, entry.getArgument(pos),
                                  lastOp->getResult(0));
      }
    }

    // Note: if we exited before here, any code that was cloned into the
    // function is still dead and can be removed by a DCE.

    // 3. Replace the block argument values with the freshly inserted new code.
    BitVector replacedArgs(processedArgs.size());
    for (auto [pos, fromVal, toVal] : replacements) {
      replacedArgs.set(pos);
      fromVal.replaceAllUsesWith(toVal);
    }

    // 4. Finish specializing func and erase any of func's arguments that were
    // substituted.
    func.eraseArguments(replacedArgs);
  }
};
} // namespace

// Helper function that takes an unzipped pair of lists of function names and
// substitution code strings. This is meant to make adding this pass to a
// pipeline easier from within a tool (such as the JIT compiler).
std::unique_ptr<mlir::Pass> cudaq::opt::createArgumentSynthesisPass(
    const ArrayRef<StringRef> &funcNames,
    const ArrayRef<StringRef> &substitutions) {
  SmallVector<std::string> pairs;
  if (funcNames.size() == substitutions.size())
    for (auto [name, text] : llvm::zip(funcNames, substitutions))
      pairs.emplace_back(name.str() + ":*" + text.str());
  return std::make_unique<ArgumentSynthesisPass>(
      ArgumentSynthesisOptions{pairs});
}
