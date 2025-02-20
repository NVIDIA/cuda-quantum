/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#include <list>

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

  void
  applySubstitutions(func::FuncOp func,
                     DenseMap<StringRef, OwningOpRef<ModuleOp>> &substModules) {
    MLIRContext *ctx = func.getContext();
    auto funcName = func.getName();
    LLVM_DEBUG(llvm::dbgs() << "processing : '" << funcName << "'\n");

    // 1. Find substitution module with argument replacements for the function.
    auto it = substModules.find(funcName);
    if (it == substModules.end()) {
      // If the function isn't on the list, do nothing.
      LLVM_DEBUG(llvm::dbgs() << funcName << " has no substitutions.\n");
      return;
    }
    auto substMod = *(it->second);

    // 2. Go through the Module and process each substitution.
    SmallVector<bool> processedArgs(func.getFunctionType().getNumInputs());
    SmallVector<std::tuple<unsigned, Value, Value>> replacements;
    BitVector replacedArgs(processedArgs.size());
    for (auto &op : substMod) {
      auto subst = dyn_cast<cudaq::cc::ArgumentSubstitutionOp>(op);
      if (!subst)
        continue;
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
      if (subst.getBody().front().empty()) {
        // No code is present. Erase the argument if it is not used.
        const auto numUses =
            std::distance(entry.getArgument(pos).getUses().begin(),
                          entry.getArgument(pos).getUses().end());
        LLVM_DEBUG(llvm::dbgs() << "maybe erasing an unused argument ("
                                << std::to_string(numUses) << ")\n");
        if (numUses == 0)
          replacedArgs.set(pos);
        continue;
      }
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
    for (auto [pos, fromVal, toVal] : replacements) {
      replacedArgs.set(pos);
      fromVal.replaceAllUsesWith(toVal);
    }

    // 4. Finish specializing func and erase any of func's arguments that were
    // substituted.
    func.eraseArguments(replacedArgs);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    // 1. Collect all substitution modules.
    std::list<std::string> funcNames;
    DenseMap<StringRef, OwningOpRef<ModuleOp>> substModules;

    for (auto &item : funcList) {
      auto pos = item.find(':');
      if (pos == std::string::npos)
        continue;

      std::string funcName = item.substr(0, pos);
      std::string text = item.substr(pos + 1);

      if (text.empty()) {
        LLVM_DEBUG(llvm::dbgs() << funcName << " has no substitutions.");
        continue;
      }

      // Create a Module with the substitutions that we'll be making.
      LLVM_DEBUG(llvm::dbgs()
                 << funcName << " : substitution pattern: '" << text << "'\n");
      auto substModule = [&]() -> OwningOpRef<ModuleOp> {
        if (text.front() == '*') {
          // Substitutions are a raw string after the '*' character.
          return parseSourceString<ModuleOp>(text.substr(1), ctx);
        }
        // Substitutions are in a text file (command-line usage).
        return parseSourceFile<ModuleOp>(text, ctx);
      }();
      assert(*substModule && "module must have been created");

      auto &name = funcNames.emplace_back(funcName);
      substModules.try_emplace(name, std::move(substModule));
    }

    // 2. Merge symbols from substitution modules into the source module.
    for (auto &[funcName, substMod] : substModules) {
      for (auto &op : *substMod) {
        if (auto symInterface = dyn_cast<SymbolOpInterface>(op)) {
          auto name = symInterface.getName();
          auto obj = mod.lookupSymbol(name);
          if (!obj)
            mod.getBody()->push_back(op.clone());
        }
      }
    }

    // 3. Apply all substitutions.
    mod->walk(
        [&](func::FuncOp func) { applySubstitutions(func, substModules); });
  }
};
} // namespace

// Helper function that takes an unzipped pair of lists of function names and
// substitution code strings. This is meant to make adding this pass to a
// pipeline easier from within a tool (such as the JIT compiler).
std::unique_ptr<mlir::Pass>
cudaq::opt::createArgumentSynthesisPass(ArrayRef<StringRef> funcNames,
                                        ArrayRef<StringRef> substitutions) {
  SmallVector<std::string> pairs;
  if (funcNames.size() == substitutions.size())
    for (auto [name, text] : llvm::zip(funcNames, substitutions))
      pairs.emplace_back(name.str() + ":*" + text.str());
  return std::make_unique<ArgumentSynthesisPass>(
      ArgumentSynthesisOptions{pairs});
}
