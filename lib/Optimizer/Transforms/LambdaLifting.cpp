/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "quake-lambda-lifting"

using namespace mlir;

static constexpr char liftedLambdaPrefix[] = "__nvqpp__lifted.lambda.";
static constexpr char thunkLambdaPrefix[] = "__nvqpp__callable.thunk.lambda.";

inline std::string getLiftedLambdaName(unsigned counter) {
  return liftedLambdaPrefix + std::to_string(counter);
}
inline std::string getThunkLambdaName(unsigned counter) {
  return thunkLambdaPrefix + std::to_string(counter);
}

inline SymbolRefAttr getLiftedLambdaSymbol(MLIRContext *ctx, unsigned counter) {
  return SymbolRefAttr::get(ctx, getLiftedLambdaName(counter));
}
inline SymbolRefAttr getThunkLambdaSymbol(MLIRContext *ctx, unsigned counter) {
  return SymbolRefAttr::get(ctx, getThunkLambdaName(counter));
}

namespace {
struct LambdaExprInfo {
  unsigned counter;              ///< Unique counter value;
  SmallVector<Value> freeValues; ///< values that are free in lambda
};

using LambdaOpAnalysisInfo = llvm::DenseMap<Operation *, LambdaExprInfo>;

/// This analysis scans the IR for `cc::CreateLambdaOp`s and gives each a unique
/// number.
struct LambdaOpAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LambdaOpAnalysis)

  LambdaOpAnalysis(ModuleOp op) : module(op) {
    performAnalysis(op.getOperation());
  }

  LambdaOpAnalysisInfo getAnalysisInfo() const { return infoMap; }

private:
  void performAnalysis(Operation *op) {
    op->walk([&](cudaq::cc::CreateLambdaOp appOp) {
      SmallVector<Value> freeValues;
      // Walk over the body of the CreateLambdaOp and find all the free values.
      // Add each of them to the list.
      appOp.walk([&](Operation *op) {
        auto addFreeValue = [&](Value oper) {
          bool found = false;
          for (auto v : freeValues)
            if (oper == v) {
              found = true;
              break;
            }
          if (!found)
            freeValues.push_back(oper);
        };
        for (Value oper : op->getOperands()) {
          if (auto *operOp = oper.getDefiningOp()) {
            if (operOp->getParentOfType<cudaq::cc::CreateLambdaOp>() != appOp)
              addFreeValue(oper);
          } else if (auto arg = dyn_cast<BlockArgument>(oper)) {
            auto *argOp = arg.getOwner()->getParentOp();
            auto argLambda = dyn_cast<cudaq::cc::CreateLambdaOp>(argOp);
            if (argLambda) {
              if (argLambda != appOp)
                addFreeValue(oper);
            } else {
              if (argOp->getParentOfType<cudaq::cc::CreateLambdaOp>() != appOp)
                addFreeValue(oper);
            }
          } else {
            // TODO: assert it is a constant?
          }
        }
      });
      // Add this CreateLambdaOp to the map with its unique number and list of
      // free values.
      LambdaExprInfo info = {counter++, freeValues};
      infoMap.insert(std::make_pair(appOp.getOperation(), std::move(info)));
    });
  }

  ModuleOp module;
  LambdaOpAnalysisInfo infoMap;
  unsigned counter = 0;
};

/// Convert a compute_action to a series of calls. A compute_action is
/// semantically a special form of a call site.
struct ComputeActionOpPattern
    : public OpRewritePattern<quake::ComputeActionOp> {
  explicit ComputeActionOpPattern(MLIRContext *ctx,
                                  const LambdaOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(quake::ComputeActionOp comAct,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = comAct.getLoc();
    rewriter.create<quake::ApplyOp>(loc, TypeRange{},
                                    getCallee(ctx, comAct.getCompute()),
                                    /*isAdjoint=*/comAct.getIsDagger(),
                                    ValueRange{}, getArgs(comAct.getCompute()));
    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{}, getCallee(ctx, comAct.getAction()),
        /*isAdjoint=*/false, ValueRange{}, getArgs(comAct.getAction()));
    rewriter.replaceOpWithNewOp<quake::ApplyOp>(
        comAct, TypeRange{}, getCallee(ctx, comAct.getCompute()),
        /*isAdjoint=*/!comAct.getIsDagger(), ValueRange{},
        getArgs(comAct.getCompute()));
    return success();
  }

  SymbolRefAttr getCallee(MLIRContext *ctx, Value val) const {
    if (auto *op = val.getDefiningOp())
      if (auto iter = infoMap.find(op); iter != infoMap.end())
        return getLiftedLambdaSymbol(ctx, iter->second.counter);
    return {};
  }

  SmallVector<Value> getArgs(Value val) const {
    if (auto *op = val.getDefiningOp())
      if (auto iter = infoMap.find(op); iter != infoMap.end())
        return iter->second.freeValues;
    return {};
  }

  const LambdaOpAnalysisInfo &infoMap;
};

/// Convert a call of a callable expression. A callable expression differs from
/// a "normal" function in that it may contain additional values captured when
/// the callable object is instantiated. These "free values/variables" are
/// dynamic to the instantiation context, will be "remembered" when the instance
/// is invoked, and need not be known at all to the invoking context.
struct CallCallableOpPattern
    : public OpRewritePattern<cudaq::cc::CallCallableOp> {
  using OpRewritePattern::OpRewritePattern;

  /// Rewrite cc.call_callable to a lowered form that extracts the trampoline
  /// from the callable closure, preps for calling the callable thunk helper,
  /// and uses a func.call_indirect Op to invoke the callable.
  LogicalResult matchAndRewrite(cudaq::cc::CallCallableOp call,
                                PatternRewriter &rewriter) const override {
    auto loc = call.getLoc();
    auto operands = call.getOperands();
    auto closure = call.getCallee();
    auto closureTy = closure.getType();

    // For a callable, call the trampoline with the closure data.
    if (auto lambTy = dyn_cast<cudaq::cc::CallableType>(closureTy)) {
      auto dynFunc = rewriter.create<cudaq::cc::CallableFuncOp>(
          loc, call.getFunctionType(), closure);
      rewriter.replaceOpWithNewOp<func::CallIndirectOp>(call, dynFunc,
                                                        operands);
      return success();
    }

    // For a normal function, there is no closure to deal with.
    if (auto sig = dyn_cast<FunctionType>(closureTy)) {
      auto dynFunc =
          rewriter.create<cudaq::cc::CallableFuncOp>(loc, sig, closure);
      rewriter.replaceOpWithNewOp<func::CallIndirectOp>(call, dynFunc,
                                                        operands.drop_front());
      return success();
    }
    return emitError(loc, "callee has unexpected type");
  }
};

class LambdaLiftingPass
    : public cudaq::opt::LambdaLiftingBase<LambdaLiftingPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();
    LambdaOpAnalysis analysis(module);
    auto lambdaInfo = analysis.getAnalysisInfo();
    if (!lambdaInfo.empty()) {
      // (1) Lift the lambda expressions to first-class functions.
      liftAllLambdas(lambdaInfo);
      LLVM_DEBUG(llvm::dbgs() << "After all lambdas lifted:\n"
                              << module << '\n');

      // (2) Rewrite the users of lambda expressions.
      RewritePatternSet patterns(ctx);
      patterns.insert<ComputeActionOpPattern>(ctx, lambdaInfo);
      patterns.insert<CallCallableOpPattern>(ctx);
      ConversionTarget target(*ctx);
      target.addLegalDialect<quake::QuakeDialect, cudaq::cc::CCDialect,
                             func::FuncDialect>();
      target.addIllegalOp<quake::ComputeActionOp, cudaq::cc::CallCallableOp>();

      if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        emitError(module.getLoc(), "failed to lift lambdas");
        signalPassFailure();
      }
    }
  }

  /// Scan the module for lambda expressions. For each lambda expression, we
  /// perform the following tasks.
  ///
  /// 1. Create a local struct, <i>s</i>.
  /// 2. Insert the callable thunk function into <i>s</i>.
  /// 3. Insert any free values into <i>s</i>.
  /// 4. Replace op with a cc.instantiate_callable.
  /// 5. Create the callable thunk function to unpack <i>s</i>.
  /// 6. Create the lifted function, a copy of the cc.create_lambda.
  void liftAllLambdas(LambdaOpAnalysisInfo &infoMap) {
    auto module = getOperation();
    auto *ctx = module.getContext();
    IRRewriter rewriter(ctx);
    rewriter.startRootUpdate(module);
    LambdaOpAnalysisInfo newInfoMap(infoMap);
    for (auto &[op, lambdaInfo] : infoMap) {
      auto lambda = dyn_cast<cudaq::cc::CreateLambdaOp>(op);
      assert(lambda && "must be a cc.create_lambda");
      auto loc = lambda.getLoc();
      auto iter = infoMap.find(lambda.getOperation());
      if (iter == infoMap.end()) {
        emitError(lambda.getLoc(), "lambda expression not analyzed");
        signalPassFailure();
        return;
      }
      rewriter.setInsertionPoint(lambda);
      cudaq::cc::CallableType lambdaTy = lambda.getType();
      auto sig = lambdaTy.getSignature();
      auto counter = iter->second.counter;
      ValueRange freeValues = iter->second.freeValues;

      // Create inline instantiation.
      OpBuilder build(module.getBodyRegion());

      // Create callable thunk function. This function binds free variables to
      // captured values in the closure then tail calls the lifted lambda.
      SmallVector<NamedAttribute> emptyDict;
      build.setInsertionPointToEnd(module.getBody());
      {
        OpBuilder::InsertionGuard guard(build);
        SmallVector<Type> argTys;
        argTys.push_back(lambdaTy);
        argTys.append(sig.getInputs().begin(), sig.getInputs().end());
        auto funTy = FunctionType::get(ctx, argTys, sig.getResults());
        auto thunk = build.create<func::FuncOp>(
            loc, getThunkLambdaName(counter), funTy, emptyDict);
        thunk.setPrivate();
        auto *entry = thunk.addEntryBlock();
        build.setInsertionPointToEnd(entry);
        SmallVector<Value> callableArgs;
        if (!freeValues.empty()) {
          auto closureData = build.create<cudaq::cc::CallableClosureOp>(
              loc, freeValues.getTypes(), thunk.getArgument(0));
          callableArgs.append(closureData.getResults().begin(),
                              closureData.getResults().end());
        }
        callableArgs.append(thunk.getArguments().begin() + 1,
                            thunk.getArguments().end());
        auto result = build.create<func::CallOp>(
            loc, sig.getResults(), getLiftedLambdaName(counter), callableArgs);
        build.create<func::ReturnOp>(loc, sig.getResults(),
                                     result.getResults());
      }

      // Create a new lambda function to lift the expression into. This function
      // should be inlined into the callable thunk function, if any, ultimately.
      {
        OpBuilder::InsertionGuard guard(build);
        SmallVector<Type> argTys(freeValues.getTypes().begin(),
                                 freeValues.getTypes().end());
        argTys.append(sig.getInputs().begin(), sig.getInputs().end());
        auto funTy = FunctionType::get(ctx, argTys, sig.getResults());
        build.setInsertionPointToEnd(module.getBody());
        auto func = build.create<func::FuncOp>(
            loc, getLiftedLambdaName(counter), funTy, emptyDict);
        func.setPrivate();
        auto *entry = func.addEntryBlock();
        // Add entry block, block arguments for free variables to a renaming
        // map.
        IRMapping blockAndValueMap;
        // TODO: Block mapping doesn't appear to work as expected.
        // auto *lambdaEntry = &lambda.getRegion().front();
        // blockAndValueMap.map(lambdaEntry, entry);

        // Add free variables to value map.
        for (auto i : llvm::enumerate(freeValues))
          blockAndValueMap.map(i.value(), entry->getArgument(i.index()));
        auto freeOffset = freeValues.size();
        // Add arguments to value map with adjusted positions.
        for (auto i :
             llvm::enumerate(lambda.getRegion().front().getArguments())) {
          blockAndValueMap.map(i.value(),
                               entry->getArgument(i.index() + freeOffset));
        }
        // Clone the lambda's region into the new function.
        rewriter.cloneRegionBefore(lambda.getRegion(), func.getRegion(),
                                   func.getRegion().end(), blockAndValueMap);
        build.setInsertionPointToEnd(entry);
        auto nextBlockIter = ++func.getBlocks().begin();
        // Connect entry block to cloned code.
        build.create<cf::BranchOp>(loc, &*nextBlockIter);
      }

      SymbolRefAttr closureSymbol =
          FlatSymbolRefAttr::get(ctx, getThunkLambdaName(counter));
      auto instantiate =
          rewriter.replaceOpWithNewOp<cudaq::cc::InstantiateCallableOp>(
              lambda, lambdaTy, closureSymbol, freeValues);
      newInfoMap[instantiate.getOperation()] = lambdaInfo;
      LLVM_DEBUG(llvm::dbgs() << module << '\n');
    }
    rewriter.finalizeRootUpdate(module);
    // Update the info map with the cc.instantiate_callable Ops.
    infoMap = newInfoMap;
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createLambdaLiftingPass() {
  return std::make_unique<LambdaLiftingPass>();
}
