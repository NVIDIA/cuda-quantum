/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LAMBDALIFTING
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "lambda-lifting"

using namespace mlir;

/**
 * \file
 *
 * The lambda lifting pass converts lambda expressions (`cc.create_lambda`) to
 * first class functions (`func.func`). It does this incrementally and
 * systematically.
 *
 * Each lambda is translated to a closure (`cc.instantiate_callable`). Each
 * lambda application (`cc.call_callable`) is translated to a regular call to a
 * trampoline. Each trampoline can unpack a closure (`cc.callable_func` and
 * `cc.callable_closure`) and tail calls the original lambda's definition, which
 * has been lifted to a regular function. This allows callable type values to be
 * first-class values that can be passed as arguments.
 *
 * See the documentation of the aforementioned `CC` dialect operations for more
 * information.
 */

static constexpr char liftedLambdaPrefix[] = "__nvqpp__lifted.lambda.";
static constexpr char thunkLambdaPrefix[] = "__nvqpp__callable.thunk.lambda.";
static constexpr char lambdaCounterAttrName[] = "cc.lambda_counter";

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

static void setLambdaCounter(cudaq::cc::CreateLambdaOp lambda,
                             unsigned &counter) {
  auto *ctx = lambda.getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto intAttr = IntegerAttr::get(i32Ty, counter++);
  lambda->setAttr(lambdaCounterAttrName, intAttr);
}

static std::optional<unsigned>
getLambdaCounter(cudaq::cc::CreateLambdaOp lambda) {
  if (auto attr = lambda->getAttr(lambdaCounterAttrName))
    return static_cast<unsigned>(
        cast<IntegerAttr>(attr).getValue().getZExtValue());
  return {};
}

static std::optional<unsigned> extractLambdaCounterFromName(StringRef symName) {
  if (!symName.starts_with(thunkLambdaPrefix))
    return {};
  if (!symName.consume_front(thunkLambdaPrefix))
    return {};
  unsigned num;
  if (symName.getAsInteger(10, num))
    return {};
  return num;
}

namespace {
struct CreateLambdaOpPattern
    : public OpRewritePattern<cudaq::cc::CreateLambdaOp> {
  explicit CreateLambdaOpPattern(MLIRContext *ctx, bool constProp)
      : OpRewritePattern(ctx), constProp(constProp) {}

  static bool defIsInside(Operation *op, cudaq::cc::CreateLambdaOp lambda) {
    auto parent = op->getParentOfType<cudaq::cc::CreateLambdaOp>();
    while (parent) {
      if (parent == lambda)
        return true;
      parent = parent->getParentOfType<cudaq::cc::CreateLambdaOp>();
    }
    return false;
  }

  static bool argIsContainedBy(BlockArgument barg,
                               cudaq::cc::CreateLambdaOp lambda) {
    auto *block = barg.getOwner();
    auto *parent = block->getParentOp();
    return (parent == lambda.getOperation()) || defIsInside(parent, lambda);
  }

  LogicalResult matchAndRewrite(cudaq::cc::CreateLambdaOp lambda,
                                PatternRewriter &rewriter) const override {
    // Get the lambda instance counter.
    auto optCounter = getLambdaCounter(lambda);
    if (!optCounter)
      return failure();
    unsigned counter = *optCounter;

    // Determine all the free values that must be lambda lifted.
    SmallVector<Value> freeVals;
    SmallVector<Value> constVals;
    auto addFreeValue = [&](Value val) {
      bool found = false;
      for (auto v : freeVals)
        if (val == v) {
          found = true;
          break;
        }
      if (!found) {
        if (constProp && cudaq::opt::factory::isConstantOp(val))
          constVals.push_back(val);
        else
          freeVals.push_back(val);
      }
    };

    // Walk over the body of the CreateLambdaOp and find all the free values.
    // Add each of them to the list.
    lambda.walk([&](Operation *op) {
      for (Value oper : op->getOperands()) {
        if (auto *use = oper.getDefiningOp()) {
          if (!defIsInside(use, lambda))
            addFreeValue(oper);
        } else {
          auto barg = cast<BlockArgument>(oper);
          if (!argIsContainedBy(barg, lambda))
            addFreeValue(oper);
        }
      }
    });

    LLVM_DEBUG({
      llvm::dbgs() << "lambda " << lambda << "\n  free:\n";
      for (auto v : freeVals)
        llvm::dbgs() << '\t' << v << '\n';
      if (constProp) {
        llvm::dbgs() << "  const:\n";
        for (auto v : constVals)
          llvm::dbgs() << '\t' << v << '\n';
      }
    });

    /// For each lambda expression, we perform the following tasks.
    ///
    /// 1. Create a local struct, <i>s</i>.
    /// 2. Insert the callable thunk function into <i>s</i>.
    /// 3. Insert any free values into <i>s</i>.
    /// 4. Replace op with a cc.instantiate_callable.
    /// 5. Create the callable thunk function to unpack <i>s</i>.
    /// 6. Create the lifted function, a copy of the cc.create_lambda.
    auto module = lambda->getParentOfType<ModuleOp>();
    auto *ctx = lambda.getContext();
    auto loc = lambda.getLoc();
    cudaq::cc::CallableType lambdaTy = lambda.getType();
    auto sig = lambdaTy.getSignature();

    // Create callable thunk function. This function binds free variables to
    // captured values in the closure then tail calls the lifted lambda.
    SmallVector<NamedAttribute> emptyDict;
    ValueRange freeValues{freeVals};
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      SmallVector<Type> argTys;
      argTys.push_back(lambdaTy);
      argTys.append(sig.getInputs().begin(), sig.getInputs().end());
      auto funTy = FunctionType::get(ctx, argTys, sig.getResults());
      auto thunk = rewriter.create<func::FuncOp>(
          loc, getThunkLambdaName(counter), funTy, emptyDict);
      thunk.setPrivate();
      thunk->setAttr(cudaq::kernelAttrName, rewriter.getUnitAttr());
      auto *entry = thunk.addEntryBlock();
      rewriter.setInsertionPointToEnd(entry);
      SmallVector<Value> callableArgs;
      if (!freeValues.empty()) {
        auto closureData = rewriter.create<cudaq::cc::CallableClosureOp>(
            loc, freeValues.getTypes(), thunk.getArgument(0));
        callableArgs.append(closureData.getResults().begin(),
                            closureData.getResults().end());
      }
      callableArgs.append(thunk.getArguments().begin() + 1,
                          thunk.getArguments().end());
      auto result = rewriter.create<func::CallOp>(
          loc, sig.getResults(), getLiftedLambdaName(counter), callableArgs);
      rewriter.create<func::ReturnOp>(loc, result.getResults());
    }

    // Create a new lambda function to lift the expression into. This function
    // should be inlined into the callable thunk function, if any, ultimately.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      SmallVector<Type> argTys(freeValues.getTypes().begin(),
                               freeValues.getTypes().end());
      argTys.append(sig.getInputs().begin(), sig.getInputs().end());
      auto funTy = FunctionType::get(ctx, argTys, sig.getResults());
      auto func = rewriter.create<func::FuncOp>(
          loc, getLiftedLambdaName(counter), funTy, emptyDict);
      func.setPrivate();
      func->setAttr(cudaq::kernelAttrName, rewriter.getUnitAttr());
      auto *entry = func.addEntryBlock();
      // Add entry block, block arguments for free variables to a renaming
      // map.
      IRMapping blockAndValueMap;

      // Clone constants and add them to the map.
      if (!constVals.empty()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(entry);
        for (auto c : constVals) {
          // A ConstantLike must have exactly 1 result and 0 arguments.
          auto *op = rewriter.clone(*c.getDefiningOp());
          blockAndValueMap.map(c, op->getResult(0));
        }
      }

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
      rewriter.setInsertionPointToEnd(entry);
      auto nextBlockIter = ++func.getBlocks().begin();
      // Connect entry block to cloned code.
      rewriter.create<cf::BranchOp>(loc, &*nextBlockIter);
    }

    SymbolRefAttr closureSymbol =
        FlatSymbolRefAttr::get(ctx, getThunkLambdaName(counter));
    [[maybe_unused]] auto instantiate =
        rewriter.replaceOpWithNewOp<cudaq::cc::InstantiateCallableOp>(
            lambda, lambdaTy, closureSymbol, freeValues);
    LLVM_DEBUG(llvm::dbgs() << instantiate << '\n');
    return success();
  }

private:
  bool constProp;
};

/// Convert a compute_action to a series of calls. A compute_action is
/// semantically a special form of a call site.
struct ComputeActionOpPattern
    : public OpRewritePattern<quake::ComputeActionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ComputeActionOp comAct,
                                PatternRewriter &rewriter) const override {
    // In lambda lifting, we only deal with rewriting the quake.compute_action
    // Op if it has lambda arguments.
    bool isComputeCallable =
        isa<cudaq::cc::CallableType>(comAct.getCompute().getType());
    bool isActionCallable =
        isa<cudaq::cc::CallableType>(comAct.getAction().getType());
    if (!(isComputeCallable || isActionCallable))
      return failure();

    // OK. At least one of the two arguments is a callable. Let's see if we can
    // identify the exact callable instantiation.
    auto instCompute =
        comAct.getCompute().getDefiningOp<cudaq::cc::InstantiateCallableOp>();
    auto instAction =
        comAct.getAction().getDefiningOp<cudaq::cc::InstantiateCallableOp>();

    // Can we identify the callable instances that we require?
    if (!(isComputeCallable && instCompute))
      return failure();
    if (!(isActionCallable && instAction))
      return failure();

    auto *ctx = rewriter.getContext();
    auto loc = comAct.getLoc();
    auto computeCallee = getCallee(ctx, comAct.getCompute());
    if (!computeCallee)
      return failure();
    auto actionCallee = getCallee(ctx, comAct.getAction());
    if (!actionCallee)
      return failure();
    auto computeArgs = getArgs(comAct.getCompute());
    rewriter.create<quake::ApplyOp>(loc, TypeRange{}, computeCallee,
                                    /*isAdjoint=*/comAct.getIsDagger(),
                                    ValueRange{}, computeArgs);
    rewriter.create<quake::ApplyOp>(loc, TypeRange{}, actionCallee,
                                    /*isAdjoint=*/false, ValueRange{},
                                    getArgs(comAct.getAction()));
    rewriter.replaceOpWithNewOp<quake::ApplyOp>(
        comAct, TypeRange{}, computeCallee,
        /*isAdjoint=*/!comAct.getIsDagger(), ValueRange{}, computeArgs);
    return success();
  }

  SymbolRefAttr getCallee(MLIRContext *ctx, Value val) const {
    if (auto inst = val.getDefiningOp<cudaq::cc::InstantiateCallableOp>()) {
      if (auto num = extractLambdaCounterFromName(
              inst.getCallee().getRootReference().getValue()))
        return getLiftedLambdaSymbol(ctx, *num);
    } else if (auto fc = val.getDefiningOp<func::ConstantOp>()) {
      return SymbolRefAttr::get(ctx, fc.getValue());
    }
    return {};
  }

  SmallVector<Value> getArgs(Value val) const {
    if (auto inst = val.getDefiningOp<cudaq::cc::InstantiateCallableOp>())
      if (!inst.getNoCapture())
        return inst.getClosureData();
    return {};
  }
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

struct CallableFuncOpPattern
    : public OpRewritePattern<cudaq::cc::CallableFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::CallableFuncOp callFunc,
                                PatternRewriter &rewriter) const override {
    auto instance = callFunc.getCallable()
                        .getDefiningOp<cudaq::cc::InstantiateCallableOp>();
    if (!instance)
      return failure();
    rewriter.replaceOpWithNewOp<func::ConstantOp>(
        callFunc, callFunc.getType(),
        instance.getCallee().getRootReference().getValue());
    return success();
  }
};

struct ReturnOpPattern : public OpRewritePattern<cudaq::cc::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::ReturnOp ret,
                                PatternRewriter &rewriter) const override {
    if (ret->getParentOfType<cudaq::cc::CreateLambdaOp>())
      return failure();
    rewriter.replaceOpWithNewOp<func::ReturnOp>(ret, ret.getOperands());
    return success();
  }
};

class LambdaLiftingPass
    : public cudaq::opt::impl::LambdaLiftingBase<LambdaLiftingPass> {
public:
  using LambdaLiftingBase::LambdaLiftingBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto *ctx = module.getContext();
    bool foundLambdas = false;

    // First scan the module to see if there are any lambdas to lift and
    // annotate them with unique indices.
    if (failed(findAndAnnotateAllLambdas(module, foundLambdas)))
      signalPassFailure();
    if (!foundLambdas) {
      // No cc.create_lambdas were found. Exit early.
      return;
    }

    // Lift the lambda expressions to first-class functions.
    // Rewrite the users and subops of those lambda expressions.
    RewritePatternSet patterns(ctx);
    patterns.insert<CreateLambdaOpPattern>(ctx, constantPropagation);
    patterns.insert<ComputeActionOpPattern, CallCallableOpPattern,
                    CallableFuncOpPattern, ReturnOpPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
  }

  LogicalResult findAndAnnotateAllLambdas(ModuleOp mod, bool &foundOne) {
    // Pre-analysis: see how many lambdas were already lifted.
    unsigned cnt = 0;
    bool incr = false;
    for (auto &a : mod)
      if (auto sym = dyn_cast<SymbolOpInterface>(a)) {
        StringRef symName = sym.getName();
        if (!symName.starts_with(liftedLambdaPrefix))
          continue;
        bool ok = symName.consume_front(liftedLambdaPrefix);
        unsigned num;
        bool err = ok ? symName.getAsInteger(10, num) : true;
        if (err)
          return a.emitOpError("lifted lambda name improperly constructed");
        cnt = std::max(cnt, num);
        incr = true;
      }

    if (incr)
      ++cnt;

    mod->walk([&](cudaq::cc::CreateLambdaOp lambda) {
      // Note setLambdaCounter increments cnt as a side effect.
      setLambdaCounter(lambda, cnt);
      foundOne = true;
    });

    return success();
  }
};
} // namespace
