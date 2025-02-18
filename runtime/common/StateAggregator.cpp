/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "StateAggregator.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Todo.h"
#include "cudaq/qis/pauli_word.h"
#include "cudaq/utils/registry.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

/// Create callee.init_N that initializes the state
/// Callee (the kernel captured by state):
// clang-format off
/// func.func @callee(%arg0: i64) {
///   %0 = cc.alloca i64
///   cc.store %arg0, %0 : !cc.ptr<i64>
///   %1 = cc.load %0 : !cc.ptr<i64>
///   %2 = quake.alloca !quake.veq<?>[%1 : i64]
///   %3 = quake.extract_ref %2[1] : (!quake.veq<?>) -> !quake.ref
///   quake.x %3 : (!quake.ref) -> ()
///   return
/// }
/// callee.init_N:
/// func.func private @callee.init_0(%arg0: !quake.veq<?>, %arg0: i64) ->
/// !!quake.veq<?> {
///   %1 = quake.extract_ref %arg0[1] : (!quake.veq<2>) -> !quake.ref
///   quake.x %1 : (f64, !quake.ref) -> ()
///   return %arg0: !quake.veq<?>
/// }
// clang-format on
static void createInitFunc(OpBuilder &builder, ModuleOp moduleOp,
                           func::FuncOp calleeFunc, StringRef initKernelName) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto ctx = builder.getContext();
  auto loc = builder.getUnknownLoc();

  auto initFunc = cast<func::FuncOp>(builder.clone(*calleeFunc));

  auto argTypes = calleeFunc.getArgumentTypes();
  auto retTy = quake::VeqType::getUnsized(ctx);
  auto funcTy = FunctionType::get(ctx, argTypes, TypeRange{retTy});

  initFunc.setName(initKernelName);
  initFunc.setType(funcTy);
  initFunc.setPrivate();

  OpBuilder newBuilder(ctx);

  auto *entryBlock = &initFunc.getRegion().front();
  newBuilder.setInsertionPointToStart(entryBlock);
  Value zero = newBuilder.create<arith::ConstantIntOp>(loc, 0, 64);
  Value one = newBuilder.create<arith::ConstantIntOp>(loc, 1, 64);
  Value begin = zero;

  auto argPos = initFunc.getArguments().size();

  // Detect errors in kernel passed to get_state.
  std::function<void(Block &)> processInner = [&](Block &block) {
    for (auto &op : block) {
      for (auto &region : op.getRegions())
        for (auto &b : region)
          processInner(b);

      // Don't allow returns in inner scopes
      if (auto retOp = dyn_cast<func::ReturnOp>(&op))
        calleeFunc.emitError("Encountered return in inner scope in a kernel "
                             "passed to get_state");
    }
  };

  for (auto &op : calleeFunc.getRegion().front())
    for (auto &region : op.getRegions())
      for (auto &b : region)
        processInner(b);

  // Process outer block to initialize the allocation passed as an argument.
  std::function<void(Block &)> process = [&](Block &block) {
    SmallVector<Operation *> cleanUps;
    Operation *replacedReturn = nullptr;

    Value arg;
    Value subArg;
    Value blockBegin = begin;
    Value blockAllocSize = zero;
    for (auto &op : block) {
      if (auto alloc = dyn_cast<quake::AllocaOp>(&op)) {
        newBuilder.setInsertionPointAfter(alloc);

        if (!arg) {
          initFunc.insertArgument(argPos, retTy, {}, loc);
          arg = initFunc.getArgument(argPos);
        }

        auto allocSize = alloc.getSize();
        auto offset = newBuilder.create<arith::SubIOp>(loc, allocSize, one);
        subArg =
            newBuilder.create<quake::SubVeqOp>(loc, retTy, arg, begin, offset);
        alloc.replaceAllUsesWith(subArg);
        cleanUps.push_back(alloc);
        begin = newBuilder.create<arith::AddIOp>(loc, begin, allocSize);
        blockAllocSize =
            newBuilder.create<arith::AddIOp>(loc, blockAllocSize, allocSize);
      }

      if (auto retOp = dyn_cast<func::ReturnOp>(&op)) {
        if (retOp != replacedReturn) {
          newBuilder.setInsertionPointAfter(retOp);

          auto offset =
              newBuilder.create<arith::SubIOp>(loc, blockAllocSize, one);
          Value ret = newBuilder.create<quake::SubVeqOp>(loc, retTy, arg,
                                                         blockBegin, offset);

          assert(arg && "No veq allocations found");
          replacedReturn = newBuilder.create<func::ReturnOp>(loc, ret);
          cleanUps.push_back(retOp);
        }
      }
    }

    for (auto &op : cleanUps) {
      op->dropAllReferences();
      op->dropAllUses();
      op->erase();
    }
  };

  // Process the function body
  process(initFunc.getRegion().front());
}

/// Create callee.num_qubits_N that calculates the number of qubits to
/// initialize the state
/// Callee: (the kernel captured by state):
// clang-format off
/// func.func @callee(%arg0: i64) {
///   %0 = cc.alloca i64
///   cc.store %arg0, %0 : !cc.ptr<i64>
///   %1 = cc.load %0 : !cc.ptr<i64>
///   %2 = quake.alloca !quake.veq<?>[%1 : i64]
///   %3 = quake.extract_ref %2[1] : (!quake.veq<?>) -> !quake.ref
///   quake.x %3 : (!quake.ref) -> ()
///   return
/// }
///
/// callee.num_qubits_0:
/// func.func private @callee.num_qubits_0(%arg0: i64) -> i64 {
///   %0 = cc.alloca i64
///   cc.store %arg0, %0 : !cc.ptr<i64>
///   %1 = cc.load %0 : !cc.ptr<i64>
///   return %1 : i64
/// }
// clang-format on
static void createNumQubitsFunc(OpBuilder &builder, ModuleOp moduleOp,
                                func::FuncOp calleeFunc,
                                StringRef numQubitsKernelName) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto ctx = builder.getContext();
  auto loc = builder.getUnknownLoc();

  auto numQubitsFunc = cast<func::FuncOp>(builder.clone(*calleeFunc));

  auto argTypes = calleeFunc.getArgumentTypes();
  auto retType = builder.getI64Type();
  auto funcTy = FunctionType::get(ctx, argTypes, TypeRange{retType});

  numQubitsFunc.setName(numQubitsKernelName);
  numQubitsFunc.setType(funcTy);
  numQubitsFunc.setPrivate();

  OpBuilder newBuilder(ctx);

  auto *entryBlock = &numQubitsFunc.getRegion().front();
  newBuilder.setInsertionPointToStart(entryBlock);
  Value size = newBuilder.create<arith::ConstantIntOp>(loc, 0, retType);

  // Process block recursively to calculate and return allocation size
  // and remove everything else.
  std::function<void(Block &)> process = [&](Block &block) {
    SmallVector<Operation *> used;
    Operation *replacedReturn = nullptr;

    for (auto &op : block) {
      // Calculate allocation size (existing allocation size plus new one)
      if (auto alloc = dyn_cast<quake::AllocaOp>(&op)) {
        auto allocSize = alloc.getSize();
        newBuilder.setInsertionPointAfter(alloc);
        size = newBuilder.create<arith::AddIOp>(loc, size, allocSize);
      }

      // Return allocation size
      if (auto retOp = dyn_cast<func::ReturnOp>(&op)) {
        if (retOp != replacedReturn) {

          newBuilder.setInsertionPointAfter(retOp);
          auto newRet = newBuilder.create<func::ReturnOp>(loc, size);
          replacedReturn = newRet;
          used.push_back(newRet);
        }
      }
    }

    // Collect all ops needed for size calculation
    SmallVector<Operation *> keep;
    while (!used.empty()) {
      auto *op = used.pop_back_val();
      keep.push_back(op);
      for (auto opnd : op->getOperands())
        if (auto defOp = opnd.getDefiningOp())
          used.push_back(defOp);
    }

    // Remove the rest of the ops
    SmallVector<Operation *> toErase;
    for (auto &op : block)
      if (std::find(keep.begin(), keep.end(), &op) == keep.end())
        toErase.push_back(&op);

    for (auto &op : toErase) {
      op->dropAllReferences();
      op->dropAllUses();
      op->erase();
    }
  };

  // Process the function body
  process(numQubitsFunc.getRegion().front());
}

void cudaq::opt::StateAggregator::collectKernelInfo(const cudaq::state *v) {
  auto simState =
      cudaq::state_helper::getSimulationState(const_cast<cudaq::state *>(v));

  // If the state has amplitude data, we materialize the data as a state
  // vector and create a new state from it in the ArgumentConverter.
  // TODO: add an option to use the kernel info if available, i.e. for
  // remote simulators
  // TODO: add an option of storing the kernel info on simulators if
  // preferred i.e. to support synthesis of density matrices.
  if (simState->hasData()) {
    return;
  }

  // Otherwise (ie quantum hardware, where getting the amplitude data is not
  // efficient) we aim at replacing states with calls to kernels (`callees`)
  // that generated them. This is done in three stages:
  //
  // 1) (done here) Generate @callee.num_qubits_0 @callee.init_0` for the callee
  //    function and its arguments stored in a state.

  //    Create two functions:
  //      - callee.num_qubits_N
  //        Calculates the number of qubits needed for the veq allocation
  //      - callee.init_N
  //        Initializes the veq passed as a parameter
  //
  // 2) (done in ArgumentConverter) Replace the state with
  //   `quake.get_state @callee.num_qubits_0 @callee.init_0`:
  //
  // clang-format off
  // ```
  // func.func @caller(%arg0: !cc.ptr<!cc.state>) {
  //   %1 = quake.get_number_of_qubits %arg0: (!cc.ptr<!cc.state>) -> i64
  //   %2 = quake.alloca !quake.veq<?>[%1 : i64]
  //   %3 = quake.init_state %2, %arg0 : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
  //   return
  // }
  //
  // func.func private @callee(%arg0: i64) {
  //   %0 = quake.alloca !quake.veq<?>[%arg0 : i64]
  //   %1 = quake.extract_ref %0[0] : (!quake.veq<2>) -> !quake.ref
  //   quake.x %1 : (!quake.ref) -> ()
  //   return
  // }
  //
  // Call from the user host code:
  // state = cudaq.get_state(callee, 2)
  // counts = cudaq.sample(caller, state)
  // ```
  // clang-format on
  //
  // => after argument synthesis:
  //
  // clang-format off
  // ```
  // func.func @caller() {
  //   %0 = quake.get_state @callee.num_qubits_0 @callee.init_state_0 : !cc.ptr<!cc.state>
  //   %1 = quake.get_number_of_qubits %0 : (!cc.ptr<!cc.state>) -> i64
  //   %2 = quake.alloca !quake.veq<?>[%1 : i64]
  //   %3 = quake.init_state %2, %0 : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
  //   return
  // }
  //
  // func.func private @callee.num_qubits_0(%arg0: i64) -> i64 {
  //   return %arg0 : i64
  // }
  //
  // func.func private @callee.init_0(%arg0: i64, %arg1: !quake.veq<?>) {
  //   %1 = quake.extract_ref %arg0[0] : (!quake.veq<2>) -> !quake.ref
  //   quake.x %1 : (f64, !quake.ref) -> ()
  //   return
  // }
  // ```
  // clang-format on
  //
  // 3) (done in ReplaceStateWithKernel) Replace the `quake.get_state` and ops
  // that use its state with calls to the generated functions, synthesized with
  // the arguments used to create the original state:
  //
  // After ReplaceStateWithKernel pass:
  //
  // clang-format off
  // ```
  // func.func @caller() {
  //   %1 = call callee.num_qubits_0() : () -> i64
  //   %2 = quake.alloca !quake.veq<?>[%1 : i64]
  //   %3 = call @callee.init_0(%2): (!quake.veq<?>) -> !quake.veq<?>
  // }
  //
  // func.func private @callee.num_qubits_0() -> i64 {
  //   %cst = arith.constant 2 : i64
  //   return %cst : i64
  // }
  //
  // func.func private @callee.init_0(%arg0: !quake.veq<?>): !quake.veq<?> {
  //   %cst = arith.constant 1.5707963267948966 : f64
  //   %1 = quake.extract_ref %arg0[0] : (!quake.veq<2>) -> !quake.ref
  //   quake.ry (%cst) %1 : (f64, !quake.ref) -> ()
  //   return %arg0
  // }
  // ```
  // clang-format on
  if (simState->getKernelInfo().has_value()) {
    auto [calleeName, calleeArgs] = simState->getKernelInfo().value();

    std::string calleeKernelName =
        cudaq::runtime::cudaqGenPrefixName + calleeName;

    auto ctx = builder.getContext();

    auto code = cudaq::get_quake_by_name(calleeName, /*throwException=*/false);
    assert(!code.empty() && "Quake code not found for callee");
    auto fromModule = parseSourceString<ModuleOp>(code, ctx);

    auto calleeFunc = fromModule->lookupSymbol<func::FuncOp>(calleeKernelName);
    assert(calleeFunc && "callee func is missing");

    // TODO: use hash of arguments instead?
    auto counter = reinterpret_cast<std::size_t>(v);
    auto initName = calleeName + ".init_" + std::to_string(counter);
    auto numQubitsName =
        calleeName + ".num_qubits_" + std::to_string(counter++);

    if (!hasKernelInfo(initName) && !hasKernelInfo(numQubitsName)) {
      auto initKernelName = cudaq::runtime::cudaqGenPrefixName + initName;
      auto numQubitsKernelName =
          cudaq::runtime::cudaqGenPrefixName + numQubitsName;

      // Create `callee.init_N` and `callee.num_qubits_N` used for
      // `quake.get_state` replacement later in ReplaceStateWithKernel pass
      createInitFunc(builder, moduleOp, calleeFunc, initKernelName);
      createNumQubitsFunc(builder, moduleOp, calleeFunc, numQubitsKernelName);

      // Store the new kernel info in the aggregator
      addKernelInfo(initName, calleeArgs);
      addKernelInfo(numQubitsName, calleeArgs);

      // Collect kernel info from the callee state recursively
      collect(initName, calleeArgs);
      collect(numQubitsName, calleeArgs);
    }
    return;
  }

  TODO("cudaq::state* argument synthesis for quantum hardware for c functions");
}

//===----------------------------------------------------------------------===//

cudaq::opt::StateAggregator::StateAggregator(ModuleOp moduleOp)
    : moduleOp(moduleOp), builder(moduleOp.getContext()) {}

void cudaq::opt::StateAggregator::collect(
    StringRef kernelName, const std::vector<void *> &arguments) {
  auto *ctx = builder.getContext();

  auto fun = moduleOp.lookupSymbol<func::FuncOp>(
      cudaq::runtime::cudaqGenPrefixName + kernelName.str());
  assert(fun && "callee func is missing in state aggregator");

  FunctionType fromFuncTy = fun.getFunctionType();
  for (auto iter :
       llvm::enumerate(llvm::zip(fromFuncTy.getInputs(), arguments))) {
    void *argPtr = std::get<1>(iter.value());
    if (!argPtr)
      continue;
    Type argTy = std::get<0>(iter.value());

    if (auto ptrTy = dyn_cast<cc::PointerType>(argTy))
      if (ptrTy.getElementType() == cc::StateType::get(ctx))
        collectKernelInfo(static_cast<const state *>(argPtr));
  }
}
