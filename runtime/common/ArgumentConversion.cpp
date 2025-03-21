/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ArgumentConversion.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
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

template <typename A>
Value genIntegerConstant(OpBuilder &builder, A v, unsigned bits) {
  return builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), v, bits);
}

static Value genConstant(OpBuilder &builder, bool v) {
  return genIntegerConstant(builder, v, 1);
}
static Value genConstant(OpBuilder &builder, char v) {
  return genIntegerConstant(builder, v, 8);
}
static Value genConstant(OpBuilder &builder, std::int16_t v) {
  return genIntegerConstant(builder, v, 16);
}
static Value genConstant(OpBuilder &builder, std::int32_t v) {
  return genIntegerConstant(builder, v, 32);
}
static Value genConstant(OpBuilder &builder, std::int64_t v) {
  return genIntegerConstant(builder, v, 64);
}

static Value genConstant(OpBuilder &builder, float v) {
  return builder.create<arith::ConstantFloatOp>(
      builder.getUnknownLoc(), APFloat{v}, builder.getF32Type());
}
static Value genConstant(OpBuilder &builder, double v) {
  return builder.create<arith::ConstantFloatOp>(
      builder.getUnknownLoc(), APFloat{v}, builder.getF64Type());
}

template <typename A>
Value genComplexConstant(OpBuilder &builder, const std::complex<A> &v,
                         FloatType fTy) {
  auto rePart = builder.getFloatAttr(fTy, APFloat{v.real()});
  auto imPart = builder.getFloatAttr(fTy, APFloat{v.imag()});
  auto complexAttr = builder.getArrayAttr({rePart, imPart});
  auto loc = builder.getUnknownLoc();
  auto ty = ComplexType::get(fTy);
  return builder.create<complex::ConstantOp>(loc, ty, complexAttr).getResult();
}

static Value genConstant(OpBuilder &builder, std::complex<float> v) {
  return genComplexConstant(builder, v, builder.getF32Type());
}
static Value genConstant(OpBuilder &builder, std::complex<double> v) {
  return genComplexConstant(builder, v, builder.getF64Type());
}
static Value genConstant(OpBuilder &builder, FloatType fltTy, long double *v) {
  return builder.create<arith::ConstantFloatOp>(
      builder.getUnknownLoc(),
      APFloat{fltTy.getFloatSemantics(), std::to_string(*v)}, fltTy);
}

static Value genConstant(OpBuilder &builder, const std::string &v,
                         ModuleOp substMod) {
  auto loc = builder.getUnknownLoc();
  auto *ctx = builder.getContext();
  auto i8Ty = builder.getI8Type();
  auto strLitTy = cudaq::cc::PointerType::get(
      cudaq::cc::ArrayType::get(ctx, i8Ty, v.size() + 1));
  auto strLit =
      builder.create<cudaq::cc::CreateStringLiteralOp>(loc, strLitTy, v);
  auto i8PtrTy = cudaq::cc::PointerType::get(i8Ty);
  auto cast = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, strLit);
  auto size = builder.create<arith::ConstantIntOp>(loc, v.size(), 64);
  auto chSpanTy = cudaq::cc::CharspanType::get(ctx);
  return builder.create<cudaq::cc::StdvecInitOp>(loc, chSpanTy, cast, size);
}

// Forward declare aggregate type builder as they can be recursive.
static Value genConstant(OpBuilder &, cudaq::cc::StdvecType, void *,
                         ModuleOp substMod, llvm::DataLayout &);
static Value genConstant(OpBuilder &, cudaq::cc::StructType, void *,
                         ModuleOp substMod, llvm::DataLayout &);
static Value genConstant(OpBuilder &, cudaq::cc::ArrayType, void *,
                         ModuleOp substMod, llvm::DataLayout &);

/// Create callee.init_N that initializes the state
///
// clang-format off
/// Callee (the kernel captured by state):
/// func.func @callee(%arg0: i64) {
///   %2 = quake.alloca !quake.veq<?>[%arg0 : i64]
///   %3 = quake.extract_ref %2[1] : (!quake.veq<?>) -> !quake.ref
///   quake.x %3 : (!quake.ref) -> ()
///   return
/// }
///
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
  auto initFunc = cast<func::FuncOp>(builder.clone(*calleeFunc));
  auto loc = initFunc.getLoc();

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
        if (!allocSize)
          allocSize = newBuilder.create<arith::ConstantIntOp>(
              loc, quake::getAllocationSize(alloc.getType()), 64);

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
///
// clang-format off
/// Callee: (the kernel captured by state):
/// func.func @callee(%arg0: i64) {
///   %2 = quake.alloca !quake.veq<?>[%arg0 : i64]
///   %3 = quake.extract_ref %2[1] : (!quake.veq<?>) -> !quake.ref
///   quake.x %3 : (!quake.ref) -> ()
///   return
/// }
///
/// callee.num_qubits_0:
/// func.func private @callee.num_qubits_0(%arg0: i64) -> i64 {
///   return %arg0 : i64
/// }
// clang-format on
static void createNumQubitsFunc(OpBuilder &builder, ModuleOp moduleOp,
                                func::FuncOp calleeFunc,
                                StringRef numQubitsKernelName) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto ctx = builder.getContext();
  auto numQubitsFunc = cast<func::FuncOp>(builder.clone(*calleeFunc));
  auto loc = numQubitsFunc.getLoc();

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
        if (!allocSize)
          allocSize = newBuilder.create<arith::ConstantIntOp>(
              loc, quake::getAllocationSize(alloc.getType()), 64);
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
      if (std::find(keep.begin(), keep.end(), op) != keep.end())
        continue;

      keep.push_back(op);

      // Collect ops creating operands used in ops we already collected
      for (auto opnd : op->getOperands())
        if (auto defOp = opnd.getDefiningOp())
          used.push_back(defOp);

      // Collect ops that store into memory used in ops we already collected.
      for (auto user : op->getUsers())
        if (auto iface = dyn_cast<MemoryEffectOpInterface>(user))
          if (iface.hasEffect<MemoryEffects::Write>() &&
              !iface.hasEffect<MemoryEffects::Allocate>())
            used.push_back(user);
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

static Value genConstant(OpBuilder &builder, const cudaq::state *v,
                         llvm::DataLayout &layout, StringRef kernelName,
                         ModuleOp substMod,
                         cudaq::opt::ArgumentConverter &converter) {
  auto simState =
      cudaq::state_helper::getSimulationState(const_cast<cudaq::state *>(v));

  // If the state has amplitude data, we materialize the data as a state
  // vector and create a new state from it.
  if (simState->hasData()) {
    // The call below might cause lazy execution of the state kernel.
    // TODO: For lazy execution scenario on remote simulators, we have the
    // kernel info available on the state as well, before we needed to run
    // the state kernel and compute its data, which might cause significant
    // data transfer). Investigate if it is more performant to use the other
    // synthesis option in that case (see the next `if`).
    auto numQubits = v->get_num_qubits();

    // We currently only synthesize small states.
    if (numQubits > 14) {
      TODO("large (>14 qubit) cudaq::state* argument synthesis for simulators");
      return {};
    }

    auto size = 1ULL << numQubits;
    auto ctx = builder.getContext();
    auto loc = builder.getUnknownLoc();
    auto is64Bit =
        v->get_precision() == cudaq::SimulationState::precision::fp64;
    auto eleTy = is64Bit ? ComplexType::get(Float64Type::get(ctx))
                         : ComplexType::get(Float32Type::get(ctx));
    auto arrTy = cudaq::cc::ArrayType::get(ctx, eleTy, size);
    static unsigned counter = 0;
    auto ptrTy = cudaq::cc::PointerType::get(arrTy);

    cudaq::IRBuilder irBuilder(ctx);
    auto genConArray = [&]<typename T>() -> Value {
      SmallVector<std::complex<T>> vec(size);
      for (std::size_t i = 0; i < size; i++) {
        vec[i] = (*v)({i}, 0);
      }
      std::string name =
          kernelName.str() + ".rodata_synth_" + std::to_string(counter++);
      irBuilder.genVectorOfConstants(loc, substMod, name, vec);
      return builder.create<cudaq::cc::AddressOfOp>(loc, ptrTy, name);
    };

    auto buffer = is64Bit ? genConArray.template operator()<double>()
                          : genConArray.template operator()<float>();

    auto arrSize = builder.create<arith::ConstantIntOp>(loc, size, 64);
    auto stateTy = quake::StateType::get(ctx);
    auto statePtrTy = cudaq::cc::PointerType::get(stateTy);

    return builder.create<quake::CreateStateOp>(loc, statePtrTy, buffer,
                                                arrSize);
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
  // 2) (done here) Replace the state with
  //   `quake.get_state @callee.num_qubits_0 @callee.init_0`:
  //
  // clang-format off
  // ```
  // func.func @caller(%arg0: !cc.ptr<!quake.state>) {
  //   %1 = quake.get_number_of_qubits %arg0: (!cc.ptr<!quake.state>) -> i64
  //   %2 = quake.alloca !quake.veq<?>[%1 : i64]
  //   %3 = quake.init_state %2, %arg0 : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
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
  //   %0 = quake.get_state @callee.num_qubits_0 @callee.init_state_0 : !cc.ptr<!quake.state>
  //   %1 = quake.get_number_of_qubits %0 : (!cc.ptr<!quake.state>) -> i64
  //   %2 = quake.alloca !quake.veq<?>[%1 : i64]
  //   %3 = quake.init_state %2, %0 : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
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
    auto loc = builder.getUnknownLoc();

    auto code = cudaq::get_quake_by_name(calleeName, /*throwException=*/false);
    assert(!code.empty() && "Quake code not found for callee");
    auto fromModule = parseSourceString<ModuleOp>(code, ctx);

    auto calleeFunc = fromModule->lookupSymbol<func::FuncOp>(calleeKernelName);
    assert(calleeFunc && "callee func is missing");

    // Use the state pointer as hash to look up the function name
    // that was created using the same hash in StateAggregator.
    auto hash = std::to_string(reinterpret_cast<std::size_t>(v));
    auto initName = calleeName + ".init_" + hash;
    auto numQubitsName = calleeName + ".num_qubits_" + hash;
    auto initKernelName = cudaq::runtime::cudaqGenPrefixName + initName;
    auto numQubitsKernelName =
        cudaq::runtime::cudaqGenPrefixName + numQubitsName;

    // Create `callee.init_N` and `callee.num_qubits_N` used to replace
    // `quake.materialize_state` in ReplaceStateWithKernel pass
    if (!converter.isRegisteredKernel(initName) ||
        !converter.isRegisteredKernel(numQubitsName)) {
      createInitFunc(builder, substMod, calleeFunc, initKernelName);
      createNumQubitsFunc(builder, substMod, calleeFunc, numQubitsKernelName);

      // Convert arguments for `callee.init_N`.
      auto registeredInitName = converter.registerKernel(initName);
      converter.gen(registeredInitName, substMod, calleeArgs);

      // Convert arguments for `callee.num_qubits_N`.
      auto registeredNumQubitsName = converter.registerKernel(numQubitsName);
      converter.gen(registeredNumQubitsName, substMod, calleeArgs);
    }

    // Create a substitution for the state pointer.
    auto statePtrTy = cudaq::cc::PointerType::get(quake::StateType::get(ctx));
    return builder.create<quake::MaterializeStateOp>(
        loc, statePtrTy, builder.getStringAttr(numQubitsKernelName),
        builder.getStringAttr(initKernelName));
  }

  TODO("cudaq::state* argument synthesis for quantum hardware for c functions");
  return {};
}

// Recursive step processing of aggregates.
Value dispatchSubtype(OpBuilder &builder, Type ty, void *p, ModuleOp substMod,
                      llvm::DataLayout &layout) {
  auto *ctx = builder.getContext();
  return TypeSwitch<Type, Value>(ty)
      .Case([&](IntegerType intTy) -> Value {
        switch (intTy.getIntOrFloatBitWidth()) {
        case 1:
          return genConstant(builder, *static_cast<bool *>(p));
        case 8:
          return genConstant(builder, *static_cast<char *>(p));
        case 16:
          return genConstant(builder, *static_cast<std::int16_t *>(p));
        case 32:
          return genConstant(builder, *static_cast<std::int32_t *>(p));
        case 64:
          return genConstant(builder, *static_cast<std::int64_t *>(p));
        default:
          return {};
        }
      })
      .Case([&](Float32Type fltTy) {
        return genConstant(builder, *static_cast<float *>(p));
      })
      .Case([&](Float64Type fltTy) {
        return genConstant(builder, *static_cast<double *>(p));
      })
      .Case([&](FloatType fltTy) {
        assert(fltTy.getIntOrFloatBitWidth() > 64);
        return genConstant(builder, fltTy, static_cast<long double *>(p));
      })
      .Case([&](ComplexType cmplxTy) -> Value {
        if (cmplxTy.getElementType() == Float32Type::get(ctx))
          return genConstant(builder, *static_cast<std::complex<float> *>(p));
        if (cmplxTy.getElementType() == Float64Type::get(ctx))
          return genConstant(builder, *static_cast<std::complex<double> *>(p));
        return {};
      })
      .Case([&](cudaq::cc::CharspanType strTy) {
        return genConstant(builder, static_cast<cudaq::pauli_word *>(p)->str(),
                           substMod);
      })
      .Case([&](cudaq::cc::StdvecType ty) {
        return genConstant(builder, ty, p, substMod, layout);
      })
      .Case([&](cudaq::cc::StructType ty) {
        return genConstant(builder, ty, p, substMod, layout);
      })
      .Case([&](cudaq::cc::ArrayType ty) {
        return genConstant(builder, ty, p, substMod, layout);
      })
      .Default({});
}

// Get the size of \p eleTy on the host side in bytes.
static std::size_t getHostSideElementSize(Type eleTy,
                                          llvm::DataLayout &layout) {
  if (isa<cudaq::cc::StdvecType>(eleTy))
    return sizeof(std::vector<int>);
  if (isa<cudaq::cc::CharspanType>(eleTy)) {
    // char span type is a std::string on host side.
    return sizeof(std::string);
  }
  // Note: we want the size on the host side, but `getDataSize()` returns the
  // size on the device side. This is ok for now since they are the same for
  // most types and the special cases are handled above.
  return cudaq::opt::getDataSize(layout, eleTy);
}

Value genConstant(OpBuilder &builder, cudaq::cc::StdvecType vecTy, void *p,
                  ModuleOp substMod, llvm::DataLayout &layout) {
  typedef const char *VectorType[3];
  VectorType *vecPtr = static_cast<VectorType *>(p);
  auto delta = (*vecPtr)[1] - (*vecPtr)[0];
  if (!delta)
    return {};
  auto eleTy = vecTy.getElementType();
  auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
  auto eleSize = getHostSideElementSize(eleTy, layout);

  assert(eleSize && "element must have a size");
  auto loc = builder.getUnknownLoc();
  std::int32_t vecSize = delta / eleSize;
  auto eleArrTy =
      cudaq::cc::ArrayType::get(builder.getContext(), eleTy, vecSize);
  auto buffer = builder.create<cudaq::cc::AllocaOp>(loc, eleArrTy);
  const char *cursor = (*vecPtr)[0];
  for (std::int32_t i = 0; i < vecSize; ++i) {
    if (Value val = dispatchSubtype(
            builder, eleTy, static_cast<void *>(const_cast<char *>(cursor)),
            substMod, layout)) {
      auto atLoc = builder.create<cudaq::cc::ComputePtrOp>(
          loc, elePtrTy, buffer, ArrayRef<cudaq::cc::ComputePtrArg>{i});
      builder.create<cudaq::cc::StoreOp>(loc, val, atLoc);
    }
    cursor += eleSize;
  }
  auto size = builder.create<arith::ConstantIntOp>(loc, vecSize, 64);
  return builder.create<cudaq::cc::StdvecInitOp>(loc, vecTy, buffer, size);
}

Value genConstant(OpBuilder &builder, cudaq::cc::StructType strTy, void *p,
                  ModuleOp substMod, llvm::DataLayout &layout) {
  if (strTy.getMembers().empty())
    return {};
  const char *cursor = static_cast<const char *>(p);
  auto loc = builder.getUnknownLoc();
  Value aggie = builder.create<cudaq::cc::UndefOp>(loc, strTy);
  for (auto iter : llvm::enumerate(strTy.getMembers())) {
    auto i = iter.index();
    if (Value v = dispatchSubtype(
            builder, iter.value(),
            static_cast<void *>(const_cast<char *>(
                cursor + cudaq::opt::getDataOffset(layout, strTy, i))),
            substMod, layout))
      aggie = builder.create<cudaq::cc::InsertValueOp>(loc, strTy, aggie, v, i);
  }
  return aggie;
}

Value genConstant(OpBuilder &builder, cudaq::cc::ArrayType arrTy, void *p,
                  ModuleOp substMod, llvm::DataLayout &layout) {
  if (arrTy.isUnknownSize())
    return {};
  auto eleTy = arrTy.getElementType();
  auto loc = builder.getUnknownLoc();
  auto eleSize = cudaq::opt::getDataSize(layout, eleTy);
  Value aggie = builder.create<cudaq::cc::UndefOp>(loc, arrTy);
  std::size_t arrSize = arrTy.getSize();
  const char *cursor = static_cast<const char *>(p);
  for (std::size_t i = 0; i < arrSize; ++i) {
    if (Value v = dispatchSubtype(
            builder, eleTy, static_cast<void *>(const_cast<char *>(cursor)),
            substMod, layout))
      aggie = builder.create<cudaq::cc::InsertValueOp>(loc, arrTy, aggie, v, i);
    cursor += eleSize;
  }
  return aggie;
}

Value genConstant(OpBuilder &builder, cudaq::cc::IndirectCallableType indCallTy,
                  void *p, ModuleOp sourceMod, ModuleOp substMod,
                  llvm::DataLayout &layout) {
  auto key = cudaq::registry::__cudaq_getLinkableKernelKey(p);
  auto *name = cudaq::registry::getLinkableKernelNameOrNull(key);
  if (!name)
    return {};
  auto code = cudaq::get_quake_by_name(name, /*throwException=*/false);
  auto *ctx = builder.getContext();
  auto fromModule = parseSourceString<ModuleOp>(code, ctx);
  OpBuilder cloneBuilder(ctx);
  cloneBuilder.setInsertionPointToStart(substMod.getBody());
  for (auto &i : *fromModule->getBody()) {
    auto s = dyn_cast_if_present<SymbolOpInterface>(i);
    if (!s || sourceMod.lookupSymbol(s.getNameAttr()) ||
        substMod.lookupSymbol(s.getNameAttr()))
      continue;
    auto clone = cloneBuilder.clone(i);
    cast<SymbolOpInterface>(clone).setPrivate();
  }
  auto loc = builder.getUnknownLoc();
  auto func = builder.create<func::ConstantOp>(
      loc, indCallTy.getSignature(),
      std::string{cudaq::runtime::cudaqGenPrefixName} + name);
  return builder.create<cudaq::cc::CastOp>(loc, indCallTy, func);
}

//===----------------------------------------------------------------------===//

cudaq::opt::ArgumentConverter::ArgumentConverter(StringRef kernelName,
                                                 ModuleOp sourceModule)
    : sourceModule(sourceModule), kernelName(kernelName) {}

void cudaq::opt::ArgumentConverter::gen(const std::vector<void *> &arguments) {
  gen(kernelName, sourceModule, arguments);
}

void cudaq::opt::ArgumentConverter::gen(StringRef kernelName,
                                        ModuleOp sourceModule,
                                        const std::vector<void *> &arguments) {
  auto *ctx = sourceModule.getContext();
  OpBuilder builder(ctx);
  ModuleOp substModule =
      builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  auto *kernelInfo = addKernelInfo(kernelName, substModule);

  // We should look up the input type signature here.
  auto fun = sourceModule.lookupSymbol<func::FuncOp>(
      cudaq::runtime::cudaqGenPrefixName + kernelName.str());
  if (!fun)
    throw std::runtime_error("missing fun in argument conversion: " +
                             kernelName.str());

  FunctionType fromFuncTy = fun.getFunctionType();
  for (auto iter :
       llvm::enumerate(llvm::zip(fromFuncTy.getInputs(), arguments))) {
    void *argPtr = std::get<1>(iter.value());
    if (!argPtr)
      continue;
    Type argTy = std::get<0>(iter.value());
    unsigned i = iter.index();
    auto buildSubst = [&, i = i]<typename... Ts>(Ts &&...ts) {
      builder.setInsertionPointToEnd(substModule.getBody());
      auto loc = builder.getUnknownLoc();
      auto result = builder.create<cc::ArgumentSubstitutionOp>(loc, i);
      auto *block = new Block();
      result.getBody().push_back(block);
      builder.setInsertionPointToEnd(block);
      [[maybe_unused]] auto val = genConstant(builder, std::forward<Ts>(ts)...);
      return result;
    };

    StringRef dataLayoutSpec = "";
    if (auto attr = sourceModule->getAttr(
            cudaq::opt::factory::targetDataLayoutAttrName))
      dataLayoutSpec = cast<StringAttr>(attr);
    llvm::DataLayout dataLayout{dataLayoutSpec};

    auto subst =
        TypeSwitch<Type, cc::ArgumentSubstitutionOp>(argTy)
            .Case([&](IntegerType intTy) -> cc::ArgumentSubstitutionOp {
              switch (intTy.getIntOrFloatBitWidth()) {
              case 1:
                return buildSubst(*static_cast<bool *>(argPtr));
              case 8:
                return buildSubst(*static_cast<char *>(argPtr));
              case 16:
                return buildSubst(*static_cast<std::int16_t *>(argPtr));
              case 32:
                return buildSubst(*static_cast<std::int32_t *>(argPtr));
              case 64:
                return buildSubst(*static_cast<std::int64_t *>(argPtr));
              default:
                return {};
              }
            })
            .Case([&](Float32Type fltTy) {
              return buildSubst(*static_cast<float *>(argPtr));
            })
            .Case([&](Float64Type fltTy) {
              return buildSubst(*static_cast<double *>(argPtr));
            })
            .Case([&](FloatType fltTy) {
              assert(fltTy.getIntOrFloatBitWidth() > 64);
              return buildSubst(fltTy, static_cast<long double *>(argPtr));
            })
            .Case([&](ComplexType cmplxTy) -> cc::ArgumentSubstitutionOp {
              if (cmplxTy.getElementType() == Float32Type::get(ctx))
                return buildSubst(*static_cast<std::complex<float> *>(argPtr));
              if (cmplxTy.getElementType() == Float64Type::get(ctx))
                return buildSubst(*static_cast<std::complex<double> *>(argPtr));
              return {};
            })
            .Case([&](cc::CharspanType strTy) {
              return buildSubst(static_cast<cudaq::pauli_word *>(argPtr)->str(),
                                substModule);
            })
            .Case([&](cc::PointerType ptrTy) -> cc::ArgumentSubstitutionOp {
              if (ptrTy.getElementType() == quake::StateType::get(ctx))
                return buildSubst(static_cast<const state *>(argPtr),
                                  dataLayout, kernelName, substModule, *this);
              return {};
            })
            .Case([&](cc::StdvecType ty) {
              return buildSubst(ty, argPtr, substModule, dataLayout);
            })
            .Case([&](cc::StructType ty) {
              return buildSubst(ty, argPtr, substModule, dataLayout);
            })
            .Case([&](cc::ArrayType ty) {
              return buildSubst(ty, argPtr, substModule, dataLayout);
            })
            .Case([&](cc::IndirectCallableType ty) {
              return buildSubst(ty, argPtr, sourceModule, substModule,
                                dataLayout);
            })
            .Default({});
    if (subst)
      kernelInfo->getSubstitutions().emplace_back(std::move(subst));
  }
}

void cudaq::opt::ArgumentConverter::gen(
    const std::vector<void *> &arguments,
    const std::unordered_set<unsigned> &exclusions) {
  std::vector<void *> partialArgs;
  for (auto iter : llvm::enumerate(arguments)) {
    if (exclusions.contains(iter.index())) {
      partialArgs.push_back(nullptr);
      continue;
    }
    partialArgs.push_back(iter.value());
  }
  gen(partialArgs);
}

void cudaq::opt::ArgumentConverter::gen_drop_front(
    const std::vector<void *> &arguments, unsigned numDrop) {
  // If we're dropping all the arguments, we're done.
  if (numDrop >= arguments.size())
    return;
  std::vector<void *> partialArgs;
  int drop = numDrop;
  for (void *arg : arguments) {
    if (drop > 0) {
      drop--;
      partialArgs.push_back(nullptr);
      continue;
    }
    partialArgs.push_back(arg);
  }
  gen(partialArgs);
}
