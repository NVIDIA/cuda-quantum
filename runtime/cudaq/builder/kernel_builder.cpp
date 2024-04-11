/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "kernel_builder.h"
#include "common/Logger.h"
#include "common/RuntimeMLIR.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

#include <numeric>

using namespace mlir;

extern "C" {
void altLaunchKernel(const char *kernelName, void (*kernelFunc)(void *),
                     void *kernelArgs, std::uint64_t argsSize);
}

namespace cudaq::details {

/// @brief Track unique measurement register names.
static std::size_t regCounter = 0;

KernelBuilderType convertArgumentTypeToMLIR(double &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) { return Float64Type::get(ctx); });
}

KernelBuilderType convertArgumentTypeToMLIR(float &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) { return Float32Type::get(ctx); });
}

KernelBuilderType convertArgumentTypeToMLIR(int &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) { return IntegerType::get(ctx, 32); });
}

KernelBuilderType convertArgumentTypeToMLIR(std::vector<double> &e) {
  return KernelBuilderType([](MLIRContext *ctx) {
    return cudaq::cc::StdvecType::get(ctx, Float64Type::get(ctx));
  });
}

KernelBuilderType convertArgumentTypeToMLIR(std::size_t &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) { return IntegerType::get(ctx, 64); });
}

KernelBuilderType convertArgumentTypeToMLIR(std::vector<int> &e) {
  return KernelBuilderType([](MLIRContext *ctx) {
    return cudaq::cc::StdvecType::get(ctx, mlir::IntegerType::get(ctx, 32));
  });
}

KernelBuilderType convertArgumentTypeToMLIR(std::vector<std::size_t> &e) {
  return KernelBuilderType([](MLIRContext *ctx) {
    return cudaq::cc::StdvecType::get(ctx, mlir::IntegerType::get(ctx, 64));
  });
}

/// Map a std::vector<float> to a KernelBuilderType
KernelBuilderType convertArgumentTypeToMLIR(std::vector<float> &e) {
  return KernelBuilderType([](MLIRContext *ctx) {
    return cudaq::cc::StdvecType::get(ctx, Float32Type::get(ctx));
  });
}

/// Map a std::vector<complex<double>> to a KernelBuilderType
KernelBuilderType
convertArgumentTypeToMLIR(std::vector<std::complex<double>> &e) {
  return KernelBuilderType([](MLIRContext *ctx) {
    return cudaq::cc::StdvecType::get(ctx,
                                      ComplexType::get(Float64Type::get(ctx)));
  });
}

/// Map a std::vector<complex<float>> to a KernelBuilderType
KernelBuilderType
convertArgumentTypeToMLIR(std::vector<std::complex<float>> &e) {
  return KernelBuilderType([](MLIRContext *ctx) {
    return cudaq::cc::StdvecType::get(ctx,
                                      ComplexType::get(Float32Type::get(ctx)));
  });
}

KernelBuilderType convertArgumentTypeToMLIR(cudaq::qubit &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) { return quake::RefType::get(ctx); });
}

KernelBuilderType convertArgumentTypeToMLIR(cudaq::qvector<> &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) { return quake::VeqType::getUnsized(ctx); });
}

MLIRContext *initializeContext() {
  cudaq::info("Initializing the MLIR infrastructure.");
  return cudaq::initializeMLIR().release();
}
void deleteContext(MLIRContext *context) { delete context; }
void deleteJitEngine(ExecutionEngine *jit) { delete jit; }

ImplicitLocOpBuilder *
initializeBuilder(MLIRContext *context,
                  std::vector<KernelBuilderType> &inputTypes,
                  std::vector<QuakeValue> &arguments, std::string &kernelName) {
  cudaq::info("Creating the MLIR ImplicitOpBuilder.");

  auto location = FileLineColLoc::get(context, "<builder>", 1, 1);
  auto *opBuilder = new ImplicitLocOpBuilder(location, context);

  auto moduleOp = opBuilder->create<ModuleOp>();
  opBuilder->setInsertionPointToEnd(moduleOp.getBody());

  // Convert our cudaq builder Types into mlir::Types
  std::vector<Type> types;
  std::transform(inputTypes.begin(), inputTypes.end(),
                 std::back_inserter(types),
                 [&](auto &&element) { return element.create(context); });

  // Make the kernel_name unique,
  std::ostringstream os;
  for (int i = 0; i < 12; ++i) {
    int digit = rand() % 10;
    os << digit;
  }

  kernelName += fmt::format("_{}", os.str());
  cudaq::info("kernel_builder name set to {}", kernelName);

  FunctionType funcTy = opBuilder->getFunctionType(types, std::nullopt);
  auto kernel = opBuilder->create<func::FuncOp>(kernelName, funcTy);
  auto *entryBlock = kernel.addEntryBlock();

  for (auto arg : entryBlock->getArguments())
    arguments.emplace_back(*opBuilder, arg);

  cudaq::info("kernel_builder has {} arguments", arguments.size());

  // Every Kernel should have a ReturnOp terminator, then we'll set the
  // insertion point to right before it.
  opBuilder->setInsertionPointToStart(entryBlock);
  auto terminator = opBuilder->create<func::ReturnOp>();
  opBuilder->setInsertionPoint(terminator);
  return opBuilder;
}
void deleteBuilder(ImplicitLocOpBuilder *builder) { delete builder; }

bool isArgStdVec(std::vector<QuakeValue> &args, std::size_t idx) {
  return args[idx].isStdVec();
}

void exp_pauli(ImplicitLocOpBuilder &builder, const QuakeValue &theta,
               const std::vector<QuakeValue> &qubits,
               const std::string &pauliWord) {
  Value qubitsVal;
  if (qubits.size() == 1)
    qubitsVal = qubits.front().getValue();
  else {
    // we have a vector of quake value qubits, need to concat them
    SmallVector<Value> values;
    for (auto &v : qubits)
      values.push_back(v.getValue());

    qubitsVal = builder.create<quake::ConcatOp>(
        quake::VeqType::get(builder.getContext(), qubits.size()), values);
  }

  auto thetaVal = theta.getValue();
  if (!isa<quake::VeqType>(qubitsVal.getType()))
    throw std::runtime_error(
        "exp_pauli must take a QuakeValue of veq type as second argument.");
  if (!thetaVal.getType().isIntOrFloat())
    throw std::runtime_error("exp_pauli must take a QuakeValue of float/int "
                             "type as first argument.");
  cudaq::info("kernel_builder apply exp_pauli {}", pauliWord);

  auto strLitTy = cc::PointerType::get(cc::ArrayType::get(
      builder.getContext(), builder.getI8Type(), pauliWord.size() + 1));
  Value stringLiteral = builder.create<cc::CreateStringLiteralOp>(
      strLitTy, builder.getStringAttr(pauliWord));
  SmallVector<Value> args{thetaVal, qubitsVal, stringLiteral};
  builder.create<quake::ExpPauliOp>(TypeRange{}, args);
}

/// @brief Search the given `FuncOp` for all `CallOps` recursively.
/// If found, see if the called function is in the current `ModuleOp` for this
/// `kernel_builder`, if so do nothing. If it is not found, then find it in the
/// other `ModuleOp`, clone it, and add it to this `ModuleOp`.
void addAllCalledFunctionRecursively(
    func::FuncOp &function, ModuleOp &currentModule,
    mlir::OwningOpRef<mlir::ModuleOp> &otherModule) {

  std::function<void(func::FuncOp func)> visitAllCallOps;
  visitAllCallOps = [&](func::FuncOp func) {
    func.walk([&](Operation *op) {
      // Check if this is a CallOp or ApplyOp
      StringRef calleeName;
      if (auto callOp = dyn_cast<func::CallOp>(op))
        calleeName = callOp.getCallee();
      else if (auto applyOp = dyn_cast<quake::ApplyOp>(op))
        calleeName = applyOp.getCalleeAttrNameStr();

      // We don't have a CallOp or an ApplyOp, drop out
      if (calleeName.empty())
        return WalkResult::skip();

      // Don't add if we already have it
      if (currentModule.lookupSymbol<func::FuncOp>(calleeName))
        return WalkResult::skip();

      // Get the called function, make sure it exists
      auto calledFunction = otherModule->lookupSymbol<func::FuncOp>(calleeName);
      if (!calledFunction)
        throw std::runtime_error(
            "Invalid called function, cannot find in ModuleOp (" +
            calleeName.str() + ")");

      // Add the called function to the list
      auto cloned = calledFunction.clone();
      // Remove entrypoint attribute if it exists
      cloned->removeAttr(cudaq::entryPointAttrName);
      currentModule.push_back(cloned);

      // Visit that new function and see if we have
      // more call operations.
      visitAllCallOps(cloned);

      // Once done, return.
      return WalkResult::advance();
    });
  };

  // Collect all called functions
  visitAllCallOps(function);
}

/// @brief Get a the function with the given name. First look in the current
/// `ModuleOp` for this `kernel_builder`, if found return it as is. If not
/// found, find it in the other `kernel_builder` `ModuleOp` and return a clone
/// of it. Throw an exception if no kernel with the given name is found
func::FuncOp
cloneOrGetFunction(StringRef name, ModuleOp &currentModule,
                   mlir::OwningOpRef<mlir::ModuleOp> &otherModule) {
  if (auto func = currentModule.lookupSymbol<func::FuncOp>(name))
    return func;

  if (auto func = otherModule->lookupSymbol<func::FuncOp>(name)) {
    auto cloned = func.clone();
    // Remove entrypoint attribute if it exists
    cloned->removeAttr(cudaq::entryPointAttrName);
    currentModule.push_back(cloned);
    return cloned;
  }

  throw std::runtime_error("Could not find function with name " + name.str());
}

void call(ImplicitLocOpBuilder &builder, std::string &name,
          std::string &quakeCode, std::vector<QuakeValue> &values) {
  // Create a ModuleOp from the other kernel's quake code
  auto otherModule =
      mlir::parseSourceString<mlir::ModuleOp>(quakeCode, builder.getContext());

  // Get our current module
  auto block = builder.getBlock();
  auto function = block->getParentOp();
  auto currentModule = function->getParentOfType<ModuleOp>();

  // We need to clone the function we care about, we need any other functions it
  // calls, so store it in a vector
  std::vector<func::FuncOp> functions;

  // Get the function with the kernel name we care about.
  auto properName = std::string(cudaq::runtime::cudaqGenPrefixName) + name;
  auto otherFuncCloned =
      cloneOrGetFunction(properName, currentModule, otherModule);

  // We need to recursively find all CallOps and add their Callee FuncOps to the
  // current Module
  addAllCalledFunctionRecursively(otherFuncCloned, currentModule, otherModule);

  // Map the QuakeValues to MLIR Values
  SmallVector<Value> mlirValues;
  for (std::size_t i = 0; auto &v : values) {
    Type argType = otherFuncCloned.getArgumentTypes()[i++];
    Value value = v.getValue();
    Type inType = value.getType();
    auto inAsVeqTy = inType.dyn_cast_or_null<quake::VeqType>();
    auto argAsVeqTy = argType.dyn_cast_or_null<quake::VeqType>();

    // If both are veqs, make sure we dont have veq<N> -> veq<?>
    if (inAsVeqTy && argAsVeqTy) {
      // make sure they are both the same veq<...> type
      if (inAsVeqTy.hasSpecifiedSize() && !argAsVeqTy.hasSpecifiedSize())
        value = builder.create<quake::RelaxSizeOp>(argAsVeqTy, value);
    } else if (inType != argType) {
      std::string inS, argS;
      {
        llvm::raw_string_ostream inOs(inS), argOs(argS);
        inType.print(inOs);
        argType.print(argOs);
      }
      throw std::runtime_error("Invalid argument type passed to kernel call (" +
                               inS + " != " + argS + ").");
    }

    mlirValues.push_back(value);
  }

  // Hook up the call op
  builder.create<func::CallOp>(otherFuncCloned, mlirValues);
}

void applyControlOrAdjoint(ImplicitLocOpBuilder &builder, std::string &name,
                           std::string &quakeCode, bool isAdjoint,
                           ValueRange controls,
                           std::vector<QuakeValue> &values) {
  // Create a ModuleOp from the other kernel's quake code
  auto otherModule =
      mlir::parseSourceString<mlir::ModuleOp>(quakeCode, builder.getContext());

  // Get our current module
  auto block = builder.getBlock();
  auto function = block->getParentOp();
  auto currentModule = function->getParentOfType<ModuleOp>();

  // Get the function with the kernel name we care about.
  auto properName = std::string(cudaq::runtime::cudaqGenPrefixName) + name;
  auto otherFuncCloned =
      cloneOrGetFunction(properName, currentModule, otherModule);

  // We need to recursively find all CallOps and add their Callee FuncOps to the
  // current Module
  addAllCalledFunctionRecursively(otherFuncCloned, currentModule, otherModule);

  SmallVector<Value> mlirValues;
  for (std::size_t i = 0; auto &v : values) {
    Type argType = otherFuncCloned.getArgumentTypes()[i];
    Value value = v.getValue();
    Type inType = value.getType();
    auto inAsVeqTy = inType.dyn_cast_or_null<quake::VeqType>();
    auto argAsVeqTy = argType.dyn_cast_or_null<quake::VeqType>();

    // If both are veqs, make sure we dont have veq<N> -> veq<?>
    if (inAsVeqTy && argAsVeqTy) {
      // make sure they are both the same veq<...> type
      if (inAsVeqTy.hasSpecifiedSize() && !argAsVeqTy.hasSpecifiedSize())
        value = builder.create<quake::RelaxSizeOp>(argAsVeqTy, value);
    } else if (inType != argType) {
      std::string inS, argS;
      {
        llvm::raw_string_ostream inOs(inS), argOs(argS);
        inType.print(inOs);
        argType.print(argOs);
      }
      throw std::runtime_error("Invalid argument type passed to kernel call (" +
                               inS + " != " + argS + ").");
    }

    mlirValues.push_back(value);
  }

  auto realName = std::string(cudaq::runtime::cudaqGenPrefixName) + name;
  builder.create<quake::ApplyOp>(
      TypeRange{}, SymbolRefAttr::get(builder.getContext(), realName),
      isAdjoint, controls, mlirValues);
}

void control(ImplicitLocOpBuilder &builder, std::string &name,
             std::string &quakeCode, QuakeValue &control,
             std::vector<QuakeValue> &values) {
  applyControlOrAdjoint(builder, name, quakeCode, /*isAdjoint*/ false,
                        control.getValue(), values);
}

void adjoint(ImplicitLocOpBuilder &builder, std::string &name,
             std::string &quakeCode, std::vector<QuakeValue> &values) {
  applyControlOrAdjoint(builder, name, quakeCode, /*isAdjoint*/ true, {},
                        values);
}

void forLoop(ImplicitLocOpBuilder &builder, Value &startVal, Value &end,
             std::function<void(QuakeValue &)> &body) {
  auto i64Ty = builder.getI64Type();
  Value castEnd = builder.create<cudaq::cc::CastOp>(
      i64Ty, end, cudaq::cc::CastOpMode::Unsigned);
  Value castStart = builder.create<cudaq::cc::CastOp>(
      i64Ty, startVal, cudaq::cc::CastOpMode::Unsigned);
  Value totalIters = builder.create<arith::SubIOp>(i64Ty, castEnd, castStart);
  cudaq::opt::factory::createInvariantLoop(
      builder, builder.getLoc(), totalIters,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Region &,
          Block &block) {
        Value iv = block.getArgument(0);
        // shift iv -> iv + start
        iv = builder.create<arith::AddIOp>(iv.getType(), iv, castStart);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        QuakeValue idxQuakeVal(builder, iv);
        body(idxQuakeVal);
      });
}

void forLoop(ImplicitLocOpBuilder &builder, QuakeValue &startVal,
             QuakeValue &end, std::function<void(QuakeValue &)> &body) {
  auto s = startVal.getValue();
  auto e = startVal.getValue();
  forLoop(builder, s, e, body);
}

void forLoop(ImplicitLocOpBuilder &builder, std::size_t start, std::size_t end,
             std::function<void(QuakeValue &)> &body) {
  Value startVal = builder.create<arith::ConstantIntOp>(start, 64);
  Value endVal = builder.create<arith::ConstantIntOp>(end, 64);
  forLoop(builder, startVal, endVal, body);
}

void forLoop(ImplicitLocOpBuilder &builder, std::size_t start, QuakeValue &end,
             std::function<void(QuakeValue &)> &body) {
  Value startVal = builder.create<arith::ConstantIntOp>(start, 64);
  auto e = end.getValue();
  forLoop(builder, startVal, e, body);
}

void forLoop(ImplicitLocOpBuilder &builder, QuakeValue &start, std::size_t end,
             std::function<void(QuakeValue &)> &body) {
  Value e = builder.create<arith::ConstantIntOp>(end, 64);
  auto s = start.getValue();
  forLoop(builder, s, e, body);
}

KernelBuilderType::KernelBuilderType(
    std::function<mlir::Type(MLIRContext *ctx)> &&f)
    : creator(f) {}

Type KernelBuilderType::create(MLIRContext *ctx) { return creator(ctx); }

QuakeValue qalloc(ImplicitLocOpBuilder &builder) {
  cudaq::info("kernel_builder allocating a single qubit");
  Value qubit = builder.create<quake::AllocaOp>();
  return QuakeValue(builder, qubit);
}

QuakeValue qalloc(ImplicitLocOpBuilder &builder, const std::size_t nQubits) {
  cudaq::info("kernel_builder allocating {} qubits", nQubits);

  auto context = builder.getContext();
  Value qubits =
      builder.create<quake::AllocaOp>(quake::VeqType::get(context, nQubits));

  return QuakeValue(builder, qubits);
}

QuakeValue qalloc(ImplicitLocOpBuilder &builder, QuakeValue &sizeOrVec) {
  cudaq::info("kernel_builder allocating qubits from quake value");
  auto value = sizeOrVec.getValue();
  auto type = value.getType();
  auto context = builder.getContext();

  if (auto stdvecTy = dyn_cast<cc::StdvecType>(type)) {
    // get the size
    Value size = builder.create<cc::StdvecSizeOp>(builder.getI64Type(), value);
    Value numQubits = builder.create<math::CountTrailingZerosOp>(size);
    auto veqTy = quake::VeqType::getUnsized(context);
    // allocate the number of qubits we need
    Value qubits = builder.create<quake::AllocaOp>(veqTy, numQubits);

    auto ptrTy = cc::PointerType::get(stdvecTy.getElementType());
    Value initials = builder.create<cc::StdvecDataOp>(ptrTy, value);
    builder.create<quake::InitializeStateOp>(veqTy, qubits, initials);
    return QuakeValue(builder, qubits);
  }

  if (!type.isIntOrIndex())
    throw std::runtime_error(
        "Invalid parameter passed to qalloc (must be integer type).");

  Value qubits = builder.create<quake::AllocaOp>(
      quake::VeqType::getUnsized(context), value);

  return QuakeValue(builder, qubits);
}

template <typename A>
std::size_t getStateVectorLength(StateVectorStorage &stateVectorStorage,
                                 std::int64_t index) {
  if (index >= static_cast<std::int64_t>(stateVectorStorage.size()))
    throw std::runtime_error("index to state initializer is out of range");
  if (!std::get<std::vector<std::complex<A>> *>(stateVectorStorage[index]))
    throw std::runtime_error("state vector cannot be null");
  auto length =
      std::get<std::vector<std::complex<A>> *>(stateVectorStorage[index])
          ->size();
  if (!std::has_single_bit(length))
    throw std::runtime_error("state initializer must be a power of 2");
  return std::countr_zero(length);
}

template <typename A>
std::complex<A> *getStateVectorData(StateVectorStorage &stateVectorStorage,
                                    std::intptr_t index) {
  // This foregoes all the checks found in getStateVectorLength because these
  // two functions are called in tandem, this one second.
  return std::get<std::vector<std::complex<A>> *>(stateVectorStorage[index])
      ->data();
}

extern "C" {
/// Runtime callback to get the log2(size) of a captured state vector.
std::size_t
__nvqpp_getStateVectorLength_fp64(StateVectorStorage &stateVectorStorage,
                                  std::int64_t index) {
  return getStateVectorLength<double>(stateVectorStorage, index);
}

std::size_t
__nvqpp_getStateVectorLength_fp32(StateVectorStorage &stateVectorStorage,
                                  std::int64_t index) {
  return getStateVectorLength<float>(stateVectorStorage, index);
}

/// Runtime callback to get the data array of a captured state vector.
std::complex<double> *
__nvqpp_getStateVectorData_fp64(StateVectorStorage &stateVectorStorage,
                                std::intptr_t index) {
  return getStateVectorData<double>(stateVectorStorage, index);
}

/// Runtime callback to get the data array of a captured state vector.
std::complex<float> *
__nvqpp_getStateVectorData_fp32(StateVectorStorage &stateVectorStorage,
                                std::intptr_t index) {
  return getStateVectorData<float>(stateVectorStorage, index);
}
}

QuakeValue qalloc(ImplicitLocOpBuilder &builder,
                  StateVectorStorage &stateVectorStorage,
                  StateVectorVariant &&state, simulation_precision precision) {
  auto *context = builder.getContext();
  auto index = stateVectorStorage.size();
  stateVectorStorage.emplace_back(std::move(state));

  // Deal with the single/double precision differences here.
  const char *getLengthCallBack;
  const char *getDataCallBack;
  Type componentTy;
  {
    auto parentModule =
        builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
    IRBuilder irb(context);
    if (precision == simulation_precision::fp64) {
      getLengthCallBack = "__nvqpp_getStateVectorLength_fp64";
      getDataCallBack = "__nvqpp_getStateVectorData_fp64";
      componentTy = irb.getF64Type();
    } else {
      getLengthCallBack = "__nvqpp_getStateVectorLength_fp32";
      getDataCallBack = "__nvqpp_getStateVectorData_fp32";
      componentTy = irb.getF32Type();
    }
    if (failed(irb.loadIntrinsic(parentModule, getLengthCallBack)) ||
        failed(irb.loadIntrinsic(parentModule, getDataCallBack)))
      throw std::runtime_error("loading callbacks should never fail");
  }

  static_assert(sizeof(std::intptr_t) * 8 == 64);
  std::intptr_t vecStor = reinterpret_cast<std::intptr_t>(&stateVectorStorage);

  auto vecPtr = builder.create<arith::ConstantIntOp>(vecStor, 64);
  auto idxOp = builder.create<arith::ConstantIntOp>(index, 64);

  // Use callback to determine the size of the captured vector `state` at
  // runtime.
  auto i64Ty = builder.getI64Type();
  auto size = builder.create<func::CallOp>(i64Ty, getLengthCallBack,
                                           ValueRange{vecPtr, idxOp});

  // Allocate the qubits
  Value qubits = builder.create<quake::AllocaOp>(
      quake::VeqType::getUnsized(context), size.getResult(0));

  // Use callback to retrieve the data pointer of the captured vector `state` at
  // runtime.
  auto complexTy = ComplexType::get(componentTy);
  auto ptrComplexTy = cc::PointerType::get(complexTy);
  auto dataPtr = builder.create<func::CallOp>(ptrComplexTy, getDataCallBack,
                                              ValueRange{vecPtr, idxOp});

  // Add the initialize state op
  qubits = builder.create<quake::InitializeStateOp>(qubits.getType(), qubits,
                                                    dataPtr.getResult(0));
  return QuakeValue(builder, qubits);
}

QuakeValue constantVal(ImplicitLocOpBuilder &builder, double val) {
  llvm::APFloat d(val);
  Value constant =
      builder.create<arith::ConstantFloatOp>(d, builder.getF64Type());
  return QuakeValue(builder, constant);
}

template <typename QuakeOp>
void handleOneQubitBroadcast(ImplicitLocOpBuilder &builder, auto param,
                             Value veq, bool adjoint = false) {
  cudaq::info("kernel_builder handling operation broadcast on qvector.");

  auto loc = builder.getLoc();
  Value rank = builder.create<quake::VeqSizeOp>(builder.getI64Type(), veq);
  auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &,
                         Block &block) {
    Value ref =
        builder.create<quake::ExtractRefOp>(loc, veq, block.getArgument(0));

    builder.create<QuakeOp>(loc, adjoint, param, ValueRange(), ref);
  };
  cudaq::opt::factory::createInvariantLoop(builder, loc, rank, bodyBuilder);
}

template <typename QuakeOp>
void applyOneQubitOp(ImplicitLocOpBuilder &builder, auto &&params, auto &&ctrls,
                     Value qubit, bool adjoint = false) {
  builder.create<QuakeOp>(adjoint, params, ctrls, qubit);
}

#define CUDAQ_ONE_QUBIT_IMPL(NAME, QUAKENAME)                                  \
  void NAME(ImplicitLocOpBuilder &builder, std::vector<QuakeValue> &ctrls,     \
            const QuakeValue &target, bool adjoint) {                          \
    cudaq::info("kernel_builder apply {}", std::string(#NAME));                \
    auto value = target.getValue();                                            \
    auto type = value.getType();                                               \
    if (type.isa<quake::VeqType>()) {                                          \
      if (!ctrls.empty())                                                      \
        throw std::runtime_error(                                              \
            "Cannot specify controls for a veq broadcast.");                   \
      handleOneQubitBroadcast<quake::QUAKENAME>(builder, ValueRange(),         \
                                                target.getValue());            \
      return;                                                                  \
    }                                                                          \
    std::vector<Value> ctrlValues;                                             \
    std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(ctrlValues), \
                   [](auto &el) { return el.getValue(); });                    \
    applyOneQubitOp<quake::QUAKENAME>(builder, ValueRange(), ctrlValues,       \
                                      value, adjoint);                         \
  }

CUDAQ_ONE_QUBIT_IMPL(h, HOp)
CUDAQ_ONE_QUBIT_IMPL(s, SOp)
CUDAQ_ONE_QUBIT_IMPL(t, TOp)
CUDAQ_ONE_QUBIT_IMPL(x, XOp)
CUDAQ_ONE_QUBIT_IMPL(y, YOp)
CUDAQ_ONE_QUBIT_IMPL(z, ZOp)

#define CUDAQ_ONE_QUBIT_PARAM_IMPL(NAME, QUAKENAME)                            \
  void NAME(ImplicitLocOpBuilder &builder, QuakeValue &parameter,              \
            std::vector<QuakeValue> &ctrls, QuakeValue &target) {              \
    cudaq::info("kernel_builder apply {}", std::string(#NAME));                \
    Value value = target.getValue();                                           \
    auto type = value.getType();                                               \
    if (type.isa<quake::VeqType>()) {                                          \
      if (!ctrls.empty())                                                      \
        throw std::runtime_error(                                              \
            "Cannot specify controls for a veq broadcast.");                   \
      handleOneQubitBroadcast<quake::QUAKENAME>(builder, parameter.getValue(), \
                                                target.getValue());            \
      return;                                                                  \
    }                                                                          \
    std::vector<Value> ctrlValues;                                             \
    std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(ctrlValues), \
                   [](auto &el) { return el.getValue(); });                    \
    applyOneQubitOp<quake::QUAKENAME>(builder, parameter.getValue(),           \
                                      ctrlValues, value, false);               \
  }

CUDAQ_ONE_QUBIT_PARAM_IMPL(rx, RxOp)
CUDAQ_ONE_QUBIT_PARAM_IMPL(ry, RyOp)
CUDAQ_ONE_QUBIT_PARAM_IMPL(rz, RzOp)
CUDAQ_ONE_QUBIT_PARAM_IMPL(r1, R1Op)

template <typename QuakeMeasureOp>
QuakeValue applyMeasure(ImplicitLocOpBuilder &builder, Value value,
                        std::string regName) {
  auto type = value.getType();
  if (!type.isa<quake::RefType, quake::VeqType>())
    throw std::runtime_error("Invalid parameter passed to mz.");

  cudaq::info("kernel_builder apply measurement");

  auto i1Ty = builder.getI1Type();
  auto strAttr = builder.getStringAttr(regName);
  auto measTy = quake::MeasureType::get(builder.getContext());
  if (type.isa<quake::RefType>()) {
    Value measureResult =
        builder.template create<QuakeMeasureOp>(measTy, value, strAttr)
            .getMeasOut();
    Value bits = builder.create<quake::DiscriminateOp>(i1Ty, measureResult);
    return QuakeValue(builder, bits);
  }

  // This must be a veq.
  auto i64Ty = builder.getIntegerType(64);
  Value vecSize = builder.template create<quake::VeqSizeOp>(i64Ty, value);
  Value size = builder.template create<cudaq::cc::CastOp>(
      i64Ty, vecSize, cudaq::cc::CastOpMode::Unsigned);
  auto buff = builder.template create<cc::AllocaOp>(i1Ty, vecSize);
  cudaq::opt::factory::createInvariantLoop(
      builder, builder.getLoc(), size,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Region &,
          Block &block) {
        Value iv = block.getArgument(0);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        Value qv =
            nestedBuilder.create<quake::ExtractRefOp>(nestedLoc, value, iv);
        Value meas =
            nestedBuilder.create<QuakeMeasureOp>(nestedLoc, measTy, qv, strAttr)
                .getMeasOut();
        Value bit =
            nestedBuilder.create<quake::DiscriminateOp>(nestedLoc, i1Ty, meas);

        auto i1PtrTy = cudaq::cc::PointerType::get(i1Ty);
        auto addr = nestedBuilder.create<cc::ComputePtrOp>(
            nestedLoc, i1PtrTy, buff, ValueRange{iv});
        nestedBuilder.create<cc::StoreOp>(nestedLoc, bit, addr);
      });
  Value ret = builder.template create<cc::StdvecInitOp>(
      cc::StdvecType::get(builder.getContext(), i1Ty), buff, vecSize);
  return QuakeValue(builder, ret);
}

QuakeValue mx(ImplicitLocOpBuilder &builder, QuakeValue &qubitOrQvec,
              std::string regName) {
  return applyMeasure<quake::MxOp>(builder, qubitOrQvec.getValue(), regName);
}

QuakeValue my(ImplicitLocOpBuilder &builder, QuakeValue &qubitOrQvec,
              std::string regName) {
  return applyMeasure<quake::MyOp>(builder, qubitOrQvec.getValue(), regName);
}

QuakeValue mz(ImplicitLocOpBuilder &builder, QuakeValue &qubitOrQvec,
              std::string regName) {
  return applyMeasure<quake::MzOp>(builder, qubitOrQvec.getValue(), regName);
}

void reset(ImplicitLocOpBuilder &builder, const QuakeValue &qubitOrQvec) {
  auto value = qubitOrQvec.getValue();
  if (isa<quake::RefType>(value.getType())) {
    builder.create<quake::ResetOp>(TypeRange{}, value);
    return;
  }

  if (isa<quake::VeqType>(value.getType())) {
    auto target = value;
    Value rank = builder.create<quake::VeqSizeOp>(builder.getI64Type(), target);
    auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &,
                           Block &block) {
      Value ref = builder.create<quake::ExtractRefOp>(loc, target,
                                                      block.getArgument(0));
      builder.create<quake::ResetOp>(loc, TypeRange{}, ref);
    };
    cudaq::opt::factory::createInvariantLoop(builder, builder.getUnknownLoc(),
                                             rank, bodyBuilder);
    return;
  }

  llvm::errs() << "Invalid type:\n";
  value.getType().dump();
  llvm::errs() << '\n';
  throw std::runtime_error("Invalid type passed to reset().");
}

void swap(ImplicitLocOpBuilder &builder, const std::vector<QuakeValue> &ctrls,
          const std::vector<QuakeValue> &qubits, bool adjoint) {
  cudaq::info("kernel_builder apply swap");
  std::vector<Value> ctrlValues;
  std::vector<Value> qubitValues;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(ctrlValues),
                 [](auto &el) { return el.getValue(); });
  std::transform(qubits.begin(), qubits.end(), std::back_inserter(qubitValues),
                 [](auto &el) { return el.getValue(); });
  builder.create<quake::SwapOp>(adjoint, ValueRange(), ctrlValues, qubitValues);
}

void checkAndUpdateRegName(quake::MeasurementInterface &measure) {
  auto regName = measure.getOptionalRegisterName();
  if (!regName.has_value() || regName.value().empty()) {
    auto regNameUpdate = "auto_register_" + std::to_string(regCounter++);
    measure.setRegisterName(regNameUpdate);
  }
}

void c_if(ImplicitLocOpBuilder &builder, QuakeValue &conditional,
          std::function<void()> &thenFunctor) {
  auto value = conditional.getValue();

  if (auto discrOp = value.getDefiningOp<quake::DiscriminateOp>())
    if (auto measureOp = discrOp.getMeasurement()
                             .getDefiningOp<quake::MeasurementInterface>())
      checkAndUpdateRegName(measureOp);

  auto type = value.getType();
  if (!type.isa<mlir::IntegerType>() || type.getIntOrFloatBitWidth() != 1)
    throw std::runtime_error("Invalid result type passed to c_if.");

  builder.create<cc::IfOp>(TypeRange{}, value,
                           [&](OpBuilder &builder, Location l, Region &region) {
                             region.push_back(new Block());
                             auto &bodyBlock = region.front();
                             OpBuilder::InsertionGuard guard(builder);
                             builder.setInsertionPointToStart(&bodyBlock);
                             thenFunctor();
                             builder.create<cc::ContinueOp>(l);
                           });
}

/// Trims off the cudaq generated prefix and the mangled suffix, if any.
std::string name(std::string_view kernelName) {
  auto nToErase = runtime::cudaqGenPrefixLength;
  std::string copy(kernelName);
  copy.erase(0, nToErase);
  auto from = copy.find('.');
  if (from != std::string::npos)
    copy.erase(from);
  return copy;
}

bool isQubitType(Type ty) {
  if (ty.isa<quake::RefType, quake::VeqType>())
    return true;
  if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(ty))
    return isQubitType(vecTy.getElementType());
  return false;
}

bool hasAnyQubitTypes(FunctionType funcTy) {
  for (auto ty : funcTy.getInputs())
    if (isQubitType(ty))
      return true;
  for (auto ty : funcTy.getResults())
    if (isQubitType(ty))
      return true;
  return false;
}

void tagEntryPoint(ImplicitLocOpBuilder &builder, ModuleOp &module,
                   StringRef symbolName) {
  module.walk([&](func::FuncOp function) {
    if (function.empty())
      return WalkResult::advance();
    if (!function->hasAttr(cudaq::entryPointAttrName) &&
        !hasAnyQubitTypes(function.getFunctionType()) &&
        (symbolName.empty() || function.getSymName().equals(symbolName)))
      function->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());

    return WalkResult::advance();
  });
}

std::tuple<bool, ExecutionEngine *>
jitCode(ImplicitLocOpBuilder &builder, ExecutionEngine *jit,
        std::unordered_map<ExecutionEngine *, std::size_t> &jitHash,
        std::string kernelName, std::vector<std::string> extraLibPaths,
        StateVectorStorage &stateVectorStorage) {

  // Start of by getting the current ModuleOp
  auto *block = builder.getBlock();
  auto *context = builder.getContext();
  auto *function = block->getParentOp();
  auto currentModule = function->getParentOfType<ModuleOp>();

  // Create a unique hash from that ModuleOp
  auto hash = llvm::hash_code{0};
  currentModule.walk([&hash](Operation *op) {
    hash = llvm::hash_combine(hash, OperationEquivalence::computeHash(op));
  });
  auto moduleHash = static_cast<size_t>(hash);

  if (jit) {
    // Have we added more instructions since the last time we jit the code? If
    // so, we need to delete this JIT engine and create a new one.
    if (moduleHash == jitHash[jit])
      return std::make_tuple(false, jit);
    else {
      // need to redo the jit, remove the old one
      jitHash.erase(jit);
    }
  }

  cudaq::info("kernel_builder running jitCode.");

  auto module = currentModule.clone();
  auto ctx = module.getContext();
  SmallVector<mlir::NamedAttribute> names;
  names.emplace_back(mlir::StringAttr::get(ctx, kernelName),
                     mlir::StringAttr::get(ctx, "BuilderKernel.EntryPoint"));
  auto mapAttr = mlir::DictionaryAttr::get(ctx, names);
  module->setAttr("quake.mangled_name_map", mapAttr);

  // Tag as an entrypoint if it is one
  tagEntryPoint(builder, module, StringRef{});

  PassManager pm(context);
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(cudaq::opt::createUnwindLoweringPass());
  cudaq::opt::addAggressiveEarlyInlining(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createApplyOpSpecializationPass());
  optPM.addPass(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addPass(cudaq::opt::createLoopNormalize());
  pm.addPass(cudaq::opt::createLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  optPM.addPass(cudaq::opt::createQuakeAddDeallocs());
  optPM.addPass(cudaq::opt::createQuakeAddMetadata());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // For some reason I get CFG ops from the LowerToCFGPass instead of the
  // unrolled cc loop if I don't run the above manually.
  if (failed(pm.run(module)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");

  // Continue on...
  pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader(/*genAsQuake=*/true));
  pm.addPass(cudaq::opt::createGenerateKernelExecution());
  optPM.addPass(cudaq::opt::createLowerToCFGPass());
  // We want quantum allocations to stay where they are if
  // we are simulating and have user-provided state vectors.
  // This check could be better / smarter probably, in tandem
  // with some synth strategy to rewrite initState with circuit
  // synthesis result
  if (stateVectorStorage.empty())
    optPM.addPass(cudaq::opt::createCombineQuantumAllocations());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(cudaq::opt::createConvertToQIRPass());
  pm.addPass(createCanonicalizerPass());

  if (failed(pm.run(module)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");

  // The "fast" instruction selection compilation algorithm is actually very
  // slow for large quantum circuits. Disable that here. Revisit this
  // decision by testing large UCCSD circuits if jitCodeGenOptLevel is changed
  // in the future. Also note that llvm::TargetMachine::setFastIsel() and
  // setO0WantsFastISel() do not retain their values in our current version of
  // LLVM. This use of LLVM command line parameters could be changed if the LLVM
  // JIT ever supports the TargetMachine options in the future.
  const char *argv[] = {"", "-fast-isel=0", nullptr};
  llvm::cl::ParseCommandLineOptions(2, argv);

  cudaq::info("- Pass manager was applied.");
  ExecutionEngineOptions opts;
  opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
  opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
  SmallVector<StringRef, 4> sharedLibs;
  for (auto &lib : extraLibPaths) {
    cudaq::info("Extra library loaded: {}", lib);
    sharedLibs.push_back(lib);
  }
  opts.sharedLibPaths = sharedLibs;
  opts.llvmModuleBuilder =
      [](Operation *module,
         llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
    llvmContext.setOpaquePointers(false);
    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
      llvm::errs() << "Failed to emit LLVM IR\n";
      return nullptr;
    }
    ExecutionEngine::setupTargetTriple(llvmModule.get());
    return llvmModule;
  };

  cudaq::info(" - Creating the MLIR ExecutionEngine");
  auto jitOrError = ExecutionEngine::create(module, opts);
  assert(!!jitOrError);

  auto uniqueJit = std::move(jitOrError.get());
  jit = uniqueJit.release();

  cudaq::info("- JIT Engine created successfully.");

  // Kernel names are __nvqpp__mlirgen__BuilderKernelPTRSTR for the following we
  // want the proper name, BuilderKernelPTRST
  std::string properName = name(kernelName);

  // Need to first invoke the init_func()
  auto kernelInitFunc = properName + ".init_func";
  auto initFuncPtr = jit->lookup(kernelInitFunc);
  if (!initFuncPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get kernelReg function.");
  }
  auto kernelInit = reinterpret_cast<void (*)()>(*initFuncPtr);
  kernelInit();

  // Need to first invoke the kernelRegFunc()
  auto kernelRegFunc = properName + ".kernelRegFunc";
  auto regFuncPtr = jit->lookup(kernelRegFunc);
  if (!regFuncPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get kernelReg function.");
  }
  auto kernelReg = reinterpret_cast<void (*)()>(*regFuncPtr);
  kernelReg();

  // Map this JIT Engine to its unique hash integer.
  jitHash.insert({jit, moduleHash});
  return std::make_tuple(true, jit);
}

void invokeCode(ImplicitLocOpBuilder &builder, ExecutionEngine *jit,
                std::string kernelName, void **argsArray,
                std::vector<std::string> extraLibPaths,
                StateVectorStorage &storage) {

  assert(jit != nullptr && "JIT ExecutionEngine was null.");
  cudaq::info("kernel_builder invoke kernel with args.");

  // Kernel names are __nvqpp__mlirgen__BuilderKernelPTRSTR for the following we
  // want the proper name, BuilderKernelPTRST
  std::string properName = name(kernelName);

  // Incoming Args... have been converted to void **, now we convert to void *
  // altLaunchKernel args.
  auto argCreatorName = properName + ".argsCreator";
  auto expectedPtr = jit->lookup(argCreatorName);
  if (!expectedPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get argsCreator function.");
  }
  auto argsCreator =
      reinterpret_cast<std::size_t (*)(void **, void **)>(*expectedPtr);
  void *rawArgs = nullptr;
  [[maybe_unused]] auto size = argsCreator(argsArray, &rawArgs);

  //  Extract the entry point, which we named.
  auto thunkName = properName + ".thunk";
  auto thunkPtr = jit->lookup(thunkName);
  if (!thunkPtr) {
    throw std::runtime_error("cudaq::builder failed to get thunk function");
  }

  // Invoke and free the args memory.
  auto thunk = reinterpret_cast<void (*)(void *)>(*thunkPtr);

  altLaunchKernel(properName.data(), thunk, rawArgs, size);
  std::free(rawArgs);
  // TODO: any return values are dropped on the floor here.
}

std::string to_quake(ImplicitLocOpBuilder &builder) {
  // Get the current ModuleOp
  auto *block = builder.getBlock();
  auto parentFunc = block->getParentOp();
  auto module = parentFunc->getParentOfType<ModuleOp>();

  // Strategy - we want to clone this ModuleOp because we have to
  // add a valid terminator (func.return), but it is not gauranteed that
  // the programmer is done building up the kernel even though they've asked
  // to look at the quake code. So we'll clone here, and add the return op
  // (we have to or the print out string will be invalid (verifier failed)).
  auto clonedModule = module.clone();

  func::FuncOp unwrappedParentFunc = llvm::cast<func::FuncOp>(parentFunc);
  llvm::StringRef symName = unwrappedParentFunc.getSymName();
  tagEntryPoint(builder, clonedModule, symName);

  // Clean up the code for print out
  PassManager pm(clonedModule.getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(clonedModule)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");

  std::string printOut;
  llvm::raw_string_ostream os(printOut);
  clonedModule->print(os);
  return printOut;
}

std::ostream &operator<<(std::ostream &stream,
                         const kernel_builder_base &builder) {
  return stream << builder.to_quake();
}

} // namespace cudaq::details
