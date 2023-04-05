/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "kernel_builder.h"
#include "common/Logger.h"
#include "common/RuntimeMLIR.h"
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

KernelBuilderType mapArgToType(double &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) mutable { return Float64Type::get(ctx); });
}

KernelBuilderType mapArgToType(float &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) mutable { return Float32Type::get(ctx); });
}

KernelBuilderType mapArgToType(int &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) mutable { return IntegerType::get(ctx, 32); });
}

KernelBuilderType mapArgToType(std::vector<double> &e) {
  return KernelBuilderType([](MLIRContext *ctx) mutable {
    return cudaq::cc::StdvecType::get(ctx, Float64Type::get(ctx));
  });
}

KernelBuilderType mapArgToType(std::size_t &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) mutable { return IntegerType::get(ctx, 64); });
}

KernelBuilderType mapArgToType(std::vector<int> &e) {
  return KernelBuilderType([](MLIRContext *ctx) mutable {
    return cudaq::cc::StdvecType::get(ctx, mlir::IntegerType::get(ctx, 32));
  });
}

KernelBuilderType mapArgToType(std::vector<std::size_t> &e) {
  return KernelBuilderType([](MLIRContext *ctx) mutable {
    return cudaq::cc::StdvecType::get(ctx, mlir::IntegerType::get(ctx, 64));
  });
}

/// Map a std::vector<float> to a KernelBuilderType
KernelBuilderType mapArgToType(std::vector<float> &e) {
  return KernelBuilderType([](MLIRContext *ctx) mutable {
    return cudaq::cc::StdvecType::get(ctx, Float32Type::get(ctx));
  });
}

KernelBuilderType mapArgToType(cudaq::qubit &e) {
  return KernelBuilderType(
      [](MLIRContext *ctx) mutable { return quake::QRefType::get(ctx); });
}

KernelBuilderType mapArgToType(cudaq::qreg<> &e) {
  return KernelBuilderType([](MLIRContext *ctx) mutable {
    return quake::QVecType::getUnsized(ctx);
  });
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
  kernelName += "_" + os.str();

  cudaq::info("kernel_builder name set to {}", kernelName);

  FunctionType funcTy = opBuilder->getFunctionType(types, std::nullopt);
  auto kernel = opBuilder->create<func::FuncOp>(kernelName, funcTy);
  auto *entryBlock = kernel.addEntryBlock();

  for (auto arg : entryBlock->getArguments())
    arguments.emplace_back(*opBuilder, arg);

  cudaq::info("kernel_builder has {} arguments", arguments.size());

  opBuilder->setInsertionPointToStart(entryBlock);

  return opBuilder;
}
void deleteBuilder(ImplicitLocOpBuilder *builder) { delete builder; }

bool isArgStdVec(std::vector<QuakeValue> &args, std::size_t idx) {
  return args[idx].isStdVec();
}

void call(ImplicitLocOpBuilder &builder, std::string &name,
          std::string &quakeCode, std::vector<QuakeValue> &values) {
  // Create a ModuleOp from the other kernel's quake code
  auto otherModule =
      mlir::parseSourceString<mlir::ModuleOp>(quakeCode, builder.getContext());

  // Get the function with the kernel name we care about
  auto otherFunc = otherModule->lookupSymbol<mlir::func::FuncOp>(
      std::string(cudaq::runtime::cudaqGenPrefixName) + name);

  // Clone it
  auto cloned = otherFunc.clone();

  // Get our current module
  auto block = builder.getBlock();
  auto function = block->getParentOp();
  auto currentModule = function->getParentOfType<ModuleOp>();

  // Add the other kernel to this ModuleOp
  currentModule.push_back(cloned);

  // Map the QuakeValues to MLIR Values
  SmallVector<Value> mlirValues;
  for (std::size_t i = 0; auto &v : values) {
    Type argType = cloned.getArgumentTypes()[i];
    Value value = v.getValue();
    Type inType = value.getType();
    auto inAsQVecTy = inType.dyn_cast_or_null<quake::QVecType>();
    auto argAsQVecTy = argType.dyn_cast_or_null<quake::QVecType>();

    // If both are qvecs, make sure we dont have qvec<N> -> qvec<?>
    if (inAsQVecTy && argAsQVecTy) {
      // make sure they are both the same qvec<...> type
      if (inAsQVecTy.hasSpecifiedSize() && !argAsQVecTy.hasSpecifiedSize())
        value = builder.create<quake::RelaxSizeOp>(argAsQVecTy, value);
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
  builder.create<func::CallOp>(cloned, mlirValues);
}

void applyControlOrAdjoint(ImplicitLocOpBuilder &builder, std::string &name,
                           std::string &quakeCode, bool isAdjoint,
                           ValueRange controls,
                           std::vector<QuakeValue> &values) {
  // Create a ModuleOp from the other kernel's quake code
  auto otherModule =
      mlir::parseSourceString<mlir::ModuleOp>(quakeCode, builder.getContext());

  // Get the function with the kernel name we care about
  auto otherFunc = otherModule->lookupSymbol<mlir::func::FuncOp>(
      std::string(cudaq::runtime::cudaqGenPrefixName) + name);

  // Get our current module
  auto block = builder.getBlock();
  auto function = block->getParentOp();
  auto currentModule = function->getParentOfType<ModuleOp>();

  func::FuncOp cloned;
  auto doWeHaveIt = currentModule.lookupSymbol<func::FuncOp>(
      std::string(cudaq::runtime::cudaqGenPrefixName) + name);
  if (!doWeHaveIt) {
    // Clone it
    cloned = otherFunc.clone();

    // Add the other kernel to this ModuleOp
    currentModule.push_back(cloned);
  } else {
    cloned = doWeHaveIt;
  }

  SmallVector<Value> mlirValues;
  for (std::size_t i = 0; auto &v : values) {
    Type argType = cloned.getArgumentTypes()[i];
    Value value = v.getValue();
    Type inType = value.getType();
    auto inAsQVecTy = inType.dyn_cast_or_null<quake::QVecType>();
    auto argAsQVecTy = argType.dyn_cast_or_null<quake::QVecType>();

    // If both are qvecs, make sure we dont have qvec<N> -> qvec<?>
    if (inAsQVecTy && argAsQVecTy) {
      // make sure they are both the same qvec<...> type
      if (inAsQVecTy.hasSpecifiedSize() && !argAsQVecTy.hasSpecifiedSize())
        value = builder.create<quake::RelaxSizeOp>(argAsQVecTy, value);
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
  Value subtracted =
      builder.create<arith::SubIOp>(end.getType(), end, startVal);
  Value totalIters =
      builder.create<arith::IndexCastOp>(builder.getIndexType(), subtracted);
  cudaq::opt::factory::createCountedLoop(
      builder, builder.getLoc(), totalIters,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Region &,
          Block &block) {
        Value iv = block.getArgument(0);
        // shift iv -> iv + start
        iv = builder.create<arith::AddIOp>(iv.getType(), iv, startVal);
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
  Value startVal = builder.create<arith::ConstantIndexOp>(start);
  Value endVal = builder.create<arith::ConstantIndexOp>(end);
  forLoop(builder, startVal, endVal, body);
}

void forLoop(ImplicitLocOpBuilder &builder, std::size_t start, QuakeValue &end,
             std::function<void(QuakeValue &)> &body) {
  Value startVal = builder.create<arith::ConstantIndexOp>(start);
  auto e = end.getValue();
  forLoop(builder, startVal, e, body);
}

void forLoop(ImplicitLocOpBuilder &builder, QuakeValue &start, std::size_t end,
             std::function<void(QuakeValue &)> &body) {
  Value e = builder.create<arith::ConstantIndexOp>(end);
  auto s = start.getValue();
  forLoop(builder, s, e, body);
}

KernelBuilderType::KernelBuilderType(
    std::function<mlir::Type(MLIRContext *ctx)> &&f)
    : creator(f) {}

Type KernelBuilderType::create(MLIRContext *ctx) { return creator(ctx); }

QuakeValue qalloc(ImplicitLocOpBuilder &builder, const std::size_t nQubits) {
  cudaq::info("kernel_builder allocating {} qubits", nQubits);

  if (nQubits == 1) {
    Value qubit = builder.create<quake::AllocaOp>();
    return QuakeValue(builder, qubit);
  }
  auto context = builder.getContext();
  Value qubits = builder.create<quake::AllocaOp>(
      quake::QVecType::get(context, nQubits),
      builder.create<arith::ConstantIntOp>(nQubits, 32));

  return QuakeValue(builder, qubits);
}

QuakeValue qalloc(ImplicitLocOpBuilder &builder, QuakeValue &size) {
  cudaq::info("kernel_builder allocating qubits from quake value");
  auto value = size.getValue();
  auto type = value.getType();
  if (!type.isIntOrIndex())
    throw std::runtime_error(
        "Invalid parameter passed to qalloc (must be integer type).");

  auto context = builder.getContext();
  Value qubits = builder.create<quake::AllocaOp>(
      quake::QVecType::getUnsized(context), value);

  return QuakeValue(builder, qubits);
}

template <typename QuakeOp>
void handleOneQubitBroadcast(ImplicitLocOpBuilder &builder, Value qvec,
                             bool adjoint = false) {
  cudaq::info("kernel_builder handling operation broadcast on qreg.");

  auto loc = builder.getLoc();
  auto indexTy = builder.getIndexType();
  auto size =
      builder.create<quake::QVecSizeOp>(builder.getIntegerType(64), qvec);
  Value rank = builder.create<arith::IndexCastOp>(indexTy, size);
  auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &,
                         Block &block) {
    Value qref =
        builder.create<quake::QExtractOp>(loc, qvec, block.getArgument(0));

    builder.create<QuakeOp>(loc, adjoint, ValueRange(), ValueRange(), qref);
  };
  cudaq::opt::factory::createCountedLoop(builder, loc, rank, bodyBuilder);
}

template <typename QuakeOp>
void applyOneQubitOp(ImplicitLocOpBuilder &builder, auto &&params, auto &&ctrls,
                     Value qubit, bool adjoint = false) {
  builder.create<QuakeOp>(adjoint, params, ctrls, qubit);
}

#define CUDAQ_ONE_QUBIT_IMPL(NAME, QUAKENAME)                                  \
  void NAME(ImplicitLocOpBuilder &builder, std::vector<QuakeValue> &ctrls,     \
            QuakeValue &target, bool adjoint) {                                \
    cudaq::info("kernel_builder apply {}", std::string(#NAME));                \
    auto value = target.getValue();                                            \
    auto type = value.getType();                                               \
    if (type.isa<quake::QVecType>()) {                                         \
      if (!ctrls.empty())                                                      \
        throw std::runtime_error(                                              \
            "Cannot specify controls for a qvec broadcast.");                  \
      handleOneQubitBroadcast<quake::QUAKENAME>(builder, target.getValue());   \
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
    std::vector<Value> ctrlValues;                                             \
    std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(ctrlValues), \
                   [](auto &el) { return el.getValue(); });                    \
    applyOneQubitOp<quake::QUAKENAME>(builder, parameter.getValue(),           \
                                      ctrlValues, value);                      \
  }

CUDAQ_ONE_QUBIT_PARAM_IMPL(rx, RxOp)
CUDAQ_ONE_QUBIT_PARAM_IMPL(ry, RyOp)
CUDAQ_ONE_QUBIT_PARAM_IMPL(rz, RzOp)
CUDAQ_ONE_QUBIT_PARAM_IMPL(r1, R1Op)

template <typename QuakeMeasureOp>
QuakeValue applyMeasure(ImplicitLocOpBuilder &builder, Value value,
                        std::string regName) {
  auto type = value.getType();
  if (!type.isa<quake::QRefType, quake::QVecType>())
    throw std::runtime_error("Invalid parameter passed to mz.");

  cudaq::info("kernel_builder apply measurement");

  auto i1Ty = builder.getI1Type();
  if (type.isa<quake::QRefType>()) {
    Value measureResult = builder.template create<QuakeMeasureOp>(
        i1Ty, value, builder.getStringAttr(regName));
    return QuakeValue(builder, measureResult);
  }

  // This must be a qvec.
  Value vecSize = builder.template create<quake::QVecSizeOp>(
      builder.getIntegerType(64), value);
  Value size = builder.template create<arith::IndexCastOp>(
      builder.getIndexType(), vecSize);
  auto i1PtrTy = cudaq::opt::factory::getPointerType(i1Ty);
  auto buff = builder.template create<LLVM::AllocaOp>(i1PtrTy, vecSize);
  cudaq::opt::factory::createCountedLoop(
      builder, builder.getLoc(), size,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Region &,
          Block &block) {
        Value iv = block.getArgument(0);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        Value qv =
            nestedBuilder.create<quake::QExtractOp>(nestedLoc, value, iv);
        Value bit = nestedBuilder.create<QuakeMeasureOp>(nestedLoc, i1Ty, qv);

        auto i64Ty = nestedBuilder.getIntegerType(64);
        auto intIv =
            nestedBuilder.create<arith::IndexCastOp>(nestedLoc, i64Ty, iv);
        auto addr = nestedBuilder.create<LLVM::GEPOp>(nestedLoc, i1PtrTy, buff,
                                                      ValueRange{intIv});
        nestedBuilder.create<LLVM::StoreOp>(nestedLoc, bit, addr);
      });
  Value ret = builder.template create<cc::StdvecInitOp>(
      cc::StdvecType::get(builder.getContext(), i1Ty), buff, vecSize);
  return QuakeValue(builder, ret);
}

QuakeValue mx(ImplicitLocOpBuilder &builder, QuakeValue &qubitOrQreg,
              std::string regName) {
  return applyMeasure<quake::MxOp>(builder, qubitOrQreg.getValue(), regName);
}

QuakeValue my(ImplicitLocOpBuilder &builder, QuakeValue &qubitOrQreg,
              std::string regName) {
  return applyMeasure<quake::MyOp>(builder, qubitOrQreg.getValue(), regName);
}

QuakeValue mz(ImplicitLocOpBuilder &builder, QuakeValue &qubitOrQreg,
              std::string regName) {
  return applyMeasure<quake::MzOp>(builder, qubitOrQreg.getValue(), regName);
}

void reset(ImplicitLocOpBuilder &builder, QuakeValue &qubitOrQreg) {
  auto value = qubitOrQreg.getValue();
  if (isa<quake::QRefType>(value.getType())) {
    builder.create<quake::ResetOp>(value);
    return;
  }

  if (isa<quake::QVecType>(value.getType())) {
    auto target = value;
    Type indexTy = builder.getIndexType();
    auto size =
        builder.create<quake::QVecSizeOp>(builder.getIntegerType(64), target);
    Value rank = builder.create<arith::IndexCastOp>(indexTy, size);
    auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &,
                           Block &block) {
      Value qref =
          builder.create<quake::QExtractOp>(loc, target, block.getArgument(0));
      builder.create<quake::ResetOp>(loc, qref);
    };
    cudaq::opt::factory::createCountedLoop(builder, builder.getUnknownLoc(),
                                           rank, bodyBuilder);
    return;
  }

  llvm::errs() << "Invalid type:\n";
  value.getType().dump();
  llvm::errs() << '\n';
  throw std::runtime_error("Invalid type passed to reset().");
}

void c_if(ImplicitLocOpBuilder &builder, QuakeValue &conditional,
          std::function<void()> &thenFunctor) {
  auto value = conditional.getValue();
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
  if (ty.isa<quake::QRefType, quake::QVecType>())
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

ExecutionEngine *jitCode(ImplicitLocOpBuilder &builder, ExecutionEngine *jit,
                         std::string kernelName,
                         std::vector<std::string> extraLibPaths) {
  if (jit)
    return jit;

  cudaq::info("kernel_builder running jitCode.");

  auto block = builder.getBlock();
  if ((block->getOperations().empty() ||
       !block->getOperations().back().hasTrait<OpTrait::IsTerminator>())) {
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
  }

  auto *context = builder.getContext();
  auto function = block->getParentOp();
  auto currentModule = function->getParentOfType<ModuleOp>();
  auto module = currentModule.clone();
  auto ctx = module.getContext();
  SmallVector<mlir::NamedAttribute> names;
  names.emplace_back(mlir::StringAttr::get(ctx, kernelName),
                     mlir::StringAttr::get(ctx, "BuilderKernel.EntryPoint"));
  auto mapAttr = mlir::DictionaryAttr::get(ctx, names);
  module->setAttr("qtx.mangled_name_map", mapAttr);

  // Tag as an entrypoint if it is one
  module.walk([&](func::FuncOp function) {
    if (function.empty())
      return WalkResult::advance();
    if (!function->hasAttr(cudaq::entryPointAttrName) &&
        !hasAnyQubitTypes(function.getFunctionType()))
      function->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());

    return WalkResult::advance();
  });

  PassManager pm(context);
  pm.addPass(createCanonicalizerPass());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createApplyOpSpecializationPass());
  pm.addPass(cudaq::opt::createLoopUnrollPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createInlinerPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // For some reason I get CFG ops from the LowerToCFGPass
  // instead of the unrolled cc loop if I don't run
  // the above manually.
  if (failed(pm.run(module)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");

  // Continue on...
  pm.addPass(createInlinerPass());
  optPM.addPass(cudaq::opt::createQuakeAddDeallocs());
  optPM.addPass(cudaq::opt::createQuakeAddMetadata());
  pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader(/*genAsQuake=*/true));
  pm.addPass(cudaq::opt::createGenerateKernelExecution());
  optPM.addPass(cudaq::opt::createLowerToCFGPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(cudaq::opt::createConvertToQIRPass());

  if (failed(pm.run(module)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");

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

  // Kernel names are __nvqpp__mlirgen__BuilderKernelPTRSTR
  // for the following we want the proper name, BuilderKernelPTRST
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

  return jit;
}

void invokeCode(ImplicitLocOpBuilder &builder, ExecutionEngine *jit,
                std::string kernelName, void **argsArray,
                std::vector<std::string> extraLibPaths) {

  assert(jit != nullptr && "JIT ExecutionEngine was null.");
  cudaq::info("kernel_builder invoke kernel with args.");

  // Kernel names are __nvqpp__mlirgen__BuilderKernelPTRSTR
  // for the following we want the proper name, BuilderKernelPTRST
  std::string properName = name(kernelName);

  // Incoming Args... have been converted to void **,
  // now we convert to void * altLaunchKernel args.
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
}

std::string to_quake(ImplicitLocOpBuilder &builder) {
  // Get the current ModuleOp
  auto *block = builder.getBlock();
  auto parentFunc = block->getParentOp();
  auto module = parentFunc->getParentOfType<ModuleOp>();

  // Strategy - we want to clone this ModuleOp because we have to
  // add a valid terminator (func.return), but it is not gauranteed that
  // the programmer is done building up the kernel even though they've asked to
  // look at the quake code. So we'll clone here, and add the return op (we have
  // to or the print out string will be invalid (verifier failed)).
  auto clonedModule = module.clone();

  // Look for the main block in the functions we have
  // add a return if it does not have one.
  clonedModule.walk([](func::FuncOp func) {
    Block &block = *func.getBlocks().begin();

    auto tmpBuilder = OpBuilder::atBlockEnd(&block);

    if (block.getOperations().empty()) {
      // If no ops, add a Return
      tmpBuilder.create<func::ReturnOp>(tmpBuilder.getUnknownLoc());
    } else if (!block.getOperations()
                    .back()
                    .hasTrait<OpTrait::IsTerminator>()) {
      // if last op is not the terminator, add the return.
      tmpBuilder.create<func::ReturnOp>(tmpBuilder.getUnknownLoc());
    }
  });

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

} // namespace cudaq::details
