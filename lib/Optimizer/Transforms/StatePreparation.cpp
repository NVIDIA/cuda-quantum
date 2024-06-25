/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include <span>
#include "StateDecomposer.h"

#include <iostream>

#define DEBUG_TYPE "state-preparation"

using namespace mlir;

/// Replace a qubit initialization from vectors with quantum gates.
/// For example:
///
///   func.func @foo(%arg0 : !cc.stdvec<complex<f32>>) {
///     %0 = cc.stdvec_size %arg0 : (!cc.stdvec<complex<f32>>) -> i64
///     %1 = math.cttz %0 : i64
///     %2 = cc.stdvec_data %arg0 : (!cc.stdvec<complex<f32>>) -> !cc.ptr<complex<f32>> 
///     %3 = quake.alloca !quake.veq<?>[%1 : i64]
///     %4 = quake.init_state %3, %2 : (!quake.veq<?>, !cc.ptr<complex<f32>>) -> !quake.veq<?>
///     return
///   }
///
/// On a call that passes std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2} as arg0:
///
///   func.func @foo(%arg0 : !cc.stdvec<complex<f32>>) {
///     %0 = quake.alloca !quake.veq<2>
///     %c0_i64 = arith.constant 0 : i64
///     %1 = quake.extract_ref %0[%c0_i64] : (!quake.veq<2>, i64) -> !quake.ref
///     %cst = arith.constant 1.5707963267948968 : f64
///     quake.ry (%cst) %1 : (f64, !quake.ref) -> ()
///     %c1_i64 = arith.constant 1 : i64
///     %2 = quake.extract_ref %0[%c1_i64] : (!quake.veq<2>, i64) -> !quake.ref
///     %cst_0 = arith.constant 1.5707963267948966 : f64
///     quake.ry (%cst_0) %2 : (f64, !quake.ref) -> ()
///     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
///     %cst_1 = arith.constant -1.5707963267948966 : f64
///     quake.ry (%cst_1) %2 : (f64, !quake.ref) -> ()
///     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
///     return
///   }
///
/// Note: the following synthesis and const prop passes will replace
/// the argument by a constant and propagate the values and vector size
/// through other instructions.

namespace {

template <typename T>
concept IntegralType = std::is_same<T, bool>::value 
    || std::is_same<T, std::int8_t>::value
    || std::is_same<T, std::int16_t>::value
    || std::is_same<T, std::int32_t>::value
    || std::is_same<T, std::int64_t>::value;

template <typename T>
concept FloatingType = std::is_same<T, float>::value;

template <typename T>
concept DoubleType = std::is_same<T, double>::value;

template <typename T>
concept ComplexDataType = FloatingType<T> || DoubleType<T> || IntegralType<T>;

/// Input was complex<float>/complex<double> but we prefer
/// complex<double>/complex<float>. Make a copy, extending or truncating the
/// values.
template <FloatingType From>
std::vector<std::complex<double>> convertToComplex(std::complex<From> *data, std::uint64_t size) {
  auto convertData = std::vector<std::complex<double>>(size);
  for (std::size_t i = 0; i < size; ++i)
    convertData[i] = std::complex<double>{static_cast<double>(data[i].real()),
                                      static_cast<double>(data[i].imag())};
  return convertData;
}

template <DoubleType From>
std::vector<std::complex<double>> convertToComplex(std::complex<From> *data, std::uint64_t size) {
    return std::vector<std::complex<From>>(data, data+size);
}

/// Input was float/double but we prefer complex<float>/complex<double>.
/// Make a copy, extending or truncating the values.
template <ComplexDataType From>
std::vector<std::complex<double>> convertToComplex(From *data, std::uint64_t size) {
  auto convertData = std::vector<std::complex<double>>(size);
  for (std::size_t i = 0; i < size; ++i)
    convertData[i] =
        std::complex<double>{static_cast<double>(data[i]), static_cast<double>(0.0)};
  return convertData;
}

LogicalResult
prepareStateFromVectorArgument(OpBuilder &builder, ModuleOp module,
                               unsigned &counter, BlockArgument argument,
                               std::vector<std::complex<double>> &vec) {
  auto *ctx = builder.getContext();
  // builder.setInsertionPointToStart(argument.getOwner());
  auto argLoc = argument.getLoc();

  // TODO: look at quake.init_state instructions from vector data and track them
  // to the argument vector, then replace the instruction by gates preparing the
  // state (or a call to a kernel with gates)

  ///   func.func @foo(%arg0 : !cc.stdvec<complex<f32>>) {
  ///     %0 = cc.stdvec_size %arg0 : (!cc.stdvec<complex<f32>>) -> i64
  ///     %2 = cc.stdvec_data %arg0 : (!cc.stdvec<complex<f32>>) ->
  ///     !cc.ptr<complex<f32>>
  ///
  ///     %3 = quake.alloca !quake.veq<?>[%1 : i64]
  ///     %4 = quake.init_state %3, %2 : (!quake.veq<?>, !cc.ptr<complex<f32>>)
  ///     -> !quake.veq<?> return
  ///   }

  /// =>

  ///     ...
  ///     %5 = quake.alloca !quake.veq<?>[%3 : i64]
  ///     %6 = quake.extract_ref %5[0] : (!quake.veq<?>) -> !quake.ref
  ///     quake.h %6 : (!quake.ref) -> ()
  ///     %7 = quake.extract_ref %5[0] : (!quake.veq<?>) -> !quake.ref
  ///     %8 = quake.extract_ref %5[1] : (!quake.veq<?>) -> !quake.ref
  ///     quake.x [%7] %8 : (!quake.ref, !quake.ref) -> ()

  auto toErase = std::vector<mlir::Operation*>();

  for (auto *argUser : argument.getUsers()) {
    // Handle the `StdvecSize` and `quake.alloca` use case:
    // - Replace a `vec.size()` with the vector length.
    // - Replace the number of qubits calculation with the vector length logarithm.
    // - Replace `quake.alloca` with a constant size qvector allocation.
    if (auto stdvecSizeOp = dyn_cast<cudaq::cc::StdvecSizeOp>(argUser)) {
      builder.setInsertionPointAfter(stdvecSizeOp);
      Value length = builder.create<arith::ConstantIntOp>(
          argLoc, vec.size(), stdvecSizeOp.getType());

      Value numQubits = builder.create<arith::ConstantIntOp>(
          argLoc, log2(vec.size()), stdvecSizeOp.getType());

      for (auto *sizeUser: argUser->getUsers()) {
        if (auto countZeroesOp = dyn_cast<mlir::math::CountTrailingZerosOp>(sizeUser)) {
          for (auto *numQubitsUser: sizeUser->getUsers()) {
            if (auto quakeAllocaOp = dyn_cast<quake::AllocaOp>(numQubitsUser)) {
              builder.setInsertionPointAfter(quakeAllocaOp);
              auto veqTy = quake::VeqType::get(ctx, log2(vec.size()));
              Value newAlloc = builder.create<quake::AllocaOp>(argLoc, veqTy);
              quakeAllocaOp.replaceAllUsesWith(newAlloc);
              toErase.push_back(quakeAllocaOp);
            }
          }
          countZeroesOp.replaceAllUsesWith(numQubits);
          toErase.push_back(countZeroesOp);
        }
      }
      
      stdvecSizeOp.replaceAllUsesWith(length);
      toErase.push_back(stdvecSizeOp);
      continue;
    }

    // Handle the `StdvecDataOp` and `quake.init_state` use case:
    // - Replace a `quake.init_state` with gates preparing the state.
    if (auto stdvecDataOp = dyn_cast<cudaq::cc::StdvecDataOp>(argUser)) {
      for (auto *dataUser : stdvecDataOp->getUsers()) {
        if (auto initOp = dyn_cast<quake::InitializeStateOp>(dataUser)) {
          builder.setInsertionPointAfter(initOp);
          // Find the qvector alloc instruction
          auto qubits = initOp.getOperand(0);

          // Prepare state from vector data.
          auto gateBuilder = StateGateBuilder(builder, argLoc, qubits);
          auto decomposer = StateDecomposer(gateBuilder, vec);
          decomposer.decompose();

          initOp.replaceAllUsesWith(qubits);
          toErase.push_back(initOp);
        }
      }
    }
  }

  for (auto& op: toErase) {
    op->erase();
  }

  return success();
}

class StatePreparation : public cudaq::opt::PrepareStateBase<StatePreparation> {
protected:
  // The name of the kernel to be synthesized
  std::string kernelName;

  // The raw pointer to the runtime arguments.
  void *args;

public:
  StatePreparation() = default;
  StatePreparation(std::string_view kernel, void *a)
      : kernelName(kernel), args(a) {}

  mlir::ModuleOp getModule() { return getOperation(); }

  std::pair<std::size_t, std::vector<std::size_t>>
  getTargetLayout(FunctionType funcTy) {
    auto bufferTy = cudaq::opt::factory::buildInvokeStructType(funcTy);
    StringRef dataLayoutSpec = "";
    if (auto attr =
            getModule()->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
      dataLayoutSpec = cast<StringAttr>(attr);
    auto dataLayout = llvm::DataLayout(dataLayoutSpec);
    // Convert bufferTy to llvm.
    llvm::LLVMContext context;
    LLVMTypeConverter converter(funcTy.getContext());
    cudaq::opt::initializeTypeConversions(converter);
    auto llvmDialectTy = converter.convertType(bufferTy);
    LLVM::TypeToLLVMIRTranslator translator(context);
    auto *llvmStructTy =
        cast<llvm::StructType>(translator.translateType(llvmDialectTy));
    auto *layout = dataLayout.getStructLayout(llvmStructTy);
    auto strSize = layout->getSizeInBytes();
    std::vector<std::size_t> fieldOffsets;
    for (std::size_t i = 0, I = bufferTy.getMembers().size(); i != I; ++i)
      fieldOffsets.emplace_back(layout->getElementOffset(i));
    return {strSize, fieldOffsets};
  }

  void runOnOperation() override final {
    std::cout << "Module before state prep " << std::endl;
    auto module = getModule();
    module.dump();
    unsigned counter = 0;

    if (args == nullptr || kernelName.empty()) {
      module.emitOpError(
          "State preparation requires a kernel and the values of the "
          "arguments passed when it is called.");
      signalPassFailure();
      return;
    }

    auto kernelNameInQuake = cudaq::runtime::cudaqGenPrefixName + kernelName;
    // Get the function we care about (the one with kernelName)
    auto funcOp = module.lookupSymbol<func::FuncOp>(kernelNameInQuake);
    if (!funcOp) {
      module.emitOpError("The kernel '" + kernelName +
                         "' was not found in the module.");
      signalPassFailure();
      return;
    }

    // Create the builder.
    auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
    auto arguments = funcOp.getArguments();
    auto structLayout = getTargetLayout(funcOp.getFunctionType());
    // Keep track of the stdVec sizes.
    std::vector<std::tuple<std::size_t, Type, std::uint64_t>> stdVecInfo;

    for (auto iter : llvm::enumerate(arguments)) {
      auto argNum = iter.index();
      auto argument = iter.value();
      std::size_t offset = structLayout.second[argNum];

      // Get the argument type
      auto type = argument.getType();
      // auto loc = argument.getLoc();

      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(type)) {
        if (isa<cudaq::cc::StateType>(ptrTy.getElementType())) {
          std::cout << "State pointer found, TODO: call a kernel that created "
                       "the state"
                    << std::endl;
        }
      }

      // If std::vector<arithmetic> type, add it to the list of vector info.
      // These will be processed when we reach the buffer's appendix.
      if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(type)) {
        auto eleTy = vecTy.getElementType();
        if (!isa<IntegerType, FloatType, ComplexType>(eleTy)) {
          funcOp.emitOpError("synthesis: unsupported argument type");
          signalPassFailure();
          return;
        }
        char *ptrToSizeInBuffer = static_cast<char *>(args) + offset;
        auto sizeFromBuffer =
            *reinterpret_cast<std::uint64_t *>(ptrToSizeInBuffer);
        unsigned bytesInType = [&eleTy]() {
          if (auto complexTy = dyn_cast<ComplexType>(eleTy))
            return 2 * cudaq::opt::convertBitsToBytes(
                           complexTy.getElementType().getIntOrFloatBitWidth());
          return cudaq::opt::convertBitsToBytes(eleTy.getIntOrFloatBitWidth());
        }();
        assert(bytesInType > 0 && "element must have a size");
        auto vectorSize = sizeFromBuffer / bytesInType;
        stdVecInfo.emplace_back(argNum, eleTy, vectorSize);
        continue;
      }
    }

    // For any `std::vector` arguments, we now know the sizes so let's replace
    // the block arg with the actual vector element data. First get the pointer
    // to the start of the buffer's appendix.
    auto structSize = structLayout.first;
    char *bufferAppendix = static_cast<char *>(args) + structSize;
    for (auto [idx, eleTy, vecLength] : stdVecInfo) {
      if (!eleTy) {
        // FIXME: Skip struct values.
        bufferAppendix += vecLength;
        funcOp.emitOpError(
            "argument to kernel may be a struct and was not synthesized");
        continue;
      }
      auto doVector = [&]<typename T>(T) {
        auto *ptr = reinterpret_cast<T *>(bufferAppendix);
        auto v = convertToComplex(ptr, vecLength);
        if (failed(prepareStateFromVectorArgument(builder, module, counter,
                                                  arguments[idx], v)))
          funcOp.emitOpError("state preparation failed for vector<T>");
        bufferAppendix += vecLength * sizeof(T);
      };
      if (auto ty = dyn_cast<IntegerType>(eleTy)) {
        switch (ty.getIntOrFloatBitWidth()) {
        case 1:
          doVector(false);
          break;
        case 8:
          doVector(std::int8_t{});
          break;
        case 16:
          doVector(std::int16_t{});
          break;
        case 32:
          doVector(std::int32_t{});
          break;
        case 64:
          doVector(std::int64_t{});
          break;
        default:
          bufferAppendix += vecLength * cudaq::opt::convertBitsToBytes(
                                            ty.getIntOrFloatBitWidth());
          funcOp.emitOpError(
              "state preparation failed for vector<integral-type>.");
          break;
        }
        continue;
      }
      if (eleTy == builder.getF32Type()) {
        doVector(float{});
        continue;
      }
      if (eleTy == builder.getF64Type()) {
        doVector(double{});
        continue;
      }
      if (eleTy == ComplexType::get(builder.getF32Type())) {
        doVector(std::complex<float>{});
        continue;
      }
      if (eleTy == ComplexType::get(builder.getF64Type())) {
        doVector(std::complex<double>{});
        continue;
      }
    }
    std::cout << "Module after state preparation " << std::endl;
    module.dump();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createStatePreparation() {
  return std::make_unique<StatePreparation>();
}

std::unique_ptr<mlir::Pass>
cudaq::opt::createStatePreparation(std::string_view kernelName, void *a) {
  return std::make_unique<StatePreparation>(kernelName, a);
}
