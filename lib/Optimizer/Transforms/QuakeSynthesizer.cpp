/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "quake-synthesizer"

using namespace mlir;

// cudaq::state is defined in the runtime. The compiler will never need to know
// about its implementation and there should not be a circular build/library
// dependence because of it. Simply forward declare it, as it is notional.
namespace cudaq {
class state;
}

/// Replace a BlockArgument of a specific type with a concrete instantiation of
/// that type, and add the generation of that constant as an MLIR Op to the
/// beginning of the function. For example
///
///   func.func @foo( %arg0 : i32) {
///     quake.op1 (%arg0)
///   }
///
/// will be updated to
///
///   func.func @foo() {
///     %0 = arith.constant CONCRETE_ARG0 : i32
///     quake.op1(%0);
///   }
///
/// This function synthesizes the constant value and replaces all uses of the
/// BlockArgument with it.
template <typename ConcreteType>
void synthesizeRuntimeArgument(
    OpBuilder &builder, BlockArgument argument, const void *args,
    std::size_t offset, std::size_t typeSize,
    std::function<Value(OpBuilder &, ConcreteType *)> &&opGenerator) {

  // Create an instance of the concrete type
  ConcreteType concrete;
  // Copy the void* struct member into that concrete instance
  std::memcpy(&concrete, ((const char *)args) + offset, typeSize);

  // Generate the MLIR Value (arith constant for example)
  auto runtimeArg = opGenerator(builder, &concrete);

  // Most of the time, this arg will have an immediate stack allocation with
  // memref, remove those load uses and replace with the concrete op.
  if (!argument.getUsers().empty()) {
    auto firstUse = *argument.user_begin();
    if (dyn_cast<cudaq::cc::StoreOp>(firstUse)) {
      auto memrefValue = firstUse->getOperand(1);
      for (auto user : memrefValue.getUsers())
        if (auto load = dyn_cast<cudaq::cc::LoadOp>(user))
          load.getResult().replaceAllUsesWith(runtimeArg);
    }
  }
  argument.replaceAllUsesWith(runtimeArg);
}

template <typename T>
Value makeIntegerElement(OpBuilder &builder, Location argLoc, T val,
                         IntegerType eleTy) {
  return builder.create<arith::ConstantIntOp>(argLoc, val, eleTy);
}

template <typename T>
Value makeFloatElement(OpBuilder &builder, Location argLoc, T val,
                       FloatType eleTy) {
  return builder.create<arith::ConstantFloatOp>(argLoc, llvm::APFloat{val},
                                                eleTy);
}

template <typename T>
Value makeComplexElement(OpBuilder &builder, Location argLoc,
                         std::complex<T> val, ComplexType complexTy) {
  auto eleTy = complexTy.getElementType();
  auto realPart = builder.getFloatAttr(eleTy, llvm::APFloat{val.real()});
  auto imagPart = builder.getFloatAttr(eleTy, llvm::APFloat{val.imag()});
  auto complexVal = builder.getArrayAttr({realPart, imagPart});
  return builder.create<complex::ConstantOp>(argLoc, eleTy, complexVal);
}

/// returns true if and only if \p argument is used by a `quake.init_state`
/// operation.
static bool hasInitStateUse(BlockArgument argument) {
  for (auto *argUser : argument.getUsers())
    if (auto stdvecDataOp = dyn_cast<cudaq::cc::StdvecDataOp>(argUser))
      for (auto *dataUser : stdvecDataOp->getUsers())
        if (isa<quake::InitializeStateOp>(dataUser))
          return true;
  return false;
}

template <typename ELETY, typename T, typename ATTR, typename MAKER>
LogicalResult
synthesizeVectorArgument(OpBuilder &builder, ModuleOp module, unsigned &counter,
                         BlockArgument argument, std::vector<T> &vec,
                         ATTR arrayAttr, MAKER makeElementValue) {
  auto *ctx = builder.getContext();
  auto argTy = argument.getType();
  assert(isa<cudaq::cc::StdvecType>(argTy) ||
         isa<cudaq::cc::CharspanType>(argTy));
  ELETY eleTy = [&]() -> ELETY {
    if (auto strTy = dyn_cast<cudaq::cc::StdvecType>(argTy))
      return cast<ELETY>(strTy.getElementType());
    // Force cast this to ELETY. This will only happen for CharspanType.
    return cast<ELETY>(cudaq::opt::factory::getCharType(ctx));
  }();
  auto strTy = cudaq::cc::StdvecType::get(ctx, eleTy);
  builder.setInsertionPointToStart(argument.getOwner());
  auto argLoc = argument.getLoc();
  auto conArray = builder.create<cudaq::cc::ConstantArrayOp>(
      argLoc, cudaq::cc::ArrayType::get(ctx, eleTy, vec.size()), arrayAttr);
  auto arrTy = cudaq::cc::ArrayType::get(ctx, eleTy, vec.size());
  std::optional<Value> arrayInMemory;
  auto ptrEleTy = cudaq::cc::PointerType::get(eleTy);
  bool generateNewValue = false;

  // Helper function that materializes the array in memory.
  auto getArrayInMemory = [&]() -> Value {
    if (arrayInMemory)
      return *arrayInMemory;
    OpBuilder::InsertionGuard guard(builder);
    Value buffer;
    if (hasInitStateUse(argument)) {
      // Stick global at end of Module.
      std::string symbol =
          "__nvqpp_rodata_init_state." + std::to_string(counter++);

      cudaq::IRBuilder irBuilder(builder);
      irBuilder.genVectorOfConstants(argLoc, module, symbol, vec);

      builder.setInsertionPointToStart(argument.getOwner());
      buffer = builder.create<cudaq::cc::AddressOfOp>(
          argLoc, cudaq::cc::PointerType::get(arrTy), symbol);
    } else {
      builder.setInsertionPointAfter(conArray);
      buffer = builder.create<cudaq::cc::AllocaOp>(argLoc, arrTy);
      builder.create<cudaq::cc::StoreOp>(argLoc, conArray, buffer);
    }

    auto ptrArrEleTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(eleTy));
    Value res = builder.create<cudaq::cc::CastOp>(argLoc, ptrArrEleTy, buffer);
    arrayInMemory = res;
    return res;
  };

  auto replaceLoads = [&](cudaq::cc::ComputePtrOp elePtrOp,
                          Value newVal) -> LogicalResult {
    bool allLoadUsers = true;
    for (auto *u : elePtrOp->getUsers()) {
      if (auto loadOp = dyn_cast<cudaq::cc::LoadOp>(u))
        loadOp.replaceAllUsesWith(newVal);
      else
        allLoadUsers = false;
    }
    return success(allLoadUsers);
  };

  // Iterate over the users of this stdvec argument.
  for (auto *argUser : argument.getUsers()) {
    // Handle the StdvecSize use case.
    // Replace a `vec.size()` with the length, which is a synthesized constant.
    if (auto stdvecSizeOp = dyn_cast<cudaq::cc::StdvecSizeOp>(argUser)) {
      Value length = builder.create<arith::ConstantIntOp>(
          argLoc, vec.size(), stdvecSizeOp.getType());
      stdvecSizeOp.replaceAllUsesWith(length);
      continue;
    }

    // Handle the StdvecDataOp use cases.  We expect `vec.data()` to be indexed
    // and the value loaded, `vec.data()[c]`. Handle the cases where the offset,
    // `c`, is a constant as well as cases when it is not. Also handle the case
    // when the `vec.data()` is used in an arbitrary pointer expression.
    if (auto stdvecDataOp = dyn_cast<cudaq::cc::StdvecDataOp>(argUser)) {
      bool replaceOtherUses = false;
      for (auto *dataUser : stdvecDataOp->getUsers()) {
        // dataUser could be a load, a computeptr, or something else. If it's a
        // load, the index is 0: get the 0-th value from the array and forward
        // it. If it's a computeptr, then we get the element from the array at
        // the index and forward it. There are two cases: (1) the element offset
        // is constant and (2) the element offset is some computed value. Both
        // cases are quite similar with the variation on the offset argument
        // being a constant or a value.
        if (auto loadOp = dyn_cast<cudaq::cc::LoadOp>(dataUser)) {
          // Load the first (0) element.
          Value runtimeParam = makeElementValue(builder, argLoc, vec[0], eleTy);
          // Replace with the constant value
          loadOp.replaceAllUsesWith(runtimeParam);
          continue;
        }
        if (auto elePtrOp = dyn_cast<cudaq::cc::ComputePtrOp>(dataUser)) {
          auto index = elePtrOp.getRawConstantIndices()[0];
          if (index == cudaq::cc::ComputePtrOp::kDynamicIndex) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(elePtrOp);
            Value getEle = builder.create<cudaq::cc::ExtractValueOp>(
                elePtrOp.getLoc(), eleTy, conArray,
                elePtrOp.getDynamicIndices()[0]);
            if (failed(replaceLoads(elePtrOp, getEle))) {
              Value memArr = getArrayInMemory();
              builder.setInsertionPoint(elePtrOp);
              Value newComputedPtr = builder.create<cudaq::cc::ComputePtrOp>(
                  argLoc, ptrEleTy, memArr, elePtrOp.getDynamicIndices()[0]);
              elePtrOp.replaceAllUsesWith(newComputedPtr);
            }
            continue;
          }
          Value runtimeParam =
              makeElementValue(builder, argLoc, vec[index], eleTy);
          if (failed(replaceLoads(elePtrOp, runtimeParam))) {
            Value memArr = getArrayInMemory();
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(elePtrOp);
            Value newComputedPtr = builder.create<cudaq::cc::ComputePtrOp>(
                argLoc, ptrEleTy, memArr,
                SmallVector<cudaq::cc::ComputePtrArg>{0, index});
            elePtrOp.replaceAllUsesWith(newComputedPtr);
          }
          continue;
        }
        replaceOtherUses = true;
      }
      // Check if there were other uses of `vec.data()` and simply forward the
      // constant array as materialized in memory.
      if (replaceOtherUses) {
        auto memArr = getArrayInMemory();
        stdvecDataOp.replaceAllUsesWith(memArr);
      }
      continue;
    }

    // In the event that the stdvec value is simply used as is, we want to
    // construct a new, constant vector in place and replace users of the
    // argument with it.
    generateNewValue = true;
  }
  if (generateNewValue) {
    Value memArr = getArrayInMemory();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(memArr.getDefiningOp());
    Value size = builder.create<arith::ConstantIntOp>(argLoc, vec.size(), 64);
    Value newVec =
        builder.create<cudaq::cc::StdvecInitOp>(argLoc, strTy, memArr, size);
    argument.replaceAllUsesWith(newVec);
  }
  return success();
}

template <typename A>
std::vector<std::int32_t> asI32(const std::vector<A> &v) {
  std::vector<std::int32_t> result(v.size());
  for (auto iter : llvm::enumerate(v))
    result[iter.index()] = static_cast<std::int32_t>(iter.value());
  return result;
}

// TODO: consider using DenseArrayAttr here instead. NB: such a change may alter
// the output of the constant array op.
static LogicalResult
synthesizeVectorArgument(OpBuilder &builder, ModuleOp module, unsigned &counter,
                         BlockArgument argument, std::vector<bool> &vec) {
  auto arrayAttr = builder.getI32ArrayAttr(asI32(vec));
  return synthesizeVectorArgument<IntegerType>(builder, module, counter,
                                               argument, vec, arrayAttr,
                                               makeIntegerElement<bool>);
}

static LogicalResult synthesizeVectorArgument(OpBuilder &builder,
                                              ModuleOp module,
                                              unsigned &counter,
                                              BlockArgument argument,
                                              std::vector<std::int8_t> &vec) {
  auto arrayAttr = builder.getI32ArrayAttr(asI32(vec));
  return synthesizeVectorArgument<IntegerType>(builder, module, counter,
                                               argument, vec, arrayAttr,
                                               makeIntegerElement<std::int8_t>);
}

static LogicalResult synthesizeVectorArgument(OpBuilder &builder,
                                              ModuleOp module,
                                              unsigned &counter,
                                              BlockArgument argument,
                                              std::vector<std::int16_t> &vec) {
  auto arrayAttr = builder.getI32ArrayAttr(asI32(vec));
  return synthesizeVectorArgument<IntegerType>(
      builder, module, counter, argument, vec, arrayAttr,
      makeIntegerElement<std::int16_t>);
}

static LogicalResult synthesizeVectorArgument(OpBuilder &builder,
                                              ModuleOp module,
                                              unsigned &counter,
                                              BlockArgument argument,
                                              std::vector<std::int32_t> &vec) {
  auto arrayAttr = builder.getI32ArrayAttr(vec);
  return synthesizeVectorArgument<IntegerType>(
      builder, module, counter, argument, vec, arrayAttr,
      makeIntegerElement<std::int32_t>);
}

static LogicalResult synthesizeVectorArgument(OpBuilder &builder,
                                              ModuleOp module,
                                              unsigned &counter,
                                              BlockArgument argument,
                                              std::vector<std::int64_t> &vec) {
  auto arrayAttr = builder.getI64ArrayAttr(vec);
  return synthesizeVectorArgument<IntegerType>(
      builder, module, counter, argument, vec, arrayAttr,
      makeIntegerElement<std::int64_t>);
}

static LogicalResult
synthesizeVectorArgument(OpBuilder &builder, ModuleOp module, unsigned &counter,
                         BlockArgument argument, std::vector<float> &vec) {
  auto arrayAttr = builder.getF32ArrayAttr(vec);
  return synthesizeVectorArgument<FloatType>(builder, module, counter, argument,
                                             vec, arrayAttr,
                                             makeFloatElement<float>);
}

static LogicalResult
synthesizeVectorArgument(OpBuilder &builder, ModuleOp module, unsigned &counter,
                         BlockArgument argument, std::vector<double> &vec) {
  auto arrayAttr = builder.getF64ArrayAttr(vec);
  return synthesizeVectorArgument<FloatType>(builder, module, counter, argument,
                                             vec, arrayAttr,
                                             makeFloatElement<double>);
}

static LogicalResult
synthesizeVectorArgument(OpBuilder &builder, ModuleOp module, unsigned &counter,
                         BlockArgument argument,
                         std::vector<std::complex<float>> &vec) {
  std::vector<float> vec2;
  for (auto c : vec) {
    vec2.push_back(c.real());
    vec2.push_back(c.imag());
  }
  auto arrayAttr = builder.getF32ArrayAttr(vec2);
  return synthesizeVectorArgument<ComplexType>(builder, module, counter,
                                               argument, vec, arrayAttr,
                                               makeComplexElement<float>);
}

static LogicalResult
synthesizeVectorArgument(OpBuilder &builder, ModuleOp module, unsigned &counter,
                         BlockArgument argument,
                         std::vector<std::complex<double>> &vec) {
  std::vector<double> vec2;
  for (auto c : vec) {
    vec2.push_back(c.real());
    vec2.push_back(c.imag());
  }
  auto arrayAttr = builder.getF64ArrayAttr(vec2);
  return synthesizeVectorArgument<ComplexType>(builder, module, counter,
                                               argument, vec, arrayAttr,
                                               makeComplexElement<double>);
}

namespace {
class QuakeSynthesizer
    : public cudaq::opt::QuakeSynthesizeBase<QuakeSynthesizer> {
protected:
  // The name of the kernel to be synthesized
  std::string kernelName;

  // The raw pointer to the runtime arguments.
  const void *args;

  // The starting argument index to synthesize. Typically 0 but may be >0 for
  // partial synthesis. If >0, it is assumed that the first argument(s) are NOT
  // in `args`.
  std::size_t startingArgIdx = 0;

  // The program is executed in the same address space as the synthesis.
  bool sameAddressSpace = false;

public:
  QuakeSynthesizer() = default;
  QuakeSynthesizer(std::string_view kernel, const void *a, std::size_t s,
                   bool sameAddrSpace)
      : kernelName(kernel), args(a), startingArgIdx(s),
        sameAddressSpace(sameAddrSpace) {}

  mlir::ModuleOp getModule() { return getOperation(); }

  std::pair<std::size_t, std::vector<std::size_t>>
  getTargetLayout(FunctionType funcTy) {
    auto bufferTy =
        cudaq::opt::factory::buildInvokeStructType(funcTy, startingArgIdx);
    StringRef dataLayoutSpec = "";
    if (auto attr =
            getModule()->getAttr(cudaq::opt::factory::targetDataLayoutAttrName))
      dataLayoutSpec = cast<StringAttr>(attr);
    llvm::DataLayout dataLayout{dataLayoutSpec};
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
    auto module = getModule();
    unsigned counter = 0;

    if (args == nullptr || kernelName.empty()) {
      module.emitOpError("Synthesis requires a kernel and the values of the "
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

    // Create the builder and get the function arguments.
    // We will remove these arguments and replace with constant ops
    auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
    auto arguments = funcOp.getArguments();
    auto structLayout = getTargetLayout(funcOp.getFunctionType());
    // Keep track of the stdVec sizes.
    std::vector<std::tuple<std::size_t, Type, std::uint64_t>> stdVecInfo;

    for (std::size_t argNum = startingArgIdx, end = arguments.size();
         argNum < end; argNum++) {
      auto argument = arguments[argNum];
      std::size_t offset = structLayout.second[argNum - startingArgIdx];

      // Get the argument type
      auto type = argument.getType();
      auto loc = argument.getLoc();

      // Based on the type, we want to replace it with a concrete constant op.
      // Process scalar integral types.
      if (auto ty = dyn_cast<IntegerType>(type)) {
        switch (ty.getIntOrFloatBitWidth()) {
        case 1:
          synthesizeRuntimeArgument<bool>(
              builder, argument, args, offset, sizeof(bool),
              [=](OpBuilder &builder, bool *concrete) {
                return builder.create<arith::ConstantIntOp>(loc, *concrete, 1);
              });
          break;
        case 8:
          synthesizeRuntimeArgument<std::uint8_t>(
              builder, argument, args, offset, sizeof(std::uint8_t),
              [=](OpBuilder &builder, std::uint8_t *concrete) {
                return builder.create<arith::ConstantIntOp>(loc, *concrete, 8);
              });
          break;
        case 16:
          synthesizeRuntimeArgument<std::int16_t>(
              builder, argument, args, offset, sizeof(std::int16_t),
              [=](OpBuilder &builder, std::int16_t *concrete) {
                return builder.create<arith::ConstantIntOp>(loc, *concrete, 16);
              });
          break;
        case 32:
          synthesizeRuntimeArgument<std::int32_t>(
              builder, argument, args, offset, sizeof(std::int32_t),
              [=](OpBuilder &builder, std::int32_t *concrete) {
                return builder.create<arith::ConstantIntOp>(loc, *concrete, 32);
              });
          break;
        case 64:
          synthesizeRuntimeArgument<std::int64_t>(
              builder, argument, args, offset, sizeof(std::int64_t),
              [=](OpBuilder &builder, std::int64_t *concrete) {
                return builder.create<arith::ConstantIntOp>(loc, *concrete, 64);
              });
          break;
        default:
          funcOp.emitOpError(
              "argument was not synthesized, unhandled integer type");
          break;
        }
        continue;
      }

      // Process scalar floating point types.
      if (type == builder.getF32Type()) {
        synthesizeRuntimeArgument<float>(
            builder, argument, args, offset,
            cudaq::opt::convertBitsToBytes(type.getIntOrFloatBitWidth()),
            [=](OpBuilder &builder, float *concrete) {
              llvm::APFloat f(*concrete);
              return builder.create<arith::ConstantFloatOp>(
                  loc, f, builder.getF32Type());
            });
        continue;
      }
      if (type == builder.getF64Type()) {
        synthesizeRuntimeArgument<double>(
            builder, argument, args, offset,
            cudaq::opt::convertBitsToBytes(type.getIntOrFloatBitWidth()),
            [=](OpBuilder &builder, double *concrete) {
              llvm::APFloat f(*concrete);
              return builder.create<arith::ConstantFloatOp>(
                  loc, f, builder.getF64Type());
            });
        continue;
      }

      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(type)) {
        if (isa<cudaq::cc::StateType>(ptrTy.getElementType())) {
          // Special case of a `cudaq::state*` which must be in the same address
          // space. This references a container to a set of simulation
          // amplitudes.
          if (sameAddressSpace) {
            synthesizeRuntimeArgument<cudaq::state *>(
                builder, argument, args, offset, sizeof(void *),
                [=](OpBuilder &builder, cudaq::state **concrete) {
                  Value rawPtr = builder.create<arith::ConstantIntOp>(
                      loc, reinterpret_cast<std::intptr_t>(*concrete),
                      sizeof(void *) * 8);
                  auto stateTy =
                      cudaq::cc::StateType::get(builder.getContext());
                  return builder.create<cudaq::cc::CastOp>(
                      loc, cudaq::cc::PointerType::get(stateTy), rawPtr);
                });
            continue;
          } else {
            funcOp.emitOpError("synthesis: unsupported argument type for "
                               "remote devices and simulators: state*");
            signalPassFailure();
          }
        }
        // N.B. Other pointers will not be materialized and may be in a
        // different address space.
      }

      // If std::vector<arithmetic> type, add it to the list of vector info.
      // These will be processed when we reach the buffer's appendix.
      if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(type)) {
        auto eleTy = vecTy.getElementType();
        if (!isa<IntegerType, FloatType, ComplexType, cudaq::cc::CharspanType>(
                eleTy)) {
          funcOp.emitOpError("synthesis: unsupported argument type");
          signalPassFailure();
          return;
        }
        const char *ptrToSizeInBuffer =
            static_cast<const char *>(args) + offset;
        auto sizeFromBuffer =
            *reinterpret_cast<const std::uint64_t *>(ptrToSizeInBuffer);
        auto bytesInType = [&eleTy]() -> unsigned {
          if (isa<cudaq::cc::CharspanType>(eleTy)) {
            /* A charspan is a struct{ ptr, i64 }, which is just an i64 in
             * pointer-free encoding. */
            return sizeof(std::int64_t);
          }
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

      if (isa<cudaq::cc::CallableType>(type)) {
        // TODO: for now we ignore the passing of callable arguments.
        continue;
      }

      // The struct type ends up as a i64 in the thunk kernel args pointer, so
      // just skip ahead. TODO: add support for struct types!
      if (auto structTy = dyn_cast<cudaq::cc::StructType>(type)) {
        if (structTy.isEmpty()) {
          // TODO: for now we can ignore empty struct types.
          continue;
        }
        const char *ptrToSizeInBuffer =
            static_cast<const char *>(args) + offset;
        auto rawSize =
            *reinterpret_cast<const std::uint64_t *>(ptrToSizeInBuffer);
        stdVecInfo.emplace_back(argNum, Type{}, rawSize);
        continue;
      }

      if (auto charSpanTy = dyn_cast<cudaq::cc::CharspanType>(type)) {
        const char *ptrToSizeInBuffer =
            static_cast<const char *>(args) + offset;
        auto sizeFromBuffer =
            *reinterpret_cast<const std::uint64_t *>(ptrToSizeInBuffer);
        std::size_t bytesInType = sizeof(char);
        auto vectorSize = sizeFromBuffer / bytesInType;
        stdVecInfo.emplace_back(
            argNum, cudaq::opt::factory::getCharType(builder.getContext()),
            vectorSize);
        continue;
      }

      funcOp.emitOpError("We cannot synthesize argument(s) of this type.");
      signalPassFailure();
      return;
    }

    // For any `std::vector` arguments, we now know the sizes so let's replace
    // the block arg with the actual vector element data. First get the pointer
    // to the start of the buffer's appendix.
    auto structSize = structLayout.first;
    const char *bufferAppendix = static_cast<const char *>(args) + structSize;
    for (auto [idx, eleTy, vecLength] : stdVecInfo) {
      if (!eleTy) {
        // FIXME: Skip struct values.
        bufferAppendix += vecLength;
        funcOp.emitOpError(
            "argument to kernel may be a struct and was not synthesized");
        continue;
      }
      auto doVector = [&]<typename T>(T) {
        auto *ptr = reinterpret_cast<const T *>(bufferAppendix);
        std::vector<T> v(ptr, ptr + vecLength);
        if (failed(synthesizeVectorArgument(builder, module, counter,
                                            arguments[idx], v)))
          funcOp.emitOpError("synthesis failed for vector<T>");
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
          funcOp.emitOpError("synthesis failed for vector<integral-type>.");
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
      if (auto charSpanTy = dyn_cast<cudaq::cc::CharspanType>(eleTy)) {
        // For this case, the message buffer contained the length of the range
        // of sizes that are encoded starting at bufferAppendix.
        // At the end of the block of sizes, the C-strings will be encoded.
        auto numberSpans = vecLength;
        auto *spanSizes =
            reinterpret_cast<const std::uint64_t *>(bufferAppendix);
        bufferAppendix += vecLength * sizeof(std::uint64_t);
        // These strings are reified in the following way:
        //   - Create an array numberSpans in length and where each element
        //     has a `{i8*, i64}` type.
        //   - Create a C-string literal constant (global) for each string in
        //     this vector for a total of numberSpans.
        //   - Save the address of the C-string to each element of the array.
        // Users of this data structure will need to use compute_ptr to access
        // the elements, which are the pointers. Each ptr element is a `char*`
        // that can be used in, say, a Pauli op.
        auto ptrTy = cudaq::cc::PointerType::get(charSpanTy);
        auto loc = arguments[idx].getLoc();
        auto ns = builder.create<arith::ConstantIntOp>(loc, numberSpans, 64);
        auto aos = builder.create<cudaq::cc::AllocaOp>(loc, charSpanTy, ns);
        auto pi8Ty = cudaq::cc::PointerType::get(charSpanTy.getElementType());
        auto ppi8Ty = cudaq::cc::PointerType::get(pi8Ty);
        auto ptrI64Ty = cudaq::cc::PointerType::get(builder.getI64Type());
        auto iaTy = cudaq::cc::PointerType::get(
            cudaq::cc::ArrayType::get(builder.getI64Type()));
        cudaq::IRBuilder irBuilder(module);
        for (decltype(numberSpans) i = 0; i < numberSpans; ++i) {
          std::size_t length = spanSizes[i];
          auto strLen = builder.create<arith::ConstantIntOp>(loc, length, 64);
          StringRef strData{bufferAppendix, length};
          auto global =
              irBuilder.genCStringLiteralAppendNul(loc, module, strData);
          auto addr = builder.create<cudaq::cc::AddressOfOp>(
              loc, cudaq::cc::PointerType::get(global.getType()),
              global.getName());
          auto str = builder.create<cudaq::cc::CastOp>(loc, pi8Ty, addr);
          auto spanp = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrTy, aos,
              ArrayRef<cudaq::cc::ComputePtrArg>{static_cast<std::int32_t>(i)});
          auto relocp = builder.create<cudaq::cc::CastOp>(loc, ppi8Ty, spanp);
          builder.create<cudaq::cc::StoreOp>(loc, str, relocp);
          auto lengthp = builder.create<cudaq::cc::CastOp>(loc, iaTy, spanp);
          auto offsetp = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrI64Ty, lengthp, ArrayRef<cudaq::cc::ComputePtrArg>{1});
          builder.create<cudaq::cc::StoreOp>(loc, strLen, offsetp);
          bufferAppendix += length;
        }
        auto svTy = cudaq::cc::StdvecType::get(ptrTy);
        auto ics = builder.create<cudaq::cc::StdvecInitOp>(loc, svTy, aos, ns);
        arguments[idx].replaceAllUsesWith(ics);
        continue;
      }
    }

    // Clean up dead code.
    {
      IRRewriter rewriter(builder);
      [[maybe_unused]] auto unused =
          simplifyRegions(rewriter, {funcOp.getBody()});
    }

    // Remove the old arguments.
    auto numArgs = funcOp.getNumArguments();
    BitVector argsToErase(numArgs);
    for (std::size_t argIndex = startingArgIdx; argIndex < numArgs;
         ++argIndex) {
      argsToErase.set(argIndex);
      if (!funcOp.getBody().front().getArgument(argIndex).getUses().empty()) {
        funcOp.emitError("argument(s) still in use after synthesis.");
        signalPassFailure();
        return;
      }
    }
    funcOp.eraseArguments(argsToErase);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQuakeSynthesizer() {
  return std::make_unique<QuakeSynthesizer>();
}

std::unique_ptr<mlir::Pass>
cudaq::opt::createQuakeSynthesizer(std::string_view kernelName, const void *a,
                                   std::size_t startingArgIdx,
                                   bool sameAddressSpace) {
  return std::make_unique<QuakeSynthesizer>(kernelName, a, startingArgIdx,
                                            sameAddressSpace);
}
