/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

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
    OpBuilder &builder, BlockArgument &argument, void *args,
    std::size_t &offset, std::size_t typeSize,
    std::function<Value(OpBuilder &, ConcreteType *)> &&opGenerator) {

  // Create an instance of the concrete type
  ConcreteType concrete;
  // Copy the void* struct member into that concrete instance
  std::memcpy(&concrete, ((char *)args) + offset, typeSize);
  // Increment the offset for the next argument
  offset += typeSize;

  // Generate the MLIR Value (arith constant for example)
  auto runtimeArg = opGenerator(builder, &concrete);

  // Most of the time, this arg will have an immediate
  // stack allocation with memref, remove those load uses
  // and replace with the concrete op.
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

static LogicalResult synthesizeVectorArgument(OpBuilder &builder,
                                              BlockArgument &argument,
                                              std::vector<double> &vec) {
  auto *ctx = builder.getContext();
  assert(isa<cudaq::cc::StdvecType>(argument.getType()));
  auto eleTy = cast<cudaq::cc::StdvecType>(argument.getType()).getElementType();
  auto arrayAttr = builder.getF64ArrayAttr(vec);
  builder.setInsertionPointToStart(argument.getOwner());
  auto conArray = builder.create<cudaq::cc::ConstantArrayOp>(
      argument.getLoc(), cudaq::cc::ArrayType::get(ctx, eleTy, vec.size()),
      arrayAttr);
  auto replaceLoads = [&](cudaq::cc::ComputePtrOp gepOp,
                          Value newVal) -> LogicalResult {
    for (auto *u : gepOp->getUsers()) {
      if (auto loadOp = dyn_cast<cudaq::cc::LoadOp>(u)) {
        loadOp.replaceAllUsesWith(newVal);
        continue;
      }
      return gepOp.emitError("Unknown gep/load configuration for quake-synth.");
    }
    return success();
  };
  for (auto *argUser : argument.getUsers()) {
    if (auto stdvecDataOp = dyn_cast<cudaq::cc::StdvecDataOp>(argUser)) {
      for (auto *dataUser : stdvecDataOp->getUsers()) {
        // could be a load, or a getelementptr.
        // if load, the index is 0
        // if getelementptr, then we get the index there to use
        if (auto loadOp = dyn_cast<cudaq::cc::LoadOp>(dataUser)) {
          llvm::APFloat f(vec[0]);
          Value runtimeParam = builder.create<arith::ConstantFloatOp>(
              argument.getLoc(), f, builder.getF64Type());
          // Replace with the constant value
          loadOp.replaceAllUsesWith(runtimeParam);
          continue;
        }
        if (auto gepOp = dyn_cast<cudaq::cc::ComputePtrOp>(dataUser)) {
          auto index = gepOp.getRawConstantIndices()[0];
          if (index == cudaq::cc::ComputePtrOp::kDynamicIndex) {
            builder.setInsertionPoint(gepOp);
            Value getEle = builder.create<cudaq::cc::GetConstantElementOp>(
                gepOp.getLoc(), eleTy, conArray, gepOp.getDynamicIndices()[0]);
            if (failed(replaceLoads(gepOp, getEle)))
              return failure();
            continue;
          }
          llvm::APFloat f(vec[index]);
          Value runtimeParam = builder.create<arith::ConstantFloatOp>(
              argument.getLoc(), f, builder.getF64Type());
          if (failed(replaceLoads(gepOp, runtimeParam)))
            return failure();
        } else {
          return dataUser->emitError(
              "unexpected use of std::vector<T>::data()");
        }
      }
    }
  }
  return success();
}

namespace {
class QuakeSynthesizer
    : public cudaq::opt::QuakeSynthesizeBase<QuakeSynthesizer> {
protected:
  // The name of the kernel to be synthesized
  std::string kernelName;

  // The raw pointer to the runtime arguments.
  void *args;

public:
  QuakeSynthesizer() = default;
  QuakeSynthesizer(std::string_view kernel, void *a)
      : kernelName(kernel), args(a) {}

  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto module = getModule();
    if (args == nullptr || kernelName.empty()) {
      emitError(
          module.getLoc(),
          "Quake Synthesis requires runtime arguments and the kernel name.\n");
      signalPassFailure();
    }

    for (auto &op : *module.getBody()) {
      // Get the function we care about (the one with kernelName)
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp)
        continue;

      if (!funcOp.getName().startswith(cudaq::runtime::cudaqGenPrefixName +
                                       kernelName))
        continue;

      // Create the builder and get the function arguments.
      // We will remove these arguments and replace with constant ops
      auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
      auto arguments = funcOp.getArguments();

      // Keep track of the stdVec sizes.
      std::vector<std::size_t> stdVecSizes;

      // For each argument, get the type and synthesize
      // the runtime constant value.
      std::size_t offset = 0;
      for (auto &argument : arguments) {
        // Get the argument type
        auto type = argument.getType();

        // Based on the type, we want to replace it
        // with a concrete constant op.
        if (type == builder.getIntegerType(1)) {
          synthesizeRuntimeArgument<bool>(
              builder, argument, args, offset, sizeof(bool),
              [](OpBuilder &builder, bool *concrete) {
                return builder.create<arith::ConstantIntOp>(
                    builder.getUnknownLoc(), *concrete, 1);
              });
        } else if (type == builder.getIntegerType(8)) {
          synthesizeRuntimeArgument<std::uint8_t>(
              builder, argument, args, offset, sizeof(std::uint8_t),
              [](OpBuilder &builder, std::uint8_t *concrete) {
                return builder.create<arith::ConstantIntOp>(
                    builder.getUnknownLoc(), *concrete, 8);
              });
        } else if (type == builder.getIntegerType(32)) {
          synthesizeRuntimeArgument<int>(
              builder, argument, args, offset, sizeof(int),
              [](OpBuilder &builder, int *concrete) {
                return builder.create<arith::ConstantIntOp>(
                    builder.getUnknownLoc(), *concrete, 32);
              });
        } else if (type == builder.getIntegerType(64)) {
          synthesizeRuntimeArgument<long>(
              builder, argument, args, offset, sizeof(long),
              [](OpBuilder &builder, long *concrete) {
                return builder.create<arith::ConstantIntOp>(
                    builder.getUnknownLoc(), *concrete, 64);
              });
        } else if (type == builder.getF32Type()) {
          synthesizeRuntimeArgument<float>(
              builder, argument, args, offset, type.getIntOrFloatBitWidth() / 8,
              [](OpBuilder &builder, float *concrete) {
                llvm::APFloat f(*concrete);
                return builder.create<arith::ConstantFloatOp>(
                    builder.getUnknownLoc(), f, builder.getF32Type());
              });
        } else if (type == builder.getF64Type()) {
          synthesizeRuntimeArgument<double>(
              builder, argument, args, offset, type.getIntOrFloatBitWidth() / 8,
              [](OpBuilder &builder, double *concrete) {
                llvm::APFloat f(*concrete);
                return builder.create<arith::ConstantFloatOp>(
                    builder.getUnknownLoc(), f, builder.getF64Type());
              });
        } else if (isa<cudaq::cc::StdvecType>(type)) {
          std::size_t vectorSize = *(std::size_t *)(((char *)args) + offset);
          vectorSize /= sizeof(double);
          offset += sizeof(std::size_t);
          stdVecSizes.push_back(vectorSize);
        } else if (isa<cudaq::cc::StructType, cudaq::cc::CallableType>(type)) {
          // The struct type ends up as a i64 in the thunk kernel
          // args pointer, so just skip ahead.
          offset += sizeof(std::size_t);
        } else {
          type.dump();
          TODO("We cannot synthesize this type of argument yet.");
        }
      }

      // For any `std::vector` arguments, we now know the sizes so let's replace
      // the block arg with the actual vector element data.
      double *ptr = (double *)(((char *)args) + offset);
      for (std::size_t idx = 0; auto &stdVecSize : stdVecSizes) {
        // FIXME: this only works with std::vector<double> for now.
        std::vector<double> v(ptr, ptr + stdVecSize);
        if (failed(synthesizeVectorArgument(builder, arguments[idx++], v))) {
          funcOp.emitError("Quake Synthesis failed for stdvec type.");
          signalPassFailure();
        }
      }

      // Clean up dead code.
      {
        IRRewriter rewriter(builder);
        (void)simplifyRegions(rewriter, {funcOp.getBody()});
      }

      // Remove the old arguments.
      auto numArgs = funcOp.getNumArguments();
      BitVector argsToErase(numArgs);
      for (std::size_t argIndex = 0; argIndex < numArgs; ++argIndex) {
        argsToErase.set(argIndex);
        if (!funcOp.getBody().front().getArgument(argIndex).getUses().empty()) {
          funcOp.emitError("argument(s) still in use after synthesis.");
          signalPassFailure();
          return;
        }
      }
      funcOp.eraseArguments(argsToErase);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQuakeSynthesizer() {
  return std::make_unique<QuakeSynthesizer>();
}

std::unique_ptr<mlir::Pass>
cudaq::opt::createQuakeSynthesizer(std::string_view kernelName, void *a) {
  return std::make_unique<QuakeSynthesizer>(kernelName, a);
}
