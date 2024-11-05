/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ArgumentConversion.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
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
  cudaq::IRBuilder irBuilder(builder);
  auto cString = irBuilder.genCStringLiteralAppendNul(loc, substMod, v);
  auto addr = builder.create<cudaq::cc::AddressOfOp>(
      loc, cudaq::cc::PointerType::get(cString.getType()), cString.getName());
  auto i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());
  auto cast = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, addr);
  auto size = builder.create<arith::ConstantIntOp>(loc, v.size(), 64);
  auto chSpanTy = cudaq::cc::CharspanType::get(builder.getContext());
  return builder.create<cudaq::cc::StdvecInitOp>(loc, chSpanTy, cast, size);
}

// Forward declare aggregate type builder as they can be recursive.
static Value genConstant(OpBuilder &, cudaq::cc::StdvecType, void *,
                         ModuleOp substMod, llvm::DataLayout &);
static Value genConstant(OpBuilder &, cudaq::cc::StructType, void *,
                         ModuleOp substMod, llvm::DataLayout &);
static Value genConstant(OpBuilder &, cudaq::cc::ArrayType, void *,
                         ModuleOp substMod, llvm::DataLayout &);

static Value genConstant(OpBuilder &builder, const cudaq::state *v,
                         ModuleOp substMod, llvm::DataLayout &layout,
                         llvm::StringRef kernelName, bool isSimulator) {
  if (isSimulator) {
    // The program is executed remotely, materialize the simulation data
    // into an array and create a new state from it.
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
    auto stateTy = cudaq::cc::StateType::get(ctx);
    auto statePtrTy = cudaq::cc::PointerType::get(stateTy);

    return builder.create<cudaq::cc::CreateStateOp>(loc, statePtrTy, buffer,
                                                    arrSize);
  }
  // The program is executed on quantum hardware, state data is not
  // available and needs to be regenerated.
  TODO("cudaq::state* argument synthesis for quantum hardware");
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

Value genConstant(OpBuilder &builder, cudaq::cc::StdvecType vecTy, void *p,
                  ModuleOp substMod, llvm::DataLayout &layout) {
  typedef const char *VectorType[3];
  VectorType *vecPtr = static_cast<VectorType *>(p);
  auto delta = (*vecPtr)[1] - (*vecPtr)[0];
  if (!delta)
    return {};
  auto eleTy = vecTy.getElementType();
  auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
  auto eleSize = cudaq::opt::getDataSize(layout, eleTy);
  if (isa<cudaq::cc::CharspanType>(eleTy)) {
    // char span type (i.e. pauli word) is a `vector<char>`
    eleSize = sizeof(VectorType);
  }

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
                                                 ModuleOp sourceModule,
                                                 bool isSimulator)
    : sourceModule(sourceModule), builder(sourceModule.getContext()),
      kernelName(kernelName), isSimulator(isSimulator) {
  substModule = builder.create<ModuleOp>(builder.getUnknownLoc());
}

void cudaq::opt::ArgumentConverter::gen(const std::vector<void *> &arguments) {
  auto *ctx = builder.getContext();
  // We should look up the input type signature here.

  auto fun = sourceModule.lookupSymbol<func::FuncOp>(
      cudaq::runtime::cudaqGenPrefixName + kernelName.str());
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
              if (ptrTy.getElementType() == cc::StateType::get(ctx))
                return buildSubst(static_cast<const state *>(argPtr),
                                  substModule, dataLayout, kernelName,
                                  isSimulator);
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
      substitutions.emplace_back(std::move(subst));
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
