/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

template <typename R>
R getParentOfType(Operation *op) {
  do {
    op = op->getParentOp();
    if (auto r = dyn_cast_or_null<R>(op))
      return r;
  } while (op);
  return {};
}

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

LogicalResult
cudaq::cc::AddressOfOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = symbolTable.lookupSymbolIn(
      getParentOfType<ModuleOp>(getOperation()), getGlobalNameAttr());

  // TODO: add globals?
  auto function = dyn_cast_or_null<func::FuncOp>(op);
  if (!function)
    return emitOpError("must reference a global defined by 'func.func'");
  return success();
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

void cudaq::cc::AllocaOp::print(OpAsmPrinter &p) {
  p << ' ' << getElementType();
  if (auto size = getSeqSize())
    p << '[' << size << " : " << size.getType() << ']';
}

ParseResult cudaq::cc::AllocaOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  Type eleTy;
  if (parser.parseType(eleTy))
    return failure();
  result.addAttribute("elementType", TypeAttr::get(eleTy));
  Type resTy;
  if (succeeded(parser.parseOptionalLSquare())) {
    OpAsmParser::UnresolvedOperand operand;
    Type operTy;
    if (parser.parseOperand(operand) || parser.parseColonType(operTy) ||
        parser.parseRSquare() ||
        parser.resolveOperand(operand, operTy, result.operands))
      return failure();
    resTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
  } else {
    resTy = cc::PointerType::get(eleTy);
  }
  if (!resTy || parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(resTy, result.types))
    return failure();
  return success();
}

OpFoldResult cudaq::cc::AllocaOp::fold(FoldAdaptor adaptor) {
  auto params = adaptor.getOperands();
  if (params.size() == 1) {
    // If allocating a contiguous block of elements and the size of the block is
    // a constant, fold the size into the cc.array type and allocate a constant
    // sized block.
    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(params[0])) {
      auto size = intAttr.getInt();
      if (size > 0) {
        auto resTy = cast<cc::ArrayType>(
            cast<cc::PointerType>(getType()).getElementType());
        auto arrTy = cc::ArrayType::get(resTy.getContext(),
                                        resTy.getElementType(), size);
        getOperation()->setAttr("elementType", TypeAttr::get(arrTy));
        getResult().setType(cc::PointerType::get(arrTy));
        getOperation()->eraseOperand(0);
        return getResult();
      }
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult cudaq::cc::CastOp::fold(FoldAdaptor) {
  // If cast is a nop, just forward the argument to the uses.
  if (getType() == getValue().getType())
    return getValue();
  return nullptr;
}

LogicalResult cudaq::cc::CastOp::verify() {
  auto inTy = getValue().getType();
  auto outTy = getType();

  // Make sure sint/zint are properly used.
  if (getSint() || getZint()) {
    if (getSint() && getZint())
      return emitOpError("cannot be both signed and unsigned.");
    if ((isa<IntegerType>(inTy) && isa<IntegerType>(outTy)) ||
        (isa<FloatType>(inTy) && isa<IntegerType>(outTy)) ||
        (isa<IntegerType>(inTy) && isa<FloatType>(outTy))) {
      // ok, do nothing.
    } else {
      return emitOpError("signed (unsigned) may only be applied to integer to "
                         "integer or integer to/from float.");
    }
  }

  // Make sure this cast can be translated to one of LLVM's instructions.
  if (isa<IntegerType>(inTy) || isa<IntegerType>(outTy)) {
    // Check casts to and from integer types.
    if (isa<IntegerType>(inTy) && isa<IntegerType>(outTy)) {
      // trunc, sext, zext, nop
      auto iTy1 = cast<IntegerType>(inTy);
      auto iTy2 = cast<IntegerType>(outTy);
      if ((iTy1.getWidth() < iTy2.getWidth()) && !getSint() && !getZint())
        return emitOpError("integer extension must be signed or unsigned.");
    } else if (isa<IntegerType>(inTy) && isa<cc::PointerType>(outTy)) {
      // ok: inttoptr
    } else if (isa<cc::PointerType>(inTy) && isa<IntegerType>(outTy)) {
      // ok: ptrtoint
    } else if (isa<IntegerType>(inTy) && isa<FloatType>(outTy)) {
      if (!getSint() && !getZint()) {
        // bitcast
        auto iTy1 = cast<IntegerType>(inTy);
        auto fTy2 = cast<FloatType>(outTy);
        if (iTy1.getWidth() != fTy2.getWidth())
          return emitOpError("bitcast must be same number of bits.");
      } else {
        // ok: sitofp, uitofp
      }
    } else if (isa<FloatType>(inTy) && isa<IntegerType>(outTy)) {
      if (!getSint() && !getZint()) {
        // bitcast
        auto iTy1 = cast<IntegerType>(outTy);
        auto fTy2 = cast<FloatType>(inTy);
        if (iTy1.getWidth() != fTy2.getWidth())
          return emitOpError("bitcast must be same number of bits.");
      } else {
        // ok: fptosi, fptoui
      }
    } else {
      return emitOpError("invalid integer cast.");
    }
  } else if (isa<FloatType>(inTy) && isa<FloatType>(outTy)) {
    // ok, floating-point casts: fptrunc, fpext, nop
  } else if (isa<cc::PointerType, LLVM::LLVMPointerType>(inTy) &&
             isa<cc::PointerType, LLVM::LLVMPointerType>(outTy)) {
    // ok, pointer casts: bitcast, nop
  } else {
    // Could support a bitcast of a float with pointer size bits to/from a
    // pointer, but that doesn't seem like it would be very common.
    return emitOpError("invalid cast.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ComputePtrOp
//===----------------------------------------------------------------------===//

static ParseResult
parseComputePtrIndices(OpAsmParser &parser,
                       SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices,
                       DenseI32ArrayAttr &rawConstantIndices) {
  SmallVector<int32_t> constantIndices;

  auto idxParser = [&]() -> ParseResult {
    int32_t constantIndex;
    OptionalParseResult parsedInteger =
        parser.parseOptionalInteger(constantIndex);
    if (parsedInteger.has_value()) {
      if (failed(parsedInteger.value()))
        return failure();
      constantIndices.push_back(constantIndex);
      return success();
    }

    constantIndices.push_back(LLVM::GEPOp::kDynamicIndex);
    return parser.parseOperand(indices.emplace_back());
  };
  if (parser.parseCommaSeparatedList(idxParser))
    return failure();

  rawConstantIndices =
      DenseI32ArrayAttr::get(parser.getContext(), constantIndices);
  return success();
}

static void printComputePtrIndices(OpAsmPrinter &printer,
                                   cudaq::cc::ComputePtrOp computePtrOp,
                                   OperandRange indices,
                                   DenseI32ArrayAttr rawConstantIndices) {
  llvm::interleaveComma(cudaq::cc::ComputePtrIndicesAdaptor<OperandRange>(
                            rawConstantIndices, indices),
                        printer, [&](PointerUnion<IntegerAttr, Value> cst) {
                          if (Value val = cst.dyn_cast<Value>())
                            printer.printOperand(val);
                          else
                            printer << cst.get<IntegerAttr>().getInt();
                        });
}

void cudaq::cc::ComputePtrOp::build(OpBuilder &builder, OperationState &result,
                                    Type resultType, Value basePtr,
                                    ValueRange indices,
                                    ArrayRef<NamedAttribute> attrs) {
  build(builder, result, resultType, basePtr,
        SmallVector<ComputePtrArg>(indices), attrs);
}

static void
destructureIndices(Type currType, ArrayRef<cudaq::cc::ComputePtrArg> indices,
                   SmallVectorImpl<std::int32_t> &rawConstantIndices,
                   SmallVectorImpl<Value> &dynamicIndices) {
  for (const cudaq::cc::ComputePtrArg &iter : indices) {
    if (Value val = iter.dyn_cast<Value>()) {
      rawConstantIndices.push_back(cudaq::cc::ComputePtrOp::kDynamicIndex);
      dynamicIndices.push_back(val);
    } else {
      rawConstantIndices.push_back(
          iter.get<cudaq::cc::ComputePtrConstantIndex>());
    }

    currType =
        TypeSwitch<Type, Type>(currType)
            .Case([](cudaq::cc::ArrayType containerType) {
              return containerType.getElementType();
            })
            .Case([&](cudaq::cc::StructType structType) -> Type {
              int64_t memberIndex = rawConstantIndices.back();
              if (memberIndex >= 0 && static_cast<std::size_t>(memberIndex) <
                                          structType.getMembers().size())
                return structType.getMembers()[memberIndex];
              return {};
            })
            .Default(Type{});
  }
}

void cudaq::cc::ComputePtrOp::build(OpBuilder &builder, OperationState &result,
                                    Type resultType, Value basePtr,
                                    ArrayRef<ComputePtrArg> cpArgs,
                                    ArrayRef<NamedAttribute> attrs) {
  SmallVector<int32_t> rawConstantIndices;
  SmallVector<Value> dynamicIndices;
  Type elementType = cast<cc::PointerType>(basePtr.getType()).getElementType();
  destructureIndices(elementType, cpArgs, rawConstantIndices, dynamicIndices);

  result.addTypes(resultType);
  result.addAttributes(attrs);
  result.addAttribute(getRawConstantIndicesAttrName(result.name),
                      builder.getDenseI32ArrayAttr(rawConstantIndices));
  result.addOperands(basePtr);
  result.addOperands(dynamicIndices);
}

OpFoldResult cudaq::cc::ComputePtrOp::fold(FoldAdaptor adaptor) {
  if (getDynamicIndices().empty())
    return nullptr;
  SmallVector<std::tuple<Attribute, std::int32_t>> pairs;
  auto params = adaptor.getOperands();
  for (auto p : llvm::zip(params.drop_front(), getRawConstantIndices()))
    pairs.push_back(p);
  auto dynIter = getDynamicIndices().begin();
  SmallVector<int32_t> newConstantIndices;
  SmallVector<Value> newIndices;
  bool changed = false;
  for (auto [paramAttr, index] : pairs) {
    if (index == kDynamicIndex) {
      std::int32_t newVal;
      if (paramAttr) {
        newVal = cast<IntegerAttr>(paramAttr).getInt();
        changed = true;
      } else {
        newVal = index;
        newIndices.push_back(*dynIter);
      }
      newConstantIndices.push_back(newVal);
      dynIter++;
    } else {
      newConstantIndices.push_back(index);
    }
  }
  if (changed) {
    assert(newConstantIndices.size() == getRawConstantIndices().size());
    assert(newIndices.size() < getDynamicIndices().size());
    getDynamicIndicesMutable().assign(newIndices);
    setRawConstantIndices(newConstantIndices);
    return Value{*this};
  }
  return nullptr;
}

namespace {
// Address arithmetic for pointers to arrays is additive.
struct FuseAddressArithmetic
    : public OpRewritePattern<cudaq::cc::ComputePtrOp> {
  using Base = OpRewritePattern<cudaq::cc::ComputePtrOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::ComputePtrOp ptrOp,
                                PatternRewriter &rewriter) const override {
    auto base = ptrOp.getBase();
    auto checkIsPtrToArr = [&](Type ty) -> bool {
      auto ptrTy = dyn_cast<cudaq::cc::PointerType>(ty);
      if (!ptrTy)
        return false;
      return true; // isa<cudaq::cc::ArrayType>(ptrTy.getElementType());
    };
    if (!checkIsPtrToArr(base.getType()))
      return success();
    if (auto origPtr =
            ptrOp.getBase().getDefiningOp<cudaq::cc::ComputePtrOp>()) {
      if (!checkIsPtrToArr(origPtr.getBase().getType()))
        return success();
      if (ptrOp.getRawConstantIndices().size() != 1 ||
          origPtr.getRawConstantIndices().size() != 1)
        return success();
      auto myOffset = ptrOp.getRawConstantIndices()[0];
      auto inOffset = origPtr.getRawConstantIndices()[0];
      auto extractConstant = [&](cudaq::cc::ComputePtrOp thisOp,
                                 std::int64_t othOffset) -> Value {
        auto v1 = thisOp.getDynamicIndices()[0];
        auto v1Ty = v1.getType();
        auto offAttr = IntegerAttr::get(v1Ty, othOffset);
        auto loc = thisOp.getLoc();
        auto newOff = rewriter.create<arith::ConstantOp>(loc, offAttr, v1Ty);
        return rewriter.create<arith::AddIOp>(loc, newOff, v1);
      };
      if (myOffset == cudaq::cc::ComputePtrOp::kDynamicIndex) {
        if (inOffset == cudaq::cc::ComputePtrOp::kDynamicIndex) {
          Value sum = rewriter.create<arith::AddIOp>(
              ptrOp.getLoc(), ptrOp.getDynamicIndices()[0],
              origPtr.getDynamicIndices()[0]);
          rewriter.replaceOpWithNewOp<cudaq::cc::ComputePtrOp>(
              ptrOp, ptrOp.getType(), origPtr.getBase(),
              ArrayRef<cudaq::cc::ComputePtrArg>{sum});
          return success();
        }
        auto sum = extractConstant(ptrOp, inOffset);
        rewriter.replaceOpWithNewOp<cudaq::cc::ComputePtrOp>(
            ptrOp, ptrOp.getType(), origPtr.getBase(),
            ArrayRef<cudaq::cc::ComputePtrArg>{sum});
        return success();
      }
      if (inOffset == cudaq::cc::ComputePtrOp::kDynamicIndex) {
        auto sum = extractConstant(origPtr, myOffset);
        rewriter.replaceOpWithNewOp<cudaq::cc::ComputePtrOp>(
            ptrOp, ptrOp.getType(), origPtr.getBase(),
            ArrayRef<cudaq::cc::ComputePtrArg>{sum});
        return success();
      }
      rewriter.replaceOpWithNewOp<cudaq::cc::ComputePtrOp>(
          ptrOp, ptrOp.getType(), origPtr.getBase(),
          ArrayRef<cudaq::cc::ComputePtrArg>{myOffset + inOffset});
    }
    return success();
  }
};
} // namespace

void cudaq::cc::ComputePtrOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseAddressArithmetic>(context);
}

//===----------------------------------------------------------------------===//
// GetConstantElementOp
//===----------------------------------------------------------------------===//

// If this operation has a constant offset, then the value can be looked up in
// the constant array and used as a scalar value directly.
OpFoldResult cudaq::cc::GetConstantElementOp::fold(FoldAdaptor adaptor) {
  auto params = adaptor.getOperands();
  if (params.size() < 2)
    return nullptr;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(params[1])) {
    auto offset = intAttr.getInt();
    auto conArr = getConstantArray().getDefiningOp<ConstantArrayOp>();
    if (!conArr)
      return nullptr;
    cudaq::cc::ArrayType arrTy = conArr.getType();
    if (arrTy.isUnknownSize())
      return nullptr;
    auto arrSize = arrTy.getSize();
    OpBuilder builder(getContext());
    builder.setInsertionPoint(getOperation());
    if (offset < arrSize) {
      auto fc = cast<FloatAttr>(conArr.getConstantValues()[offset]).getValue();
      auto f64Ty = builder.getF64Type();
      Value val = builder.create<arith::ConstantFloatOp>(getLoc(), fc, f64Ty);
      return val;
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// StdvecDataOp
//===----------------------------------------------------------------------===//

namespace {
struct FuseStdvecInitData : public OpRewritePattern<cudaq::cc::StdvecDataOp> {
  using Base = OpRewritePattern<cudaq::cc::StdvecDataOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::StdvecDataOp data,
                                PatternRewriter &rewriter) const override {
    // Bypass the std::vector wrappers for the creation of an abstract
    // subvector. This is possible because copies of std::vector data aren't
    // created but instead passed around like std::span objects. Specifically, a
    // pointer to the data and a length. Thus the pointer wrapped by stdvec_init
    // and unwrapped by stdvec_data is the same pointer value. This pattern will
    // arise after inlining, for example.
    if (auto ini = data.getStdvec().getDefiningOp<cudaq::cc::StdvecInitOp>()) {
      Value cast = rewriter.create<cudaq::cc::CastOp>(
          data.getLoc(), data.getType(), ini.getBuffer());
      rewriter.replaceOp(data, cast);
    }
    return success();
  }
};
} // namespace

void cudaq::cc::StdvecDataOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseStdvecInitData>(context);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

// Override the default.
Region &cudaq::cc::LoopOp::getLoopBody() { return getBodyRegion(); }

// The basic block of the step region must end in a continue op, which need not
// be pretty printed if the loop has no block arguments. This ensures the step
// block is properly terminated.
static void ensureStepTerminator(OpBuilder &builder, OperationState &result,
                                 Region *stepRegion) {
  if (stepRegion->empty())
    return;
  auto *block = &stepRegion->back();
  auto addContinue = [&]() {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(block);
    builder.create<cudaq::cc::ContinueOp>(result.location);
  };
  if (block->empty()) {
    addContinue();
  } else {
    auto *term = &block->back();
    if (!term->hasTrait<OpTrait::IsTerminator>())
      addContinue();
  }
}

void cudaq::cc::LoopOp::build(OpBuilder &builder, OperationState &result,
                              TypeRange resultTypes, ValueRange iterArgs,
                              bool postCond, RegionBuilderFn whileBuilder,
                              RegionBuilderFn bodyBuilder,
                              RegionBuilderFn stepBuilder) {
  auto *whileRegion = result.addRegion();
  auto *bodyRegion = result.addRegion();
  auto *stepRegion = result.addRegion();
  whileBuilder(builder, result.location, *whileRegion);
  bodyBuilder(builder, result.location, *bodyRegion);
  if (stepBuilder) {
    stepBuilder(builder, result.location, *stepRegion);
    ensureStepTerminator(builder, result, stepRegion);
  }
  result.addAttribute(postCondAttrName(), builder.getBoolAttr(postCond));
  result.addOperands(iterArgs);
  result.addTypes(resultTypes);
}

void cudaq::cc::LoopOp::build(OpBuilder &builder, OperationState &result,
                              ValueRange iterArgs, bool postCond,
                              RegionBuilderFn whileBuilder,
                              RegionBuilderFn bodyBuilder,
                              RegionBuilderFn stepBuilder) {
  build(builder, result, TypeRange{}, iterArgs, postCond, whileBuilder,
        bodyBuilder, stepBuilder);
}

LogicalResult cudaq::cc::LoopOp::verify() {
  const auto initArgsSize = getInitialArgs().size();
  if (getResults().size() != initArgsSize)
    return emitOpError("size of init args and outputs must be equal");
  if (getWhileRegion().front().getArguments().size() != initArgsSize)
    return emitOpError("size of init args and while region args must be equal");
  if (auto condOp = dyn_cast<cudaq::cc::ConditionOp>(
          getWhileRegion().front().getTerminator())) {
    if (condOp.getResults().size() != initArgsSize)
      return emitOpError("size of init args and condition op must be equal");
  } else {
    return emitOpError("while region must end with condition op");
  }
  if (getBodyRegion().front().getArguments().size() != initArgsSize)
    return emitOpError("size of init args and body region args must be equal");
  if (!getStepRegion().empty()) {
    if (getStepRegion().front().getArguments().size() != initArgsSize)
      return emitOpError(
          "size of init args and step region args must be equal");
    if (auto contOp = dyn_cast<cudaq::cc::ContinueOp>(
            getStepRegion().front().getTerminator())) {
      if (contOp.getOperands().size() != initArgsSize)
        return emitOpError("size of init args and continue op must be equal");
    } else {
      return emitOpError("step region must end with continue op");
    }
  }
  return success();
}

static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    Operation::operand_range initializers) {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << "((";
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ") -> (" << initializers.getTypes() << ")) ";
}

void cudaq::cc::LoopOp::print(OpAsmPrinter &p) {
  if (isPostConditional()) {
    p << " do ";
    printInitializationList(p, getBodyRegion().front().getArguments(),
                            getOperands());
    p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
    p << " while ";
    p.printRegion(getWhileRegion(), /*printEntryBlockArgs=*/hasArguments(),
                  /*printBlockTerminators=*/true);
  } else {
    p << " while ";
    printInitializationList(p, getWhileRegion().front().getArguments(),
                            getOperands());
    p.printRegion(getWhileRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
    p << " do ";
    p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/hasArguments(),
                  /*printBlockTerminators=*/true);
    if (!getStepRegion().empty()) {
      p << " step ";
      p.printRegion(getStepRegion(), /*printEntryBlockArgs=*/hasArguments(),
                    /*printBlockTerminators=*/hasArguments());
    }
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {postCondAttrName()});
}

ParseResult cudaq::cc::LoopOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  auto &builder = parser.getBuilder();
  bool isPostCondition = false;
  auto *cond = result.addRegion();
  auto *body = result.addRegion();
  auto *step = result.addRegion();
  auto parseOptBlockArgs =
      [&](SmallVector<OpAsmParser::Argument, 4> &regionArgs) {
        SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
        if (succeeded(parser.parseOptionalLParen())) {
          // Parse assignment list and results type list.
          if (parser.parseAssignmentList(regionArgs, operands) ||
              parser.parseArrowTypeList(result.types) || parser.parseRParen())
            return true;

          // Resolve input operands.
          for (auto argOperandType :
               llvm::zip(regionArgs, operands, result.types)) {
            auto type = std::get<2>(argOperandType);
            std::get<0>(argOperandType).type = type;
            if (parser.resolveOperand(std::get<1>(argOperandType), type,
                                      result.operands))
              return true;
          }
        }
        return false;
      };
  if (succeeded(parser.parseOptionalKeyword("while"))) {
    SmallVector<OpAsmParser::Argument, 4> regionArgs;
    if (parseOptBlockArgs(regionArgs) || parser.parseRegion(*cond, regionArgs))
      return failure();
    SmallVector<OpAsmParser::Argument, 4> emptyArgs;
    if (parser.parseKeyword("do") || parser.parseRegion(*body, emptyArgs))
      return failure();
    if (succeeded(parser.parseOptionalKeyword("step"))) {
      if (parser.parseRegion(*step, emptyArgs))
        return failure();
      OpBuilder opBuilder(builder.getContext());
      ensureStepTerminator(opBuilder, result, step);
    }
  } else if (succeeded(parser.parseOptionalKeyword("do"))) {
    isPostCondition = true;
    SmallVector<OpAsmParser::Argument, 4> regionArgs;
    if (parseOptBlockArgs(regionArgs) || parser.parseRegion(*body, regionArgs))
      return failure();
    SmallVector<OpAsmParser::Argument, 4> emptyArgs;
    if (parser.parseKeyword("while") || parser.parseRegion(*cond, emptyArgs))
      return failure();
  } else {
    return parser.emitError(parser.getNameLoc(), "expected 'while' or 'do'");
  }
  result.addAttribute(
      postCondAttrName(),
      builder.getIntegerAttr(builder.getI1Type(), isPostCondition));
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

bool cudaq::cc::LoopOp::hasBreakInBody() {
  // Note: the lowering of unwinds should've taken place for this to be
  // accurate. Add an assertion?
  for (Block &block : getBodyRegion())
    for (Operation &op : block)
      if (isa<BreakOp>(op))
        return true;
  return false;
}

void cudaq::cc::LoopOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (!index) {
    // loop op - successor is either the while region or, if a post conditional
    // loop, the do region.
    if (isPostConditional())
      regions.push_back(
          RegionSuccessor(&getBodyRegion(), getDoEntryArguments()));
    else
      regions.push_back(
          RegionSuccessor(&getWhileRegion(), getWhileArguments()));
    return;
  }
  switch (index.value()) {
  case 0:
    // While region = successors are the owning loop op and the do region.
    regions.push_back(RegionSuccessor(&getBodyRegion(), getDoEntryArguments()));
    regions.push_back(RegionSuccessor(getResults()));
    break;
  case 1:
    // do region - Successor is step region (2) if present or while region (0)
    // if step is absent.
    if (hasStep())
      regions.push_back(RegionSuccessor(&getStepRegion(), getStepArguments()));
    else
      regions.push_back(
          RegionSuccessor(&getWhileRegion(), getWhileArguments()));
    // If the body contains a break, then the loop op is also a successor.
    if (hasBreakInBody())
      regions.push_back(RegionSuccessor(getResults()));
    break;
  case 2:
    // step region - if present, while region is always successor.
    if (hasStep())
      regions.push_back(
          RegionSuccessor(&getWhileRegion(), getWhileArguments()));
    break;
  }
}

OperandRange
cudaq::cc::LoopOp::getSuccessorEntryOperands(std::optional<unsigned> index) {
  return getInitialArgs();
}

namespace {
// If an argument to a LoopOp traverses the loop unchanged then it is invariant
// across all iterations of the loop and can be hoisted out of the loop. This
// pattern detects invariant arguments and removes them from the LoopOp. This
// performs the following rewrite, where the notation `⟦x := y⟧` means all uses
// of `x` are replaced with `y`.
//
//    %result = cc.loop while ((%invariant = %1) -> (T)) {
//      ...
//      cc.condition %cond(%invariant : T)
//    } do {
//    ^bb1(%invariant : T):
//      ...
//      cc.continue %invariant : T
//    } step {
//    ^bb1(%invariant : T):
//      ...
//      cc.continue %invariant : T
//    }
//  ──────────────────────────────────────
//    cc.loop while {
//      ...⟦%invariant := %1⟧...
//      cc.condition %cond
//    } do {
//    ^bb1:
//      ...⟦%invariant := %1⟧...
//      cc.continue
//    } step {
//    ^bb1:
//      ...⟦%invariant := %1⟧...
//      cc.continue
//    }
//    ...⟦%result := %1⟧...

struct HoistLoopInvariantArgs : public OpRewritePattern<cudaq::cc::LoopOp> {
  using Base = OpRewritePattern<cudaq::cc::LoopOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    // 1. Find all the terminators.
    SmallVector<Operation *> terminators;
    for (auto *reg : loop.getRegions())
      for (auto &block : *reg)
        if (block.hasNoSuccessors())
          terminators.push_back(block.getTerminator());

    // 2. Determine if any arguments are invariant.
    SmallVector<bool> invariants;
    bool hasInvariants = false;
    for (auto iter : llvm::enumerate(loop.getInitialArgs())) {
      bool isInvar = true;
      auto i = iter.index();
      for (auto *term : terminators) {
        Value blkArg = term->getBlock()->getParent()->front().getArgument(i);
        if (auto cond = dyn_cast<cudaq::cc::ConditionOp>(term)) {
          if (cond.getResults()[i] != blkArg)
            isInvar = false;
        } else if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(term)) {
          if (cont.getOperands()[i] != blkArg)
            isInvar = false;
        } else if (auto brk = dyn_cast<cudaq::cc::BreakOp>(term)) {
          if (brk.getOperands()[i] != blkArg)
            isInvar = false;
        }
        if (!isInvar)
          break;
      }
      if (isInvar)
        hasInvariants = true;
      invariants.push_back(isInvar);
    }

    // 3. For each invariant argument replace the uses with the original
    // invariant value throughout.
    if (hasInvariants) {
      for (auto iter : llvm::enumerate(invariants)) {
        if (iter.value()) {
          auto i = iter.index();
          Value initialVal = loop.getInitialArgs()[i];
          loop.getResult(i).replaceAllUsesWith(initialVal);
          for (auto *reg : loop.getRegions()) {
            if (reg->empty())
              continue;
            auto &entry = reg->front();
            entry.getArgument(i).replaceAllUsesWith(initialVal);
          }
        }
      }
    }

    return success();
  }
};
} // namespace

void cudaq::cc::LoopOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add<HoistLoopInvariantArgs>(context);
}

//===----------------------------------------------------------------------===//
// ScopeOp
//===----------------------------------------------------------------------===//

void cudaq::cc::ScopeOp::build(OpBuilder &builder, OperationState &result,
                               BodyBuilderFn bodyBuilder) {
  auto *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  auto &bodyBlock = bodyRegion->front();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  if (bodyBuilder)
    bodyBuilder(builder, result.location);
}

void cudaq::cc::ScopeOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = getRegion().getBlocks().size() > 1;
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ")";
    // Print terminator explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);
  p.printOptionalAttrDict((*this)->getAttrs());
}

static void ensureScopeRegionTerminator(OpBuilder &builder,
                                        OperationState &result,
                                        Region *region) {
  auto *block = region->empty() ? nullptr : &region->back();
  if (!block)
    return;
  if (!block->empty()) {
    auto *term = &block->back();
    if (term->hasTrait<OpTrait::IsTerminator>())
      return;
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(block);
  builder.create<cudaq::cc::ContinueOp>(result.location);
}

ParseResult cudaq::cc::ScopeOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  OpBuilder opBuilder(parser.getContext());
  ensureScopeRegionTerminator(opBuilder, result, body);
  return success();
}

void cudaq::cc::ScopeOp::getRegionInvocationBounds(
    ArrayRef<Attribute> attrs, SmallVectorImpl<InvocationBounds> &bounds) {}

void cudaq::cc::ScopeOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (!index) {
    regions.push_back(RegionSuccessor(&getRegion()));
    return;
  }
  regions.push_back(RegionSuccessor(getResults()));
}

namespace {
// If there are no allocations in the scope, then the scope is not needed as
// there is nothing to deallocate. This transformation does the following
// rewrite.
//
//    op1
//    <vals> = cc.scope {
//      sop1; ...; sopN;
//      cc.continue <args>
//    }
//    op2
//  ──────────────────────────────────────
//    op1
//    br bb1^
//  ^bb1:
//    sop1; ...; sopN;
//    br bb2^(<args>)
//  ^bb2(<vals>):
//    op2
//
// The canonicalizer will then fuse these blocks appropriately.
struct EraseScopeWhenNotNeeded : public OpRewritePattern<cudaq::cc::ScopeOp> {
  using Base = OpRewritePattern<cudaq::cc::ScopeOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::ScopeOp scope,
                                PatternRewriter &rewriter) const override {
    for (auto &reg : scope->getRegions())
      if (hasAllocation(reg))
        return success();

    // scope does not allocate, so the region can be inlined into the parent.
    auto loc = scope.getLoc();
    auto *scopeBlock = rewriter.getInsertionBlock();
    auto scopePt = rewriter.getInsertionPoint();
    // Split the block at the cc.scope. Make sure to maintain any values that
    // escape the cc.scope as block arguments.
    auto *splitBlock = rewriter.splitBlock(scopeBlock, scopePt);
    Block *succBlock;
    if (scope.getNumResults() == 0) {
      succBlock = splitBlock;
    } else {
      succBlock = rewriter.createBlock(
          splitBlock, scope.getResultTypes(),
          SmallVector<Location>(scope.getNumResults(), loc));
      rewriter.create<cf::BranchOp>(loc, splitBlock);
    }
    // Inline the cc.scope's region into the parent and create a branch to the
    // new successor block.
    auto &initRegion = scope.getInitRegion();
    auto *initBlock = &initRegion.front();
    auto *initTerminator = initRegion.back().getTerminator();
    auto initTerminatorOperands = initTerminator->getOperands();
    rewriter.setInsertionPointToEnd(&initRegion.back());
    rewriter.create<cf::BranchOp>(loc, succBlock, initTerminatorOperands);
    rewriter.eraseOp(initTerminator);
    rewriter.inlineRegionBefore(initRegion, succBlock);
    // Replace the cc.scope with a branch to the newly inlined region's entry
    // block.
    rewriter.setInsertionPointToEnd(scopeBlock);
    rewriter.create<cf::BranchOp>(loc, initBlock, ValueRange{});
    rewriter.replaceOp(scope, succBlock->getArguments());
    return success();
  }

  static bool hasAllocation(Region &region) {
    for (auto &block : region)
      for (auto &op : block) {
        if (auto mem = dyn_cast<MemoryEffectOpInterface>(op))
          if (mem.hasEffect<MemoryEffects::Allocate>())
            return true;
        for (auto &opReg : op.getRegions())
          if (hasAllocation(opReg))
            return true;
      }
    return false;
  }
};
} // namespace

void cudaq::cc::ScopeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<EraseScopeWhenNotNeeded>(context);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void cudaq::cc::IfOp::build(OpBuilder &builder, OperationState &result,
                            TypeRange resultTypes, Value cond,
                            RegionBuilderFn thenBuilder,
                            RegionBuilderFn elseBuilder) {
  auto *thenRegion = result.addRegion();
  auto *elseRegion = result.addRegion();
  thenBuilder(builder, result.location, *thenRegion);
  if (elseBuilder)
    elseBuilder(builder, result.location, *elseRegion);
  result.addOperands(cond);
  result.addTypes(resultTypes);
}

LogicalResult cudaq::cc::IfOp::verify() {
  if (getNumResults() != 0 && getElseRegion().empty())
    return emitOpError("must have an else block if defining values");
  return success();
}

void cudaq::cc::IfOp::print(OpAsmPrinter &p) {
  p << '(' << getCondition() << ')';
  p.printOptionalArrowTypeList(getResultTypes());
  p << ' ';
  const bool printBlockTerminators =
      !getThenRegion().hasOneBlock() || (getNumResults() > 0);
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);
  if (!getElseRegion().empty()) {
    p << " else ";
    const bool printBlockTerminators =
        !getElseRegion().hasOneBlock() || (getNumResults() > 0);
    p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false,
                  printBlockTerminators);
  }
  p.printOptionalAttrDict((*this)->getAttrs());
}

static void ensureIfRegionTerminator(OpBuilder &builder, OperationState &result,
                                     Region *ifRegion) {
  if (ifRegion->empty())
    return;
  auto *block = &ifRegion->back();
  if (!block)
    return;
  if (!block->empty()) {
    auto *term = &block->back();
    if (term->hasTrait<OpTrait::IsTerminator>())
      return;
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(block);
  builder.create<cudaq::cc::ContinueOp>(result.location);
}

ParseResult cudaq::cc::IfOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  auto &builder = parser.getBuilder();
  auto *thenRegion = result.addRegion();
  auto *elseRegion = result.addRegion();
  OpAsmParser::UnresolvedOperand cond;
  auto i1Type = builder.getIntegerType(1);
  if (parser.parseLParen() || parser.parseOperand(cond) ||
      parser.parseRParen() ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  OpBuilder opBuilder(parser.getContext());
  ensureIfRegionTerminator(opBuilder, result, thenRegion);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    ensureIfRegionTerminator(opBuilder, result, elseRegion);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void cudaq::cc::IfOp::getRegionInvocationBounds(
    ArrayRef<Attribute> attrs, SmallVectorImpl<InvocationBounds> &bounds) {
  // Assume non-constant condition. Each region may be executed 0 or 1 times.
  bounds.assign(2, {0, 1});
}

void cudaq::cc::IfOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (index) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }
  // TODO: can constant fold if the condition is a constant here.
  regions.push_back(RegionSuccessor(&getThenRegion()));
  if (!getElseRegion().empty())
    regions.push_back(RegionSuccessor(&getElseRegion()));
}

//===----------------------------------------------------------------------===//
// CreateLambdaOp
//===----------------------------------------------------------------------===//

void cudaq::cc::CreateLambdaOp::build(OpBuilder &builder,
                                      OperationState &result,
                                      cudaq::cc::CallableType lambdaTy,
                                      BodyBuilderFn bodyBuilder) {
  auto *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  result.addTypes(TypeRange{lambdaTy});
  auto &bodyBlock = bodyRegion->front();
  auto argTys = lambdaTy.getSignature().getInputs();
  SmallVector<Location> locations(argTys.size(), result.location);
  bodyBlock.addArguments(argTys, locations);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  if (bodyBuilder)
    bodyBuilder(builder, result.location);
}

void cudaq::cc::CreateLambdaOp::print(OpAsmPrinter &p) {
  p << ' ';
  bool hasArgs = getRegion().getNumArguments() != 0;
  bool hasRes =
      getType().cast<cudaq::cc::CallableType>().getSignature().getNumResults();
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/hasArgs,
                /*printBlockTerminators=*/hasRes);
  p << " : " << getType();
  p.printOptionalAttrDict((*this)->getAttrs(), {"signature"});
}

ParseResult cudaq::cc::CreateLambdaOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  auto *body = result.addRegion();
  Type lambdaTy;
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}) ||
      parser.parseColonType(lambdaTy) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("signature", TypeAttr::get(lambdaTy));
  result.addTypes(lambdaTy);
  CreateLambdaOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// CallCallableOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::CallCallableOp::verify() {
  FunctionType funcTy;
  auto ty = getCallee().getType();
  if (auto lambdaTy = dyn_cast<cudaq::cc::CallableType>(ty))
    funcTy = lambdaTy.getSignature();
  else if (auto fTy = dyn_cast<FunctionType>(ty))
    funcTy = fTy;
  else
    return emitOpError("callee has unexpected type");

  // Check argument types.
  auto argTys = funcTy.getInputs();
  if (argTys.size() != getArgOperands().size())
    return emitOpError("call has incorrect arity");
  for (auto [targArg, argVal] : llvm::zip(argTys, getArgOperands()))
    if (targArg != argVal.getType())
      return emitOpError("argument type mismatch");

  // Check return types.
  auto resTys = funcTy.getResults();
  if (resTys.size() != getResults().size())
    return emitOpError("call has incorrect coarity");
  for (auto [targRes, callVal] : llvm::zip(resTys, getResults()))
    if (targRes != callVal.getType())
      return emitOpError("result type mismatch");
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::ReturnOp::verify() {
  auto *op = getOperation();
  auto resultTypes = [&]() {
    if (auto func = op->getParentOfType<CreateLambdaOp>()) {
      auto lambdaTy = cast<CallableType>(func->getResult(0).getType());
      return SmallVector<Type>(lambdaTy.getSignature().getResults());
    }
    if (auto func = op->getParentOfType<func::FuncOp>())
      return SmallVector<Type>(func.getResultTypes());
    return SmallVector<Type>();
  }();

  // The operand number and types must match the function signature.
  if (getNumOperands() != resultTypes.size())
    return emitOpError("has ")
           << getNumOperands()
           << " operands, but enclosing function/lambda returns "
           << resultTypes.size();
  for (auto ep :
       llvm::enumerate(llvm::zip(getOperands().getTypes(), resultTypes))) {
    auto p = ep.value();
    auto i = ep.index();
    if (std::get<0>(p) != std::get<1>(p))
      return emitOpError("type of return operand ")
             << i << " (" << std::get<0>(p)
             << ") doesn't match function/lambda result type ("
             << std::get<1>(p) << ')';
  }
  return success();
}

// Replace an Op of type FROM with an Op of type WITH if the Op appears to be
// directly owned by a func::FuncOp. This is required to replace cc.return with
// func.return.
template <typename FROM, typename WITH>
struct ReplaceInFunc : public OpRewritePattern<FROM> {
  using Base = OpRewritePattern<FROM>;
  using Base::Base;

  LogicalResult matchAndRewrite(FROM fromOp,
                                PatternRewriter &rewriter) const override {
    if (isa_and_nonnull<func::FuncOp>(fromOp->getParentOp()))
      rewriter.replaceOpWithNewOp<WITH>(fromOp, fromOp.getOperands());
    return success();
  }
};

void cudaq::cc::ReturnOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ReplaceInFunc<cudaq::cc::ReturnOp, func::ReturnOp>>(context);
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::ConditionOp::verify() {
  Operation *self = getOperation();
  Region *region = self->getBlock()->getParent();
  auto parentOp = self->getParentOfType<LoopOp>();
  assert(parentOp); // checked by tablegen constraints
  if (&parentOp.getWhileRegion() != region)
    return emitOpError("only valid in the while region of a loop");
  return success();
}

MutableOperandRange cudaq::cc::ConditionOp::getMutableSuccessorOperands(
    std::optional<unsigned> index) {
  return getResultsMutable();
}

//===----------------------------------------------------------------------===//
// UnwindBreakOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::UnwindBreakOp::verify() {
  // The arguments to this op must correspond to the LoopOp's results.
  bool foundFunc = true;
  auto *op = getOperation();
  auto resultTypes = [&]() {
    if (auto func = op->getParentOfType<LoopOp>())
      return SmallVector<Type>(func->getResultTypes());
    foundFunc = false;
    return SmallVector<Type>();
  }();
  if (!foundFunc)
    return emitOpError("cannot find nearest enclosing loop");
  if (getOperands().size() != resultTypes.size())
    return emitOpError("arity of arguments and loop result mismatch");
  for (auto p : llvm::zip(getOperands().getTypes(), resultTypes))
    if (std::get<0>(p) != std::get<1>(p))
      return emitOpError("argument type mismatch with loop result");
  return success();
}

// Replace an Op of type FROM with an Op of type WITH if the Op appears to be
// directly owned by a cc::LoopOp. This is required to replace unwind breaks and
// unwind continues with breaks and continues, resp., when a cc::ScopeOp is
// erased.
template <typename FROM, typename WITH>
struct ReplaceInLoop : public OpRewritePattern<FROM> {
  using Base = OpRewritePattern<FROM>;
  using Base::Base;

  LogicalResult matchAndRewrite(FROM fromOp,
                                PatternRewriter &rewriter) const override {
    if (isa_and_nonnull<cudaq::cc::LoopOp>(fromOp->getParentOp())) {
      auto *scopeBlock = rewriter.getInsertionBlock();
      auto scopePt = rewriter.getInsertionPoint();
      rewriter.splitBlock(scopeBlock, scopePt);
      rewriter.setInsertionPointToEnd(scopeBlock);
      rewriter.replaceOpWithNewOp<WITH>(fromOp, fromOp.getOperands());
    }
    return success();
  }
};

void cudaq::cc::UnwindBreakOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ReplaceInLoop<cudaq::cc::UnwindBreakOp, cudaq::cc::BreakOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// UnwindContinueOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::UnwindContinueOp::verify() {
  // The arguments to this op must correspond to the LoopOp's results.
  bool foundFunc = true;
  auto *op = getOperation();
  auto resultTypes = [&]() {
    if (auto func = op->getParentOfType<LoopOp>())
      return SmallVector<Type>(func->getResultTypes());
    foundFunc = false;
    return SmallVector<Type>();
  }();
  if (!foundFunc)
    return emitOpError("cannot find nearest enclosing loop");
  if (getOperands().size() != resultTypes.size())
    return emitOpError("arity of arguments and loop result mismatch");
  for (auto p : llvm::zip(getOperands().getTypes(), resultTypes))
    if (std::get<0>(p) != std::get<1>(p))
      return emitOpError("argument type mismatch with loop result");
  return success();
}

void cudaq::cc::UnwindContinueOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns
      .add<ReplaceInLoop<cudaq::cc::UnwindContinueOp, cudaq::cc::ContinueOp>>(
          context);
}

//===----------------------------------------------------------------------===//
// UnwindReturnOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::UnwindReturnOp::verify() {
  // The arguments to this op must correspond to the FuncOp's results.
  bool foundFunc = true;
  auto *op = getOperation();
  auto resultTypes = [&]() {
    if (auto func = op->getParentOfType<CreateLambdaOp>()) {
      auto lambdaTy = cast<CallableType>(func->getResult(0).getType());
      return SmallVector<Type>(lambdaTy.getSignature().getResults());
    }
    if (auto func = op->getParentOfType<func::FuncOp>())
      return SmallVector<Type>(func.getResultTypes());
    foundFunc = false;
    return SmallVector<Type>();
  }();
  if (!foundFunc)
    return emitOpError("cannot find nearest enclosing function/lambda");
  if (getOperands().size() != resultTypes.size())
    return emitOpError(
        "arity of arguments and function/lambda result mismatch");
  for (auto p : llvm::zip(getOperands().getTypes(), resultTypes))
    if (std::get<0>(p) != std::get<1>(p))
      return emitOpError("argument type mismatch with function/lambda result");
  return success();
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/CC/CCOps.cpp.inc"
