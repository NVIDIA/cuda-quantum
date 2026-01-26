/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
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

std::optional<std::int64_t> cudaq::opt::factory::getIntIfConstant(Value value) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return {constant.getSExtValue()};
  return {};
}

std::optional<APFloat> cudaq::opt::factory::getDoubleIfConstant(Value value) {
  APFloat constant{0.0};
  if (matchPattern(value, m_ConstantFloat(&constant)))
    return {constant};
  return {};
}

Value cudaq::cc::getByteSizeOfType(OpBuilder &builder, Location loc, Type ty,
                                   bool useSizeOf) {
  auto createInt = [&](std::int32_t byteWidth) -> Value {
    return builder.create<arith::ConstantIntOp>(loc, byteWidth, 64);
  };

  // Handle primitive types with constant sizes.
  auto primSize = [](auto ty) -> unsigned {
    return (ty.getIntOrFloatBitWidth() + 7) / 8;
  };
  auto rawSize =
      TypeSwitch<Type, std::optional<std::int32_t>>(ty)
          .Case([&](IntegerType intTy) -> std::optional<std::int32_t> {
            return {primSize(intTy)};
          })
          .Case([&](FloatType fltTy) -> std::optional<std::int32_t> {
            return {primSize(fltTy)};
          })
          .Case([&](ComplexType complexTy) -> std::optional<std::int32_t> {
            auto eleTy = complexTy.getElementType();
            if (isa<IntegerType, FloatType>(eleTy))
              return {2 * primSize(eleTy)};
            return {};
          })
          .Case(
              [](cudaq::cc::PointerType ptrTy) -> std::optional<std::int32_t> {
                // TODO: get this from the target specification. For now
                // we're assuming pointers are 64 bits.
                return {8};
              })
          .Default({});

  if (rawSize)
    return createInt(*rawSize);

  // Handle aggregate types.
  return TypeSwitch<Type, Value>(ty)
      .Case([&](cudaq::cc::StructType strTy) -> Value {
        if (std::size_t bitWidth = strTy.getBitSize()) {
          assert(bitWidth % 8 == 0 && "struct ought to be in bytes");
          std::size_t byteWidth = bitWidth / 8;
          return createInt(byteWidth);
        }
        if (useSizeOf)
          return builder.create<cudaq::cc::SizeOfOp>(loc, builder.getI64Type(),
                                                     strTy);
        return {};
      })
      .Case([&](cudaq::cc::ArrayType arrTy) -> Value {
        if (arrTy.isUnknownSize())
          return {};
        auto v =
            getByteSizeOfType(builder, loc, arrTy.getElementType(), useSizeOf);
        if (!v)
          return {};
        auto scale = createInt(arrTy.getSize());
        return builder.create<arith::MulIOp>(loc, builder.getI64Type(), v,
                                             scale);
      })
      .Case([&](cudaq::cc::SpanLikeType) -> Value {
        // Uniformly on the device size: {ptr, i64}
        return createInt(16);
      })
      .Default({});
}

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

LogicalResult
cudaq::cc::AddressOfOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = symbolTable.lookupSymbolIn(
      getParentOfType<ModuleOp>(getOperation()), getGlobalNameAttr());

  if (!isa_and_nonnull<func::FuncOp, GlobalOp, LLVM::GlobalOp>(op))
    return emitOpError("must reference a global");
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

namespace {
struct FuseAllocLength : public OpRewritePattern<cudaq::cc::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp alloca,
                                PatternRewriter &rewriter) const override {
    auto params = alloca.getOperands();
    if (params.size() == 1) {
      // If allocating a contiguous block of elements and the size of the block
      // is a constant, fold the size into the cc.array type and allocate a
      // constant sized block.
      if (auto size = cudaq::opt::factory::getIntIfConstant(params[0]))
        if (*size > 0) {
          auto loc = alloca.getLoc();
          auto *context = rewriter.getContext();
          Type oldTy = alloca.getElementType();
          auto arrTy = cudaq::cc::ArrayType::get(context, oldTy, *size);
          Type origTy = alloca.getType();
          auto newAlloc = rewriter.create<cudaq::cc::AllocaOp>(loc, arrTy);
          rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(alloca, origTy,
                                                         newAlloc);
          return success();
        }
    }
    return failure();
  }
};
} // namespace

void cudaq::cc::AllocaOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseAllocLength>(context);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult cudaq::cc::CastOp::fold(FoldAdaptor adaptor) {
  // If cast is a nop, just forward the argument to the uses.
  if (getType() == getValue().getType())
    return getValue();
  if (auto optConst = adaptor.getValue()) {
    // Replace a constant + cast with a new constant of an updated type.
    auto ty = getType();
    OpBuilder builder(*this);
    auto fltTy = builder.getF32Type();
    auto dblTy = builder.getF64Type();
    auto loc = getLoc();
    auto truncate = [&](std::int64_t val) -> std::int64_t {
      auto srcTy = getValue().getType();
      auto srcWidth = srcTy.getIntOrFloatBitWidth();
      // Zero-extend to get the original integer value.
      if (srcWidth < 64)
        val &= ((1UL << srcWidth) - 1);
      return val;
    };

    if (auto attr = dyn_cast<IntegerAttr>(optConst)) {
      auto val = attr.getInt();
      if (isa<IntegerType>(ty)) {
        auto width = ty.getIntOrFloatBitWidth();

        if (getZint())
          val = truncate(val);

        if (width == 1) {
          bool v = val != 0;
          return builder.create<arith::ConstantIntOp>(loc, v, width)
              .getResult();
        }
        return builder.create<arith::ConstantIntOp>(loc, val, width)
            .getResult();

      } else if (ty == fltTy) {
        if (getZint()) {
          val = truncate(val);
          APFloat fval(static_cast<float>(static_cast<std::uint64_t>(val)));
          return builder.create<arith::ConstantFloatOp>(loc, fval, fltTy)
              .getResult();
        }
        if (getSint()) {
          APFloat fval(static_cast<float>(val));
          return builder.create<arith::ConstantFloatOp>(loc, fval, fltTy)
              .getResult();
        }
      } else if (ty == dblTy) {
        if (getZint()) {
          val = truncate(val);
          APFloat fval(static_cast<double>(static_cast<std::uint64_t>(val)));
          return builder.create<arith::ConstantFloatOp>(loc, fval, dblTy)
              .getResult();
        }
        if (getSint()) {
          APFloat fval(static_cast<double>(val));
          return builder.create<arith::ConstantFloatOp>(loc, fval, dblTy)
              .getResult();
        }
      }
    }

    // %5 = arith.constant ... : F1
    // %6 = cc.cast %5 : (F1) -> F2
    // ────────────────────────────
    // %6 = arith.constant ... : F2
    if (auto attr = dyn_cast<FloatAttr>(optConst)) {
      auto val = attr.getValue();
      if (ty == fltTy) {
        float f = val.convertToDouble();
        APFloat fval(f);
        return builder.create<arith::ConstantFloatOp>(loc, fval, fltTy)
            .getResult();
      }
      if (ty == dblTy) {
        APFloat fval{val.convertToDouble()};
        return builder.create<arith::ConstantFloatOp>(loc, fval, dblTy)
            .getResult();
      }
      if (isa<IntegerType>(ty)) {
        auto width = ty.getIntOrFloatBitWidth();
        if (getZint()) {
          std::uint64_t v = val.convertToDouble();
          return builder.create<arith::ConstantIntOp>(loc, v, width)
              .getResult();
        }
        if (getSint()) {
          std::int64_t v = val.convertToDouble();
          return builder.create<arith::ConstantIntOp>(loc, v, width)
              .getResult();
        }
      }
    }

    // %5 = complex.constant ... : complex<T>
    // %6 = cc.cast %5 : (complex<T>) -> complex<U>
    // ────────────────────────────────────────────
    // %6 = complex.constant ... : complex<U>
    if (auto attr = dyn_cast<ArrayAttr>(optConst)) {
      auto eleTy = cast<ComplexType>(ty).getElementType();
      auto reFp = dyn_cast<FloatAttr>(attr[0]);
      auto imFp = dyn_cast<FloatAttr>(attr[1]);
      if (reFp && imFp) {
        if (eleTy == fltTy) {
          float reVal = reFp.getValue().convertToDouble();
          float imVal = imFp.getValue().convertToDouble();
          auto rePart = builder.getFloatAttr(eleTy, APFloat{reVal});
          auto imPart = builder.getFloatAttr(eleTy, APFloat{imVal});
          auto cv = builder.getArrayAttr({rePart, imPart});
          return builder.create<complex::ConstantOp>(loc, ty, cv).getResult();
        }
        if (eleTy == dblTy) {
          double reVal = reFp.getValue().convertToDouble();
          double imVal = imFp.getValue().convertToDouble();
          auto rePart = builder.getFloatAttr(eleTy, APFloat{reVal});
          auto imPart = builder.getFloatAttr(eleTy, APFloat{imVal});
          auto cv = builder.getArrayAttr({rePart, imPart});
          return builder.create<complex::ConstantOp>(loc, ty, cv).getResult();
        }
      }
    }
  }
  return nullptr;
}

LogicalResult cudaq::cc::CastOp::verify() {
  auto inTy = getValue().getType();
  auto outTy = getType();

  // Make sure sint/zint are properly used.
  if (getSint() || getZint()) {
    if (getSint() && getZint())
      return emitOpError("cannot be both signed and unsigned.");
    if (isa<IntegerType>(inTy) && isa<IntegerType>(outTy)) {
      if (cast<IntegerType>(inTy).getWidth() >
          cast<IntegerType>(outTy).getWidth())
        return emitOpError("signed (unsigned) may only be applied to integer "
                           "to integer extension, not truncation.");
    } else if ((isa<FloatType>(inTy) && isa<IntegerType>(outTy)) ||
               (isa<IntegerType>(inTy) && isa<FloatType>(outTy))) {
      // ok, do nothing.
    } else if (isa<ComplexType>(inTy) && isa<ComplexType>(outTy)) {
      auto inEleTy = cast<ComplexType>(inTy).getElementType();
      auto outEleTy = cast<ComplexType>(outTy).getElementType();
      if ((isa<IntegerType>(inEleTy) && isa<IntegerType>(outEleTy)) ||
          (isa<FloatType>(inEleTy) && isa<IntegerType>(outEleTy)) ||
          (isa<IntegerType>(inEleTy) && isa<FloatType>(outEleTy))) {
      } else {
        return emitOpError(
            "signed (unsigned) may only be applied to complex of integer "
            "to/from complex of integer or complex of float.");
      }
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
    } else if (isa<IntegerType>(inTy) && isa<cc::IndirectCallableType>(outTy)) {
      // ok: nop
      // the indirect callable value is an integer key on the device side.
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
  } else if (isa<cc::PointerType, LLVM::LLVMPointerType>(inTy)) {
    // ok, will become pointer casts: nop
  } else if (isa<ComplexType>(inTy) && isa<ComplexType>(outTy)) {
    auto inEleTy = cast<ComplexType>(inTy).getElementType();
    auto outEleTy = cast<ComplexType>(outTy).getElementType();
    if (isa<FloatType>(inEleTy) && isa<FloatType>(outEleTy)) {
      // ok, type conversion of a complex floating-point value
      // NB: use complex.re or complex.im to convert (extract) a fp value.
    } else {
      // TODO: For now, disable complex<int>. All variants of complex<int>
      // require a signed/unsigned modifier. These include to/from complex<int>
      // and to/from complex<fp>.
      return emitOpError("invalid complex cast.");
    }
  } else if (isa<FunctionType>(inTy) && isa<cc::IndirectCallableType>(outTy)) {
    // ok, type conversion of a function to an indirect callable
    // Folding will remove this.
  } else if (isa<FunctionType>(inTy) && isa<cc::PointerType>(outTy)) {
    auto ptrTy = cast<cc::PointerType>(outTy);
    auto eleTy = ptrTy.getElementType();
    auto *ctx = eleTy.getContext();
    if (eleTy == NoneType::get(ctx) || eleTy == IntegerType::get(ctx, 8)) {
      // ok, type conversion of a function to a pointer.
    } else {
      return emitOpError("invalid cast.");
    }
  } else {
    // Could support a bitcast of a float with pointer size bits to/from a
    // pointer, but that doesn't seem like it would be very common.
    return emitOpError("invalid cast.");
  }
  return success();
}

namespace {
// There are a number of series of casts that can be fused. For now, fuse
// pointer cast chains.
struct FuseCastCascade : public OpRewritePattern<cudaq::cc::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (auto castToCast = castOp.getValue().getDefiningOp<cudaq::cc::CastOp>())
      if (isa<cudaq::cc::PointerType>(castOp.getType()) &&
          isa<cudaq::cc::PointerType>(castToCast.getType())) {
        // %4 = cc.cast %3 : (!cc.ptr<T>) -> !cc.ptr<U>
        // %5 = cc.cast %4 : (!cc.ptr<U>) -> !cc.ptr<V>
        // ────────────────────────────────────────────
        // %5 = cc.cast %3 : (!cc.ptr<T>) -> !cc.ptr<V>
        rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(castOp, castOp.getType(),
                                                       castToCast.getValue());
        return success();
      }
    return failure();
  }
};

// Ad hoc pattern to erase casts used by arith.cmpi.
struct SimplifyIntegerCompare : public OpRewritePattern<arith::CmpIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::CmpIOp compare,
                                PatternRewriter &rewriter) const override {
    auto lhsCast = compare.getLhs().getDefiningOp<cudaq::cc::CastOp>();
    auto rhsCast = compare.getRhs().getDefiningOp<cudaq::cc::CastOp>();
    // %4 = cc.cast %2 ...
    // %5 = cc.cast %3 ...
    // %6 = arith.cmpi %4, %5 ...
    //      and
    // type(%2) == type(%3)
    //      and
    // %4 and %5 are compatible casts
    // ──────────────────────────────
    // %5 = arith.cmpi %2, %3 ...
    if (lhsCast && rhsCast) {
      auto lhsVal = lhsCast.getValue();
      auto rhsVal = rhsCast.getValue();
      if (lhsVal.getType() == rhsVal.getType() &&
          lhsCast.getSint() == rhsCast.getSint() &&
          lhsCast.getZint() == rhsCast.getZint()) {
        rewriter.replaceOpWithNewOp<arith::CmpIOp>(
            compare, compare.getType(), compare.getPredicate(), lhsVal, rhsVal);
        return success();
      }
    }
    return failure();
  }
};
} // namespace

namespace {
// Ad hoc pattern to erase complex.create. (MLIR doesn't do this.) This pattern
// gets piggybacked into the canonicalizations, but does NOT have anything to do
// with cc::CastOp.
struct FuseComplexCreate : public OpRewritePattern<complex::CreateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(complex::CreateOp create,
                                PatternRewriter &rewriter) const override {
    auto reVal = cudaq::opt::factory::getDoubleIfConstant(create.getReal());
    auto imVal =
        cudaq::opt::factory::getDoubleIfConstant(create.getImaginary());
    if (reVal && imVal) {
      auto eleTy = cast<ComplexType>(create.getType()).getElementType();
      auto rePart = rewriter.getFloatAttr(eleTy, *reVal);
      auto imPart = rewriter.getFloatAttr(eleTy, *imVal);
      auto arrAttr = rewriter.getArrayAttr({rePart, imPart});
      rewriter.replaceOpWithNewOp<complex::ConstantOp>(
          create, ComplexType::get(eleTy), arrAttr);
      return success();
    }
    return failure();
  }
};

struct FuseComplexRe : public OpRewritePattern<complex::ReOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(complex::ReOp reop,
                                PatternRewriter &rewriter) const override {
    auto comcon = reop.getReal().getDefiningOp<complex::ConstantOp>();
    if (comcon) {
      FloatType fltTy = reop.getType();
      APFloat reVal = cast<FloatAttr>(comcon.getValue()[0]).getValue();
      rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(reop, reVal, fltTy);
      return success();
    }
    return failure();
  }
};

struct FuseComplexIm : public OpRewritePattern<complex::ImOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(complex::ImOp imop,
                                PatternRewriter &rewriter) const override {
    auto comcon = imop.getImaginary().getDefiningOp<complex::ConstantOp>();
    if (comcon) {
      FloatType fltTy = imop.getType();
      APFloat imVal = cast<FloatAttr>(comcon.getValue()[1]).getValue();
      rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(imop, imVal, fltTy);
      return success();
    }
    return failure();
  }
};
} // namespace

static void
getArbitraryCustomCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<FuseComplexCreate, FuseComplexRe, FuseComplexIm>(context);
}

void cudaq::cc::CastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add<FuseCastCascade, SimplifyIntegerCompare>(context);
  getArbitraryCustomCanonicalizationPatterns(patterns, context);
}

//===----------------------------------------------------------------------===//
// Support for operations with interleaved indices.
//===----------------------------------------------------------------------===//

template <typename A>
ParseResult parseInterleavedIndices(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices,
    DenseI32ArrayAttr &rawConstantIndices) {
  SmallVector<std::int32_t> constantIndices;

  auto idxParser = [&]() -> ParseResult {
    std::int32_t constantIndex;
    OptionalParseResult parsedInteger =
        parser.parseOptionalInteger(constantIndex);
    if (parsedInteger.has_value()) {
      if (failed(parsedInteger.value()))
        return failure();
      constantIndices.push_back(constantIndex);
      return success();
    }

    constantIndices.push_back(A::kDynamicIndex);
    return parser.parseOperand(indices.emplace_back());
  };
  if (parser.parseCommaSeparatedList(idxParser))
    return failure();

  rawConstantIndices =
      DenseI32ArrayAttr::get(parser.getContext(), constantIndices);
  return success();
}

template <typename Adaptor, typename B>
void printInterleavedIndices(OpAsmPrinter &printer, B computePtrOp,
                             OperandRange indices,
                             DenseI32ArrayAttr rawConstantIndices) {
  llvm::interleaveComma(Adaptor{rawConstantIndices, indices}, printer,
                        [&](PointerUnion<IntegerAttr, Value> cst) {
                          if (Value val = dyn_cast<Value>(cst))
                            printer.printOperand(val);
                          else
                            printer << cst.get<IntegerAttr>().getInt();
                        });
}

//===----------------------------------------------------------------------===//
// ComputePtrOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::ComputePtrOp::verify() {
  auto basePtrTy = cast<cc::PointerType>(getBase().getType());
  Type eleTy = basePtrTy.getElementType();
  auto resultTy = cast<cc::PointerType>(getResult().getType());
  for (std::int32_t i : getRawConstantIndices()) {
    if (auto arrTy = dyn_cast<cc::ArrayType>(eleTy)) {
      if (i != kDynamicIndex && !arrTy.isUnknownSize() &&
          (i < 0 || i > arrTy.getSize())) {
        // Note: allow indexing of last element + 1 so we can compute a
        // pointer to `end()` of a stdvec buffer. Consider adding a flag
        // to cc.compute_ptr explicitly for this or using casts.
        return emitOpError("array cannot index out of bounds elements");
      }
      eleTy = arrTy.getElementType();
    } else if (auto strTy = dyn_cast<cc::StructType>(eleTy)) {
      if (i == kDynamicIndex)
        return emitOpError("struct cannot have non-constant index");
      if (i < 0 || static_cast<std::size_t>(i) >= strTy.getMembers().size())
        return emitOpError("struct cannot index out of bounds members");
      eleTy = strTy.getMember(i);
    } else if (auto complexTy = dyn_cast<ComplexType>(eleTy)) {
      if (!(i == 0 || i == 1 || i == kDynamicIndex))
        return emitOpError("complex index is out of bounds");
      eleTy = complexTy.getElementType();
    } else {
      return emitOpError("too many indices (" +
                         std::to_string(getRawConstantIndices().size()) +
                         ") for the source pointer");
    }
  }
  if (eleTy != resultTy.getElementType())
    return emitOpError("result type does not match input");
  return success();
}

// Is this `cc.compute_ptr` in LLVM normal form?
// To be in LLVM normal form, the base object must have a type of
// `!cc.ptr<!cc.array<T x ?>>`, which corresponds 1:1 with LLVM's GEP semantics.
bool cudaq::cc::ComputePtrOp::llvmNormalForm() {
  if (auto ptrTy = dyn_cast<PointerType>(getBase().getType()))
    if (auto arrTy = dyn_cast<ArrayType>(ptrTy.getElementType()))
      return arrTy.isUnknownSize();
  return false;
}

static ParseResult
parseComputePtrIndices(OpAsmParser &parser,
                       SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices,
                       DenseI32ArrayAttr &rawConstantIndices) {
  return parseInterleavedIndices<cudaq::cc::ComputePtrOp>(parser, indices,
                                                          rawConstantIndices);
}

static void printComputePtrIndices(OpAsmPrinter &printer,
                                   cudaq::cc::ComputePtrOp computePtrOp,
                                   OperandRange indices,
                                   DenseI32ArrayAttr rawConstantIndices) {
  printInterleavedIndices<cudaq::cc::ComputePtrIndicesAdaptor<OperandRange>>(
      printer, computePtrOp, indices, rawConstantIndices);
}

void cudaq::cc::ComputePtrOp::build(OpBuilder &builder, OperationState &result,
                                    Type resultType, Value basePtr,
                                    ValueRange indices,
                                    ArrayRef<NamedAttribute> attrs) {
  build(builder, result, resultType, basePtr,
        SmallVector<ComputePtrArg>(indices), attrs);
}

template <typename A, typename B>
void destructureIndices(Type currType, ArrayRef<B> indices,
                        SmallVectorImpl<std::int32_t> &rawConstantIndices,
                        SmallVectorImpl<Value> &dynamicIndices) {
  for (const B &iter : indices) {
    if (Value val = iter.template dyn_cast<Value>()) {
      rawConstantIndices.push_back(A::kDynamicIndex);
      dynamicIndices.push_back(val);
    } else {
      rawConstantIndices.push_back(
          iter.template get<cudaq::cc::InterleavedArgumentConstantIndex>());
    }

    currType =
        TypeSwitch<Type, Type>(currType)
            .Case([](cudaq::cc::ArrayType containerType) {
              return containerType.getElementType();
            })
            .Case([&](cudaq::cc::StructType structType) -> Type {
              auto memberIndex = rawConstantIndices.back();
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
  SmallVector<std::int32_t> rawConstantIndices;
  SmallVector<Value> dynamicIndices;
  Type elementType = cast<cc::PointerType>(basePtr.getType()).getElementType();
  destructureIndices<cudaq::cc::ComputePtrOp>(
      elementType, cpArgs, rawConstantIndices, dynamicIndices);

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
  // Params is a list of possible substitutions (Attributes) the length of the
  // SSA arguments. Skip the first one, which is the base pointer argument.
  auto paramIter = adaptor.getOperands().begin();
  ++paramIter;

  auto dynamicIndexIter = getDynamicIndices().begin();
  SmallVector<std::int32_t> newConstantIndices;
  SmallVector<Value> newIndices;
  bool changed = false;

  // Build lists of raw constants and SSA values with the SSA values that have
  // substituions omitted and properly interleaved in as constants in the first
  // list.
  for (auto index : getRawConstantIndices()) {
    if (index != kDynamicIndex) {
      newConstantIndices.push_back(index);
      continue;
    }
    if (auto newVal = dyn_cast_if_present<IntegerAttr>(*paramIter)) {
      newConstantIndices.push_back(newVal.getInt());
      changed = true;
    } else {
      newConstantIndices.push_back(kDynamicIndex);
      newIndices.push_back(*dynamicIndexIter);
    }
    ++dynamicIndexIter;
    ++paramIter;
  }

  // If any new constants were found, update the cc.compute_ptr in place, adding
  // the new constants and dropping any unneeded SSA arguments on the floor.
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
/// If two (or more) `cc.compute_ptr` are chained then they can be fused into a
/// single `cc.compute_ptr`.
struct FuseAddressArithmetic
    : public OpRewritePattern<cudaq::cc::ComputePtrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::ComputePtrOp ptrOp,
                                PatternRewriter &rewriter) const override {
    auto base = ptrOp.getBase();
    if (auto prev = base.getDefiningOp<cudaq::cc::ComputePtrOp>()) {
      // %prev = cc.compute_ptr %pb[<pargs>] : (!ptr<U>, Ts) -> !ptr<V>
      // %this = cc.compute_ptr %prev[<targs>] : (!ptr<V>, Ss) -> !ptr<W>
      // ────────────────────────────────────────────────────────────────
      // %prev = <left as is>
      // %this' = cc.compute_ptr %pb[<pargs>, <targs>] :
      //                                      (!ptr<U>, Ts Ss) -> !ptr<W>
      auto newBase = prev.getBase();
      SmallVector<Value> newDynamics = prev.getDynamicIndices();
      newDynamics.append(ptrOp.getDynamicIndices().begin(),
                         ptrOp.getDynamicIndices().end());
      SmallVector<std::int32_t> newConstants{
          prev.getRawConstantIndices().begin(),
          prev.getRawConstantIndices().end()};
      newConstants.append(ptrOp.getRawConstantIndices().begin(),
                          ptrOp.getRawConstantIndices().end());
      rewriter.replaceOpWithNewOp<cudaq::cc::ComputePtrOp>(
          ptrOp, ptrOp.getType(), newBase, newDynamics, newConstants);
      return success();
    }

    // We always favor the more restricted array type over an open array type.
    // Consider tagged the compute_ptr so a less restrictive correctness check
    // might be made.
    if (auto cast = base.getDefiningOp<cudaq::cc::CastOp>()) {
      // %cast = cc.cast %p : (!ptr<array<U x n>>) -> !ptr<array<U x ?>>
      // %this = cc.compute_ptr %cast[<targs>] : (!ptr<U x ?>, Ts) -> !ptr<V>
      // ────────────────────────────────────────────────────────────────────
      // %cast = <left as is>
      // %this' = cc.compute_ptr %p[<targs>] : (!ptr<U x n>, Ts) -> !ptr<V>
      auto fromTy = dyn_cast<cudaq::cc::PointerType>(cast.getValue().getType());
      auto toTy = dyn_cast<cudaq::cc::PointerType>(cast.getType());
      if (fromTy && toTy) {
        auto fromArrTy =
            dyn_cast<cudaq::cc::ArrayType>(fromTy.getElementType());
        auto toArrTy = dyn_cast<cudaq::cc::ArrayType>(toTy.getElementType());
        if (fromArrTy && toArrTy &&
            fromArrTy.getElementType() == toArrTy.getElementType() &&
            !fromArrTy.isUnknownSize() && toArrTy.isUnknownSize()) {
          rewriter.replaceOpWithNewOp<cudaq::cc::ComputePtrOp>(
              ptrOp, ptrOp.getType(), cast.getValue(),
              ptrOp.getDynamicIndices(), ptrOp.getRawConstantIndices());
          return success();
        }
      }
    }

    if (ptrOp.getRawConstantIndices().empty()) {
      // This is a degenerate form and really a cast.
      rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(ptrOp, ptrOp.getType(),
                                                     ptrOp.getBase());
      return success();
    }

    if (ptrOp.getDynamicIndices().empty()) {
      bool allZeros = true;
      for (std::int32_t i : ptrOp.getRawConstantIndices())
        if (i != 0) {
          allZeros = false;
          break;
        }
      if (allZeros) {
        // This is really a cast. Replace it with a cast.
        rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(ptrOp, ptrOp.getType(),
                                                       ptrOp.getBase());
        return success();
      }
    }

    if (ptrOp.llvmNormalForm() && ptrOp.getRawConstantIndices()[0] == 0) {
      // This is in LLVM normal form. Simplify it using the following rule.
      //
      // %8 = cc.compute_ptr %7[0, ...] :
      //                    (!cc.ptr<!cc.array<T x ?>, ...) -> !cc.ptr<U>
      // ────────────────────────────────────────────────────────────────
      // %new = cc.cast %7 : (!cc.ptr<!cc.array<T x ?>) -> !cc.ptr<T>
      // %8 = cc.compute_ptr %new[...] : (!cc.ptr<T>, ...) -> !cc.ptr<U>

      // We want to avoid expanding the code and adding more casts.
      if (auto castOp = ptrOp.getBase().getDefiningOp<cudaq::cc::CastOp>())
        if (isa<cudaq::cc::PointerType>(castOp.getValue().getType())) {
          auto ptrTy = cast<cudaq::cc::PointerType>(ptrOp.getBase().getType());
          auto eleTy = cast<cudaq::cc::ArrayType>(ptrTy.getElementType());
          auto subTy = eleTy.getElementType();
          auto simpleTy = cudaq::cc::PointerType::get(subTy);
          auto simple = rewriter.create<cudaq::cc::CastOp>(
              ptrOp.getLoc(), simpleTy, ptrOp.getBase());

          // Collect indices.
          auto iter = ptrOp.getDynamicIndices().begin();
          SmallVector<cudaq::cc::ComputePtrArg> indices;
          for (auto i : ptrOp.getRawConstantIndices().drop_front(1)) {
            if (i == cudaq::cc::ComputePtrOp::getDynamicIndexValue())
              indices.push_back(*iter++);
            else
              indices.push_back(i);
          }

          rewriter.replaceOpWithNewOp<cudaq::cc::ComputePtrOp>(
              ptrOp, ptrOp.getType(), simple, indices);
          return success();
        }
    }
    return failure();
  }
};
} // namespace

void cudaq::cc::ComputePtrOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseAddressArithmetic>(context);
}

std::optional<std::int32_t>
cudaq::cc::ComputePtrOp::getConstantIndex(std::size_t arg) {
  if (arg >= getRawConstantIndices().size())
    return {};
  std::int32_t result = getRawConstantIndices()[arg];
  if (result == getDynamicIndexValue())
    return {};
  return {result};
}

//===----------------------------------------------------------------------===//
// ExtractValueOp
//===----------------------------------------------------------------------===//

// Recursively determine if \p ty1 and \p ty2 are compatible. Ordinarily we can
// use type equality, but for StructType we may have a named and unnamed struct
// that are equivalent structurally.
static bool isCompatible(Type ty1, Type ty2) {
  if (ty1 == ty2)
    return true;
  auto sty1 = dyn_cast<cudaq::cc::StructType>(ty1);
  auto sty2 = dyn_cast<cudaq::cc::StructType>(ty2);
  if (sty1 && sty2) {
    if (sty1.getMembers().size() != sty2.getMembers().size() ||
        sty1.getPacked() != sty2.getPacked())
      return false;
    for (auto [a, b] : llvm::zip(sty1.getMembers(), sty2.getMembers()))
      if (!isCompatible(a, b))
        return false;
    return true;
  }
  return false;
}

LogicalResult cudaq::cc::ExtractValueOp::verify() {
  Type eleTy = getAggregate().getType();
  auto resultTy = getResult().getType();
  for (std::int32_t i : getRawConstantIndices()) {
    if (auto arrTy = dyn_cast<cc::ArrayType>(eleTy)) {
      if (arrTy.isUnknownSize())
        return emitOpError("array must have constant size");
      if (i != kDynamicIndex && (i < 0 || i >= arrTy.getSize()))
        return emitOpError("array cannot index out of bounds elements");
      eleTy = arrTy.getElementType();
    } else if (auto strTy = dyn_cast<cc::StructType>(eleTy)) {
      if (i == kDynamicIndex)
        return emitOpError("struct cannot have non-constant index");
      if (i < 0 || static_cast<std::size_t>(i) >= strTy.getMembers().size())
        return emitOpError("struct cannot index out of bounds members");
      eleTy = strTy.getMember(i);
    } else if (auto complexTy = dyn_cast<ComplexType>(eleTy)) {
      if (!(i == 0 || i == 1))
        return emitOpError("complex index is out of bounds");
      eleTy = complexTy.getElementType();
    } else {
      return emitOpError("too many indices (" +
                         std::to_string(getRawConstantIndices().size()) +
                         ") for the source pointer");
    }
  }
  if (!isCompatible(eleTy, resultTy))
    return emitOpError("result type does not match input");
  return success();
}

OpFoldResult cudaq::cc::ExtractValueOp::fold(FoldAdaptor adaptor) {
  if (indicesAreConstant())
    return nullptr;

  // Params is a list of possible substitutions (Attributes) the length of the
  // SSA arguments. Skip the first one, which is the base pointer argument.
  auto paramIter = adaptor.getOperands().begin();
  ++paramIter;

  auto dynamicIndexIter = getDynamicIndices().begin();
  SmallVector<std::int32_t> newConstantIndices;
  SmallVector<Value> newIndices;
  bool changed = false;

  // Build lists of raw constants and SSA values with the SSA values that have
  // substituions omitted and properly interleaved in as constants in the first
  // list.
  for (auto index : getRawConstantIndices()) {
    if (index != kDynamicIndex) {
      newConstantIndices.push_back(index);
      continue;
    }
    if (auto newVal = dyn_cast_if_present<IntegerAttr>(*paramIter)) {
      newConstantIndices.push_back(newVal.getInt());
      changed = true;
    } else {
      newConstantIndices.push_back(kDynamicIndex);
      newIndices.push_back(*dynamicIndexIter);
    }
    ++dynamicIndexIter;
    ++paramIter;
  }

  // If any new constants were found, update the cc.compute_ptr in place, adding
  // the new constants and dropping any unneeded SSA arguments on the floor.
  if (changed) {
    assert(newConstantIndices.size() == getRawConstantIndices().size());
    assert(newIndices.size() < getDynamicIndices().size());
    getDynamicIndicesMutable().assign(newIndices);
    setRawConstantIndices(newConstantIndices);
    return Value{*this};
  }
  return nullptr;
}

static ParseResult parseExtractValueIndices(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices,
    DenseI32ArrayAttr &rawConstantIndices) {
  return parseInterleavedIndices<cudaq::cc::ExtractValueOp>(parser, indices,
                                                            rawConstantIndices);
}

static void printExtractValueIndices(OpAsmPrinter &printer,
                                     cudaq::cc::ExtractValueOp extractValueOp,
                                     OperandRange indices,
                                     DenseI32ArrayAttr rawConstantIndices) {
  printInterleavedIndices<cudaq::cc::ExtractValueIndicesAdaptor<OperandRange>>(
      printer, extractValueOp, indices, rawConstantIndices);
}

void cudaq::cc::ExtractValueOp::build(OpBuilder &builder,
                                      OperationState &result, Type resultType,
                                      Value aggregate,
                                      ArrayRef<ExtractValueArg> indices,
                                      ArrayRef<NamedAttribute> attrs) {
  SmallVector<std::int32_t> rawConstantIndices;
  SmallVector<Value> dynamicIndices;
  Type elementType = aggregate.getType();
  destructureIndices<cudaq::cc::ExtractValueOp>(
      elementType, indices, rawConstantIndices, dynamicIndices);

  result.addTypes(resultType);
  result.addAttributes(attrs);
  result.addAttribute(getRawConstantIndicesAttrName(result.name),
                      builder.getDenseI32ArrayAttr(rawConstantIndices));
  result.addOperands(aggregate);
  result.addOperands(dynamicIndices);
}
void cudaq::cc::ExtractValueOp::build(OpBuilder &builder,
                                      OperationState &result, Type resultType,
                                      Value aggregate, ValueRange indices,
                                      ArrayRef<NamedAttribute> attrs) {
  SmallVector<ExtractValueArg> args{indices.begin(), indices.end()};
  build(builder, result, resultType, aggregate, args, attrs);
}
void cudaq::cc::ExtractValueOp::build(OpBuilder &builder,
                                      OperationState &result, Type resultType,
                                      Value aggregate, std::int32_t index,
                                      ArrayRef<NamedAttribute> attrs) {
  build(builder, result, resultType, aggregate,
        ArrayRef<ExtractValueArg>{index}, attrs);
}
void cudaq::cc::ExtractValueOp::build(OpBuilder &builder,
                                      OperationState &result, Type resultType,
                                      Value aggregate, Value index,
                                      ArrayRef<NamedAttribute> attrs) {
  build(builder, result, resultType, aggregate, ValueRange{index}, attrs);
}

namespace {
struct FuseWithConstantArray
    : public OpRewritePattern<cudaq::cc::ExtractValueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::ExtractValueOp extval,
                                PatternRewriter &rewriter) const override {
    if (auto conarr =
            extval.getAggregate().getDefiningOp<cudaq::cc::ConstantArrayOp>())
      if (extval.indicesAreConstant() &&
          extval.getRawConstantIndices().size() == 1) {
        if (auto intTy = dyn_cast<IntegerType>(extval.getType())) {
          std::int32_t i = extval.getRawConstantIndices()[0];
          auto cval = cast<IntegerAttr>(conarr.getConstantValues()[i]).getInt();
          rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(extval, cval,
                                                            intTy);

          return success();
        }
        if (auto fltTy = dyn_cast<FloatType>(extval.getType())) {
          std::int32_t i = extval.getRawConstantIndices()[0];
          auto cval = cast<FloatAttr>(conarr.getConstantValues()[i]).getValue();
          rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(extval, cval,
                                                              fltTy);

          return success();
        }
        if (auto complexTy = dyn_cast<ComplexType>(extval.getType())) {
          std::int32_t i = extval.getRawConstantIndices()[0];
          auto cval = cast<ArrayAttr>(conarr.getConstantValues()[i]);
          rewriter.replaceOpWithNewOp<complex::ConstantOp>(extval, complexTy,
                                                           cval);
          return success();
        }
      }
    return failure();
  }
};
} // namespace

void cudaq::cc::ExtractValueOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseWithConstantArray>(context);
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

ParseResult cudaq::cc::GlobalOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  // Check for the `extern` optional keyword first.
  if (succeeded(parser.parseOptionalKeyword("extern")))
    result.addAttribute(getExternalAttrName(result.name),
                        parser.getBuilder().getUnitAttr());

  // Check for the `constant` optional keyword second.
  if (succeeded(parser.parseOptionalKeyword("constant")))
    result.addAttribute(getConstantAttrName(result.name),
                        parser.getBuilder().getUnitAttr());

  // Check for the visibility optional keyword third.
  StringRef visibility;
  if (parser.parseOptionalKeyword(&visibility, {"public", "private", "nested"}))
    return failure();

  StringAttr visibilityAttr = parser.getBuilder().getStringAttr(visibility);
  result.addAttribute(SymbolTable::getVisibilityAttrName(), visibilityAttr);

  // Parse the rest of the global.
  //   @<symbol> ( <initializer-attr> ) : <result-type>
  StringAttr name;
  if (parser.parseSymbolName(name, getSymNameAttrName(result.name),
                             result.attributes))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    Attribute value;
    if (parser.parseAttribute(value, getValueAttrName(result.name),
                              result.attributes) ||
        parser.parseRParen())
      return failure();
  }
  SmallVector<Type, 1> types;
  if (parser.parseOptionalColonTypeList(types) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (types.size() > 1)
    return parser.emitError(parser.getNameLoc(), "expected zero or one type");
  result.addAttribute(getGlobalTypeAttrName(result.name),
                      TypeAttr::get(types[0]));
  return success();
}

void cudaq::cc::GlobalOp::print(OpAsmPrinter &p) {
  p << ' ';
  if (getExternal())
    p << "extern ";
  if (getConstant())
    p << "constant ";

  if (auto visibility = getSymVisibility())
    if (visibility.has_value())
      p << visibility.value().str() << ' ';

  p.printSymbolName(getSymName());
  if (auto value = getValue()) {
    p << " (";
    p.printAttribute(*value);
    p << ")";
  }
  p << " : " << getGlobalType();

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getSymNameAttrName(), getValueAttrName(),
                           getGlobalTypeAttrName(), getConstantAttrName(),
                           getExternalAttrName(), getSymVisibilityAttrName()});
}

//===----------------------------------------------------------------------===//
// InsertValueOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::InsertValueOp::verify() {
  Type eleTy = getContainer().getType();
  auto resultTy = getResult().getType();

  if (!isCompatible(eleTy, resultTy))
    return emitOpError("result type does not match input");

  for (std::int32_t i : getPosition()) {
    if (auto arrTy = dyn_cast<cc::ArrayType>(eleTy)) {
      if (arrTy.isUnknownSize())
        return emitOpError("array must have constant size");
      if (i < 0 || static_cast<std::int64_t>(i) >= arrTy.getSize())
        return emitOpError("array cannot index out of bounds elements");
      eleTy = arrTy.getElementType();
    } else if (auto strTy = dyn_cast<cc::StructType>(eleTy)) {
      if (i < 0 || static_cast<std::size_t>(i) >= strTy.getMembers().size())
        return emitOpError("struct cannot index out of bounds members");
      eleTy = strTy.getMember(i);
    } else if (auto complexTy = dyn_cast<ComplexType>(eleTy)) {
      if (!(i == 0 || i == 1))
        return emitOpError("complex index is out of bounds");
      eleTy = complexTy.getElementType();
    } else {
      return emitOpError(std::string{"too many indices ("} +
                         std::to_string(getPosition().size()) +
                         ") for the source pointer");
    }
  }

  Type valTy = getValue().getType();
  if (!isCompatible(valTy, eleTy))
    return emitOpError("value type does not match selected element");
  return success();
}

//===----------------------------------------------------------------------===//
// StdvecInitOp
//===----------------------------------------------------------------------===//

namespace {
struct CollapseCastToStdvecInit
    : public OpRewritePattern<cudaq::cc::StdvecInitOp> {
  using Base = OpRewritePattern<cudaq::cc::StdvecInitOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::StdvecInitOp init,
                                PatternRewriter &rewriter) const override {
    if (auto buff = init.getBuffer().getDefiningOp<cudaq::cc::CastOp>()) {
      auto castVal = buff.getValue();
      auto fromPtrTy = dyn_cast<cudaq::cc::PointerType>(castVal.getType());
      if (!fromPtrTy)
        return failure();
      auto fromTy = fromPtrTy.getElementType();
      auto toTy = cast<cudaq::cc::PointerType>(buff.getType()).getElementType();
      if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(fromTy))
        if (!isa<cudaq::cc::ArrayType>(toTy)) {
          if (arrTy.isUnknownSize())
            rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
                init, init.getType(), castVal, init.getLength());
          else
            rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
                init, init.getType(), castVal);
          return success();
        }
    }
    return failure();
  }
};

struct FoldStdvecInit : public OpRewritePattern<cudaq::cc::StdvecInitOp> {
  using Base = OpRewritePattern<cudaq::cc::StdvecInitOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::StdvecInitOp init,
                                PatternRewriter &rewriter) const override {
    if (auto arrTy =
            dyn_cast<cudaq::cc::ArrayType>(init.getBuffer().getType())) {
      if (arrTy.isUnknownSize())
        return failure();
      if (auto len = init.getLength())
        if (auto optInt = cudaq::opt::factory::getIntIfConstant(len))
          if (*optInt == arrTy.getSize()) {
            rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
                init, init.getType(), init.getBuffer());
            return success();
          }
    }
    return failure();
  }
};
} // namespace

void cudaq::cc::StdvecInitOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<CollapseCastToStdvecInit, FoldStdvecInit>(context);
}

LogicalResult cudaq::cc::StdvecInitOp::verify() {
  Value buff = getBuffer();
  auto buffTy = cast<cc::PointerType>(buff.getType());
  auto buffEleTy = buffTy.getElementType();
  if (auto arrTy = dyn_cast<cc::ArrayType>(buffEleTy)) {
    if (arrTy.isUnknownSize()) {
      if (!getLength())
        return emitOpError("must specify a length.");
    } else {
      // Input buffer is an array of constant length. If there is a length
      // argument provided, it must not exceed the length of the buffer.
      if (auto len = getLength())
        if (auto optInt = opt::factory::getIntIfConstant(len))
          if (*optInt > arrTy.getSize())
            return emitOpError("length override exceeds array length.");
    }
    buffEleTy = arrTy.getElementType();
  }
  // FIXME: For now leave the loophole that the input buffer may be a "void*" or
  // "char*" and implicitly casted by this operation.
  if (buffEleTy != NoneType::get(getContext()) &&
      buffEleTy != IntegerType::get(getContext(), 8) &&
      buffEleTy != cast<cc::SpanLikeType>(getType()).getElementType())
    return emitOpError("element types must be the same.");
  return success();
}

//===----------------------------------------------------------------------===//
// StdvecDataOp
//===----------------------------------------------------------------------===//

namespace {
struct ForwardStdvecInitData
    : public OpRewritePattern<cudaq::cc::StdvecDataOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::StdvecDataOp data,
                                PatternRewriter &rewriter) const override {
    // Bypass the std::vector wrappers for the creation of an abstract
    // subvector. This is possible because copies of std::vector data aren't
    // created but instead passed around like std::span objects. Specifically, a
    // pointer to the data and a length. Thus the pointer wrapped by stdvec_init
    // and unwrapped by stdvec_data is the same pointer value. This pattern will
    // arise after inlining, for example.
    if (auto ini = data.getStdvec().getDefiningOp<cudaq::cc::StdvecInitOp>()) {
      rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(data, data.getType(),
                                                     ini.getBuffer());
      return success();
    }
    return failure();
  }
};
} // namespace

void cudaq::cc::StdvecDataOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ForwardStdvecInitData>(context);
}

//===----------------------------------------------------------------------===//
// StdvecSizeOp
//===----------------------------------------------------------------------===//

namespace {
struct ForwardStdvecInitSize
    : public OpRewritePattern<cudaq::cc::StdvecSizeOp> {
  using Base = OpRewritePattern<cudaq::cc::StdvecSizeOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::StdvecSizeOp size,
                                PatternRewriter &rewriter) const override {
    if (auto init = size.getStdvec().getDefiningOp<cudaq::cc::StdvecInitOp>()) {
      auto ty = size.getType();
      if (Value len = init.getLength()) {
        rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(size, ty, len);
        return success();
      }
      if (auto arrTy =
              dyn_cast<cudaq::cc::ArrayType>(init.getBuffer().getType()))
        if (!arrTy.isUnknownSize()) {
          rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(
              size, arrTy.getSize(), ty);
          return success();
        }
    }
    return failure();
  }
};
} // namespace

void cudaq::cc::StdvecSizeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ForwardStdvecInitSize>(context);
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
  [[maybe_unused]] auto *elseRegion = result.addRegion();
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
  build(builder, result, iterArgs.getTypes(), iterArgs, postCond, whileBuilder,
        bodyBuilder, stepBuilder);
}

void cudaq::cc::LoopOp::build(OpBuilder &builder, OperationState &result,
                              TypeRange resultTypes, ValueRange iterArgs,
                              RegionBuilderFn whileBuilder,
                              RegionBuilderFn bodyBuilder,
                              RegionBuilderFn stepBuilder,
                              RegionBuilderFn elseBuilder) {
  auto *whileRegion = result.addRegion();
  auto *bodyRegion = result.addRegion();
  auto *stepRegion = result.addRegion();
  auto *elseRegion = result.addRegion();
  whileBuilder(builder, result.location, *whileRegion);
  bodyBuilder(builder, result.location, *bodyRegion);
  stepBuilder(builder, result.location, *stepRegion);
  ensureStepTerminator(builder, result, stepRegion);
  elseBuilder(builder, result.location, *elseRegion);
  result.addAttribute(postCondAttrName(), builder.getBoolAttr(false));
  result.addOperands(iterArgs);
  result.addTypes(resultTypes);
}

void cudaq::cc::LoopOp::build(OpBuilder &builder, OperationState &result,
                              ValueRange iterArgs, RegionBuilderFn whileBuilder,
                              RegionBuilderFn bodyBuilder,
                              RegionBuilderFn stepBuilder,
                              RegionBuilderFn elseBuilder) {
  build(builder, result, iterArgs.getTypes(), iterArgs, whileBuilder,
        bodyBuilder, stepBuilder, elseBuilder);
}

LogicalResult cudaq::cc::LoopOp::verify() {
  const auto initArgsSize = getInitialArgs().size();
  if (getResults().size() != initArgsSize)
    return emitOpError("size of init args and outputs must be equal");
  if (getWhileArguments().size() != initArgsSize)
    return emitOpError("size of init args and while region args must be equal");
  if (auto condOp = dyn_cast<ConditionOp>(getWhileBlock()->getTerminator())) {
    if (condOp.getResults().size() != initArgsSize)
      return emitOpError("size of init args and condition op must be equal");
  } else {
    return emitOpError("while region must end with condition op");
  }
  if (getDoEntryArguments().size() != initArgsSize)
    return emitOpError("size of init args and body region args must be equal");
  if (hasStep()) {
    if (isPostConditional())
      return emitOpError("post-conditional loop cannot have a step region");
    if (getStepArguments().size() != initArgsSize)
      return emitOpError(
          "size of init args and step region args must be equal");
    if (auto contOp = dyn_cast<ContinueOp>(getStepBlock()->getTerminator())) {
      if (contOp.getOperands().size() != initArgsSize)
        return emitOpError("size of init args and continue op must be equal");
    } else {
      return emitOpError("step region must end with continue op");
    }
  }
  if (hasPythonElse()) {
    if (isPostConditional())
      return emitOpError("post-conditional loop cannot have an else region");
    if (getElseEntryArguments().size() != initArgsSize)
      return emitOpError(
          "size of init args and else region args must be equal");
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
    printInitializationList(p, getDoEntryArguments(), getOperands());
    p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
    p << " while ";
    p.printRegion(getWhileRegion(), /*printEntryBlockArgs=*/hasArguments(),
                  /*printBlockTerminators=*/true);
  } else {
    p << " while ";
    printInitializationList(p, getWhileArguments(), getOperands());
    p.printRegion(getWhileRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
    p << " do ";
    p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/hasArguments(),
                  /*printBlockTerminators=*/true);
    if (hasStep()) {
      p << " step ";
      p.printRegion(getStepRegion(), /*printEntryBlockArgs=*/hasArguments(),
                    /*printBlockTerminators=*/hasArguments());
    }
    if (hasPythonElse()) {
      p << " else ";
      p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/hasArguments(),
                    /*printBlockTerminators=*/true);
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
  auto *elseReg = result.addRegion();
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
    if (succeeded(parser.parseOptionalKeyword("else"))) {
      if (parser.parseRegion(*elseReg, emptyArgs))
        return failure();
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
    // loop op, successor is either the WHILE region, or the DO region if loop
    // is post conditional.
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
    // WHILE region, successors are the DO region and either the owning loop op
    // (if no else region is present) or the else region.
    regions.push_back(RegionSuccessor(&getBodyRegion(), getDoEntryArguments()));
    if (hasPythonElse())
      regions.push_back(
          RegionSuccessor(&getElseRegion(), getElseEntryArguments()));
    else
      regions.push_back(RegionSuccessor(getResults()));
    break;
  case 1:
    // DO region, successor is STEP region (2) if present, or WHILE region (0)
    // if STEP is absent.
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
    // STEP region, if present, WHILE region is always successor.
    if (hasStep())
      regions.push_back(
          RegionSuccessor(&getWhileRegion(), getWhileArguments()));
    break;
  case 3:
    // ELSE region, successors are the owning loop op.
    if (hasPythonElse())
      regions.push_back(RegionSuccessor(getResults()));
    break;
  }
}

OperandRange
cudaq::cc::LoopOp::getSuccessorEntryOperands(std::optional<unsigned> index) {
  assert(index && "invalid index region");
  switch (*index) {
  case 0:
    if (!isPostConditional())
      return getInitialArgs();
    break;
  case 1:
    if (isPostConditional())
      return getInitialArgs();
    break;
  }
  return {nullptr, 0};
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
//    } else {
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
//    } else {
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
      return success();
    }
    return failure();
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
  if (parser.parseRegion(*body, /*arguments=*/{}) ||
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

// If quantumAllocs, then just look for any allocate memory effect. Otherwise,
// look for any allocate memory other than from the quake dialect.
template <bool quantumAllocs>
bool hasAllocation(Region &region) {
  for (auto &block : region)
    for (auto &op : block) {
      if (auto mem = dyn_cast<MemoryEffectOpInterface>(op))
        if (mem.hasEffect<MemoryEffects::Allocate>())
          if (quantumAllocs || !isa<quake::AllocaOp>(op))
            return true;
      if (!isa<cudaq::cc::ScopeOp>(op))
        for (auto &opReg : op.getRegions())
          if (hasAllocation<quantumAllocs>(opReg))
            return true;
    }
  return false;
}

bool cudaq::cc::ScopeOp::hasAllocation(bool quantumAllocs) {
  if (quantumAllocs)
    return ::hasAllocation</*quantumAllocs=*/true>(getRegion());
  return ::hasAllocation</*quantumAllocs=*/false>(getRegion());
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
    if (scope.hasAllocation())
      return failure();

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

void cudaq::cc::IfOp::build(OpBuilder &builder, OperationState &result,
                            TypeRange resultTypes, Value cond,
                            ValueRange linearVals, RegionBuilderFn thenBuilder,
                            RegionBuilderFn elseBuilder) {
  auto *thenRegion = result.addRegion();
  auto *elseRegion = result.addRegion();
  thenBuilder(builder, result.location, *thenRegion);
  if (elseBuilder)
    elseBuilder(builder, result.location, *elseRegion);
  result.addOperands(cond);
  result.addOperands(linearVals);
  result.addTypes(resultTypes);
}

LogicalResult cudaq::cc::IfOp::verify() {
  if (getNumResults() != 0 && getElseRegion().empty())
    return emitOpError("must have an else block if defining values");
  return success();
}

void cudaq::cc::IfOp::print(OpAsmPrinter &p) {
  p << '(' << getCondition() << ')';
  if (!getLinearArgs().empty()) {
    p << " ((";
    llvm::interleaveComma(
        llvm::zip(getThenEntryArguments(), getLinearArgs()), p,
        [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
    p << "))";
  }
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
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    // Linear type arguments.
    SmallVector<OpAsmParser::UnresolvedOperand, 4> linearOperands;
    if (parser.parseAssignmentList(regionArgs, linearOperands) ||
        parser.parseRParen())
      return failure();
    Type wireTy = quake::WireType::get(builder.getContext());
    for (auto argOperand : llvm::zip(regionArgs, linearOperands)) {
      std::get<0>(argOperand).type = wireTy;
      if (parser.resolveOperand(std::get<1>(argOperand), wireTy,
                                result.operands))
        return failure();
    }
  }
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  if (!regionArgs.empty()) {
    // Check that the result.types is compatible with the regionArgs. To be
    // compatible, it must have at least as many linear types as there are
    // region arguments. (It can have more.)
    std::int64_t numRegionArgs = regionArgs.size();
    std::for_each(result.types.begin(), result.types.end(), [&](Type t) {
      if (quake::isLinearType(t))
        --numRegionArgs;
    });
    if (numRegionArgs > 0)
      return failure();
  }
  if (parser.parseRegion(*thenRegion, regionArgs))
    return failure();
  OpBuilder opBuilder(parser.getContext());
  ensureIfRegionTerminator(opBuilder, result, thenRegion);

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, regionArgs))
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

template <typename A>
long countLinearArgs(const A &iterable) {
  return std::count_if(iterable.begin(), iterable.end(),
                       [](Type t) { return quake::isLinearType(t); });
}

LogicalResult cudaq::cc::verifyConvergentLinearTypesInRegions(Operation *op) {
  auto regionOp = dyn_cast_if_present<RegionBranchOpInterface>(op);
  if (!regionOp)
    return failure();
  SmallVector<RegionSuccessor> successors;
  regionOp.getSuccessorRegions(std::nullopt, {}, successors);
  // For each region successor, determine the number of distinct linear-typed
  // definitions in the region.
  long linearMax = -1;
  for (auto iter : successors)
    if (iter.getSuccessor())
      for (Block &block : *iter.getSuccessor())
        if (auto term = dyn_cast<cc::ContinueOp>(block.getTerminator())) {
          auto numLinearArgs = countLinearArgs(term.getOperands().getTypes());
          if (numLinearArgs > linearMax)
            linearMax = numLinearArgs;
        }

  // All regions must have the same number of linear-type arguments and the
  // region with the maximal number of distinct linear-typed definitions.
  for (auto iter : successors)
    if (iter.getSuccessor()) {
      auto *block = &iter.getSuccessor()->front();
      if (static_cast<long>(block->getNumArguments()) != linearMax)
        return failure();
    }

  return success();
}

namespace {
struct KillRegionIfConstant : public OpRewritePattern<cudaq::cc::IfOp> {
  using Base = OpRewritePattern<cudaq::cc::IfOp>;
  using Base::Base;

  // This rewrite will determine if the condition is constant. If it is, then it
  // will elide the true or false region completely, depending on the constant's
  // value.
  LogicalResult matchAndRewrite(cudaq::cc::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto cond = ifOp.getCondition();
    if (!ifOp.getResults().empty())
      return failure();
    auto con = cond.getDefiningOp<arith::ConstantIntOp>();
    if (!con)
      return failure();
    auto val = con.value();
    auto loc = ifOp.getLoc();
    auto truth = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    Region *newRegion = nullptr;
    if (val) {
      // The else block, if any, is dead.
      if (ifOp.getElseRegion().empty())
        return failure();
      newRegion = &ifOp.getThenRegion();
    } else {
      // The then block is dead.
      newRegion = &ifOp.getElseRegion();
      if (newRegion->empty()) {
        // If there was no else, then build an empty dummy Region.
        OpBuilder::InsertionGuard guard(rewriter);
        Block *block = new Block();
        rewriter.setInsertionPointToEnd(block);
        rewriter.create<cudaq::cc::ContinueOp>(loc);
        newRegion->push_back(block);
      }
    }
    rewriter.replaceOpWithNewOp<cudaq::cc::IfOp>(
        ifOp, ifOp.getResultTypes(), truth,
        [&](OpBuilder &, Location, Region &region) {
          region.takeBody(*newRegion);
        });
    return success();
  }
};
} // namespace

void cudaq::cc::IfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<KillRegionIfConstant>(context);
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
  const bool hasArgs = getRegion().getNumArguments() != 0;
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/hasArgs,
                /*printBlockTerminators=*/true);
  p << " : " << getType();
  p.printOptionalAttrDict((*this)->getAttrs(), {"signature"});
}

ParseResult cudaq::cc::CreateLambdaOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  auto *body = result.addRegion();
  Type lambdaTy;
  if (parser.parseRegion(*body, /*arguments=*/{}) ||
      parser.parseColonType(lambdaTy) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("signature", TypeAttr::get(lambdaTy));
  result.addTypes(lambdaTy);
  return success();
}

//===----------------------------------------------------------------------===//
// CallableFuncOp
//===----------------------------------------------------------------------===//

namespace {
// FIXME: Same rewrite pattern as appears in LambdaLifting.cpp. Share it!
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
} // namespace

void cudaq::cc::CallableFuncOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<CallableFuncOpPattern>(context);
}

//===----------------------------------------------------------------------===//
// CallCallableOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::CallCallableOp::verify() {
  FunctionType funcTy;
  auto ty = getCallee().getType();
  if (auto lambdaTy = dyn_cast<CallableType>(ty))
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
// CallIndirectCallableOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::CallIndirectCallableOp::verify() {
  FunctionType funcTy =
      cast<IndirectCallableType>(getCallee().getType()).getSignature();

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

namespace {
struct MakeDirectCall
    : public OpRewritePattern<cudaq::cc::CallIndirectCallableOp> {
  using Base = OpRewritePattern<cudaq::cc::CallIndirectCallableOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::CallIndirectCallableOp indCall,
                                PatternRewriter &rewriter) const override {
    if (auto cast = indCall.getCallee().getDefiningOp<cudaq::cc::CastOp>())
      if (auto fn = cast.getValue().getDefiningOp<func::ConstantOp>()) {
        rewriter.replaceOpWithNewOp<func::CallIndirectOp>(indCall, fn,
                                                          indCall.getArgs());
        return success();
      }
    return failure();
  }
};
} // namespace

void cudaq::cc::CallIndirectCallableOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<MakeDirectCall>(context);
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
// NoInlineCallOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::NoInlineCallOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

//===----------------------------------------------------------------------===//
// DeviceCallOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::DeviceCallOp::verify() {
  if (getNumBlocks().size() > 3)
    return emitOpError(
        "the number of blocks  must have a maximum dimension of 3");
  if (getNumThreadsPerBlock().size() > 3)
    return emitOpError(
        "the number of threads per block must have a maximum dimension of 3");
  return success();
}

LogicalResult
cudaq::cc::DeviceCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  func::FuncOp fn =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getArgs().size())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getArgs()[i].getType() != fnType.getInput(i)) {
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getArgs()[i].getType() << " for operand number " << i;
    }

  if (fnType.getResults().empty() && getNumResults() == 0)
    return success();

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("number of results does not agree");

  for (auto [myRes, resTy] : llvm::zip(getResults(), fnType.getResults()))
    if (myRes.getType() != resTy) {
      auto diag = emitOpError("result type mismatch ");
      diag.attachNote() << "      op result types: " << myRes.getType();
      diag.attachNote() << "function result types: " << resTy;
      return diag;
    }
  return success();
}

//===----------------------------------------------------------------------===//
// OffsetOfOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::OffsetOfOp::verify() {
  Type ty = getInputType();
  for (std::int32_t i : getConstantIndices()) {
    if (auto strTy = dyn_cast<cc::StructType>(ty)) {
      if (i < 0 || static_cast<std::size_t>(i) >= strTy.getMembers().size())
        return emitOpError("out of bounds for struct");
      ty = strTy.getMembers()[i];
      continue;
    }
    if (auto arrTy = dyn_cast<cc::ArrayType>(ty)) {
      if (arrTy.isUnknownSize())
        return emitOpError("array must have bounds");
      if (i < 0 || i >= arrTy.getSize())
        return emitOpError("out of bounds for array");
      ty = arrTy.getElementType();
      continue;
    }
    if (auto complexTy = dyn_cast<ComplexType>(ty)) {
      if (i < 0 || i > 1)
        return emitOpError("out of bounds for complex");
      ty = complexTy.getElementType();
      continue;
    }
    // Walking off the end of the type.
    return failure();
  }
  return success();
}

namespace {
struct FoldTrivialOffsetOf : public OpRewritePattern<cudaq::cc::OffsetOfOp> {
  using Base = OpRewritePattern<cudaq::cc::OffsetOfOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::OffsetOfOp offOp,
                                PatternRewriter &rewriter) const override {
    // If there are no offsets, the offset is 0.
    if (offOp.getConstantIndices().empty()) {
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(offOp, 0,
                                                        offOp.getType());
      return success();
    }

    // If all indices are 0, then the offset is necessarily 0.
    if (std::all_of(offOp.getConstantIndices().begin(),
                    offOp.getConstantIndices().end(),
                    [](std::int32_t i) { return i == 0; })) {
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(offOp, 0,
                                                        offOp.getType());
      return success();
    }

    return failure();
  }
};
} // namespace

void cudaq::cc::OffsetOfOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldTrivialOffsetOf>(context);
}

//===----------------------------------------------------------------------===//
// ReifySpanOp
//===----------------------------------------------------------------------===//

LogicalResult cudaq::cc::ReifySpanOp::verify() {
  auto conArr = getElements().getDefiningOp<cudaq::cc::ConstantArrayOp>();
  if (!conArr && !isa<BlockArgument>(getElements()))
    return emitOpError("requires a constant array argument.");
  if (conArr.arrayDimension() != spanDimension())
    return emitOpError("input array dimension must be same as span dimension.");
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
  patterns.add<ReplaceInFunc<ReturnOp, func::ReturnOp>>(context);
}

//===----------------------------------------------------------------------===//
// SizeOfOp
//===----------------------------------------------------------------------===//

namespace {
struct ReplaceConstantSizes : public OpRewritePattern<cudaq::cc::SizeOfOp> {
  using Base = OpRewritePattern<cudaq::cc::SizeOfOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::SizeOfOp sizeOp,
                                PatternRewriter &rewriter) const override {
    // TODO: Add handling of more types.
    auto inpTy = sizeOp.getInputType();
    if (Value v = cudaq::cc::getByteSizeOfType(rewriter, sizeOp.getLoc(), inpTy,
                                               /*useSizeOf=*/false)) {
      if (v.getType() != sizeOp.getType()) {
        auto vSz = v.getType().getIntOrFloatBitWidth();
        auto sizeOpSz = sizeOp.getType().getIntOrFloatBitWidth();
        auto loc = sizeOp.getLoc();
        if (sizeOpSz < vSz)
          v = rewriter.create<cudaq::cc::CastOp>(loc, sizeOp.getType(), v);
        else
          v = rewriter.create<cudaq::cc::CastOp>(
              loc, sizeOp.getType(), v, cudaq::cc::CastOpMode::Unsigned);
      }
      rewriter.replaceOp(sizeOp, v);
      return success();
    }
    return failure();
  }
};
} // namespace

void cudaq::cc::SizeOfOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ReplaceConstantSizes>(context);
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
      return success();
    }
    return failure();
  }
};

void cudaq::cc::UnwindBreakOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ReplaceInLoop<UnwindBreakOp, BreakOp>>(context);
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
  patterns.add<ReplaceInLoop<UnwindContinueOp, ContinueOp>>(context);
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
// VarargCallOp
//===----------------------------------------------------------------------===//

LogicalResult
cudaq::cc::VarargCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  LLVM::LLVMFuncOp fn =
      symbolTable.lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid LLVM function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumParams() > getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumParams(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getParams()[i]) {
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getParams()[i] << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;
    }

  if (fnType.getReturnType() == LLVM::LLVMVoidType::get(getContext()) &&
      getNumResults() == 0)
    return success();

  if (getNumResults() > 1)
    return emitOpError("wrong number of result types: ") << getNumResults();

  if (getResult(0).getType() != fnType.getReturnType()) {
    auto diag = emitOpError("result type mismatch ");
    diag.attachNote() << "      op result types: " << getResultTypes();
    diag.attachNote() << "function result types: " << fnType.getReturnType();
    return diag;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/CC/CCOps.cpp.inc"
