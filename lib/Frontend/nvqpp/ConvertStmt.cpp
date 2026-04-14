/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "lower-ast-stmt"

using namespace mlir;

namespace cudaq::details {

bool QuakeBridgeVisitor::hasTerminator(Block &block) {
  return !block.empty() && block.back().hasTrait<OpTrait::IsTerminator>();
}

bool QuakeBridgeVisitor::VisitBreakStmt(clang::BreakStmt *x) {
  // It is a C++ syntax error if a break statement is not in a loop or switch
  // statement. The bridge does not currently support switch statements.
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (builder.getBlock())
    cc::UnwindBreakOp::create(builder, toLocation(x));
  return true;
}

bool QuakeBridgeVisitor::VisitContinueStmt(clang::ContinueStmt *x) {
  // It is a C++ syntax error if a continue statement is not in a loop.
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (builder.getBlock())
    cc::UnwindContinueOp::create(builder, toLocation(x));
  return true;
}

bool QuakeBridgeVisitor::TraverseDeclStmt(clang::DeclStmt *x,
                                          DataRecursionQueue *q) {
  const auto depthBefore = valueStack.size();
  auto result = Base::TraverseDeclStmt(x, q);
  const auto depthAfter = valueStack.size();
  if (result && depthAfter > depthBefore) {
    [[maybe_unused]] auto unused = lastValues(depthAfter - depthBefore);
  }
  return result;
}

bool QuakeBridgeVisitor::VisitCompoundAssignOperator(
    clang::CompoundAssignOperator *x) {
  auto loc = toLocation(x->getSourceRange());
  auto rhs = popValue();
  auto lhsPtr = popValue();
  auto lhs = loadLValue(lhsPtr);

  // Coerce the rhs to be the same sized type as the lhs.
  if (x->getType()->isIntegerType())
    rhs = integerCoercion(loc, x->getRHS()->getType(), lhs.getType(), rhs);
  else if (x->getType()->isFloatingType())
    rhs = floatingPointCoercion(loc, lhs.getType(), rhs);

  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  auto result = [&]() -> mlir::Value {
    switch (x->getOpcode()) {
    case clang::BinaryOperatorKind::BO_AddAssign: {
      if (x->getType()->isIntegerType())
        return mlir::arith::AddIOp::create(builder, loc, lhs, rhs);
      if (x->getType()->isFloatingType())
        return mlir::arith::AddFOp::create(builder, loc, lhs, rhs);
      TODO_loc(loc, "Unknown type in assignment operator");
    }
    case clang::BinaryOperatorKind::BO_SubAssign: {
      if (x->getType()->isIntegerType())
        return mlir::arith::SubIOp::create(builder, loc, lhs, rhs);
      if (x->getType()->isFloatingType())
        return mlir::arith::SubFOp::create(builder, loc, lhs, rhs);
      TODO_loc(loc, "Unknown type in assignment operator");
    }
    case clang::BinaryOperatorKind::BO_MulAssign: {
      if (x->getType()->isIntegerType())
        return mlir::arith::MulIOp::create(builder, loc, lhs, rhs);
      if (x->getType()->isFloatingType())
        return mlir::arith::MulFOp::create(builder, loc, lhs, rhs);
      TODO_loc(loc, "Unknown type in assignment operator");
    }
    case clang::BinaryOperatorKind::BO_DivAssign: {
      if (x->getType()->isIntegerType())
        if (x->getType()->isUnsignedIntegerOrEnumerationType())
          return mlir::arith::DivUIOp::create(builder, loc, lhs, rhs);
      return mlir::arith::DivSIOp::create(builder, loc, lhs, rhs);
      if (x->getType()->isFloatingType())
        return mlir::arith::DivFOp::create(builder, loc, lhs, rhs);
      TODO_loc(loc, "Unknown type in assignment operator");
    }
    case clang::BinaryOperatorKind::BO_ShlAssign:
      return mlir::arith::ShLIOp::create(builder, loc, lhs, rhs);
    case clang::BinaryOperatorKind::BO_ShrAssign:
      if (x->getType()->isUnsignedIntegerOrEnumerationType())
        return mlir::arith::ShRUIOp::create(builder, loc, lhs, rhs);
      return mlir::arith::ShRSIOp::create(builder, loc, lhs, rhs);
    case clang::BinaryOperatorKind::BO_OrAssign:
      return mlir::arith::OrIOp::create(builder, loc, lhs, rhs);
    case clang::BinaryOperatorKind::BO_XorAssign:
      return mlir::arith::XOrIOp::create(builder, loc, lhs, rhs);
    case clang::BinaryOperatorKind::BO_AndAssign:
      return mlir::arith::AndIOp::create(builder, loc, lhs, rhs);
    default:
      break;
    }
    TODO_loc(loc, "assignment operator");
  }();

  cudaq::cc::StoreOp::create(builder, loc, result, lhsPtr);
  return pushValue(lhsPtr);
}

bool QuakeBridgeVisitor::TraverseAsmStmt(clang::AsmStmt *x,
                                         DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "asm statement");
  return false;
}

bool QuakeBridgeVisitor::TraverseCXXCatchStmt(clang::CXXCatchStmt *x,
                                              DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "catch statement");
  return false;
}

bool QuakeBridgeVisitor::TraverseCXXForRangeStmt(clang::CXXForRangeStmt *x,
                                                 DataRecursionQueue *) {
  auto loc = toLocation(x);
  if (!TraverseStmt(x->getRangeInit()))
    return false;
  Value buffer = popValue();
  bool result = true;
  auto *body = x->getBody();
  auto *loopVar = x->getLoopVariable();
  auto i64Ty = builder.getI64Type();
  if (auto stdvecTy = dyn_cast<cc::SpanLikeType>(buffer.getType())) {
    auto eleTy = stdvecTy.getElementType();
    const bool isBool = eleTy == builder.getI1Type();
    if (isBool)
      eleTy = builder.getI8Type();
    auto dataPtrTy = cc::PointerType::get(eleTy);
    auto dataArrPtrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
    auto [iters, ptr, initial,
          stepBy] = [&]() -> std::tuple<Value, Value, Value, Value> {
      if (auto call = buffer.getDefiningOp<func::CallOp>()) {
        if (call.getCallee() == setCudaqRangeVector) {
          // The std::vector was produced by cudaq::range(). Optimize this
          // special case to use the loop control directly. Erase the transient
          // buffer and call here since neither is required.
          Value i = call.getOperand(1);
          if (auto alloc = call.getOperand(0).getDefiningOp<cc::AllocaOp>()) {
            call->erase(); // erase call must be first
            alloc->erase();
          } else {
            // shouldn't get here, but we can erase the call at minimum
            call->erase();
          }
          return {i, {}, {}, {}};
        } else if (call.getCallee() == setCudaqRangeVectorTriple) {
          // Save operands before erasing the call.
          Value initial = call.getOperand(1);
          Value i = call.getOperand(2);
          Value stepBy = call.getOperand(3);
          if (auto alloc = call.getOperand(0).getDefiningOp<cc::AllocaOp>()) {
            Operation *callGetSizeOp = nullptr;
            if (auto seqSize = alloc.getSeqSize()) {
              if (auto callSize = seqSize.getDefiningOp<func::CallOp>())
                if (callSize.getCallee() == getCudaqSizeFromTriple)
                  callGetSizeOp = callSize.getOperation();
            }
            call->erase(); // erase call must be first
            alloc->erase();
            if (callGetSizeOp)
              callGetSizeOp->erase();
          } else {
            // shouldn't get here, but we can erase the call at minimum
            call->erase();
          }
          return {i, {}, initial, stepBy};
        }
      }
      Value i = cc::StdvecSizeOp::create(builder, loc, i64Ty, buffer);
      Value p = cc::StdvecDataOp::create(builder, loc, dataArrPtrTy, buffer);
      return {i, p, {}, {}};
    }();

    auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region,
                           Block &block) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&block);
      Value index = block.getArgument(0);
      // May need to create a temporary for the loop variable. Create a new
      // scope.
      auto scopeBuilder = [&](OpBuilder &builder, Location loc) {
        if (!ptr) {
          // cudaq::range(N): not necessary to collect values from a buffer, the
          // values are the same as the index.
          symbolTable.insert(loopVar->getName(), index);
        } else {
          Value addr =
              cc::ComputePtrOp::create(builder, loc, dataPtrTy, ptr, index);
          if (loopVar->getType().isConstQualified()) {
            // Read-only binding, so omit copy.
            symbolTable.insert(loopVar->getName(), addr);
          } else if (loopVar->getType().getTypePtr()->isReferenceType()) {
            // Bind to location of the value in the container, std::vector<T>.
            symbolTable.insert(loopVar->getName(), addr);
          } else {
            // Create a local copy of the value from the container.
            if (!TraverseVarDecl(loopVar)) {
              result = false;
              return;
            }
            auto iterVar = popValue();
            Value atOffset = cc::LoadOp::create(builder, loc, addr);
            if (isBool)
              atOffset = cc::CastOp::create(builder, loc, builder.getI1Type(),
                                            atOffset);
            cc::StoreOp::create(builder, loc, atOffset, iterVar);
          }
        }
        if (!TraverseStmt(static_cast<clang::Stmt *>(body))) {
          result = false;
          return;
        }
        cc::ContinueOp::create(builder, loc);
      };
      cc::ScopeOp::create(builder, loc, scopeBuilder);
    };

    if (!initial) {
      auto idxIters = cudaq::cc::CastOp::create(
          builder, loc, i64Ty, iters, cudaq::cc::CastOpMode::Unsigned);
      opt::factory::createInvariantLoop(builder, loc, idxIters, bodyBuilder);
    } else {
      auto idxIters = cudaq::cc::CastOp::create(builder, loc, i64Ty, iters,
                                                cudaq::cc::CastOpMode::Signed);
      opt::factory::createMonotonicLoop(builder, loc, initial, idxIters, stepBy,
                                        bodyBuilder);
    }
  } else if (auto veqTy = dyn_cast<quake::VeqType>(buffer.getType());
             veqTy && veqTy.hasSpecifiedSize()) {
    Value iters = arith::ConstantIntOp::create(
        builder, loc, i64Ty, static_cast<int64_t>(veqTy.getSize()));
    auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region,
                           Block &block) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&block);
      Value index = block.getArgument(0);
      Value ref = quake::ExtractRefOp::create(builder, loc, buffer, index);
      symbolTable.insert(loopVar->getName(), ref);
      if (!TraverseStmt(static_cast<clang::Stmt *>(body)))
        result = false;
    };
    auto idxIters = cudaq::cc::CastOp::create(builder, loc, i64Ty, iters,
                                              cudaq::cc::CastOpMode::Unsigned);
    opt::factory::createInvariantLoop(builder, loc, idxIters, bodyBuilder);
  } else {
    TODO_x(toLocation(x), x, mangler, "ranged for statement");
  }
  return result;
}

bool QuakeBridgeVisitor::TraverseCXXTryStmt(clang::CXXTryStmt *x,
                                            DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "try statement");
  return false;
}

bool QuakeBridgeVisitor::TraverseCapturedStmt(clang::CapturedStmt *x,
                                              DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "captured statement");
  return false;
}

bool QuakeBridgeVisitor::TraverseCoreturnStmt(clang::CoreturnStmt *x,
                                              DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "coreturn statement");
  return false;
}

bool QuakeBridgeVisitor::TraverseCoroutineBodyStmt(clang::CoroutineBodyStmt *x,
                                                   DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "coroutine body statement");
  return false;
}

bool QuakeBridgeVisitor::TraverseGotoStmt(clang::GotoStmt *x,
                                          DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "goto statement");
  return false;
}

bool QuakeBridgeVisitor::TraverseIndirectGotoStmt(clang::IndirectGotoStmt *x,
                                                  DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "indirect goto statement");
  return false;
}

bool QuakeBridgeVisitor::TraverseSwitchStmt(clang::SwitchStmt *x,
                                            DataRecursionQueue *q) {
  TODO_x(toLocation(x), x, mangler, "switch statement");
  return false;
}

bool QuakeBridgeVisitor::VisitReturnStmt(clang::ReturnStmt *x) {
  auto loc = toLocation(x->getSourceRange());
  bool isFuncScope = [&]() {
    if (auto *block = builder.getBlock())
      if (auto *region = block->getParent())
        if (auto *op = region->getParentOp())
          return isa<func::FuncOp, cc::CreateLambdaOp>(op);
    return false;
  }();
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (x->getRetValue()) {
    auto result = popValue();
    auto resTy = result.getType();
    if (isa<cc::PointerType>(resTy)) {
      // Promote reference (T&) to value (T) on a return. (There is not
      // necessarily an explicit cast or promotion node in the AST.)
      auto load = cc::LoadOp::create(builder, loc, result);
      result = load.getResult();
      if (load.getType() == builder.getI8Type()) {
        auto fnTy = load->getParentOfType<func::FuncOp>().getFunctionType();
        auto i1Ty = builder.getI1Type();
        if (fnTy.getNumResults() == 1 && fnTy.getResult(0) == i1Ty)
          result = cc::CastOp::create(builder, loc, i1Ty, result);
      }
    }
    if (auto vecTy = dyn_cast<cc::SpanLikeType>(resTy)) {
      // Returning vector data that was allocated on the stack is not valid.
      // Allocate space on the heap and make a copy of the vector instead. It
      // will be the responsibility of the calling side to free this memory.
      auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
      if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_vectorCopyCtor")))
        module.emitError("failed to load intrinsic");
      auto eleTy = vecTy.getElementType();
      auto createVectorInit = [&](Value eleSize) {
        auto ptrTy = cudaq::cc::PointerType::get(builder.getI8Type());
        Value resBuff = cc::StdvecDataOp::create(builder, loc, ptrTy, result);
        Value dynSize = cc::StdvecSizeOp::create(builder, loc,
                                                 builder.getI64Type(), result);
        Value heapCopy =
            func::CallOp::create(builder, loc, ptrTy, "__nvqpp_vectorCopyCtor",
                                 ValueRange{resBuff, dynSize, eleSize})
                .getResult(0);
        return cc::StdvecInitOp::create(builder, loc, resTy,
                                        ValueRange{heapCopy, dynSize});
      };
      IRBuilder irb(builder);
      Value tySize;
      if (!cudaq::cc::isDynamicType(eleTy))
        tySize = irb.getByteSizeOfType(loc, eleTy);
      if (!tySize) {
        // TODO: we need to recursively create copies of all
        // dynamic memory used within the type. See the
        // implementation of `visit_Return` in the Python bridge.
        TODO_x(toLocation(x), x, mangler, "unhandled vector element type");
        return false;
      }
      result = createVectorInit(tySize);
    }
    if (isFuncScope)
      cc::ReturnOp::create(builder, loc, result);
    else
      cc::UnwindReturnOp::create(builder, loc, result);
    return true;
  }
  if (isFuncScope)
    cc::ReturnOp::create(builder, loc);
  else
    cc::UnwindReturnOp::create(builder, loc);
  return true;
}

bool QuakeBridgeVisitor::TraverseCompoundStmt(clang::CompoundStmt *stmt,
                                              DataRecursionQueue *q) {
  auto loc = toLocation(stmt->getSourceRange());
  SymbolTableScope var_scope(symbolTable);
  auto traverseAndCheck = [&](clang::Stmt *cs) {
    LLVM_DEBUG(llvm::dbgs() << "[[[\n"; cs->dump());
    if (!TraverseStmt(cs))
      reportClangError(cs, mangler, "statement not supported in qpu kernel");
    LLVM_DEBUG({
      if (!typeStack.empty()) {
        llvm::dbgs() << "\n\nERROR: type stack has garbage after stmt:\n";
        for (auto t : llvm::reverse(typeStack))
          t.dump();
        typeStack.clear();
      }
      llvm::dbgs() << "]]]\n";
    });
  };
  if (skipCompoundScope) {
    skipCompoundScope = false;
    for (auto *cs : stmt->body())
      traverseAndCheck(static_cast<clang::Stmt *>(cs));
    return true;
  }
  cc::ScopeOp::create(builder, loc, [&](OpBuilder &builder, Location loc) {
    for (auto *cs : stmt->body())
      traverseAndCheck(static_cast<clang::Stmt *>(cs));
    cc::ContinueOp::create(builder, loc);
  });
  return true;
}

// Shared implementation for lowering of `do while` and `while` loops.
template <bool postCondition, typename S>
bool QuakeBridgeVisitor::traverseDoOrWhileStmt(S *x) {
  bool result = true;
  auto loc = toLocation(x);
  auto *cond = x->getCond();
  auto whileBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
    if (!result)
      return;
    region.push_back(new Block());
    auto &bodyBlock = region.front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    if (!TraverseStmt(static_cast<clang::Stmt *>(cond))) {
      result = false;
      return;
    }
    auto val = popValue();
    cc::ConditionOp::create(builder, loc, val, ValueRange{});
  };
  auto *body = x->getBody();
  auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
    if (!result)
      return;
    region.push_back(new Block());
    auto &bodyBlock = region.front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    if (!TraverseStmt(static_cast<clang::Stmt *>(body))) {
      result = false;
      return;
    }
    if (!hasTerminator(region.back()))
      cc::ContinueOp::create(builder, loc);
  };
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  cc::LoopOp::create(builder, loc, ValueRange{}, postCondition, whileBuilder,
                     bodyBuilder);
  return result;
}

bool QuakeBridgeVisitor::TraverseDoStmt(clang::DoStmt *x,
                                        DataRecursionQueue *) {
  return traverseDoOrWhileStmt</*postCondition=*/true>(x);
}

bool QuakeBridgeVisitor::TraverseWhileStmt(clang::WhileStmt *x,
                                           DataRecursionQueue *) {
  return traverseDoOrWhileStmt</*postCondition=*/false>(x);
}

bool QuakeBridgeVisitor::TraverseIfStmt(clang::IfStmt *x,
                                        DataRecursionQueue *q) {
  bool result = true;
  auto loc = toLocation(x);
  auto stmtBuilder = [&](clang::Stmt *stmt) {
    return [&, stmt](OpBuilder &builder, Location loc, Region &region) {
      if (!result)
        return;
      region.push_back(new Block());
      auto &bodyBlock = region.front();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&bodyBlock);
      if (!TraverseStmt(stmt)) {
        result = false;
        return;
      }
      if (!hasTerminator(region.back()))
        cc::ContinueOp::create(builder, loc);
    };
  };
  auto *cond = x->getCond();
  assert(cond && "if statement should have a condition");
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (auto *init = x->getInit()) {
    cc::ScopeOp::create(builder, loc, [&](OpBuilder &builder, Location loc) {
      SymbolTableScope varScope(symbolTable);
      if (!TraverseStmt(init) || !TraverseStmt(cond)) {
        result = false;
        return;
      }
      if (x->getElse())
        cc::IfOp::create(builder, loc, TypeRange{}, popValue(),
                         stmtBuilder(x->getThen()), stmtBuilder(x->getElse()));
      else
        cc::IfOp::create(builder, loc, TypeRange{}, popValue(),
                         stmtBuilder(x->getThen()));
      cc::ContinueOp::create(builder, loc);
    });
  } else {
    // If there is no initialization expression, skip creating an `if` scope.
    if (!TraverseStmt(cond))
      return false;

    // For something like an `operator[]` the TOS value (likely) is a
    // pointer to the indexed element in a vector. Since there may not be a cast
    // node in the AST to make that a RHS value, we must explicitly check here
    // and add the required a load and cast.
    if (auto ptrTy = dyn_cast<cc::PointerType>(peekValue().getType())) {
      Value v = popValue();
      pushValue(cc::LoadOp::create(builder, loc, v));
      if (ptrTy != builder.getI1Type()) {
        reportClangError(x, mangler,
                         "expression in condition not yet supported");
      }
    }
    if (x->getElse())
      cc::IfOp::create(builder, loc, TypeRange{}, popValue(),
                       stmtBuilder(x->getThen()), stmtBuilder(x->getElse()));
    else
      cc::IfOp::create(builder, loc, TypeRange{}, popValue(),
                       stmtBuilder(x->getThen()));
  }
  return result;
}

bool QuakeBridgeVisitor::TraverseForStmt(clang::ForStmt *x,
                                         DataRecursionQueue *q) {
  bool result = true;
  auto loc = toLocation(x);
  auto *cond = x->getCond();
  const auto initialValueDepth = valueStack.size();
  auto whileBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
    if (!result)
      return;
    region.push_back(new Block());
    auto &bodyBlock = region.front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    if (!TraverseStmt(static_cast<clang::Stmt *>(cond))) {
      result = false;
      return;
    }
    auto val = popValue();
    cc::ConditionOp::create(builder, loc, val, ValueRange{});
  };
  auto *body = x->getBody();
  auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
    if (!result)
      return;
    region.push_back(new Block());
    auto &bodyBlock = region.front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    if (!TraverseStmt(static_cast<clang::Stmt *>(body))) {
      result = false;
      return;
    }
    if (!hasTerminator(region.back()))
      cc::ContinueOp::create(builder, loc);
  };
  auto *incr = x->getInc();
  auto stepBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
    if (!result)
      return;
    region.push_back(new Block());
    auto &bodyBlock = region.front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    if (!TraverseStmt(static_cast<clang::Stmt *>(incr)))
      result = false;
  };

  constexpr bool postCondition = false;
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (auto *init = x->getInit()) {
    SymbolTableScope var_scope(symbolTable);
    cc::ScopeOp::create(builder, loc, [&](OpBuilder &builder, Location loc) {
      if (!TraverseStmt(static_cast<clang::Stmt *>(init))) {
        result = false;
        return;
      }
      cc::LoopOp::create(builder, loc, ValueRange{}, postCondition,
                         whileBuilder, bodyBuilder, stepBuilder);
      cc::ContinueOp::create(builder, loc);
    });
  } else {
    // If there is no initialization expression, skip creating a `for` scope.
    cc::LoopOp::create(builder, loc, ValueRange{}, postCondition, whileBuilder,
                       bodyBuilder);
  }
  const auto finalValueDepth = valueStack.size();
  if (finalValueDepth > initialValueDepth) {
    [[maybe_unused]] auto vals =
        lastValues(finalValueDepth - initialValueDepth);
    LLVM_DEBUG({
      llvm::dbgs() << "Garbage left after loop:\n";
      for (auto v : vals)
        v.dump();
      llvm::dbgs() << "\n";
    });
  }

  return result;
}

} // namespace cudaq::details
