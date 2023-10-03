/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Builder/CUDAQBuilder.h"
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
    builder.create<cc::UnwindBreakOp>(toLocation(x));
  return true;
}

bool QuakeBridgeVisitor::VisitContinueStmt(clang::ContinueStmt *x) {
  // It is a C++ syntax error if a continue statement is not in a loop.
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (builder.getBlock())
    builder.create<cc::UnwindContinueOp>(toLocation(x));
  return true;
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
        return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
      if (x->getType()->isFloatingType())
        return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
      TODO_loc(loc, "Unknown type in assignment operator");
    }
    case clang::BinaryOperatorKind::BO_SubAssign: {
      if (x->getType()->isIntegerType())
        return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
      if (x->getType()->isFloatingType())
        return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
      TODO_loc(loc, "Unknown type in assignment operator");
    }
    case clang::BinaryOperatorKind::BO_MulAssign: {
      if (x->getType()->isIntegerType())
        return builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
      if (x->getType()->isFloatingType())
        return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
      TODO_loc(loc, "Unknown type in assignment operator");
    }
    case clang::BinaryOperatorKind::BO_DivAssign: {
      if (x->getType()->isIntegerType())
        if (x->getType()->isUnsignedIntegerOrEnumerationType())
          return builder.create<mlir::arith::DivUIOp>(loc, lhs, rhs);
      return builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
      if (x->getType()->isFloatingType())
        return builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
      TODO_loc(loc, "Unknown type in assignment operator");
    }
    case clang::BinaryOperatorKind::BO_ShlAssign:
      return builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
    case clang::BinaryOperatorKind::BO_ShrAssign:
      if (x->getType()->isUnsignedIntegerOrEnumerationType())
        return builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs);
      return builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
    case clang::BinaryOperatorKind::BO_OrAssign:
      return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
    case clang::BinaryOperatorKind::BO_XorAssign:
      return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
    case clang::BinaryOperatorKind::BO_AndAssign:
      return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
    default:
      break;
    }
    TODO_loc(loc, "assignment operator");
  }();

  builder.create<cudaq::cc::StoreOp>(loc, result, lhsPtr);
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
  auto stdvecTy = dyn_cast<cc::StdvecType>(buffer.getType());
  assert(stdvecTy && "expected a std::vector<T>");
  auto eleTy = stdvecTy.getElementType();
  auto i64Ty = builder.getI64Type();
  auto iters = builder.create<cc::StdvecSizeOp>(loc, i64Ty, buffer);
  auto dataPtrTy = cc::PointerType::get(eleTy);
  auto dataArrPtrTy = cc::PointerType::get(cc::ArrayType::get(eleTy));
  auto ptr = builder.create<cc::StdvecDataOp>(loc, dataArrPtrTy, buffer);
  bool result = true;
  auto *body = x->getBody();
  auto *loopVar = x->getLoopVariable();

  auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region,
                         Block &block) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);
    auto iterIdx = block.getArgument(0);
    Value index = builder.create<arith::IndexCastOp>(loc, i64Ty, iterIdx);
    // May need to create a temporary for the loop variable. Create a new scope.
    auto scopeBuilder = [&](OpBuilder &builder, Location loc) {
      Value addr = builder.create<cc::ComputePtrOp>(loc, dataPtrTy, ptr, index);
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
        Value atOffset = builder.create<cc::LoadOp>(loc, addr);
        builder.create<cc::StoreOp>(loc, atOffset, iterVar);
      }
      if (!TraverseStmt(static_cast<clang::Stmt *>(body))) {
        result = false;
        return;
      }
      builder.create<cc::ContinueOp>(loc);
    };
    builder.create<cc::ScopeOp>(loc, scopeBuilder);
  };

  auto idxTy = builder.getIndexType();
  auto idxIters = builder.create<arith::IndexCastOp>(loc, idxTy, iters);
  opt::factory::createInvariantLoop(builder, loc, idxIters, bodyBuilder);
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

bool QuakeBridgeVisitor::VisitReturnStmt(clang::ReturnStmt *stmt) {
  auto loc = toLocation(stmt->getSourceRange());
  bool isFuncScope = [&]() {
    if (auto *block = builder.getBlock())
      if (auto *region = block->getParent())
        if (auto *op = region->getParentOp())
          return isa<func::FuncOp, cc::CreateLambdaOp>(op);
    return false;
  }();
  LLVM_DEBUG(llvm::dbgs() << "%% "; stmt->dump());
  if (stmt->getRetValue()) {
    auto result = popValue();
    auto resTy = result.getType();
    if (isa<cc::PointerType>(resTy)) {
      // Promote reference (T&) to value (T) on a return. (There is not
      // necessarily an explicit cast or promotion node in the AST.)
      result = builder.create<cc::LoadOp>(loc, result);
    }
    if (auto vecTy = dyn_cast<cc::StdvecType>(resTy)) {
      // Returning vector data that was allocated on the stack is not valid.
      // Allocate space on the heap and make a copy of the vector instead. It
      // will be the responsibility of the calling side to free this memory.
      auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
      if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_vectorCopyCtor")))
        module.emitError("failed to load intrinsic");
      auto eleTy = vecTy.getElementType();
      auto ptrTy = cudaq::cc::PointerType::get(builder.getI8Type());
      Value resBuff = builder.create<cc::StdvecDataOp>(loc, ptrTy, result);
      std::size_t byteWidth = (eleTy.getIntOrFloatBitWidth() + 7) / 8;
      Value dynSize =
          builder.create<cc::StdvecSizeOp>(loc, builder.getI64Type(), result);
      auto eleSize = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(byteWidth));
      Value heapCopy =
          builder
              .create<func::CallOp>(loc, ptrTy, "__nvqpp_vectorCopyCtor",
                                    ValueRange{resBuff, dynSize, eleSize})
              .getResult(0);
      result = builder.create<cc::StdvecInitOp>(loc, resTy,
                                                ValueRange{heapCopy, dynSize});
    }
    if (isFuncScope)
      builder.create<cc::ReturnOp>(loc, result);
    else
      builder.create<cc::UnwindReturnOp>(loc, result);
    return true;
  }
  if (isFuncScope)
    builder.create<cc::ReturnOp>(loc);
  else
    builder.create<cc::UnwindReturnOp>(loc);
  return true;
}

bool QuakeBridgeVisitor::TraverseCompoundStmt(clang::CompoundStmt *stmt,
                                              DataRecursionQueue *q) {
  auto loc = toLocation(stmt->getSourceRange());
  SymbolTableScope var_scope(symbolTable);
  if (skipCompoundScope) {
    skipCompoundScope = false;
    for (auto *cs : stmt->body()) {
      LLVM_DEBUG(llvm::dbgs() << "[[[\n"; cs->dump());
      if (TraverseStmt(static_cast<clang::Stmt *>(cs))) {
        LLVM_DEBUG({
          if (!typeStack.empty()) {
            llvm::dbgs() << "\n\nERROR: type stack has garbage after stmt:\n";
            for (auto t : llvm::reverse(typeStack))
              t.dump();
            typeStack.clear();
          }
          llvm::dbgs() << "]]]\n";
        });
      }
    }
    return true;
  }
  bool result = true;
  builder.create<cc::ScopeOp>(loc, [&](OpBuilder &builder, Location loc) {
    for (auto *cs : stmt->body())
      if (!TraverseStmt(static_cast<clang::Stmt *>(cs))) {
        result = false;
        return;
      }
    builder.create<cc::ContinueOp>(loc);
  });
  return result;
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
    builder.create<cc::ConditionOp>(loc, val, ValueRange{});
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
      builder.create<cc::ContinueOp>(loc);
  };
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  builder.create<cc::LoopOp>(loc, ValueRange{}, postCondition, whileBuilder,
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
        builder.create<cc::ContinueOp>(loc);
    };
  };
  auto *cond = x->getCond();
  assert(cond && "if statement should have a condition");
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (auto *init = x->getInit()) {
    builder.create<cc::ScopeOp>(loc, [&](OpBuilder &builder, Location loc) {
      SymbolTableScope varScope(symbolTable);
      if (!TraverseStmt(init) || !TraverseStmt(cond)) {
        result = false;
        return;
      }
      if (x->getElse())
        builder.create<cc::IfOp>(loc, TypeRange{}, popValue(),
                                 stmtBuilder(x->getThen()),
                                 stmtBuilder(x->getElse()));
      else
        builder.create<cc::IfOp>(loc, TypeRange{}, popValue(),
                                 stmtBuilder(x->getThen()));
      builder.create<cc::ContinueOp>(loc);
    });
  } else {
    // If there is no initialization expression, skip creating an `if` scope.
    if (!TraverseStmt(cond))
      return false;
    if (x->getElse())
      builder.create<cc::IfOp>(loc, TypeRange{}, popValue(),
                               stmtBuilder(x->getThen()),
                               stmtBuilder(x->getElse()));
    else
      builder.create<cc::IfOp>(loc, TypeRange{}, popValue(),
                               stmtBuilder(x->getThen()));
  }
  return result;
}

bool QuakeBridgeVisitor::TraverseForStmt(clang::ForStmt *x,
                                         DataRecursionQueue *q) {
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
    builder.create<cc::ConditionOp>(loc, val, ValueRange{});
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
      builder.create<cc::ContinueOp>(loc);
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
    builder.create<cc::ScopeOp>(loc, [&](OpBuilder &builder, Location loc) {
      if (!TraverseStmt(static_cast<clang::Stmt *>(init))) {
        result = false;
        return;
      }
      builder.create<cc::LoopOp>(loc, ValueRange{}, postCondition, whileBuilder,
                                 bodyBuilder, stepBuilder);
      builder.create<cc::ContinueOp>(loc);
    });
  } else {
    // If there is no initialization expression, skip creating a `for` scope.
    builder.create<cc::LoopOp>(loc, ValueRange{}, postCondition, whileBuilder,
                               bodyBuilder);
  }
  return result;
}

} // namespace cudaq::details
