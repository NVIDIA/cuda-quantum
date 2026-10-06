/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "lower-ast-stmt"

using namespace mlir;

namespace cudaq::detail {

bool QuakeBridgeVisitor::hasTerminator(Block &block) {
  return !block.empty() && block.back().hasTrait<OpTrait::IsTerminator>();
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::BreakStmt *x) {
  // It is a C++ syntax error if a break statement is not in a loop or switch
  // statement. The bridge does not currently support switch statements.
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (builder.getBlock())
    cc::UnwindBreakOp::create(builder, toLocation(x), currentLoopArgs());
  return std::nullopt;
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::ContinueStmt *x) {
  // It is a C++ syntax error if a continue statement is not in a loop.
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (builder.getBlock())
    cc::UnwindContinueOp::create(builder, toLocation(x), currentLoopArgs());
  return std::nullopt;
}

static_assert(derivedStmtsAreKnown<clang::AsmStmt, clang::GCCAsmStmt,
                                   clang::MSAsmStmt>());

void QuakeBridgeVisitor::reportUnsupportedNode(clang::Stmt *x) {
  auto &de = astContext->getDiagnostics();
  const auto id = de.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                     "'%0' is not yet supported in a kernel");
  de.Report(x->getBeginLoc(), id) << x->getStmtClassName();
  fail();
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::DeclStmt *x) {
  // The declarations contain their initializers.
  bool ok = true;
  for (auto *decl : x->decls()) {
    // A variable has code to generate, and a declaration of a function is a
    // reference to the function. The other declarations in a kernel declare
    // types: classes, enumerations, aliases, and so on. They generate no code.
    // A type is converted where a variable (or an expression) uses it, whether
    // it is declared here or not.
    if (!isa<clang::VarDecl, clang::FunctionDecl>(decl))
      continue;
    if (!traverseDecl(decl)) {
      ok = false;
      break;
    }
  }
  return finish(ok);
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::visit(clang::CompoundAssignOperator *x) {
  auto lhsPtrValue = traverseValue(x->getLHS());
  if (!lhsPtrValue)
    return std::nullopt;
  auto rhsValue = traverseValue(x->getRHS());
  if (!rhsValue)
    return std::nullopt;
  auto loc = toLocation(x->getSourceRange());
  auto rhs = *rhsValue;
  auto lhsPtr = *lhsPtrValue;
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
  return BridgeResult{lhsPtr};
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::AsmStmt *x) {
  // AsmStmt does not know where it is. The statement that it is does.
  auto *stmt = static_cast<clang::Stmt *>(x);
  TODO_x(toLocation(stmt), stmt, mangler, "asm statement");
  return fail();
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::CXXCatchStmt *x) {
  TODO_x(toLocation(x), x, mangler, "catch statement");
  return fail();
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::visit(clang::CXXForRangeStmt *x) {
  auto loc = toLocation(x);
  auto rangeValue = traverseValue(x->getRangeInit());
  if (!rangeValue)
    return std::nullopt;
  // `std::vector<measure_handle>` locals are stack-allocated by
  // `ConvertDecl.cpp` and arrive here as `!cc.ptr<!cc.sequence<...>>`; the
  // `SpanLikeType` dispatch below needs the descriptor value, not the slot
  // pointer. Other handle-vec consumers in `ConvertExpr.cpp` call the same
  // helper. The `quake::VeqType` arm is unaffected.
  Value buffer = loadHandleVectorIfPointer(builder, loc, *rangeValue);
  bool result = true;
  auto *body = x->getBody();
  auto *loopVar = x->getLoopVariable();
  auto i64Ty = builder.getI64Type();
  if (auto sequenceTy = dyn_cast<cc::SpanLikeType>(buffer.getType())) {
    auto eleTy = sequenceTy.getElementType();
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
      Value i = cc::SequenceSizeOp::create(builder, loc, i64Ty, buffer);
      Value p = cc::SequenceDataOp::create(builder, loc, dataArrPtrTy, buffer);
      return {i, p, {}, {}};
    }();

    auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region,
                           Block &block) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&block);
      LoopArgsScope loopArgsScope(*this, block.getArguments());
      Value index = initial ? block.getArgument(1) : block.getArgument(0);
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
            auto iterVarValue = traverseValue(loopVar);
            if (!iterVarValue) {
              result = false;
              return;
            }
            auto iterVar = *iterVarValue;
            Value atOffset = cc::LoadOp::create(builder, loc, addr);
            if (isBool) {
              atOffset = cc::CastOp::create(builder, loc, builder.getI1Type(),
                                            atOffset);
            } else if (isa<cc::MeasureHandleType>(atOffset.getType())) {
              // `for (bool b : mz(reg))` binds an `i1` loop variable to a
              // container of `!cc.measure_handle`. Mirror
              // `measure_handle::operator bool()` and lower the handle through
              // `quake.discriminate` before storing into the `i1` slot.
              // Without it the `cc.store` value type (`!cc.measure_handle`) and
              // pointer element type (`i1`) disagree and the verifier rejects
              // the module. A `measure_handle` loop variable
              // (`for (auto b : mz(reg))`) keeps the handle and is unaffected.
              if (auto iterPtrTy = dyn_cast<cc::PointerType>(iterVar.getType()))
                if (iterPtrTy.getElementType() == builder.getI1Type()) {
                  llvm::SmallPtrSet<Value, 4> visited;
                  if (isBoundHandleVector(buffer, visited)) {
                    atOffset = cudaq::quake::DiscriminateOp::create(
                        builder, loc, builder.getI1Type(), atOffset);
                  } else {
                    reportClangError(
                        x, mangler, "discriminating an unbound measure_handle");
                    // Substitute a well-typed `i1` so the shared store below
                    // stays valid, and let traversal succeed.
                    atOffset = arith::ConstantIntOp::create(
                        builder, loc, builder.getI1Type(), /*value=*/0);
                  }
                }
            }
            cc::StoreOp::create(builder, loc, atOffset, iterVar);
          }
        }
        if (!traverseStmt(static_cast<clang::Stmt *>(body))) {
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
  } else if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(buffer.getType());
             veqTy && veqTy.hasSpecifiedSize()) {
    Value iters = arith::ConstantIntOp::create(
        builder, loc, i64Ty, static_cast<int64_t>(veqTy.getSize()));
    auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region,
                           Block &block) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&block);
      LoopArgsScope loopArgsScope(*this, block.getArguments());
      Value index = block.getArgument(0);
      Value ref =
          cudaq::quake::ExtractRefOp::create(builder, loc, buffer, index);
      symbolTable.insert(loopVar->getName(), ref);
      if (!traverseStmt(static_cast<clang::Stmt *>(body)))
        result = false;
    };
    auto idxIters = cudaq::cc::CastOp::create(builder, loc, i64Ty, iters,
                                              cudaq::cc::CastOpMode::Unsigned);
    opt::factory::createInvariantLoop(builder, loc, idxIters, bodyBuilder);
  } else {
    TODO_x(toLocation(x), x, mangler, "ranged for statement");
  }
  return finish(result);
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::CXXTryStmt *x) {
  TODO_x(toLocation(x), x, mangler, "try statement");
  return fail();
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::CapturedStmt *x) {
  TODO_x(toLocation(x), x, mangler, "captured statement");
  return fail();
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::CoreturnStmt *x) {
  TODO_x(toLocation(x), x, mangler, "coreturn statement");
  return fail();
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::visit(clang::CoroutineBodyStmt *x) {
  TODO_x(toLocation(x), x, mangler, "coroutine body statement");
  return fail();
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::GotoStmt *x) {
  TODO_x(toLocation(x), x, mangler, "goto statement");
  return fail();
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::visit(clang::IndirectGotoStmt *x) {
  TODO_x(toLocation(x), x, mangler, "indirect goto statement");
  return fail();
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::SwitchStmt *x) {
  TODO_x(toLocation(x), x, mangler, "switch statement");
  return fail();
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::ReturnStmt *x) {
  std::optional<Value> returned;
  if (x->getRetValue()) {
    returned = traverseValue(x->getRetValue());
    if (!returned)
      return std::nullopt;
  }
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
    auto result = *returned;
    auto resTy = result.getType();
    if (isa<cc::PointerType>(resTy)) {
      // Promote reference (T&) to value (T) on a return. (There is not
      // necessarily an explicit cast or promotion node in the AST.)
      auto load = cc::LoadOp::create(builder, loc, result);
      result = load.getResult();
      // A `std::vector<measure_handle>` local is a descriptor slot, so it
      // arrives here in pointer form. After promoting it to a value, refresh
      // `resTy` to the loaded vector type so the `SpanLikeType` branch below
      // copies the vector contents to the heap before returning. Without the
      // refresh that branch tests the stale pointer type and is skipped,
      // returning a descriptor that aliases a buffer freed when the callee
      // returns.
      if (auto sv = dyn_cast<cc::SequenceType>(result.getType());
          sv && isa<cc::MeasureHandleType>(sv.getElementType()))
        resTy = result.getType();
      if (load.getType() == builder.getI8Type()) {
        auto fnTy = load->getParentOfType<func::FuncOp>().getFunctionType();
        auto i1Ty = builder.getI1Type();
        if (fnTy.getNumResults() == 1 && fnTy.getResult(0) == i1Ty)
          result = cc::CastOp::create(builder, loc, i1Ty, result);
      }
    }
    if (isa<cc::SpanLikeType>(resTy) ||
        (isa<cc::StructType>(result.getType()) &&
         cc::isDynamicType(result.getType()))) {
      // Returning vector data that was allocated on the stack is not valid.
      // Allocate space on the heap and make a copy of the vector instead. It
      // will be the responsibility of the calling side to free this memory.
      auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
      if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_vectorCopyCtor")) ||
          failed(irBuilder.loadIntrinsic(module, "malloc")))
        module.emitError("failed to load intrinsic");
      result = opt::marshal::copyDynamicValueToHeap(loc, builder, result);
    }
    if (isFuncScope)
      cc::ReturnOp::create(builder, loc, result);
    else
      cc::UnwindReturnOp::create(builder, loc, result);
    return std::nullopt;
  }
  if (isFuncScope)
    cc::ReturnOp::create(builder, loc);
  else
    cc::UnwindReturnOp::create(builder, loc);
  return std::nullopt;
}

QuakeBridgeVisitor::Result
QuakeBridgeVisitor::visit(clang::CompoundStmt *stmt) {
  auto loc = toLocation(stmt->getSourceRange());
  SymbolTableScope var_scope(symbolTable);
  auto traverseAndCheck = [&](clang::Stmt *cs) {
    LLVM_DEBUG(llvm::dbgs() << "[[[\n"; cs->dump());
    if (!traverseStmt(cs)) {
      reportClangError(cs, mangler, "statement not supported in qpu kernel");
      // Carry on with the next statement.
      clearFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "]]]\n");
  };
  if (skipCompoundScope) {
    skipCompoundScope = false;
    for (auto *cs : stmt->body())
      traverseAndCheck(static_cast<clang::Stmt *>(cs));
    return std::nullopt;
  }
  cc::ScopeOp::create(builder, loc, [&](OpBuilder &builder, Location loc) {
    for (auto *cs : stmt->body())
      traverseAndCheck(static_cast<clang::Stmt *>(cs));
    cc::ContinueOp::create(builder, loc);
  });
  return std::nullopt;
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
    auto val = traverseValue(static_cast<clang::Stmt *>(cond));
    if (!val) {
      result = false;
      return;
    }
    cc::ConditionOp::create(builder, loc, *val, ValueRange{});
  };
  auto *body = x->getBody();
  auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
    if (!result)
      return;
    region.push_back(new Block());
    auto &bodyBlock = region.front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    LoopArgsScope loopArgsScope(*this, ValueRange{});
    if (!traverseStmt(static_cast<clang::Stmt *>(body))) {
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

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::DoStmt *x) {
  return finish(traverseDoOrWhileStmt</*postCondition=*/true>(x));
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::WhileStmt *x) {
  return finish(traverseDoOrWhileStmt</*postCondition=*/false>(x));
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::IfStmt *x) {
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
      if (!traverseStmt(stmt)) {
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
      if (!traverseStmt(init)) {
        result = false;
        return;
      }
      auto condValue = traverseValue(cond);
      if (!condValue) {
        result = false;
        return;
      }
      if (x->getElse())
        cc::IfOp::create(builder, loc, TypeRange{}, *condValue,
                         stmtBuilder(x->getThen()), stmtBuilder(x->getElse()));
      else
        cc::IfOp::create(builder, loc, TypeRange{}, *condValue,
                         stmtBuilder(x->getThen()));
      cc::ContinueOp::create(builder, loc);
    });
  } else {
    // If there is no initialization expression, skip creating an `if` scope.
    auto condValue = traverseValue(cond);
    if (!condValue)
      return std::nullopt;
    Value condition = *condValue;

    // For something like an `operator[]` the value of the condition (likely)
    // is a pointer to the indexed element in a vector. Since there may not be a
    // cast node in the AST to make that a RHS value, we must explicitly check
    // here and add the required a load and cast.
    if (auto ptrTy = dyn_cast<cc::PointerType>(condition.getType())) {
      condition = cc::LoadOp::create(builder, loc, condition);
      if (ptrTy != builder.getI1Type()) {
        reportClangError(x, mangler,
                         "expression in condition not yet supported");
      }
    }
    if (x->getElse())
      cc::IfOp::create(builder, loc, TypeRange{}, condition,
                       stmtBuilder(x->getThen()), stmtBuilder(x->getElse()));
    else
      cc::IfOp::create(builder, loc, TypeRange{}, condition,
                       stmtBuilder(x->getThen()));
  }
  return finish(result);
}

QuakeBridgeVisitor::Result QuakeBridgeVisitor::visit(clang::ForStmt *x) {
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
    auto val = traverseValue(static_cast<clang::Stmt *>(cond));
    if (!val) {
      result = false;
      return;
    }
    cc::ConditionOp::create(builder, loc, *val, ValueRange{});
  };
  auto *body = x->getBody();
  auto bodyBuilder = [&](OpBuilder &builder, Location loc, Region &region) {
    if (!result)
      return;
    region.push_back(new Block());
    auto &bodyBlock = region.front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    LoopArgsScope loopArgsScope(*this, ValueRange{});
    if (!traverseStmt(static_cast<clang::Stmt *>(body))) {
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
    if (!traverseStmt(static_cast<clang::Stmt *>(incr)))
      result = false;
  };

  constexpr bool postCondition = false;
  LLVM_DEBUG(llvm::dbgs() << "%% "; x->dump());
  if (auto *init = x->getInit()) {
    SymbolTableScope var_scope(symbolTable);
    cc::ScopeOp::create(builder, loc, [&](OpBuilder &builder, Location loc) {
      if (!traverseStmt(static_cast<clang::Stmt *>(init))) {
        result = false;
        return;
      }
      cc::LoopOp::create(builder, loc, ValueRange{}, postCondition,
                         whileBuilder, bodyBuilder, stepBuilder);
      cc::ContinueOp::create(builder, loc);
    });
  } else {
    // If there is no initialization expression, skip creating a `for` scope.
    // The step builder is still needed regardless of whether there's an init
    // clause -- an empty init clause says nothing about whether an increment
    // clause exists (e.g. `for (; i < 4; ++i)`).
    cc::LoopOp::create(builder, loc, ValueRange{}, postCondition, whileBuilder,
                       bodyBuilder, stepBuilder);
  }
  return finish(result);
}

} // namespace cudaq::detail
