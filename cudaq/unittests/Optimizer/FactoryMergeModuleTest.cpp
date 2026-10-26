/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Factory.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;
using cudaq::opt::factory::mergeModules;

static mlir::OwningOpRef<mlir::ModuleOp> parse(MLIRContext &ctx,
                                               llvm::StringRef ir) {
  return mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
}

TEST(FactoryMergeModuleTest, CopiesMissingFunction) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
  MLIRContext ctx(registry);

  auto dst = parse(ctx, R"mlir(
        module {
            func.func @alreadyThere() { return }
        }
    )mlir");
  ASSERT_TRUE(dst);

  auto src = parse(ctx, R"mlir(
        module {
            func.func @newFunc() { return }
            func.func @alreadyThere() { return }
        }
    )mlir");
  ASSERT_TRUE(src);

  mergeModules(*dst, *src);

  auto newFunc = dst->lookupSymbol<mlir::func::FuncOp>("newFunc");
  EXPECT_TRUE(newFunc);

  int countAlreadyThere = 0;
  dst->walk([&](mlir::func::FuncOp f) {
    if (f.getSymName() == "alreadyThere")
      countAlreadyThere++;
  });
  EXPECT_EQ(countAlreadyThere, 1);
}

TEST(FactoryMergeModuleTest, RetainOriginalModuleSymbols) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
  MLIRContext ctx(registry);

  auto dst = parse(ctx, R"mlir(
        module attributes { test.attr = "keepme" } {
            func.func @a() { return }
        }
    )mlir");
  ASSERT_TRUE(dst);

  auto src = parse(ctx, R"mlir(
        module {
            func.func @b() { return }
        }
    )mlir");
  ASSERT_TRUE(src);

  mergeModules(*dst, *src);

  // Verify the destination attribute remains
  auto sattr =
      dst->getOperation()->getAttrOfType<mlir::StringAttr>("test.attr");
  ASSERT_TRUE(sattr);
  ASSERT_EQ(sattr.getValue(), "keepme");

  // Both symbols exists exactly once
  EXPECT_TRUE(dst->lookupSymbol<mlir::func::FuncOp>("a"));
  EXPECT_TRUE(dst->lookupSymbol<mlir::func::FuncOp>("b"));

  int countA = 0, countB = 0;
  dst->walk([&](mlir::func::FuncOp f) {
    if (f.getSymName() == "a")
      countA++;
    if (f.getSymName() == "b")
      countB++;
  });
  EXPECT_EQ(countA, 1);
  EXPECT_EQ(countB, 1);
}
