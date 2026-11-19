/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CodeGenConfig.h"
#include "cudaq/Verifier/QIRSpec.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

namespace {
std::unique_ptr<llvm::Module> parse(llvm::LLVMContext &ctx,
                                    const std::string &text) {
  llvm::SMDiagnostic err;
  auto module = llvm::parseAssemblyString(text, err, ctx);
  EXPECT_TRUE(module) << err.getMessage().str();
  return module;
}

cudaq::verifier::LLVMVerifierOptions options(bool base, bool adaptive,
                                             bool irProfile) {
  return {.isBaseProfile = base,
          .isAdaptiveProfile = adaptive,
          .allowAllInstructions = false,
          .irProfile = irProfile,
          .integerComputations = false,
          .floatComputations = false};
}

// The entry point calls a function defined in the module. That function keeps
// its data in a stack slot, which no profile allows in a function body.
const char *callsDefinedFunction = R"(
declare void @__quantum__qis__h__body(ptr)

define internal i64 @helper(i64 %n) {
  %slot = alloca [4 x i64]
  store i64 %n, ptr %slot
  %v = load i64, ptr %slot
  ret i64 %v
}

define void @entry() #0 {
  %r = call i64 @helper(i64 3)
  call void @__quantum__qis__h__body(ptr null)
  ret void
}

attributes #0 = { "entry_point" }
)";

// The entry point itself has a stack slot.
const char *entryHasStackSlot = R"(
define void @entry() #0 {
  %slot = alloca i64
  ret void
}

attributes #0 = { "entry_point" }
)";

// A call argument that is a constant expression other than a GEP, an `inttoptr`
// or a bitcast.
const char *callWithUnusualArgument = R"(
@g = global i64 0
declare void @consume(i64)

define void @entry() #0 {
  call void @consume(i64 ptrtoint (ptr @g to i64))
  ret void
}

attributes #0 = { "entry_point" }
)";
} // namespace

TEST(QIRSpecVerifier, definedFunctionIsRejectedWithoutIrProfile) {
  llvm::LLVMContext ctx;
  auto module = parse(ctx, callsDefinedFunction);
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::failed(cudaq::verifier::verifyLLVMInstructions(
      module.get(), options(false, true, false))));
}

TEST(QIRSpecVerifier, definedFunctionIsAcceptedWithIrProfile) {
  llvm::LLVMContext ctx;
  auto module = parse(ctx, callsDefinedFunction);
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::succeeded(cudaq::verifier::verifyLLVMInstructions(
      module.get(), options(false, true, true))));
}

// `ir_profile` is a property of the adaptive profile only.
TEST(QIRSpecVerifier, baseProfileIgnoresIrProfile) {
  llvm::LLVMContext ctx;
  auto module = parse(ctx, callsDefinedFunction);
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::failed(cudaq::verifier::verifyLLVMInstructions(
      module.get(), options(true, false, true))));
}

// `ir_profile` frees the functions that are called and the calls. The body of
// the entry point is still held to the profile.
TEST(QIRSpecVerifier, entryPointBodyIsStillChecked) {
  llvm::LLVMContext ctx;
  auto module = parse(ctx, entryHasStackSlot);
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::failed(cudaq::verifier::verifyLLVMInstructions(
      module.get(), options(false, true, true))));
}

TEST(QIRSpecVerifier, unusualCallArgumentNeedsIrProfile) {
  llvm::LLVMContext ctx;
  auto module = parse(ctx, callWithUnusualArgument);
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::failed(cudaq::verifier::verifyLLVMInstructions(
      module.get(), options(false, true, false))));
  EXPECT_TRUE(mlir::succeeded(cudaq::verifier::verifyLLVMInstructions(
      module.get(), options(false, true, true))));
}

TEST(QIRSpecVerifier, irProfileIsParsedAsAnAdaptiveProperty) {
  auto config = cudaq::parseCodeGenTranslation(
      "qir-adaptive:1.0:int_computations,ir_profile");
  EXPECT_TRUE(config.isAdaptiveProfile);
  EXPECT_TRUE(config.irProfile);
  EXPECT_TRUE(config.integerComputations);

  auto without = cudaq::parseCodeGenTranslation("qir-adaptive:1.0");
  EXPECT_FALSE(without.irProfile);

  // Only the adaptive profile has properties.
  EXPECT_THROW(cudaq::parseCodeGenTranslation("qir-base:1.0:ir_profile"),
               std::runtime_error);
}
