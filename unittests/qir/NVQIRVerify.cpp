/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Verifier/NVQIRCalls.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>

using namespace mlir;

static void doCommonSetup(StringRef theQuake) {
  DialectRegistry registry;
  registry.insert<LLVM::LLVMDialect>();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  llvm::SourceMgr sourceMgr;
  auto memBuf = llvm::MemoryBuffer::getMemBuffer(
      theQuake, /*BufferName=*/"test.qke", /*RequiresNullTerminator=*/false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuf), llvm::SMLoc());
  SourceMgrDiagnosticVerifierHandler verifierHandler(sourceMgr, &ctx);
  ParserConfig config(&ctx);
  auto module = parseSourceFile<ModuleOp>(sourceMgr, config);
  EXPECT_TRUE(failed(cudaq::verifier::checkNvqirCalls(module.get())));
  EXPECT_TRUE(succeeded(verifierHandler.verify()));
}

TEST(NVQIRVerify, check1) {
  StringRef theQuake = R"#(
    llvm.func @indirectCallFunc() -> i32
    llvm.func @entryPoint() {
      %0 = llvm.mlir.addressof @indirectCallFunc : !llvm.ptr
      // expected-error @+2 {{unexpected indirect call in NVQIR}}
      // expected-note @+1 {{}}
      %1 = llvm.call %0() : !llvm.ptr, () -> i32
      llvm.return
    }
    )#";
  doCommonSetup(theQuake);
}

TEST(NVQIRVerify, check2) {
  StringRef theQuake = R"#(
    llvm.func @directUndefCallFunc() -> i32
    llvm.func @entryPoint() {
      // expected-error @+2 {{unexpected function call in NVQIR: directUndefCallFunc}}
      // expected-note @+1 {{}}
      %1 = llvm.call @directUndefCallFunc() : () -> i32
      llvm.return
    }
    )#";
  doCommonSetup(theQuake);
}

TEST(NVQIRVerify, check3) {
  StringRef theQuake = R"#(
    llvm.func @entryPoint() {
      // expected-error @+2 {{unexpected op in NVQIR}}
      // expected-note @+1 {{}}
      llvm.inline_asm "asm_string", "constraints" : () -> i32
      llvm.return
    }
    )#";
  doCommonSetup(theQuake);
}
