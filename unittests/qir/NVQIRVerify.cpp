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

static void doCommonSetup(StringRef theQuake, bool expectSuccess = false) {
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
  ASSERT_TRUE(module);
  EXPECT_EQ(succeeded(cudaq::verifier::checkNvqirCalls(module.get())),
            expectSuccess);
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

TEST(NVQIRVerify, controlValueCalls) {
  StringRef theQuake = R"#(
    llvm.func @generalizedInvokeWithRotationsControlsTargets(i64, i64, i64, i1, !llvm.ptr, ...)
    llvm.func @invokeCustomUnitaryWithControlValues(i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, ...)
    llvm.func @invokeExpPauliWithControlValues(i64, f64, !llvm.ptr, !llvm.ptr, ...)
    llvm.func @entryPoint(%callback: !llvm.ptr, %matrix: !llvm.ptr,
                         %targets: !llvm.ptr, %name: !llvm.ptr, %theta: f64,
                         %word: !llvm.ptr) {
      %zero = llvm.mlir.constant(0 : i64) : i64
      %one = llvm.mlir.constant(1 : i64) : i64
      %with_values = llvm.mlir.constant(true) : i1
      llvm.call @generalizedInvokeWithRotationsControlsTargets(%zero, %zero, %one, %with_values, %callback, %targets)
          vararg(!llvm.func<void (i64, i64, i64, i1, ptr, ...)>)
          : (i64, i64, i64, i1, !llvm.ptr, !llvm.ptr) -> ()
      llvm.call @invokeCustomUnitaryWithControlValues(%zero, %matrix, %targets, %name, %callback)
          vararg(!llvm.func<void (i64, ptr, ptr, ptr, ptr, ...)>)
          : (i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
      llvm.call @invokeExpPauliWithControlValues(%zero, %theta, %targets, %word)
          vararg(!llvm.func<void (i64, f64, ptr, ptr, ...)>)
          : (i64, f64, !llvm.ptr, !llvm.ptr) -> ()
      llvm.return
    }
    )#";
  doCommonSetup(theQuake, /*expectSuccess=*/true);
}
