// ========================================================================== //
// Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                 //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: cudaq-opt --verify-nvqir-call-ops %s -split-input-file -verify-diagnostics 

// -----

llvm.func @indirectCallFunc() -> i32
// expected-error @+1 {{function entryPoint not compatible with NVQIR}}
llvm.func @entryPoint() {
  %0 = llvm.mlir.addressof @indirectCallFunc : !llvm.ptr<func<i32()>>
  // expected-error @+1 {{op unexpected indirect call in NVQIR}}
  %1 = llvm.call %0() : () -> i32
  llvm.return
}

// -----

llvm.func @directUndefCallFunc() -> i32
// expected-error @+1 {{function entryPoint not compatible with NVQIR}}
llvm.func @entryPoint() {
  // expected-error @+1 {{op unexpected function call in NVQIR: directUndefCallFunc}}
  %1 = llvm.call @directUndefCallFunc() : () -> i32
  llvm.return
}

// -----

// expected-error @+1 {{function entryPoint not compatible with NVQIR}}
llvm.func @entryPoint() {
  // expected-error @+1 {{op unexpected op in NVQIR}}
  llvm.inline_asm "asm_string", "constraints" : () -> i32
  llvm.return
}
