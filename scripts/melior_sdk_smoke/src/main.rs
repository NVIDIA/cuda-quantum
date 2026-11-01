// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

//! Standalone smoke test of the CUDA-Q SDK's upstream MLIR providers through
//! an unmodified, pinned Melior. Exercises the register-all APIs, an
//! arith/func round-trip, a standard pass, an external Rust pass, and a small
//! execution-engine call. Quake/choice registration is Onsen's concern; this
//! crate proves the SDK alone is Melior-consumable.

use melior::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    ir::{Module, OperationRef, r#type::TypeId},
    pass::{ExternalPass, PassManager, create_external, transform},
    utility::{register_all_dialects, register_all_llvm_translations, register_all_passes},
};
use std::cell::Cell;

const ARITH_FUNC_MODULE: &str = r"module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}";

const LLVM_MODULE: &str = r"module {
  llvm.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = llvm.add %arg0, %arg1 : i32
    llvm.return %0 : i32
  }
}";

#[repr(align(8))]
struct SmokePassId;

static SMOKE_PASS_ID: SmokePassId = SmokePassId;

fn main() {
    // Register-all APIs: dialects, passes, and LLVM translations.
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    register_all_passes();
    register_all_llvm_translations(&context);
    println!("ok: register-all APIs");

    // arith/func round-trip: parse, print, reparse, compare.
    let module = Module::parse(&context, ARITH_FUNC_MODULE).expect("parse arith/func module");
    let printed = module.as_operation().to_string();
    let reparsed = Module::parse(&context, &printed).expect("reparse printed module");
    assert_eq!(printed, reparsed.as_operation().to_string());
    println!("ok: arith/func round-trip");

    // A standard pass: the canonicalizer over the module.
    let mut canon_module = Module::parse(&context, ARITH_FUNC_MODULE).expect("parse for passes");
    let manager = PassManager::new(&context);
    manager.add_pass(transform::create_canonicalizer());
    manager.run(&mut canon_module).expect("canonicalizer run");
    println!("ok: standard pass (canonicalizer)");

    // An external Rust pass created through the upstream C API.
    let ran = Cell::new(false);
    let external = create_external(
        |operation: OperationRef, _pass: ExternalPass| {
            assert!(operation.to_string().contains("func.func"));
            ran.set(true);
        },
        TypeId::create(&SMOKE_PASS_ID),
        "sdk-smoke-pass",
        "sdk-smoke-pass",
        "CUDA-Q SDK smoke external pass",
        "",
        &[],
    );
    let external_manager = PassManager::new(&context);
    external_manager.add_pass(external);
    let mut external_module = Module::parse(&context, ARITH_FUNC_MODULE).expect("parse for external pass");
    external_manager.run(&mut external_module).expect("external pass run");
    assert!(ran.get(), "external pass did not run");
    println!("ok: external Rust pass");

    // A small execution-engine call: JIT and invoke add(2, 3).
    let engine_module = Module::parse(&context, LLVM_MODULE).expect("parse llvm module");
    let engine = ExecutionEngine::new(&engine_module, 0, &[], false, false);
    let symbol = engine.lookup("add");
    assert!(!symbol.is_null(), "execution engine lost symbol `add`");
    let add: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(symbol) };
    assert_eq!(add(2, 3), 5);
    println!("ok: execution-engine call (add(2, 3) == 5)");

    println!("PASS: CUDA-Q SDK Melior smoke test");
}
