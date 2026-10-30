/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef QLX_CONVERSION_PHYSTOQUAKE_PASSES
#define QLX_CONVERSION_PHYSTOQUAKE_PASSES

include "mlir/Pass/PassBase.td"

def PhysToQuake : Pass<"phys-to-quake", "mlir::ModuleOp"> {
  let summary = "Lower a P3 physical graph to a Quake entry point";
  let description = [{
    Translates the `phys.graph` named by `entry` into one `func.func` with the
    `cudaq-entrypoint` attribute, in value-semantics Quake. The function is
    named `__nvqpp__mlirgen__<entry>`, as the CUDA-Q front-ends name kernels.
    The module then holds only that function and the intrinsics it calls.
    The pass removes the `qlx.*` module attributes, keeps the others, and
    adds `quake.noOptimization`. Other graphs in the module are erased
    without being translated.

    Each P3 operation becomes the Quake below, in program order:
      - `phys.acquire` and `phys.release` of a carrier become
        `quake.null_wire` and `quake.sink` at the same program point.
      - `phys.prepare` in state `zero`, `0`, or `loaded`, and `phys.reset` to
        `zero`, become `quake.reset`. The states `plus` and `+` add `h`.
      - `phys.apply` becomes the Quake gate that its action's builtin
        `process` names, once per lane.
      - `phys.measure` becomes `quake.mz`. When the instrument's builtin
        `process` is `measure_x`, `h` comes before the measurement, and
        after it when the state survives. A destructive measurement ends
        with `quake.sink`.
      - `phys.measure_product` collects the parity of its Pauli product on a
        fresh ancilla: each operand is changed to Z around its CX into the
        ancilla, and the ancilla is measured and freed. With `invert`, the
        bit is complemented where it is read.
      - `phys.rotate_product` is `exp(-i angle P / 2)`: basis changes and a
        CX ladder around `rz`, at every angle.
      - `phys.call` is inlined. `phys.call_template` replays the body of the
        call that `template_event` names, on the carriers that
        `state_aliases` maps the body's resources to. A template with
        `state_boundary_elided` advances its carriers without new SSA
        states, so the pass tracks each carrier's wire by its resource.
      - `cflow.if` becomes `cc.if`, and `cflow.repeat` and `cflow.while`
        become `cc.loop`. Each carries every live carrier's wire, then its
        classical values. A record crosses a region boundary as its `i1`.
        A `cflow.while` with `max_iterations` is rejected, because Logical
        does not define what happens at the bound.
      - Actions and instruments resolve in the graph's machine first, then
        in the module. The Phys graph verifier resolves actions the same way.
      - `phys.condition`, `phys.xor`, and `phys.all_false` become `i1` logic.
        A measurement stays a `!cc.measure_handle` until a Boolean is needed.
      - `phys.delay`, `phys.barrier`, `phys.move`, and `event.fence` emit
        nothing and forward their states.
      - `arith` operations are cloned. A record operand becomes its `i1`, so
        an `arith.select` between records selects their bits.

    The function's signature follows the graph's:
      - An input of type `index`, `f32`, `f64`, or a signless integer of
        width 1, 8, 16, 32, or 64 becomes a parameter of the same type,
        except that an `index` input becomes an `i64`.
      - A graph with no results gives a function with no results.
      - One Boolean result, a measurement record of any schema or an `i1`,
        gives an `i1` result.
      - Two or more Boolean results give one `!cc.sequence<i1>` result in
        `phys.return` order, in heap storage that the caller owns.

    Any other form fails with a diagnostic on its operation. That covers
    retry, post-selection, resource and factory flows, code epoch
    transitions, patch-granularity resources, parameterized or non-builtin
    processes, other input types, and non-Boolean results. It also covers a
    call template that replays the call enclosing it, or whose canonical body
    reads a value that is out of scope at the template.
  }];
  let options = [
    Option<"entry", "entry", "std::string", /*default=*/"\"\"",
           "Symbol of the phys.graph to translate">
  ];
  let dependentDialects = [
    "cudaq::cc::CCDialect",
    "cudaq::quake::QuakeDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // QLX_CONVERSION_PHYSTOQUAKE_PASSES
