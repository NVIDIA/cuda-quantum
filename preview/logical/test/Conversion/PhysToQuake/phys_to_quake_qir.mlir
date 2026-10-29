// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake, cudaq-opt
// RUN: qlx-opt --split-input-file --phys-to-quake=entry=entry %s | \
// RUN:   %cudaq_opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(codegen-for-qir{convert-to=qir})' | \
// RUN:   FileCheck --check-prefix=QIS %s
// RUN: qlx-opt --split-input-file --phys-to-quake=entry=entry %s | \
// RUN:   %cudaq_opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(aot-prep-pipeline,codegen-for-qir{convert-to=qir-adaptive})' | \
// RUN:   FileCheck --check-prefix=PROFILE %s

// The final QIS calls keep P3's allocation points and measurement basis.
// @a is allocated at its acquire and released by its destructive
// measurement, and only then is @b allocated. The reset to plus is
// `reset; h`, and the retained X measurement is `h; mz; h`.
//
// The returned list survives QIR profile output. The `.run` function that
// `aot-prep-pipeline` generates records the list as an array of three
// Booleans, then each element in return order. Profile output records a list
// only when its length folds to a constant, and drops it without a
// diagnostic otherwise. `aot-prep-pipeline` generates `.run` only for a
// module with a `quake.mangled_name_map`, so the input carries one, and the
// pass keeps it.
module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__entry = "entry_host"}} {
  phys.machine @machine {
    phys.resource_class @qubits {count = 2 : i64, kind = "qubit", native_actions = [@x], native_instruments = [@mz, @mx]}
  }
  phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.instrument @mx {arity = 1 : i64, kind = "measure_x", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_x\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource @a {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @b {architecture = @machine, index = 1 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @entry on @machine : () -> (!phys.record<@bit>, !phys.record<@bit>, !phys.record<@bit>) {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    %1 = phys.reset %0 {event_id = "e1", state = "plus"} : (!phys.state<@a>) -> !phys.state<@a>
    %2, %kept = phys.measure @mx(%1) {event_id = "e2", record_id = "kept"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    %last = phys.measure @mx(%2) {destructive, event_id = "e3", record_id = "last"} : (!phys.state<@a>) -> !phys.record<@bit>
    %3 = phys.acquire [@b] {event_id = "e4"} : !phys.state<@b>
    %4 = phys.apply @x(%3) {event_id = "e5"} : (!phys.state<@b>) -> !phys.state<@b>
    %flip = phys.measure @mz(%4) {destructive, event_id = "e6", record_id = "flip"} : (!phys.state<@b>) -> !phys.record<@bit>
    phys.return %kept, %last, %flip : !phys.record<@bit>, !phys.record<@bit>, !phys.record<@bit>
  }
}

// QIS-LABEL: llvm.func @__nvqpp__mlirgen__entry()
// QIS:         %[[A:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    llvm.call @__quantum__qis__reset(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[A]], %{{.*}}) {registerName = "r00000"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[A]], %{{.*}}) {registerName = "r00001"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    %[[B:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    llvm.call @__quantum__qis__x(%[[B]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[B]], %{{.*}}) {registerName = "r00002"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[B]]) : (!llvm.ptr) -> ()

// PROFILE-LABEL: llvm.func @__nvqpp__mlirgen__entry.run()
// PROFILE:         %[[LENGTH:.*]] = llvm.mlir.constant(3 : i64) : i64
// PROFILE-NOT:     record_output
// PROFILE:         llvm.mlir.addressof @[[ARRAY:cstr\.[0-9A-F]+]] : !llvm.ptr
// PROFILE:         llvm.call @__quantum__rt__array_record_output(%[[LENGTH]], %{{.*}}) : (i64, !llvm.ptr) -> ()
// PROFILE:         llvm.getelementptr %{{.*}}[0] : (!llvm.ptr) -> !llvm.ptr, i1
// PROFILE:         llvm.mlir.addressof @[[ELEMENT0:cstr\.[0-9A-F]+]] : !llvm.ptr
// PROFILE:         llvm.call @__quantum__rt__bool_record_output(%{{.*}}, %{{.*}}) : (i1, !llvm.ptr) -> ()
// PROFILE:         llvm.getelementptr %{{.*}}[1] : (!llvm.ptr) -> !llvm.ptr, i1
// PROFILE:         llvm.mlir.addressof @[[ELEMENT1:cstr\.[0-9A-F]+]] : !llvm.ptr
// PROFILE:         llvm.call @__quantum__rt__bool_record_output(%{{.*}}, %{{.*}}) : (i1, !llvm.ptr) -> ()
// PROFILE:         llvm.getelementptr %{{.*}}[2] : (!llvm.ptr) -> !llvm.ptr, i1
// PROFILE:         llvm.mlir.addressof @[[ELEMENT2:cstr\.[0-9A-F]+]] : !llvm.ptr
// PROFILE:         llvm.call @__quantum__rt__bool_record_output(%{{.*}}, %{{.*}}) : (i1, !llvm.ptr) -> ()
// PROFILE-NOT:     record_output
// PROFILE:         llvm.return
// PROFILE-LABEL: llvm.func @__nvqpp__mlirgen__entry.run.entry()
// PROFILE-DAG:     llvm.mlir.global private constant @[[ARRAY]]("array<i1 x 3>\00")
// PROFILE-DAG:     llvm.mlir.global private constant @[[ELEMENT0]]("[0]\00")
// PROFILE-DAG:     llvm.mlir.global private constant @[[ELEMENT1]]("[1]\00")
// PROFILE-DAG:     llvm.mlir.global private constant @[[ELEMENT2]]("[2]\00")

// -----

// Each product measurement allocates its ancilla, changes each operand to Z
// around the operand's CX into the ancilla, then measures and releases the
// ancilla. The X@X parity is complete before Z@Y begins.
phys.instrument @mpp {kind = "measure_product", variadic, record_schema = "bit", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_product\",\"parameters\":{}}"}
phys.machine @machine {
  phys.resource_class @qubits {count = 2 : i64, kind = "qubit", native_actions = [], native_instruments = [@mpp]}
}
phys.resource @a {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
phys.resource @b {architecture = @machine, index = 1 : i64, kind = "qubit", resource_class = @qubits}
phys.graph @entry on @machine : () -> (!phys.record<@bit>, !phys.record<@bit>) {
  %0:2 = phys.acquire [@a, @b] {event_id = "e0"} : !phys.state<@a>, !phys.state<@b>
  %1:3 = phys.measure_product %0#0, %0#1 {event_id = "e1", instrument = @mpp, paulis = ["X", "X"], record_id = "xx"} : (!phys.state<@a>, !phys.state<@b>) -> (!phys.state<@a>, !phys.state<@b>, !phys.record<@bit>)
  %2:3 = phys.measure_product %1#0, %1#1 {event_id = "e2", instrument = @mpp, paulis = ["Z", "Y"], record_id = "zy"} : (!phys.state<@a>, !phys.state<@b>) -> (!phys.state<@a>, !phys.state<@b>, !phys.record<@bit>)
  phys.release %2#0, %2#1 {event_id = "e3"} : !phys.state<@a>, !phys.state<@b>
  phys.return %1#2, %2#2 : !phys.record<@bit>, !phys.record<@bit>
}

// QIS-LABEL: llvm.func @__nvqpp__mlirgen__entry()
// QIS:         %[[A:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    %[[B:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    %[[XX:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    %[[C0:.*]] = llvm.bitcast %[[A]] : !llvm.ptr to !llvm.ptr
// QIS:         %[[T0:.*]] = llvm.bitcast %[[XX]] : !llvm.ptr to !llvm.ptr
// QIS-NEXT:    llvm.call @generalizedInvokeWithRotationsControlsTargets({{.*}}, %[[C0]], %[[T0]])
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[B]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    %[[C1:.*]] = llvm.bitcast %[[B]] : !llvm.ptr to !llvm.ptr
// QIS:         %[[T1:.*]] = llvm.bitcast %[[XX]] : !llvm.ptr to !llvm.ptr
// QIS-NEXT:    llvm.call @generalizedInvokeWithRotationsControlsTargets({{.*}}, %[[C1]], %[[T1]])
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[B]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[XX]], %{{.*}}) {registerName = "r00000"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[XX]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    %[[ZY:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    %[[C2:.*]] = llvm.bitcast %[[A]] : !llvm.ptr to !llvm.ptr
// QIS:         %[[T2:.*]] = llvm.bitcast %[[ZY]] : !llvm.ptr to !llvm.ptr
// QIS-NEXT:    llvm.call @generalizedInvokeWithRotationsControlsTargets({{.*}}, %[[C2]], %[[T2]])
// QIS-NEXT:    llvm.call @__quantum__qis__s__adj(%[[B]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[B]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    %[[C3:.*]] = llvm.bitcast %[[B]] : !llvm.ptr to !llvm.ptr
// QIS:         %[[T3:.*]] = llvm.bitcast %[[ZY]] : !llvm.ptr to !llvm.ptr
// QIS-NEXT:    llvm.call @generalizedInvokeWithRotationsControlsTargets({{.*}}, %[[C3]], %[[T3]])
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[B]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__qis__s(%[[B]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[ZY]], %{{.*}}) {registerName = "r00001"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[ZY]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[A]]) : (!llvm.ptr) -> ()
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[B]]) : (!llvm.ptr) -> ()

// -----

// A dynamic loop keeps its back-edge: the body measures again and returns to
// the condition, which reads the new bit. The loop exits only on 0, so the
// returned record folds to `false`.
phys.machine @machine {
  phys.resource_class @qubits {count = 1 : i64, kind = "qubit", native_actions = [@h], native_instruments = [@mz]}
}
phys.action @h {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"h\",\"parameters\":{}}"}
phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
phys.resource @a {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
phys.graph @entry on @machine : () -> !phys.record<@bit> {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  %1 = phys.apply @h(%0) {event_id = "h0"} : (!phys.state<@a>) -> !phys.state<@a>
  %2, %first = phys.measure @mz(%1) {event_id = "m0", record_id = "m"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
  %3:2 = cflow.while(%2, %first) event_id = "until_zero" : (!phys.state<@a>, !phys.record<@bit>) -> (!phys.state<@a>, !phys.record<@bit>) {
  ^bb0(%s: !phys.state<@a>, %r: !phys.record<@bit>):
    %one = phys.condition %r {event_id = "is_one"} : !phys.record<@bit> -> i1
    cflow.while_condition %one, %s, %r : !phys.state<@a>, !phys.record<@bit>
  } do {
  ^bb0(%s: !phys.state<@a>, %r: !phys.record<@bit>):
    %4 = phys.apply @h(%s) {event_id = "h1"} : (!phys.state<@a>) -> !phys.state<@a>
    %5, %m = phys.measure @mz(%4) {event_id = "m1", record_id = "m"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    cflow.yield %5, %m : !phys.state<@a>, !phys.record<@bit>
  }
  phys.release %3#0 {event_id = "e1"} : !phys.state<@a>
  phys.return %3#1 : !phys.record<@bit>
}

// QIS-LABEL: llvm.func @__nvqpp__mlirgen__entry()
// QIS:         %[[A:.*]] = llvm.call @__quantum__rt__qubit_allocate() : () -> !llvm.ptr
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[A]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[A]], %{{.*}}) {registerName = "r00000"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS:         %[[FIRST:.*]] = llvm.call @__quantum__rt__read_result(%{{.*}}) : (!llvm.ptr) -> i1
// QIS-NEXT:    llvm.br ^[[COND:.*]](%[[A]], %[[FIRST]] : !llvm.ptr, i1)
// QIS:       ^[[COND]](%[[Q:.*]]: !llvm.ptr, %[[ONE:.*]]: i1):
// QIS-NEXT:    llvm.cond_br %[[ONE]], ^[[BODY:.*]], ^[[EXIT:[^ ]*]]
// QIS:       ^[[BODY]]:
// QIS-NEXT:    llvm.call @__quantum__qis__h(%[[Q]]) : (!llvm.ptr) -> ()
// QIS:         llvm.call @__quantum__qis__mz_handle__to__register(%[[Q]], %{{.*}}) {registerName = "r00001"} : (!llvm.ptr, !llvm.ptr) -> i64
// QIS:         %[[AGAIN:.*]] = llvm.call @__quantum__rt__read_result(%{{.*}}) : (!llvm.ptr) -> i1
// QIS-NEXT:    llvm.br ^[[COND]](%[[Q]], %[[AGAIN]] : !llvm.ptr, i1)
// QIS:       ^[[EXIT]]:
// QIS-NEXT:    llvm.call @__quantum__rt__qubit_release(%[[Q]]) : (!llvm.ptr) -> ()
