// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --split-input-file --phys-to-quake=entry=entry \
// RUN:   --verify-diagnostics %s

// Each case is valid P3 with one form that the pass rejects.

phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@op], native_instruments = []}
}
// expected-error@+1 {{'phys.action' op has a process that is not valid JSON}}
phys.action @op {arity = 1 : i64, process = "h"}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  %1 = phys.apply @op(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
  phys.release %1 {event_id = "e2"} : !phys.state<@a>
  phys.return
}

// -----

phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@op], native_instruments = []}
}
// expected-error@+1 {{'phys.action' op has a process that is not a builtin process, which has no exact lowering to Quake}}
phys.action @op {arity = 1 : i64, process = "{\"kind\":\"unitary\",\"matrix\":[[0,1],[1,0]]}"}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  %1 = phys.apply @op(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
  phys.release %1 {event_id = "e2"} : !phys.state<@a>
  phys.return
}

// -----

phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@op], native_instruments = []}
}
// expected-error@+1 {{'phys.action' op has a parameterized process, which has no lowering to Quake}}
phys.action @op {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"h\",\"parameters\":{\"angle\":0.5}}"}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  %1 = phys.apply @op(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
  phys.release %1 {event_id = "e2"} : !phys.state<@a>
  phys.return
}

// -----

// The pass lowers a builtin process by name, and `rz` names no fixed gate.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@rz], native_instruments = []}
}
phys.action @rz {arity = 1 : i64, parameters = ["angle"], process = "{\"kind\":\"builtin\",\"name\":\"rz\",\"parameters\":{}}"}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.apply' op applies builtin process 'rz' to 1 states, which has no lowering to Quake}}
  %1 = phys.apply @rz(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
  phys.release %1 {event_id = "e2"} : !phys.state<@a>
  phys.return
}

// -----

// Reservations carry values that the process does not state.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [@x], native_instruments = []}
}
phys.action @x {arity = 1 : i64, process = "{\"kind\":\"builtin\",\"name\":\"x\",\"parameters\":{}}"}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.apply' op carries reservations ["angle=0.3"], which have no lowering to Quake}}
  %1 = phys.apply @x(%0) {event_id = "e1", reservations = ["angle=0.3"]} : (!phys.state<@a>) -> !phys.state<@a>
  phys.release %1 {event_id = "e2"} : !phys.state<@a>
  phys.return
}

// -----

// An unresolved action or instrument verifies in a partially linked module.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.apply' op names action @h, which resolves to no phys.action}}
  %1 = phys.apply @h(%0) {event_id = "e1"} : (!phys.state<@a>) -> !phys.state<@a>
  phys.release %1 {event_id = "e2"} : !phys.state<@a>
  phys.return
}

// -----

phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.measure' op names instrument @mz, which resolves to no phys.instrument}}
  %record = phys.measure @mz(%0) {destructive, event_id = "e1", record_id = "r"} : (!phys.state<@a>) -> !phys.record<@bit>
  phys.return
}

// -----

// The Phys verifier does not check an instrument declared in the machine.
phys.machine @m {
  phys.instrument @probe {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"h\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = [@probe]}
}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.measure' op measures with builtin process 'h', which has no lowering to Quake}}
  %record = phys.measure @probe(%0) {destructive, event_id = "e1", record_id = "r"} : (!phys.state<@a>) -> !phys.record<@bit>
  phys.return
}

// -----

// One state of a patch-granularity resource stands for a whole code patch.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, footprint_evidence = "one rotated surface-code patch", granularity = "patch", kind = "surface_code_patch", native_actions = [], native_instruments = [], physical_unit_kind = "qubit", physical_units = 17 : i64}
}
phys.resource @p {architecture = @m, index = 0 : i64, kind = "surface_code_patch", resource_class = @q}
phys.graph @entry on @m : () -> () {
  // expected-error@+1 {{'phys.acquire' op acquires resource @p of patch granularity, which `phys-to-quake` does not support}}
  %0 = phys.acquire [@p] {event_id = "e0"} : !phys.state<@p>
  phys.release %0 {event_id = "e1"} : !phys.state<@p>
  phys.return
}

// -----

// The verifier accepts any prepared state name.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
phys.resource @a {architecture = @m, index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.prepare' op prepares state 'minus', which has no lowering to Quake}}
  %1 = phys.prepare %0 {event_id = "e1", state = "minus"} : (!phys.state<@a>) -> !phys.state<@a>
  phys.release %1 {event_id = "e2"} : !phys.state<@a>
  phys.return
}

// -----

// An input type that a CUDA-Q kernel does not take as an argument, such as
// i128, is rejected.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
// expected-error@+1 {{'phys.graph' op takes input 0 of type 'i128', but `phys-to-quake` translates only inputs of type index, f32, f64, and signless i1, i8, i16, i32, and i64}}
phys.graph @entry on @m : (i128) -> () {
^bb0(%arg0: i128):
  phys.return
}

// -----

phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
phys.graph @entry on @m : (i1, index) -> (i1, index) {
^bb0(%arg0: i1, %arg1: index):
  // expected-error@+1 {{'phys.return' op returns result 1 of type 'index', but `phys-to-quake` translates only Boolean results: measurement records and i1 values}}
  phys.return %arg0, %arg1 : i1, index
}

// -----

// A selection sidecar post-selects the graph's shots. The graph and sidecar
// follow test/Model/p3-qec-sidecars.mlir.
module attributes {qlx.profiles = ["p3"]} {
  fabric.code @source_code {distance = 1 : i64, n = 1 : i64, k = 1 : i64, r = 0 : i64, partitions = {data = 1 : i64, sx = 0 : i64, sz = 0 : i64}}
  fabric.gadget @source(%patch: !fabric.patch<@source_code>) -> (!fabric.patch<@source_code>, i1) {
    %next, %record = fabric.measure_product %patch {logical_indices = array<i64: 0>, patch_indices = array<i64: 0>, pauli_product = "Z", record = "final"} : (!fabric.patch<@source_code>) -> (!fabric.patch<@source_code>, i1)
    fabric.return %next, %record : !fabric.patch<@source_code>, i1
  }
  fabric.gadget_profile @analysis for @source {
    fabric.success {records = ["source.final.outcome"]}
  }
  phys.machine @arch {
    phys.resource_class @qubits {count = 1 : i64, kind = "qubit", native_actions = ["measure_z"]}
  }
  phys.resource @q0 {index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @entry on @arch : () -> !phys.state<@q0> attributes {source_protocol = @source} {
    %0 = phys.acquire [@q0] {event_id = "acquire0"} : !phys.state<@q0>
    %1:2 = phys.measure @measure_z(%0) {event_id = "m0", record_id = "readout.final.data0"} : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %1#0 : !phys.state<@q0>
  }
  phys.record_projection @projection for @entry from @source {
    entries = [{instance = "source", physical_record = "readout.final.data0", source_record = "source.final.outcome"}]
  }
  // expected-error@+1 {{'phys.selection_sidecar' op post-selects shots of the entry graph, which `phys-to-quake` does not support}}
  phys.selection_sidecar @success for @entry {constant = false, expected = false, input_syndromes = [], projection_indices = array<i64: 0>, record_projection = @projection, records = ["readout.final.data0"], source_kind = "profile", source_instance = "source", source_profile = @analysis, source_records = ["source.final.outcome"], source_row = 0 : i64}
}

// -----

// In-graph post-selection, as the native projector emits it.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
phys.graph @entry on @m : (i1) -> () {
^bb0(%arg0: i1):
  // expected-error@+1 {{'event.selection' op post-selects shots, which `phys-to-quake` does not support}}
  event.selection %arg0 {accept_when = false, mode = "abort_on"} : i1
  phys.return
}

// -----

phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
phys.graph @entry on @m : () -> () {
  // expected-error@+1 {{'phys.produce_resource' op is part of a resource or factory flow, which `phys-to-quake` does not support}}
  %0 = phys.produce_resource @t_state at @factory {protocol = #fabric.spec_only<"distill-15to1">, event_id = "e0"} : !phys.resource_payload<@t_state>
  phys.discard_resource_payload %0 {event_id = "e1"} : !phys.resource_payload<@t_state>
  phys.return
}

// -----

// The transition follows test/Model/p3-epoch-transition.mlir.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = []}
}
phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> () {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.epoch_transition' op changes the code epoch, and `phys-to-quake` cannot check that the operations before it realize the declared logical action}}
  %1 = phys.epoch_transition @even to @odd(%0) {event_id = "e1", evidence = "verified", logical_map = {q0 = "q0"}} : (!phys.state<@a>) -> !phys.state<@a>
  phys.release %1 {event_id = "e2"} : !phys.state<@a>
  phys.return
}

// -----

// Retry replays an attempt until it succeeds. The attempt's call lowers, and
// the retry stops the pass. It follows test/Model/p3-structured-control.mlir.
module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget_spec @attempt_spec for @objective : () -> () {encodings = []}
  fabric.gadget @attempt() {
    fabric.return
  } {realization_boundary = {}, spec = @attempt_spec}
  fabric.gadget_profile @attempt_profile for @attempt {}
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = [@mz]}
  }
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
    %1:2 = "phys.call"(%0) <{callee = @attempt, event_id = "attempt0", instance = "attempt.0", profile = @attempt_profile}> ({
    ^bb0(%current: !phys.state<@a>):
      %o, %r = phys.measure @mz(%current) {event_id = "m0", record_id = "attempt_result"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
      phys.yield %o, %r : !phys.state<@a>, !phys.record<@bit>
    }) : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
    %2 = phys.condition %1#1 {event_id = "decision0"} : !phys.record<@bit> -> i1
    // expected-error@+1 {{'phys.retry' op has no lowering to Quake}}
    %3 = phys.retry %2 carries (%1#0) {attempt = @attempt, attempt_event = "attempt0", decision_event = "decision0", event_id = "retry0", max_attempts = 3 : i64, profile = @attempt_profile} : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %3 {event_id = "e1"} : !phys.state<@a>
    phys.return
  }
}

// -----

// A template after a region cannot replay a call whose body reads a value
// from inside that region: the value no longer exists.
module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget_spec @gate_spec for @objective : () -> () {encodings = []}
  fabric.gadget @gate() {
    fabric.return
  } {realization_boundary = {}, spec = @gate_spec}
  phys.machine @m {
    phys.resource_class @q {count = 2 : i64, kind = "qubit", native_actions = [], native_instruments = [@mz]}
  }
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.resource @b {index = 1 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : (i1) -> i1 {
  ^bb0(%c: i1):
    %0 = phys.acquire [@a] {event_id = "acquire_a"} : !phys.state<@a>
    %1 = phys.acquire [@b] {event_id = "acquire_b"} : !phys.state<@b>
    %2 = cflow.if %c event_id = "if" -> !phys.state<@a> {
      %3, %r = phys.measure @mz(%0) {event_id = "m", record_id = "m"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
      %4:2 = "phys.call"(%3) <{callee = @gate, event_id = "canon", instance = "gate.0"}> ({
      ^bb0(%s: !phys.state<@a>):
        %p = phys.condition %r {event_id = "p"} : !phys.record<@bit> -> i1
        phys.yield %s, %p : !phys.state<@a>, i1
      }) : (!phys.state<@a>) -> (!phys.state<@a>, i1)
      cflow.yield %4#0 : !phys.state<@a>
    } else {
      cflow.yield %0 : !phys.state<@a>
    }
    // expected-error@+1 {{'phys.call_template' op replays the phys.call 'canon', whose body reads a value that is out of scope here}}
    %5:2 = phys.call_template %1 {callee = @gate, event_id = "replay", instance = "gate.1", state_aliases = [{alias = @b, template = @a}], template_event = "canon"} : (!phys.state<@b>) -> (!phys.state<@b>, i1)
    phys.release %2, %5#0 {event_id = "release"} : !phys.state<@a>, !phys.state<@b>
    phys.return %5#1 : i1
  }
}

// -----

// A template inside its own canonical call would replay without end.
module attributes {qlx.profiles = ["p3"]} {
  fabric.gadget_spec @gate_spec for @objective : () -> () {encodings = []}
  fabric.gadget @gate() {
    fabric.return
  } {realization_boundary = {}, spec = @gate_spec}
  phys.machine @m {
    phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
  }
  phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
  phys.graph @entry on @m : () -> () {
    %0 = phys.acquire [@a] {event_id = "acquire_a"} : !phys.state<@a>
    %1 = "phys.call"(%0) <{callee = @gate, event_id = "canon", instance = "gate.0"}> ({
    ^bb0(%t: !phys.state<@a>):
      // expected-error@+1 {{'phys.call_template' op replays the phys.call 'canon' that encloses it, which an inlined replay cannot express}}
      phys.call_template {callee = @gate, event_id = "self", instance = "gate.1", state_boundary_elided, template_event = "canon"} : () -> ()
      phys.yield %t : !phys.state<@a>
    }) : (!phys.state<@a>) -> !phys.state<@a>
    phys.release %1 {event_id = "release"} : !phys.state<@a>
    phys.return
  }
}

// -----

// An unresolved product instrument verifies in a partially linked module.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = [@mpp]}
}
phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> !phys.record<@bit> {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.measure_product' op names instrument @mpp, which resolves to no phys.instrument}}
  %1:2 = phys.measure_product %0 {event_id = "e1", instrument = @mpp, paulis = ["Z"], record_id = "r"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
  phys.release %1#0 {event_id = "e2"} : !phys.state<@a>
  phys.return %1#1 : !phys.record<@bit>
}

// -----

// The verifier checks the instrument's kind, not its process.
phys.instrument @mpp {kind = "measure_product", variadic, record_schema = "bit", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}"}
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = [@mpp]}
}
phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : () -> !phys.record<@bit> {
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'phys.measure_product' op measures with builtin process 'measure_z', which has no lowering to Quake}}
  %1:2 = phys.measure_product %0 {event_id = "e1", instrument = @mpp, paulis = ["Z"], record_id = "r"} : (!phys.state<@a>) -> (!phys.state<@a>, !phys.record<@bit>)
  phys.release %1#0 {event_id = "e2"} : !phys.state<@a>
  phys.return %1#1 : !phys.record<@bit>
}

// -----

// Logical does not define what happens when a dynamic loop reaches
// max_iterations. The loop follows test/Model/p3-structured-control.mlir.
phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
phys.resource @a {index = 0 : i64, kind = "qubit", resource_class = @q}
phys.graph @entry on @m : (i1) -> () {
^bb0(%arg0: i1):
  %0 = phys.acquire [@a] {event_id = "e0"} : !phys.state<@a>
  // expected-error@+1 {{'cflow.while' op bounds the loop by max_iterations, and Logical does not define what happens at the bound}}
  %1:2 = "cflow.while"(%0, %arg0) <{event_id = "while0", max_iterations = 4 : i64}> ({
  ^bb0(%current: !phys.state<@a>, %predicate: i1):
    "cflow.while_condition"(%predicate, %current, %predicate) : (i1, !phys.state<@a>, i1) -> ()
  }, {
  ^bb0(%current: !phys.state<@a>, %predicate: i1):
    cflow.yield %current, %predicate : !phys.state<@a>, i1
  }) : (!phys.state<@a>, i1) -> (!phys.state<@a>, i1)
  phys.release %1#0 {event_id = "e1"} : !phys.state<@a>
  phys.return
}

// -----

phys.machine @m {
  phys.resource_class @q {count = 1 : i64, kind = "qubit", native_actions = [], native_instruments = []}
}
phys.graph @entry on @m : (i1) -> () {
^bb0(%arg0: i1):
  // expected-error@+1 {{'cf.assert' op has no lowering to Quake}}
  cf.assert %arg0, "unsupported"
  phys.return
}
