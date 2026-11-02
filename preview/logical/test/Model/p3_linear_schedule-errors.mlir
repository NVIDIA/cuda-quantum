// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{graph inputs cannot contain more than one physical-state owner for resource @q}}
  phys.graph @duplicate_inputs on @arch :
      (!phys.state<@q>, !phys.state<@q>) -> () {
  ^bb0(%first: !phys.state<@q>, %second: !phys.state<@q>):
    phys.release %first : !phys.state<@q>
    phys.release %second : !phys.state<@q>
    phys.return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "first"}
    event.fence {effects = ["event"], event_id = "second"}
    phys.return
  }
  // expected-error @+1 {{schedule event 'second' domain_deps must exactly match its graph clock frontier}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order",
    optimization_status = "not_applicable",
    entries = [
      "first|fence|0|0|control:first|deps=|data_deps=|resource_deps=|domain_deps=",
      "second|fence|0|0|control:second|deps=first|data_deps=|resource_deps=|domain_deps=first"
    ],
    makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.operating_point @point for @arch {timing = {cycle_ns = "invalid"}}
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () attributes {operating_point = @point} {
    %acquired = phys.acquire [@q] {event_id = "acquire"}
      : !phys.state<@q>
    %prepared = phys.prepare %acquired {
      event_id = "prepare", state = "zero"
    } : (!phys.state<@q>) -> !phys.state<@q>
    phys.release %prepared {event_id = "release"} : !phys.state<@q>
    phys.return
  }
  // expected-error @+1 {{graph operating-point timing fact 'cycle_ns' must be finite, nonnegative, and numeric}}
  phys.schedule @invalid_cycle for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=",
      "prepare|prepare|0|1|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=acquire",
      "release|release|1|0|qubits[0]|deps=prepare|data_deps=prepare|resource_deps=prepare"
    ],
    makespan_ns = 1.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = [@h]
    }
  }
  phys.operating_point @point for @arch {
    timing = {cycle_ns = 1.0 : f64, h_ns = -1.0 : f64}
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () attributes {operating_point = @point} {
    %acquired = phys.acquire [@q] {event_id = "acquire"}
      : !phys.state<@q>
    %applied = phys.apply @h(%acquired) {event_id = "h"}
      : (!phys.state<@q>) -> !phys.state<@q>
    phys.release %applied {event_id = "release"} : !phys.state<@q>
    phys.return
  }
  // expected-error @+1 {{graph operating-point timing fact 'h_ns' must be finite, nonnegative, and numeric}}
  phys.schedule @invalid_action for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=",
      "h|apply|0|1|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=acquire",
      "release|release|1|0|qubits[0]|deps=h|data_deps=h|resource_deps=h"
    ],
    makespan_ns = 1.0 : f64
  }
}

// -----

// A legal but shifted schedule is not the canonical stable greedy-ASAP result.
module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () {
    %acquired = phys.acquire [@q] {event_id = "acquire"}
      : !phys.state<@q>
    %delayed = phys.delay %acquired {
      duration_ns = 2.0 : f64, event_id = "idle"
    } : (!phys.state<@q>) -> !phys.state<@q>
    phys.release %delayed {event_id = "release"} : !phys.state<@q>
    phys.return
  }
  // expected-error @+1 {{schedule event 'acquire' must start at its canonical earliest legal time 0}}
  phys.schedule @shifted for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire|acquire|5|0|qubits[0]|deps=|data_deps=|resource_deps=",
      "idle|delay|5|2|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=acquire",
      "release|release|7|0|qubits[0]|deps=idle|data_deps=idle|resource_deps=idle"
    ],
    makespan_ns = 7.0 : f64
  }
}

// -----

// Structured durations are authenticated from the graph template and bound.
module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () {
    %acquired = phys.acquire [@q] {event_id = "acquire"}
      : !phys.state<@q>
    %repeated = "cflow.repeat"(%acquired) <{
      count = 3 : i64, event_id = "repeat"
    }> ({
    ^bb0(%current: !phys.state<@q>):
      %delayed = phys.delay %current {
        duration_ns = 2.0 : f64, event_id = "idle"
      } : (!phys.state<@q>) -> !phys.state<@q>
      cflow.yield %delayed : !phys.state<@q>
    }) : (!phys.state<@q>) -> !phys.state<@q>
    phys.release %repeated {event_id = "release"} : !phys.state<@q>
    phys.return
  }
  // expected-error @+1 {{structured schedule event 'repeat' duration must equal its canonical folded envelope duration 6}}
  phys.schedule @inflated for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=",
      "repeat|repeat|0|8|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=|repeat_count=3|repeat_period_ns=2|repeat_epilogue_ns=2",
      "idle|delay|0|2|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=acquire|parent=repeat|branch=body",
      "release|release|8|0|qubits[0]|deps=repeat|data_deps=repeat|resource_deps=repeat"
    ],
    makespan_ns = 8.0 : f64
  }
}

// -----

// The graph commits both the machine and the timing source used by scheduling.
module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.operating_point @point for @arch {timing = {cycle_ns = 2.0 : f64}}
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () attributes {operating_point = @point} {
    %acquired = phys.acquire [@q] {event_id = "acquire"}
      : !phys.state<@q>
    %prepared = phys.prepare %acquired {
      event_id = "prepare", state = "zero"
    } : (!phys.state<@q>) -> !phys.state<@q>
    phys.release %prepared {event_id = "release"} : !phys.state<@q>
    phys.return
  }
  // expected-error @+1 {{timing_profile fact 'cycle_ns' for event 'prepare' must equal the graph's committed operating-point timing}}
  phys.schedule @forged_timing for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 3.0 : f64}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=",
      "prepare|prepare|0|3|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=acquire",
      "release|release|3|0|qubits[0]|deps=prepare|data_deps=prepare|resource_deps=prepare"
    ],
    makespan_ns = 3.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{graph results cannot contain more than one physical-state owner for resource @q}}
  phys.graph @duplicate_results on @arch :
      () -> (!phys.state<@q>, !phys.state<@q>) {
    %state = phys.acquire [@q] : !phys.state<@q>
    phys.return %state, %state : !phys.state<@q>, !phys.state<@q>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 3 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q2 {
    index = 2 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{physical linear type crosses unsupported regionless operation builtin.unrealized_conversion_cast}}
  phys.graph @foreign_cast on @arch : () -> () {
    %state = phys.acquire [@q0] : !phys.state<@q0>
    %first, %second = builtin.unrealized_conversion_cast %state
      : !phys.state<@q0> to !phys.state<@q1>, !phys.state<@q2>
    phys.release %first : !phys.state<@q1>
    phys.release %second : !phys.state<@q2>
    phys.return
  }
}

// -----

module {
  func.func @missing_successor(%state: !phys.state<@q>) {
    // expected-error @+1 {{destructive measurement must omit its state result and non-destructive measurement must return one state}}
    %record = phys.measure @mz(%state) {record_id = "r"}
      : (!phys.state<@q>) -> !phys.record<@bit>
    return
  }
}

// -----

module {
  func.func @destructive_successor(%state: !phys.state<@q>) {
    // expected-error @+1 {{destructive measurement must omit its state result and non-destructive measurement must return one state}}
    %next, %record = phys.measure @mz(%state) {
      destructive, record_id = "r"
    } : (!phys.state<@q>) -> (!phys.state<@q>, !phys.record<@bit>)
    return
  }
}

// -----

module {
  func.func @wrong_successor(%state: !phys.state<@q>) {
    // expected-error @+1 {{measurement input/output state types must match}}
    %next, %record = phys.measure @mz(%state) {record_id = "r"}
      : (!phys.state<@q>) -> (!phys.state<@other>, !phys.record<@bit>)
    return
  }
}

// -----

module {
  phys.instrument @mz {
    kind = "measure", arity = 1 : i64, record_schema = "bit",
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_z\22}"
  }
  func.func @instrument_disagrees(%state: !phys.state<@q>) {
    // expected-error @+1 {{state-preserving phys.measure requires a preserves_inputs instrument}}
    %next, %record = phys.measure @mz(%state) {record_id = "r"}
      : (!phys.state<@q>) -> (!phys.state<@q>, !phys.record<@bit>)
    return
  }
}

// -----

module {
  func.func @empty_record_id(%state: !phys.state<@q>) {
    // expected-error @+1 {{record_id must be nonempty}}
    %record = phys.measure @mz(%state) {
      destructive, record_id = ""
    } : (!phys.state<@q>) -> !phys.record<@bit>
    return
  }
}

// -----

module {
  phys.action @pulse {
    arity = 1 : i64,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22h\22}"
  }
  func.func @wrong_measurement_symbol(%state: !phys.state<@q>) {
    // expected-error @+1 {{measurement reference must resolve to phys.instrument}}
    %record = phys.measure @pulse(%state) {
      destructive, record_id = "r"
    } : (!phys.state<@q>) -> !phys.record<@bit>
    return
  }
}

// -----

module {
  phys.instrument @not_measure {
    kind = "measure_product", arity = 1 : i64, preserves_inputs,
    record_schema = "bit",
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_product\22}"
  }
  func.func @wrong_measurement_kind(%state: !phys.state<@q>) {
    // expected-error @+1 {{instrument must implement measure}}
    %record = phys.measure @not_measure(%state) {
      destructive, record_id = "r"
    } : (!phys.state<@q>) -> !phys.record<@bit>
    return
  }
}

// -----

module {
  phys.instrument @binary_measure {
    kind = "measure", arity = 2 : i64, preserves_inputs,
    record_schema = "bit",
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_z\22}"
  }
  func.func @wrong_measurement_arity(%state: !phys.state<@q>) {
    // expected-error @+1 {{phys.measure requires a fixed unary instrument}}
    %record = phys.measure @binary_measure(%state) {
      destructive, record_id = "r"
    } : (!phys.state<@q>) -> !phys.record<@bit>
    return
  }
}

// -----

module {
  phys.instrument @analog_measure {
    kind = "measure", arity = 1 : i64, preserves_inputs,
    record_schema = "analog",
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_z\22}"
  }
  func.func @wrong_measurement_schema(%state: !phys.state<@q>) {
    // expected-error @+1 {{record type does not match instrument record_schema}}
    %record = phys.measure @analog_measure(%state) {
      destructive, record_id = "r"
    } : (!phys.state<@q>) -> !phys.record<@bit>
    return
  }
}

// -----

module {
  phys.instrument @preserving_measure {
    kind = "measure", arity = 1 : i64, preserves_inputs,
    record_schema = "bit",
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_z\22}"
  }
  func.func @drop_preserved_successor(%state: !phys.state<@q>) {
    %record = phys.measure @preserving_measure(%state) {
      destructive, record_id = "r"
    } : (!phys.state<@q>) -> !phys.record<@bit>
    return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  // expected-error @+1 {{kind must match resource class @qubits kind 'qubit'}}
  phys.resource @wrong_kind {
    index = 0 : i64, kind = "atom", resource_class = @qubits
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  // expected-error @+1 {{index is outside resource class @qubits capacity 1}}
  phys.resource @out_of_capacity {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
}

// -----

module {
  // expected-error @+1 {{resource_class @missing must resolve in a phys.machine}}
  phys.resource @unresolved {
    index = 0 : i64, kind = "qubit", resource_class = @missing
  }
}

// -----

module {
  phys.machine @first {
    phys.resource_class @q {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.machine @second {
    phys.resource_class @q {
      count = 1 : i64, kind = "atom", native_actions = []
    }
  }
  // expected-error @+1 {{resource_class @q is ambiguous across physical architectures}}
  phys.resource @unused {
    index = 0 : i64, kind = "qubit", resource_class = @q
  }
}

// -----

module {
  phys.machine @first {
    phys.resource_class @q {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.machine @second {
    phys.resource_class @q {
      count = 1 : i64, kind = "atom", native_actions = []
    }
  }
  phys.resource @used {
    index = 0 : i64, kind = "qubit", resource_class = @q
  }
  phys.graph @scoped on @first :
      (!phys.state<@used>) -> !phys.state<@used> {
  ^bb0(%state: !phys.state<@used>):
    phys.return %state : !phys.state<@used>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{linear physical event is consumed more than once by event.cancel and event.await}}
  phys.graph @double_event_consume on @arch :
      (!phys.state<@q>) -> !phys.state<@q> {
  ^bb0(%state: !phys.state<@q>):
    %event = phys.resource_request "t_state" from @resources::@magic
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
    %resource = event.await %event
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
        -> !phys.resource_payload<@t_state>
    %cancelled = event.cancel %event
      : !event.handle<!phys.resource_payload<@t_state>, "linear"> -> i8
    phys.discard_resource_payload %resource
      : !phys.resource_payload<@t_state>
    phys.return %state : !phys.state<@q>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{physical resource payload is consumed more than once by phys.discard_resource_payload and phys.discard_resource_payload}}
  phys.graph @double_resource_consume on @arch :
      (!phys.resource_payload<@t_state>, !phys.state<@q>)
        -> !phys.state<@q> {
  ^bb0(%resource: !phys.resource_payload<@t_state>,
       %state: !phys.state<@q>):
    phys.discard_resource_payload %resource
      : !phys.resource_payload<@t_state>
    phys.discard_resource_payload %resource
      : !phys.resource_payload<@t_state>
    phys.return %state : !phys.state<@q>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch :
      (!phys.record<@bit>) -> () {
  ^bb0(%record: !phys.record<@bit>):
    %predicate = phys.condition %record {event_id = "condition"}
      : !phys.record<@bit> -> i1
    "cflow.if"(%predicate) <{event_id = "if"}> ({
      cflow.yield
    }, {
      cflow.yield
    }) : (i1) -> ()
    phys.return
  }
  // expected-error @+1 {{schedule event 'if' condition must exactly match graph control}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "condition|condition|0|1|control:condition|deps=|data_deps=|resource_deps=|parent=|branch=|condition=|max_attempts=|repeat_count=|max_iterations=|callee=|instance=|exhaustion=",
      "if|if|1|0|control:if|deps=condition|data_deps=condition|resource_deps=|parent=|branch=|condition=if|max_attempts=|repeat_count=|max_iterations=|callee=|instance=|exhaustion="
    ],
    makespan_ns = 1.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "e"}
    phys.return
  }
  // expected-error @+1 {{schedule entry 'forged' is not backed by an event in graph @g}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "e|fence|0|0|control:e",
      "forged|fence|0|0|control:forged"
    ],
    makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @observed_then_consumed on @arch :
      (!event.handle<!phys.resource_payload<@t_state>, "linear">,
       !phys.state<@q>) -> !phys.state<@q> {
  ^bb0(%event: !event.handle<!phys.resource_payload<@t_state>, "linear">,
       %state: !phys.state<@q>):
    %ready = event.test %event {event_id = "test"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear"> -> i1
    %status = event.poll %event {event_id = "poll"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear"> -> i8
    %resource = event.await %event {event_id = "await"}
      : !event.handle<!phys.resource_payload<@t_state>, "linear">
        -> !phys.resource_payload<@t_state>
    phys.discard_resource_payload %resource {event_id = "discard"}
      : !phys.resource_payload<@t_state>
    phys.return %state : !phys.state<@q>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // `event.cancel` requires a linear event owner unconditionally, so
  // canceling a "shared" event is rejected here directly, before the
  // graph-level physical-linearity check (which only tracks "linear"-owned
  // events, and would otherwise silently let this "shared" event through)
  // ever runs.
  phys.graph @shared_event_not_linear on @arch :
      (!event.handle<i8, "shared">, !phys.state<@q>) -> !phys.state<@q> {
  ^bb0(%event: !event.handle<i8, "shared">, %state: !phys.state<@q>):
    // expected-error @+1 {{cancellation requires a linear event owner, but got 'shared'}}
    %first = event.cancel %event {event_id = "cancel.first"}
      : !event.handle<i8, "shared"> -> i8
    phys.return %state : !phys.state<@q>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{event_id values must be nonempty}}
  phys.graph @empty_event_id on @arch : () -> () {
    %state = phys.acquire [@q] {event_id = ""}
      : !phys.state<@q>
    phys.release %state : !phys.state<@q>
    phys.return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{event_id values must be graph-global and unique; duplicate 'same'}}
  phys.graph @duplicate_event_id on @arch : () -> () {
    %state = phys.acquire [@q] {event_id = "same"}
      : !phys.state<@q>
    phys.release %state {event_id = "same"} : !phys.state<@q>
    phys.return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{every schedulable graph operation requires a stable event_id; missing on phys.acquire}}
  phys.graph @missing_event_id on @arch : () -> () {
    %state = phys.acquire [@q] : !phys.state<@q>
    phys.release %state {event_id = "release"} : !phys.state<@q>
    phys.return
  }
}

// -----

module {
  // expected-error @+1 {{graph must resolve to phys.graph}}
  phys.schedule @unscoped for @missing {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable", entries = [], makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    phys.return
  }
  // expected-error @+1 {{makespan_ns must be finite and nonnegative}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [],
    makespan_ns = -1.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () {
    %acquired = phys.acquire [@q] {event_id = "acquire"}
      : !phys.state<@q>
    %prepared = phys.prepare %acquired {
      event_id = "prepare", state = "zero"
    } : (!phys.state<@q>) -> !phys.state<@q>
    phys.release %prepared {event_id = "release"} : !phys.state<@q>
    phys.return
  }
  // expected-error @+1 {{schedule rows must follow deterministic stable graph order}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "release|release|0|0|qubits[0]",
      "prepare|prepare|1|1|qubits[0]",
      "acquire|acquire|2|0|qubits[0]"
    ],
    makespan_ns = 2.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "e"}
    phys.return
  }
  // expected-error @+1 {{schedule event_id and kind must be nonempty}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = ["|fence|0|0|control:e"],
    makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "e"}
    phys.return
  }
  // expected-error @+1 {{schedule event IDs must be unique; duplicate 'e'}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = ["e|fence|0|0|control:e", "e|fence|0|0|control:e"],
    makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "e"}
    phys.return
  }
  // expected-error @+1 {{schedule start and duration must be finite nonnegative numbers}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = ["e|fence|-1|0|control:e"],
    makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "e"}
    phys.return
  }
  // expected-error @+1 {{schedule entry contains unknown detail field 'forged'}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = ["e|fence|0|0|control:e|forged=value"],
    makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "e"}
    phys.return
  }
  // expected-error @+1 {{schedule dependency 'missing' must name another entry}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = ["e|fence|0|0|control:e|deps=missing"],
    makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "first"}
    event.fence {effects = ["event"], event_id = "second"}
    phys.return
  }
  // expected-error @+1 {{schedule event 'first' resources must exactly match its resolved physical resource identities}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "first|fence|0|0|controller",
      "second|fence|0|0|controller"
    ],
    makespan_ns = 0.0 : f64
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.graph @g on @arch : () -> () {
    event.fence {effects = ["event"], event_id = "e"}
    phys.return
  }
  // expected-error @+1 {{makespan_ns must equal the latest top-level schedule finish}}
  phys.schedule @bad for @g {
    strategy = "greedy_asap", strategy_domain = "physical", provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1", constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = ["e|fence|0|0|control:e"],
    makespan_ns = 1.0 : f64
  }
}
