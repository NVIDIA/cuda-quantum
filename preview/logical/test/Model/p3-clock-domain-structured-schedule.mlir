// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module attributes {qlx.profiles = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @none {
      count = 0 : i64, kind = "qubit", native_actions = []
    }
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @branch_clock on @arch : () -> () {
    %condition = "arith.constant"() {
      event_id = "condition", value = true
    } : () -> i1
    "cflow.if"(%condition) <{event_id = "if0"}> ({
      phys.barrier {domains = ["clock"], event_id = "branch_tick"}
        : () -> ()
      cflow.yield
    }, {
      cflow.yield
    }) : (i1) -> ()
    event.fence {effects = ["event"], event_id = "after_if"}
    phys.return
  }
  phys.schedule @branch_clock_schedule for @branch_clock {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64},
    tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "condition|arith.constant|0|1|control:condition|deps=|data_deps=|resource_deps=|domain_deps=",
      "if0|if|1|0|control:if0|deps=condition|data_deps=condition|resource_deps=|domain_deps=|condition=condition",
      "branch_tick|barrier|1|0|control:branch_tick|deps=|data_deps=|resource_deps=|domain_deps=|parent=if0|branch=then|condition=condition",
      "after_if|fence|1|0|control:after_if|deps=if0|data_deps=|resource_deps=|domain_deps=if0"
    ],
    makespan_ns = 1.0 : f64
  }

  phys.graph @zero_repeat_clock on @arch : () -> () {
    "cflow.repeat"() <{count = 0 : i64, event_id = "repeat0"}> ({
      phys.barrier {domains = ["clock"], event_id = "inactive_tick"}
        : () -> ()
      cflow.yield
    }) : () -> ()
    event.fence {effects = ["event"], event_id = "after_repeat"}
    phys.return
  }
  phys.schedule @zero_repeat_clock_schedule for @zero_repeat_clock {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {}, tie_break = "stable_graph_order",
    optimization_status = "not_applicable",
    entries = [
      "repeat0|repeat|0|0|control:repeat0|deps=|data_deps=|resource_deps=|domain_deps=|repeat_count=0|repeat_period_ns=0|repeat_epilogue_ns=0",
      "inactive_tick|barrier|0|0|control:inactive_tick|deps=|data_deps=|resource_deps=|domain_deps=|parent=repeat0|branch=body",
      "after_repeat|fence|0|0|control:after_repeat|deps=|data_deps=|resource_deps=|domain_deps="
    ],
    makespan_ns = 0.0 : f64
  }

  phys.graph @branch_clock_with_live_resource on @arch : () -> () {
    %q = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %p = phys.prepare %q {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %r = phys.reset %p {event_id = "reset", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %condition = "arith.constant"() {
      event_id = "resource_condition", value = true
    } : () -> i1
    "cflow.if"(%condition) <{event_id = "resource_if"}> ({
      phys.barrier {domains = ["clock"], event_id = "resource_tick"}
        : () -> ()
      cflow.yield
    }, {
      cflow.yield
    }) : (i1) -> ()
    phys.release %r {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
  phys.schedule @branch_clock_with_live_resource_schedule
      for @branch_clock_with_live_resource {
    strategy = "greedy_asap", strategy_domain = "physical",
    provider = "qlx.compiler.greedy_asap", provider_version = "1",
    constraint_profile = "qlx.physical_schedule.constraints/v1",
    constraints = ["graph_ssa_dependencies", "physical_resource_exclusion", "allocation_mapping_after", "structured_control_exclusivity", "folded_region_bounds", "resolved_event_durations"],
    timing_profile = {cycle_ns = 1.0 : f64},
    tie_break = "stable_graph_order", optimization_status = "not_applicable",
    entries = [
      "acquire|acquire|0|0|qubits[0]|deps=|data_deps=|resource_deps=|domain_deps=",
      "prepare|prepare|0|1|qubits[0]|deps=acquire|data_deps=acquire|resource_deps=acquire|domain_deps=",
      "reset|reset|1|1|qubits[0]|deps=prepare|data_deps=prepare|resource_deps=prepare|domain_deps=",
      "resource_condition|arith.constant|0|1|control:resource_condition|deps=|data_deps=|resource_deps=|domain_deps=",
      "resource_if|if|1|1|control:resource_if|deps=resource_condition|data_deps=resource_condition|resource_deps=|domain_deps=|condition=resource_condition",
      "resource_tick|barrier|2|0|control:resource_tick|deps=reset|data_deps=|resource_deps=|domain_deps=reset|parent=resource_if|branch=then|condition=resource_condition",
      "release|release|2|0|qubits[0]|deps=reset,resource_if|data_deps=reset|resource_deps=reset|domain_deps=resource_if"
    ],
    makespan_ns = 2.0 : f64
  }
}

// CHECK: "after_if|fence|1|0|control:after_if|deps=if0|data_deps=|resource_deps=|domain_deps=if0"
// CHECK: "after_repeat|fence|0|0|control:after_repeat|deps=|data_deps=|resource_deps=|domain_deps="
// CHECK: "resource_if|if|1|1|control:resource_if|deps=resource_condition
// CHECK: "release|release|2|0|qubits[0]|deps=reset,resource_if|data_deps=reset|resource_deps=reset|domain_deps=resource_if"
