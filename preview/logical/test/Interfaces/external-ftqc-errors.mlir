// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, test-ftqc-plugin
// RUN: qlx-opt --load-dialect-plugin=%test_ftqc_plugin \
// RUN:   --allow-unregistered-dialect --split-input-file \
// RUN:   --verify-diagnostics %s

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  // expected-error @+1 {{physical linear type crosses unsupported regionless operation test_ftqc.opaque_measure}}
  phys.graph @missing_interface on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    %next, %record = test_ftqc.opaque_measure %state {
      record_id = "round.result"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 2 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q1 {
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @measurement_transfers_owner on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q1>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    // expected-error @+1 {{physical measurement successor states must preserve input physical resources}}
    %next, %record = test_ftqc.measure_round %state {
      record_id = "owner-transfer"
    } : (!phys.state<@q0>) -> (!phys.state<@q1>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q1>, !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @measurement_hides_record on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    // expected-error @+1 {{physical measurement interface must report its only !phys.record result}}
    %next, %record, %extra = "test_ftqc.measure_with_extra_record"(%state) {
      record_id = "hidden-record"
    } : (!phys.state<@q0>) ->
        (!phys.state<@q0>, !phys.record<@bit>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @measurement_hides_payload on @arch :
      (!phys.state<@q0>, !phys.resource_payload<@magic>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>,
       %payload: !phys.resource_payload<@magic>):
    // expected-error @+1 {{physical measurement interface cannot carry unmodelled physical linear values}}
    %next, %record = "test_ftqc.measure_with_payload"(%state, %payload) {
      record_id = "hidden-payload"
    } : (!phys.state<@q0>, !phys.resource_payload<@magic>) ->
        (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  // expected-error @+1 {{physical linear type crosses unsupported region control foreign.region}}
  phys.graph @forged_payload on @arch : () ->
      !phys.resource_payload<@magic> {
    %forged = "foreign.region"() ({
      "foreign.yield"() : () -> ()
    }) : () -> !phys.resource_payload<@magic>
    phys.return %forged : !phys.resource_payload<@magic>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  // expected-error @+1 {{physical linear type crosses unsupported region control foreign.region}}
  phys.graph @forged_event on @arch : () -> () {
    "foreign.region"() ({
    ^bb0(%forged: !event.handle<!phys.resource_payload<@magic>, "linear">):
      "foreign.yield"() : () -> ()
    }) : () -> ()
    phys.return
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @duplicate_successor on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    // expected-error @+1 {{physical measurement interface reported a duplicate successor state}}
    %next, %record = test_ftqc.measure_round %state {
      record_id = "duplicate", test_mode = "duplicate_output"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @forged_state on @arch : () -> !phys.state<@q0> {
    %constant = arith.constant 0 : i32
    // expected-error @+1 {{physical measurement interface state inputs must have !phys.state type}}
    %junk, %forged = test_ftqc.measure_round %constant {
      record_id = "forged"
    } : (i32) -> (i32, !phys.state<@q0>)
    phys.return %forged : !phys.state<@q0>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @wrong_record on @arch : (!phys.state<@q0>) ->
      !phys.state<@q0> {
  ^bb0(%state: !phys.state<@q0>):
    // expected-error @+1 {{physical measurement interface record must have !phys.record type}}
    %next, %not_record = test_ftqc.measure_round %state {
      record_id = "wrong-record"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, i32)
    phys.return %next : !phys.state<@q0>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @omitted_state on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    // expected-error @+1 {{physical measurement interface must report every !phys.state result exactly once}}
    %next, %record = test_ftqc.measure_round %state {
      record_id = "omitted", test_mode = "omit_output"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @duplicate_state on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    // expected-error @+1 {{physical measurement interface reported a duplicate state input}}
    %next, %record = test_ftqc.measure_round %state {
      record_id = "duplicate", test_mode = "duplicate_input"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @overlapping_roles on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    // expected-error @+1 {{physical measurement record cannot also be a successor state}}
    %next, %record = test_ftqc.measure_round %state {
      record_id = "overlap", test_mode = "record_as_output"
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}

// -----

module {
  phys.machine @arch {
    phys.resource_class @qubits {
      kind = "qubit", count = 1 : i64, native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @invalid_interface on @arch : (!phys.state<@q0>) ->
      (!phys.state<@q0>, !phys.record<@bit>) {
  ^bb0(%state: !phys.state<@q0>):
    // expected-error @+1 {{physical measurement interface requires a nonempty record identity}}
    %next, %record = test_ftqc.measure_round %state {
      record_id = ""
    } : (!phys.state<@q0>) -> (!phys.state<@q0>, !phys.record<@bit>)
    phys.return %next, %record : !phys.state<@q0>, !phys.record<@bit>
  }
}
