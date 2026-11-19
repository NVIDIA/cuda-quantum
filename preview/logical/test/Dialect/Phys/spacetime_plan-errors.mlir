// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s --split-input-file --verify-diagnostics

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.protocol @p2 : () -> () { fabric.protocol_return }
  phys.machine @arch {
    phys.resource_class @patches {
      count = 1 : i64, footprint_evidence = "test patch",
      granularity = "patch", kind = "patch", native_actions = [],
      physical_unit_kind = "qubit", physical_units = 32 : i64
    }
  }
  phys.operating_point @point for @arch
  // expected-error @+1 {{body must contain at least one spacetime operation}}
  phys.spacetime_plan @empty attributes {
    architecture = @arch, derivation = "test", derivation_version = 1 : i64,
    evidence = "invalid", operating_point = @point, provider = "qlx.test",
    provider_version = "1", source_protocol = @p2
  } {
  ^bb0:
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.protocol @p2 : () -> () { fabric.protocol_return }
  phys.machine @arch {
    phys.resource_class @patches {
      count = 1 : i64, footprint_evidence = "test patch",
      granularity = "patch", kind = "patch", native_actions = [],
      physical_unit_kind = "qubit", physical_units = 32 : i64
    }
  }
  phys.operating_point @point for @arch
  // expected-error @+1 {{unregistered spacetime-plan provider/derivation 'qlx.test/test'}}
  phys.spacetime_plan @bad_dependency attributes {
    architecture = @arch, derivation = "test", derivation_version = 1 : i64,
    evidence = "invalid", operating_point = @point, provider = "qlx.test",
    provider_version = "1", source_protocol = @p2
  } {
    phys.spacetime_phase @first {
      after = [@later], factory_models = [],
      resource_classes = [@arch::@patches],
      step_duration_ns = 1.000000e+00 : f64, steps = 1 : i64
    }
    phys.spacetime_phase @later {
      after = [], factory_models = [],
      resource_classes = [@arch::@patches],
      step_duration_ns = 1.000000e+00 : f64, steps = 1 : i64
    }
  }
}

// -----

module attributes {qlx.profiles = ["p2n", "p3"]} {
  fabric.protocol @p2 : () -> () { fabric.protocol_return }
  phys.machine @arch {
    phys.resource_class @patches {
      count = 1 : i64, footprint_evidence = "test patch",
      granularity = "patch", kind = "patch", native_actions = [],
      physical_unit_kind = "qubit", physical_units = 32 : i64
    }
  }
  phys.operating_point @point for @arch
  phys.graph @graph on @arch : () -> () attributes {
    operating_point = @point, source_protocol = @p2
  } {
    // expected-error @+1 {{plan must resolve to phys.spacetime_plan}}
    "phys.spacetime_call"() <{
      event_id = "bad", instance = "root.bad", plan = @missing
    }> : () -> ()
    phys.return
  }
}
