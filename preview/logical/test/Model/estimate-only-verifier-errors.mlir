// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt --split-input-file %s -verify-diagnostics

module attributes {qlx.profiles = ["p1"]} {
  lvm.domain @machine {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  // expected-error @+1 {{estimate_only requires an input_p0 refinement}}
  lvm.kernel @demand on @machine : () -> () attributes {estimate_only} {
    lvm.return
  }
}

// -----

module attributes {qlx.profiles = ["p0", "p1"]} {
  qlx.program @demand : () -> () attributes {
    estimate_only,
    qlx.profile = "p0",
    specialization = {rounds = 3 : i64}
  } {
    qlx.return
  }
  lvm.domain @machine {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  // expected-error @+1 {{estimate_only must exactly match the input_p0 program}}
  lvm.kernel @demand_placed on @machine : () -> () attributes {
    input_p0 = @demand,
    placement_witness_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    specialization = {rounds = 3 : i64}
  } {
    lvm.return
  }
}

// -----

module attributes {qlx.profiles = ["p0", "p1"]} {
  qlx.program @executable : () -> () attributes {qlx.profile = "p0"} {
    qlx.return
  }
  lvm.domain @machine {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  // expected-error @+1 {{estimate_only must exactly match the input_p0 program}}
  lvm.kernel @executable_placed on @machine : () -> () attributes {
    estimate_only,
    input_p0 = @executable,
    placement_witness_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  } {
    lvm.return
  }
}

// -----

module attributes {qlx.profiles = ["p0", "p1"]} {
  qlx.program @demand : () -> () attributes {
    estimate_only,
    qlx.profile = "p0",
    specialization = {rounds = 3 : i64}
  } {
    qlx.return
  }
  lvm.domain @machine {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
  }
  // expected-error @+1 {{specialization must exactly match the input_p0 program}}
  lvm.kernel @demand_placed on @machine : () -> () attributes {
    estimate_only,
    input_p0 = @demand,
    placement_witness_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    specialization = {rounds = 4 : i64}
  } {
    lvm.return
  }
}

// -----

module attributes {qlx.profiles = ["p0"]} {
  qlx.program @paper_count : (index) -> index attributes {
    estimate_only,
    qlx.profile = "p0"
  } {
  ^bb0(%rounds: index):
    qlx.return %rounds : index
  }
  qlx.program @executable : (index) -> index attributes {qlx.profile = "p0"} {
  ^bb0(%rounds: index):
    // expected-error @+1 {{executable qlx.program cannot call estimate-only @paper_count}}
    %count = qlx.call @paper_count(%rounds) : (index) -> index
    qlx.return %count : index
  }
}
