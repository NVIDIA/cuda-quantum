// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  func.func @bad_factory() {
    // expected-error @+1 {{resource_kind must match the result resource type}}
    %0 = phys.produce_resource @ccz_state at @factory {
      protocol = #fabric.spec_only<"wrong-kind">
    } : !phys.resource_payload<@t_state>
    return
  }
}
