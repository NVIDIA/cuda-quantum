// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  func.func @empty_instance(%q: !phys.state<@q0>) {
    // expected-error @+1 {{instance must be nonempty}}
    %0 = "phys.call"(%q) <{callee = @g, instance = ""}> ({
    ^bb0(%current: !phys.state<@q0>):
      phys.yield %current : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    return
  }
}
