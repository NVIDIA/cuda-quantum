// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt --split-input-file --verify-diagnostics %s

module {
  lvm.domain @machine {
    lvm.space @factory {
      capabilities = [#lvm.capability<"qlx.machine/logical_factory">]
    }
    lvm.stream @states {
      produces = @state, backing_region = @factory,
      produced_by = @ghost,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
    // expected-error @+1 {{QLX resource-supply produced_by must resolve to a typed fabric.protocol}}
    lvm.channel @forged_supply {
      from = @factory, to = @states,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
}

// -----

module {
  lvm.domain @machine {
    lvm.space @factory {
      capabilities = [#lvm.capability<"qlx.machine/logical_factory">]
    }
    lvm.stream @states {
      produces = @state, backing_region = @factory
    }
    // expected-error @+1 {{QLX resource-supply destination requires authenticated produced_by provenance}}
    lvm.channel @forged_supply {
      from = @factory, to = @states,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
}
