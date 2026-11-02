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
    lvm.space @not_a_protocol {capabilities = []}
    // expected-error @+1 {{produced_by must resolve to a typed fabric.protocol}}
    lvm.stream @states {
      produces = @state, backing_region = @factory,
      produced_by = @not_a_protocol,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
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
    // expected-error @+1 {{produced_by requires a canonical sha256 payload commitment}}
    lvm.stream @states {
      produces = @state, backing_region = @factory,
      produced_by = @later_stage_provider,
      produced_by_sha256 = "not-a-canonical-digest"
    }
    lvm.channel @forged_supply {
      from = @factory, to = @states,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
}
