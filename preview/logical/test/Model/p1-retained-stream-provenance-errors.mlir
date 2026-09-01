// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  lvm.domain @logical {
    lvm.space @factory {capabilities = [#lvm.capability<"qlx.machine/logical_factory">]}
    lvm.stream @magic {
      backing_region = @factory, produced_by = @handoff,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      produces = @t_state
    }
    lvm.channel @supply {
      from = @factory, to = @magic,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
  qlx.action @transport_t : (!fabric.resource<@t_state>) -> !fabric.resource<@t_state> {
    kind = "transport_t_state"
  }
  // expected-error @+1 {{retained stream produced_by function type must have the exact resource-flow boundary for @t_state}}
  fabric.protocol @handoff : (!fabric.resource<@t_state>) -> !fabric.resource<@t_state> attributes {
    objective = @transport_t
  } {
  ^bb0(%state: !fabric.resource<@t_state>):
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
}

// -----

module {
  lvm.domain @logical {
    // expected-error @+1 {{produced_by requires a canonical sha256 payload commitment}}
    lvm.stream @magic {
      external, produced_by = @selected_producer, produces = @t_state
    }
  }
}

// -----

module {
  lvm.domain @logical {
    // expected-error @+1 {{produced_by_sha256 requires the matching protocol identity}}
    lvm.stream @magic {
      external, produces = @t_state,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
  }
}

// -----

// A consumer/injection protocol is the other exact transfer role: one matching
// resource plus the built-in action's patch boundary.
module {
  lvm.domain @logical {
    lvm.stream @magic {
      external, produces = @t_state, transfer = @inject,
      transfer_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
  }
  fabric.protocol @inject : (!fabric.patch<@code>, !fabric.resource<@t_state>) -> !fabric.patch<@code> attributes {
    objective = #qlx.action<t>
  } {
  ^bb0(%patch: !fabric.patch<@code>, %state: !fabric.resource<@t_state>):
    fabric.discard_resource %state : !fabric.resource<@t_state>
    fabric.protocol_return %patch : !fabric.patch<@code>
  }
}

module {
  lvm.domain @logical {
    lvm.space @factory {capabilities = [#lvm.capability<"qlx.machine/logical_factory">]}
    lvm.stream @magic {
      backing_region = @factory, produced_by = @wrong_objective,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      produces = @t_state
    }
    lvm.channel @supply {
      from = @factory, to = @magic,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
  qlx.action @same_shape_wrong_role : () -> !fabric.resource<@t_state> {
    kind = "transport_t_state"
  }
  // expected-error @+1 {{retained stream produced_by objective must be the exact 'produce_t_state' qlx.action}}
  fabric.protocol @wrong_objective : () -> !fabric.resource<@t_state> attributes {
    objective = @same_shape_wrong_role
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @t_state
    } : !fabric.resource<@t_state>
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
}

// -----

module {
  lvm.domain @logical {
    lvm.space @factory {capabilities = [#lvm.capability<"qlx.machine/logical_factory">]}
    lvm.stream @magic {
      backing_region = @factory, produced_by = @wrong_resource,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      produces = @t_state
    }
    lvm.channel @supply {
      from = @factory, to = @magic,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
  qlx.action @produce_y : () -> !fabric.resource<@y_state> {
    kind = "produce_y_state"
  }
  // expected-error @+1 {{retained stream produced_by function type must have the exact resource-flow boundary for @t_state}}
  fabric.protocol @wrong_resource : () -> !fabric.resource<@y_state> attributes {
    objective = @produce_y
  } {
    %state = fabric.produce_resource {
      region = @factory, resource_kind = @y_state
    } : !fabric.resource<@y_state>
    fabric.protocol_return %state : !fabric.resource<@y_state>
  }
}

// -----

module {
  lvm.domain @logical {
    lvm.space @factory {capabilities = [#lvm.capability<"qlx.machine/logical_factory">]}
    lvm.space @other_factory {capabilities = [#lvm.capability<"qlx.machine/logical_factory">]}
    lvm.stream @magic {
      backing_region = @factory, produced_by = @producer,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      produces = @t_state
    }
    lvm.channel @supply {
      from = @factory, to = @magic,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
  qlx.action @produce_t : () -> !fabric.resource<@t_state> {
    kind = "produce_t_state"
  }
  // expected-error @+1 {{retained producer factory-region references must exactly equal the stream backing_region}}
  fabric.protocol @producer : () -> !fabric.resource<@t_state> attributes {
    objective = @produce_t
  } {
    %state = fabric.produce_resource {
      region = @other_factory, resource_kind = @t_state
    } : !fabric.resource<@t_state>
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
}

// -----

module {
  lvm.domain @logical {
    lvm.space @factory {capabilities = [#lvm.capability<"qlx.machine/logical_factory">]}
    lvm.space @compute {capabilities = [#lvm.capability<"qlx.machine/logical_compute">]}
    // expected-error @+1 {{backing_region must equal the retained supply channel source}}
    lvm.stream @magic {backing_region = @compute, produces = @t_state}
    lvm.channel @supply {
      from = @factory, to = @magic,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
}

// -----

module {
  lvm.domain @logical {
    lvm.space @factory {capabilities = [#lvm.capability<"qlx.machine/logical_factory">]}
    lvm.space @other_factory {capabilities = [#lvm.capability<"qlx.machine/logical_factory">]}
    // expected-error @+1 {{backing_region must equal the retained supply channel source}}
    lvm.stream @magic {backing_region = @other_factory, produces = @t_state}
    lvm.channel @supply {
      from = @factory, to = @magic,
      capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
    }
  }
}

// -----

// External streams deliberately have no retained factory relation. Their
// typed producer and transfer roles are still authenticated.
module {
  lvm.domain @logical {
    lvm.stream @magic {
      external, produced_by = @producer, produces = @t_state,
      produced_by_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000",
      transfer = @handoff,
      transfer_sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    }
  }
  qlx.action @produce_t : () -> !fabric.resource<@t_state> {
    kind = "produce_t_state"
  }
  fabric.protocol @producer : () -> !fabric.resource<@t_state> attributes {
    objective = @produce_t
  } {
    %state = fabric.produce_resource {
      region = @out_of_frame_factory, resource_kind = @t_state
    } : !fabric.resource<@t_state>
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
  qlx.action @transport_t : (!fabric.resource<@t_state>) -> !fabric.resource<@t_state> {
    kind = "transport_t_state"
  }
  fabric.protocol @handoff : (!fabric.resource<@t_state>) -> !fabric.resource<@t_state> attributes {
    objective = @transport_t
  } {
  ^bb0(%state: !fabric.resource<@t_state>):
    fabric.protocol_return %state : !fabric.resource<@t_state>
  }
}
