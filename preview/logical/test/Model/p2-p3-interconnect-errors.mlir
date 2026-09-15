// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt --split-input-file --verify-diagnostics %s

fabric.machine @qec {
  fabric.region @factory {
    code = @encoding,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @compute {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  // expected-error@+1 {{port_a must be less than region @factory block_capacity 1}}
  fabric.interconnect @bad_port {
    region_a = @factory,
    port_a = 1 : i64,
    region_b = @compute,
    port_b = 0 : i64
  }
}

// -----

lvm.domain @logical {
  lvm.space @left {capabilities = [], capacity = 1 : i64}
  lvm.space @middle {capabilities = [], capacity = 1 : i64}
  lvm.space @right {capabilities = [], capacity = 1 : i64}
  lvm.channel @left_middle {
    from = @left,
    to = @middle,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">]
  }
  lvm.channel @middle_right {
    from = @middle,
    to = @right,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">]
  }
}
fabric.machine @qec {
  fabric.region @left {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @middle {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @right {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  // expected-error@+1 {{persistent channel port 'shared_port' has inconsistent region, slot, capability, concurrency, provider, or metadata facts}}
  fabric.interconnect @left_middle {
    region_a = @left,
    port_a = 0 : i64,
    region_b = @middle,
    port_b = 0 : i64,
    logical_channel = @logical::@left_middle,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    direction = "forward",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "left_port",
    port_b_name = "shared_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64,
    port_a_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_b_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_a_provider = "example.port@1",
    port_b_provider = "example.port@1"
  }
  fabric.interconnect @middle_right {
    region_a = @middle,
    port_a = 0 : i64,
    region_b = @right,
    port_b = 0 : i64,
    logical_channel = @logical::@middle_right,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    direction = "forward",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "shared_port",
    port_b_name = "right_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64,
    port_a_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_b_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_a_provider = "different.port@1",
    port_b_provider = "example.port@1"
  }
}

// -----

lvm.domain @logical {
  lvm.space @left {capabilities = [], capacity = 1 : i64}
  lvm.space @right {capabilities = [], capacity = 1 : i64}
  lvm.channel @remote {
    from = @left,
    to = @right,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">]
  }
}
fabric.machine @qec {
  fabric.region @left {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @right {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  // expected-error@+1 {{channel-port capability #lvm.capability<"qlx.machine/state_transport"> is absent from the P1 logical channel}}
  fabric.interconnect @remote {
    region_a = @left,
    port_a = 0 : i64,
    region_b = @right,
    port_b = 0 : i64,
    logical_channel = @logical::@remote,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    direction = "forward",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "left_port",
    port_b_name = "right_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64,
    port_a_capabilities = [
      #lvm.capability<"qlx.machine/observable_remote">,
      #lvm.capability<"qlx.machine/state_transport">
    ],
    port_b_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_a_provider = "example.port@1",
    port_b_provider = "example.port@1"
  }
}

// -----

lvm.domain @logical {
  lvm.space @factory {capabilities = [], capacity = 1 : i64}
  lvm.space @compute {capabilities = [], capacity = 1 : i64}
  lvm.stream @encoded_y {produces = @y_state, capacity = 1 : i64}
  lvm.channel @encoded_y_supply {
    from = @factory,
    to = @encoded_y,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
  }
  lvm.channel @delivery {
    from = @encoded_y,
    to = @compute,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    direction = "forward"
  }
}
fabric.machine @qec {
  fabric.region @factory {
    code = @encoding,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @compute {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  // expected-error@+1 {{resource-transfer refinement requires a typed fabric.protocol}}
  fabric.interconnect @delivery {
    region_a = @factory,
    port_a = 0 : i64,
    region_b = @compute,
    port_b = 0 : i64,
    logical_channel = @logical::@delivery,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    direction = "forward",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "factory_port",
    port_b_name = "compute_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64,
    port_a_capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    port_b_capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    port_a_provider = "example.port@1",
    port_b_provider = "example.port@1"
  }
}

// -----

lvm.domain @logical {
  lvm.space @factory {capabilities = [], capacity = 1 : i64}
  lvm.space @compute {capabilities = [], capacity = 1 : i64}
  lvm.stream @encoded_y {produces = @y_state, capacity = 1 : i64}
  lvm.channel @encoded_y_supply {
    from = @factory,
    to = @encoded_y,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">]
  }
  lvm.channel @delivery {
    from = @encoded_y,
    to = @compute,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    direction = "forward"
  }
}
fabric.protocol @delivery_adapter : (!fabric.resource<@y_state>) -> !fabric.resource<@y_state> {
^bb0(%state: !fabric.resource<@y_state>):
  fabric.protocol_return %state : !fabric.resource<@y_state>
}
fabric.machine @qec {
  fabric.region @factory {
    code = @encoding,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @compute {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  // expected-error@+1 {{P2 interconnect regions must refine the P1 channel endpoints}}
  fabric.interconnect @delivery {
    region_a = @compute,
    port_a = 0 : i64,
    region_b = @factory,
    port_b = 0 : i64,
    logical_channel = @logical::@delivery,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    direction = "forward",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "factory_port",
    port_b_name = "compute_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64,
    port_a_capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    port_b_capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    port_a_provider = "example.port@1",
    port_b_provider = "example.port@1",
    protocol = @delivery_adapter
  }
}
qlx.logical_to_qec @map {
  logical = @logical,
  qec = @qec,
  entries = [
    {logical = "factory", qec = "factory"},
    {logical = "compute", qec = "compute"}
  ]
}
qlx.device @device {
  logical = @logical,
  qec = @qec,
  logical_to_qec = @map
}

// -----

lvm.domain @logical {
  lvm.space @factory {capabilities = [], capacity = 1 : i64}
  lvm.space @compute {capabilities = [], capacity = 1 : i64}
  lvm.channel @remote {
    from = @factory,
    to = @compute,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    direction = "bidirectional"
  }
}
fabric.machine @qec {
  fabric.region @factory {
    code = @encoding,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @compute {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.interconnect @remote {
    region_a = @factory,
    port_a = 0 : i64,
    region_b = @compute,
    port_b = 0 : i64,
    logical_channel = @logical::@remote,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    direction = "bidirectional",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "factory_port",
    port_b_name = "compute_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64,
    port_a_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_b_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_a_provider = "example.port@1",
    port_b_provider = "example.port@1"
  }
}
phys.machine @physical {
  phys.resource_class @qubits {
    kind = "qubit",
    count = 2 : i64,
    native_actions = []
  }
}
qlx.logical_to_qec @map {
  logical = @logical,
  qec = @qec,
  entries = [
    {logical = "factory", qec = "factory"},
    {logical = "compute", qec = "compute"}
  ]
}
qlx.qec_to_physical @physical_map {
  qec = @qec,
  physical = @physical,
  entries = []
}
// expected-error@+1 {{selected P2 interconnect @remote must have exactly one P3 qec_channel_binding; found 0}}
qlx.device @device {
  logical = @logical,
  qec = @qec,
  physical = @physical,
  logical_to_qec = @map,
  qec_to_physical = @physical_map
}

// -----

lvm.domain @logical {
  lvm.space @left {capabilities = [], capacity = 1 : i64}
  lvm.space @right {capabilities = [], capacity = 1 : i64}
  lvm.channel @remote {
    from = @left,
    to = @right,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    direction = "bidirectional"
  }
}
fabric.machine @qec {
  fabric.region @left {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @right {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.interconnect @remote {
    region_a = @left,
    port_a = 0 : i64,
    region_b = @right,
    port_b = 0 : i64,
    logical_channel = @logical::@remote,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    direction = "bidirectional",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "left_port",
    port_b_name = "right_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64,
    port_a_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_b_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_a_provider = "example.port@1",
    port_b_provider = "example.port@1"
  }
}
phys.machine @physical {
  phys.resource_class @qubits {
    kind = "qubit",
    count = 2 : i64,
    native_actions = []
  }
  phys.qec_channel_binding @first {
    qec_channel = @qec::@remote,
    resources = [@qubits]
  }
  phys.qec_channel_binding @second {
    qec_channel = @qec::@remote,
    resources = [@qubits]
  }
}
qlx.logical_to_qec @map {
  logical = @logical,
  qec = @qec,
  entries = [
    {logical = "left", qec = "left"},
    {logical = "right", qec = "right"}
  ]
}
qlx.qec_to_physical @physical_map {
  qec = @qec,
  physical = @physical,
  entries = []
}
// expected-error@+1 {{selected P2 interconnect @remote must have exactly one P3 qec_channel_binding; found 2}}
qlx.device @device {
  logical = @logical,
  qec = @qec,
  physical = @physical,
  logical_to_qec = @map,
  qec_to_physical = @physical_map
}

// -----

fabric.machine @qec {
  fabric.region @factory {
    code = @encoding,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @compute {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  // expected-error@+1 {{selected P2 channel requires nonempty capabilities}}
  fabric.interconnect @incomplete_refinement {
    region_a = @factory,
    port_a = 0 : i64,
    region_b = @compute,
    port_b = 0 : i64,
    logical_channel = @logical::@delivery,
    direction = "forward",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "factory_port",
    port_b_name = "compute_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64
  }
}

// -----

fabric.machine @qec {
  fabric.region @factory {
    code = @encoding,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
  fabric.region @compute {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
  fabric.interconnect @delivery {
    region_a = @factory,
    port_a = 0 : i64,
    region_b = @compute,
    port_b = 0 : i64
  }
}
phys.machine @physical {
  phys.resource_class @qubits {
    kind = "qubit",
    count = 2 : i64,
    native_actions = []
  }
  // expected-error@+1 {{requires at least one bound resource class}}
  phys.qec_channel_binding @missing_resources {
    qec_channel = @qec::@delivery,
    resources = []
  }
}

// -----

lvm.domain @logical {
  lvm.space @factory {capabilities = [], capacity = 1 : i64}
  lvm.space @compute {capabilities = [], capacity = 1 : i64}
  lvm.channel @delivery {
    from = @factory,
    to = @compute,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    capacity = 1 : i64,
    direction = "forward"
  }
}
fabric.machine @qec {
  fabric.region @factory {
    code = @encoding,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @compute {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  // expected-error@+1 {{concurrency exceeds the P1 logical channel capacity}}
  fabric.interconnect @too_wide {
    region_a = @factory,
    port_a = 0 : i64,
    region_b = @compute,
    port_b = 0 : i64,
    logical_channel = @logical::@delivery,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    direction = "forward",
    concurrency = 2 : i64,
    provider = "example@1",
    port_a_name = "factory_port",
    port_b_name = "compute_port",
    port_a_concurrency = 2 : i64,
    port_b_concurrency = 2 : i64,
    port_a_capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    port_b_capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    port_a_provider = "example.port@1",
    port_b_provider = "example.port@1"
  }
}

// -----

lvm.domain @logical {
  lvm.space @factory {capabilities = [], capacity = 1 : i64}
  lvm.space @compute {capabilities = [], capacity = 1 : i64}
  lvm.channel @delivery {
    from = @factory,
    to = @compute,
    capabilities = [#lvm.capability<"qlx.machine/resource_transfer">],
    direction = "forward"
  }
}
fabric.machine @qec {
  fabric.region @factory {
    code = @encoding,
    role = #fabric.role<factory>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  fabric.region @compute {
    code = @encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>,
    block_capacity = 1 : i64
  }
  // expected-error@+1 {{selected channel capability #lvm.capability<"qlx.machine/observable_remote"> is absent from the P1 logical channel}}
  fabric.interconnect @wrong_capability {
    region_a = @factory,
    port_a = 0 : i64,
    region_b = @compute,
    port_b = 0 : i64,
    logical_channel = @logical::@delivery,
    capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    direction = "forward",
    concurrency = 1 : i64,
    provider = "example@1",
    port_a_name = "factory_port",
    port_b_name = "compute_port",
    port_a_concurrency = 1 : i64,
    port_b_concurrency = 1 : i64,
    port_a_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_b_capabilities = [#lvm.capability<"qlx.machine/observable_remote">],
    port_a_provider = "example.port@1",
    port_b_provider = "example.port@1"
  }
}
