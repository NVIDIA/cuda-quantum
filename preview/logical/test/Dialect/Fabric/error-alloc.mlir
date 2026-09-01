// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.code @auxiliary {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.machine @device {
  fabric.region @data_region {
    code = @data,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

fabric.protocol @wrong_region_code : () -> () {
  // expected-error@+1 {{code @auxiliary does not match resolved region @data_region code @data}}
  %0 = fabric.alloc {code = @auxiliary, region = @data_region, strict_region} : !fabric.patch<@auxiliary>
  fabric.dealloc %0 : !fabric.patch<@auxiliary>
  fabric.protocol_return
}

// -----

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.machine @device {
  fabric.region @data_region {
    code = @data,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

fabric.protocol @missing_region : () -> () {
  // expected-error@+1 {{region @missing does not resolve in the linked fabric.machine}}
  %0 = fabric.alloc {code = @data, region = @missing, strict_region} : !fabric.patch<@data>
  fabric.dealloc %0 : !fabric.patch<@data>
  fabric.protocol_return
}

// -----

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.machine @device {
  fabric.region @data_region {
    code = @data,
    encoding = @data_encoding,
    epoch = @data_epoch,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

fabric.protocol @wrong_region_encoding : () -> () {
  // expected-error@+1 {{patch encoding @other_encoding does not match resolved region @data_region encoding @data_encoding}}
  %0 = fabric.alloc {code = @data, region = @data_region, strict_region} : !fabric.patch<@data, @other_encoding, @data_epoch>
  fabric.dealloc %0 : !fabric.patch<@data, @other_encoding, @data_epoch>
  fabric.protocol_return
}

// -----

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.machine @device {
  fabric.region @data_region {
    code = @data,
    encoding = @data_encoding,
    epoch = @data_epoch,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

fabric.protocol @wrong_region_epoch : () -> () {
  // expected-error@+1 {{patch epoch @other_epoch does not match resolved region @data_region epoch @data_epoch}}
  %0 = fabric.alloc {code = @data, region = @data_region, strict_region} : !fabric.patch<@data, @data_encoding, @other_epoch>
  fabric.dealloc %0 : !fabric.patch<@data, @data_encoding, @other_epoch>
  fabric.protocol_return
}

// -----

// A detached reusable protocol may carry a provider-facing region hint. It is
// resolved by a later projection and therefore deliberately omits
// `strict_region`.
fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.code @auxiliary {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.machine @device {
  fabric.region @provider_factory {
    code = @auxiliary,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

fabric.protocol @provider_region_hint : () -> () {
  %0 = fabric.alloc {code = @data, region = @provider_factory} : !fabric.patch<@data>
  fabric.dealloc %0 : !fabric.patch<@data>
  fabric.protocol_return
}

// -----

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.machine @left {
  fabric.region @shared {
    code = @data,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

fabric.machine @right {
  fabric.region @shared {
    code = @data,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

fabric.protocol @ambiguous_strict_region : () -> () {
  // expected-error@+1 {{region @shared is ambiguous across linked fabric.machine ops}}
  %0 = fabric.alloc {code = @data, region = @shared, strict_region} : !fabric.patch<@data>
  fabric.dealloc %0 : !fabric.patch<@data>
  fabric.protocol_return
}

// -----

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.code @auxiliary {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.code_profile @auxiliary_profile {code = @auxiliary}

fabric.encoding @auxiliary_encoding {
  block = "auxiliary",
  code = @auxiliary,
  logical_ports = ["q0"],
  profile = @auxiliary_profile
}

fabric.machine @wrong_encoding_code {
  // expected-error@+1 {{encoding-qualified type code @data disagrees with encoding @auxiliary_encoding code @auxiliary}}
  fabric.region @data_region {
    code = @data,
    encoding = @auxiliary_encoding,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

// -----

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.code @auxiliary {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.code_profile @auxiliary_profile {code = @auxiliary}

fabric.encoding @auxiliary_encoding {
  block = "auxiliary",
  code = @auxiliary,
  logical_ports = ["q0"],
  profile = @auxiliary_profile
}

fabric.machine @device {
  fabric.region @data_region {
    code = @data,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

fabric.protocol @wrong_allocation_encoding_code : () -> () {
  // expected-error@+1 {{encoding-qualified type code @data disagrees with encoding @auxiliary_encoding code @auxiliary}}
  %0 = fabric.alloc {code = @data, region = @data_region, strict_region} : !fabric.patch<@data, @auxiliary_encoding>
  fabric.dealloc %0 : !fabric.patch<@data, @auxiliary_encoding>
  fabric.protocol_return
}

// -----

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.machine @missing_region_encoding {
  // expected-error@+1 {{epoch requires an encoding-qualified region}}
  fabric.region @data_region {
    code = @data,
    epoch = @orphan_epoch,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}

// -----

fabric.code @data {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.code_profile @data_profile {code = @data}

fabric.encoding_epoch_schema @epochs {
  phases = ["initial"],
  initial = "initial",
  transitions = []
}

fabric.encoding @data_encoding {
  block = "data",
  code = @data,
  epoch_schema = @epochs,
  initial_epoch = @data_initial,
  logical_ports = ["q0"],
  profile = @data_profile
}

fabric.encoding @other_encoding {
  block = "other",
  code = @data,
  epoch_schema = @epochs,
  initial_epoch = @other_initial,
  logical_ports = ["q0"],
  profile = @data_profile
}

fabric.encoding_epoch @data_initial {
  encoding = @data_encoding,
  index = 0 : i64,
  phase = "initial",
  schema = @epochs
}

fabric.encoding_epoch @other_initial {
  encoding = @other_encoding,
  index = 0 : i64,
  phase = "initial",
  schema = @epochs
}

fabric.machine @wrong_region_epoch_owner {
  // expected-error@+1 {{epoch @other_initial belongs to encoding @other_encoding, not @data_encoding}}
  fabric.region @data_region {
    code = @data,
    encoding = @data_encoding,
    epoch = @other_initial,
    role = #fabric.role<compute>,
    floorplan = #fabric.floorplan<linear, [1]>
  }
}
