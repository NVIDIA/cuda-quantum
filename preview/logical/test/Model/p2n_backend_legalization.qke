// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt --fabric-lower-protocols='root-symbol=main' %s | FileCheck %s

module attributes {qlx.profiles = ["p2n"]} {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}

  fabric.gadget @h(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    %0 = fabric.h %arg0 data : !fabric.patch<@c>
    fabric.return %0 : !fabric.patch<@c>
  }

  fabric.protocol @helper : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%arg0: !fabric.patch<@c>):
    %0 = fabric.call @h(%arg0) : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %0 : !fabric.patch<@c>
  }

  fabric.protocol @main : () -> () {
    %0 = fabric.alloc {code = @c, region = @compute} : !fabric.patch<@c>
    %1 = fabric.call @helper(%0) : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.dealloc %1 : !fabric.patch<@c>
    fabric.protocol_return
  }
}

// CHECK-NOT: fabric.protocol
// CHECK-NOT: fabric.protocol_return
// CHECK: fabric.gadget @helper(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c>
// CHECK: fabric.return
// CHECK: realization_kind = "protocol"
// CHECK: fabric.gadget @main {entry} on @__qlx_target()
// CHECK: fabric.call @helper
// CHECK: fabric.return
// CHECK: realization_kind = "protocol"
