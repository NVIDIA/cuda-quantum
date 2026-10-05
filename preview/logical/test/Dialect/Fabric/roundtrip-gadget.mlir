// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

fabric.code @steane {
  distance = 3 : i64,
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64}
}

// CHECK-LABEL: fabric.gadget @simple
// CHECK-SAME: (%arg0: !fabric.patch<@steane>)
// CHECK-SAME: -> !fabric.patch<@steane>
fabric.gadget @simple(%p: !fabric.patch<@steane>)
    -> !fabric.patch<@steane> {
  fabric.return %p : !fabric.patch<@steane>
}

// CHECK-LABEL: fabric.gadget @with_flow
// CHECK-SAME: (%arg0: !fabric.patch<@steane>, %arg1: !fabric.syndrome<@steane>)
// CHECK-SAME: -> (!fabric.patch<@steane>, !fabric.syndrome<@steane>)
// CHECK-SAME: flow #fabric.flow<{x = "z", z = "x"}>
fabric.gadget @with_flow(
    %p: !fabric.patch<@steane>, %syn: !fabric.syndrome<@steane>)
    -> (!fabric.patch<@steane>, !fabric.syndrome<@steane>)
    flow #fabric.flow<{x = "z", z = "x"}>
{
  fabric.return %p, %syn : !fabric.patch<@steane>, !fabric.syndrome<@steane>
}

// CHECK-LABEL: fabric.gadget @no_return
fabric.gadget @no_return(%p: !fabric.patch<@steane>) {
  fabric.return
}

// CHECK-LABEL: fabric.gadget @call_test
fabric.gadget @call_test(%p: !fabric.patch<@steane>)
    -> !fabric.patch<@steane> {
  // CHECK: fabric.call @simple
  %r = fabric.call @simple(%p) : (!fabric.patch<@steane>) -> !fabric.patch<@steane>
  fabric.return %r : !fabric.patch<@steane>
}
