// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s | qlx-opt | FileCheck %s

fabric.code @sc { distance = 3 : i64, partitions = {data = 9 : i64, sx = 4 : i64, sz = 4 : i64} }

fabric.gadget @extract(
    %p: !fabric.patch<@sc>, %s: !fabric.syndrome<@sc>)
    -> (!fabric.patch<@sc>, !fabric.syndrome<@sc>)
    flow #fabric.flow<{x = "x", z = "z"}>
{
  %p1, %syn = fabric.read_syndrome_ancillas %p
      : !fabric.patch<@sc> -> !fabric.syndrome<@sc>
  fabric.return %p1, %syn : !fabric.patch<@sc>, !fabric.syndrome<@sc>
}

// CHECK-LABEL: fabric.gadget @test_repeat
fabric.gadget @test_repeat(
    %p: !fabric.patch<@sc>, %syn_in: !fabric.syndrome<@sc>)
    -> (!fabric.patch<@sc>, !fabric.syndrome<@sc>)
    flow #fabric.flow<{x = "x", z = "z"}>
{
  // CHECK: fabric.repeat 3
  // CHECK-NEXT: iter(%arg2: !fabric.patch<@sc> = %arg0,
  // CHECK-NEXT:      %arg3: !fabric.syndrome<@sc> = %arg1)
  %p_out, %syn_out = fabric.repeat 3
      iter(%pi : !fabric.patch<@sc> = %p,
           %si : !fabric.syndrome<@sc> = %syn_in) {
    %pn, %sn = fabric.call @extract(%pi, %si)
        : (!fabric.patch<@sc>, !fabric.syndrome<@sc>)
        -> (!fabric.patch<@sc>, !fabric.syndrome<@sc>)
    fabric.yield %pn, %sn : !fabric.patch<@sc>, !fabric.syndrome<@sc>
  }

  fabric.return %p_out, %syn_out
      : !fabric.patch<@sc>, !fabric.syndrome<@sc>
}

// CHECK-LABEL: fabric.gadget @test_repeat_single_arg
fabric.gadget @test_repeat_single_arg(%p: !fabric.patch<@sc>)
    -> !fabric.patch<@sc>
{
  // CHECK: fabric.repeat 5
  %p_out = fabric.repeat 5
      iter(%pi : !fabric.patch<@sc> = %p) {
    %pn = fabric.h %pi data : !fabric.patch<@sc>
    fabric.yield %pn : !fabric.patch<@sc>
  }

  fabric.return %p_out : !fabric.patch<@sc>
}
