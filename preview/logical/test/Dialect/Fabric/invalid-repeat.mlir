// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt -split-input-file -verify-diagnostics %s

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 1 : i64}
}

fabric.gadget @negative_repeat(%p: !fabric.patch<@c>)
    -> !fabric.patch<@c> {
  // expected-error @+1 {{count must be non-negative}}
  %next = fabric.repeat -1 iter(%arg: !fabric.patch<@c> = %p) {
    fabric.yield %arg : !fabric.patch<@c>
  }
  fabric.return %next : !fabric.patch<@c>
}
