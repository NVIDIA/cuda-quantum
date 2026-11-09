// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  fabric.code @c {distance = 1 : i64, n = 3 : i64, partitions = {data = 3 : i64}}
  fabric.gadget @duplicate(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    // expected-error @+1 {{permutation entries must be unique}}
    %0 = fabric.permute %arg0 perm [0, 0, 2] : !fabric.patch<@c>
    fabric.return %0 : !fabric.patch<@c>
  }
}

// -----

module {
  fabric.code @c {distance = 1 : i64, n = 3 : i64, partitions = {data = 3 : i64}}
  fabric.gadget @short(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    // expected-error @+1 {{permutation length must equal the code's carrier count}}
    %0 = fabric.permute %arg0 perm [0, 1] : !fabric.patch<@c>
    fabric.return %0 : !fabric.patch<@c>
  }
}

// -----

module {
  fabric.code @c {distance = 1 : i64, n = 3 : i64, partitions = {data = 3 : i64}}
  fabric.gadget @range(%arg0: !fabric.patch<@c>) -> !fabric.patch<@c> {
    // expected-error @+1 {{permutation entries must lie in [0, n)}}
    %0 = fabric.permute %arg0 perm [0, 1, 3] : !fabric.patch<@c>
    fabric.return %0 : !fabric.patch<@c>
  }
}
