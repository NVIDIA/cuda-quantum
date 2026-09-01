// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s --pass-pipeline='builtin.module(fabric-count{root=roundtrip result=counts})' | FileCheck %s

fabric.code @bare {
  distance = 1 : i64,
  partitions = {data = 1 : i64},
  k = 1 : i64
}

fabric.protocol @roundtrip :
    (!fabric.patch<@bare, @bare_encoding, @bare_epoch>,
     !fabric.resource<@t_state>)
    -> (!fabric.patch<@bare, @bare_encoding, @bare_epoch>,
        !fabric.resource<@t_state>) {
^bb0(%anchor: !fabric.patch<@bare, @bare_encoding, @bare_epoch>,
     %state: !fabric.resource<@t_state>):
  %next, %payload = fabric.unpack_resource %state like(%anchor)
      : (!fabric.resource<@t_state>,
         !fabric.patch<@bare, @bare_encoding, @bare_epoch>)
        -> (!fabric.patch<@bare, @bare_encoding, @bare_epoch>,
            !fabric.patch<@bare, @bare_encoding, @bare_epoch>)
  %packed = fabric.pack_resource %payload as @t_state
      : !fabric.patch<@bare, @bare_encoding, @bare_epoch>
        -> !fabric.resource<@t_state>
  fabric.protocol_return %next, %packed
      : !fabric.patch<@bare, @bare_encoding, @bare_epoch>,
        !fabric.resource<@t_state>
}

// CHECK: fabric.counts = {
// CHECK-SAME: logical_qubits_peak = 2 : i64
