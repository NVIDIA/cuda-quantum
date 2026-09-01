// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -verify-diagnostics

module {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.gadget @attempt(%patch: !fabric.patch<@c>)
      -> (!fabric.patch<@c>, i1) {
    %ok = arith.constant true
    fabric.return %patch, %ok : !fabric.patch<@c>, i1
  }
  fabric.protocol @checks : (!fabric.patch<@c>) -> !fabric.patch<@c> {
  ^bb0(%patch: !fabric.patch<@c>):
    %next, %ok = fabric.call @attempt(%patch)
        : (!fabric.patch<@c>) -> (!fabric.patch<@c>, i1)
    // expected-error @+1 {{max_attempts must be positive}}
    %bad = fabric.retry %ok carries(%next) {
      attempt = @attempt, max_attempts = 0 : i64
    } : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %bad : !fabric.patch<@c>
  }
}

// -----

module {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.protocol @checks : (!fabric.patch<@c>, i1) -> !fabric.patch<@c> {
  ^bb0(%patch: !fabric.patch<@c>, %ok: i1):
    // expected-error @+1 {{requires an explicit attempt gadget or protocol}}
    %bad = fabric.retry %ok carries(%patch) {max_attempts = 2 : i64}
        : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %bad : !fabric.patch<@c>
  }
}

// -----

module {
  fabric.code @c {distance = 3 : i64, partitions = {data = 1 : i64}}
  fabric.gadget @attempt(%patch: !fabric.patch<@c>)
      -> (!fabric.patch<@c>, i1) {
    %ok = arith.constant true
    fabric.return %patch, %ok : !fabric.patch<@c>, i1
  }
  fabric.protocol @checks : (!fabric.patch<@c>, i1) -> !fabric.patch<@c> {
  ^bb0(%patch: !fabric.patch<@c>, %ok: i1):
    // expected-error @+1 {{success_probability requires attempt-bound nonempty provenance}}
    %bad = fabric.retry %ok carries(%patch) {
      attempt = @attempt, max_attempts = 2 : i64,
      success_probability = 0.5 : f64
    } : (!fabric.patch<@c>) -> !fabric.patch<@c>
    fabric.protocol_return %bad : !fabric.patch<@c>
  }
}
