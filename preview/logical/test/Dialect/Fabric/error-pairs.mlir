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
  partitions = {data = 3 : i64, sx = 2 : i64, sz = 1 : i64}
}

func.func @malformed(%p: !fabric.patch<@c>) {
  // expected-error@+1 {{'pairs' entries must use the canonical c:t grammar}}
  %0 = fabric.cx %p data -> sx {pairs = "0-1"}
      : !fabric.patch<@c>
  return
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 3 : i64, sx = 2 : i64, sz = 1 : i64}
}

func.func @out_of_range(%p: !fabric.patch<@c>) {
  // expected-error@+1 {{'pairs' target index 2 is outside partition width 2}}
  %0 = fabric.cx %p data -> sx {pairs = "0:2"}
      : !fabric.patch<@c>
  return
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 3 : i64, sx = 2 : i64, sz = 1 : i64}
}

func.func @noncanonical_index(%p: !fabric.patch<@c>) {
  // expected-error@+1 {{'pairs' must not contain surrounding whitespace}}
  %0 = fabric.cx %p data -> sx {pairs = " index "}
      : !fabric.patch<@c>
  return
}

// -----

// `all` spans every declared carrier partition, including custom scratch roles.
fabric.code @with_flag {
  distance = 1 : i64,
  partitions = {data = 2 : i64, sx = 1 : i64, sz = 0 : i64, flag = 1 : i64}
}

func.func @custom_partition_in_all(%p: !fabric.patch<@with_flag>) {
  %0 = fabric.cx %p sx -> all {pairs = "0:3"}
      : !fabric.patch<@with_flag>
  return
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 2 : i64, sx = 1 : i64, sz = 0 : i64}
}

func.func @self_alias_through_all(%p: !fabric.patch<@c>) {
  // expected-error@+1 {{control index 0 and target index 2 alias the same patch carrier}}
  %0 = fabric.cx %p sx -> all {pairs = "0:2"}
      : !fabric.patch<@c>
  return
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 2 : i64, sx = 1 : i64, sz = 0 : i64}
}

func.func @self_alias_index(%p: !fabric.patch<@c>) {
  // expected-error@+1 {{control index 0 and target index 0 alias the same patch carrier}}
  %0 = fabric.cz %p data -> all {pairs = "index"}
      : !fabric.patch<@c>
  return
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 3 : i64, sx = 1 : i64, sz = 1 : i64}
}

func.func @unknown_schedule(%p: !fabric.patch<@c>) {
  // expected-error@+1 {{'schedule' must be 'hx' or 'hz', got 'round_robin'}}
  %0 = fabric.cx %p sx -> data {schedule = "round_robin"}
      : !fabric.patch<@c>
  return
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 3 : i64, sx = 1 : i64, sz = 1 : i64}
}

func.func @missing_checks(%p: !fabric.patch<@c>) {
  // expected-error@+1 {{schedule 'hx' requires nonempty hx checks on code @c}}
  %0 = fabric.cx %p sx -> data {schedule = "hx"}
      : !fabric.patch<@c>
  return
}

// -----

fabric.code @c {
  distance = 1 : i64,
  partitions = {data = 3 : i64, sx = 1 : i64, sz = 1 : i64},
  hx = [array<i64: 0>]
}

func.func @wrong_schedule_partitions(%p: !fabric.patch<@c>) {
  // expected-error@+1 {{schedule 'hx' requires sx -> data, got data -> data}}
  %0 = fabric.cx %p data -> data {schedule = "hx"}
      : !fabric.patch<@c>
  return
}
