// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s

module {
  func.func @physical_mpp_selection(
      %patch: !fabric.patch<@code, @encoding, @epoch>,
      %a: tensor<1xi1>, %b: tensor<1xi1>)
      -> (!fabric.patch<@code, @encoding, @epoch>, i1) {
    fabric.tick
    %next, %measured = fabric.mpp %patch data
      indices [0, 2] paulis "XY" {record = "physical_product"}
      : !fabric.patch<@code, @encoding, @epoch> -> tensor<1xi1>
    %event = fabric.parity %a, %b, %measured
      : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> i1
    %accepted = fabric.all_false %event : (i1) -> i1
    return %next, %accepted
      : !fabric.patch<@code, @encoding, @epoch>, i1
  }
}

// CHECK: fabric.tick
// CHECK: fabric.mpp %{{.*}} data indices [0, 2] paulis "XY"
// CHECK: fabric.parity
// CHECK: fabric.all_false
