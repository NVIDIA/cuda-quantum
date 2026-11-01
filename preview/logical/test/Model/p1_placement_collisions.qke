// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  lvm.domain @collision {
    lvm.space @compute {capabilities = [], capacity = 2 : i64}
    // expected-error @+1 {{ordinary local residency must reference lvm.space directly}}
    lvm.placement @p0 {space = @compute, slot = 0 : i64}
  }
}

// -----

module {
  lvm.domain @collision {
    lvm.space @compute {capabilities = [], capacity = 1 : i64}
    lvm.placement @p0 {
      space = @compute, slot = 0 : i64,
      binding = {kind = "unresolved", constraints = ["c0"],
                 dimensions = ["space"], reason = "test"}
    }
    // expected-error @+1 {{slot is already occupied by another logical placement}}
    lvm.placement @p1 {
      space = @compute, slot = 0 : i64,
      binding = {kind = "unresolved", constraints = ["c1"],
                 dimensions = ["space"], reason = "test"}
    }
  }
}
