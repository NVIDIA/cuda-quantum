// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: qlx-opt %s -verify-diagnostics

lvm.domain @logical {
  // expected-error @+1 {{capacity must be nonnegative}}
  lvm.stream @invalid {produces = @t_state, capacity = -1 : i64}
}
