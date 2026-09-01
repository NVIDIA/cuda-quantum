// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error @+1 {{interaction patches must name nodes in the patch graph}}
  fabric.patch_graph @bad {
    root = @protocol,
    nodes = [{id = "patch0"}],
    interactions = [
      {id = "interaction0", action = "cx", patches = ["patch0", "patch1"]}
    ]
  }
}
