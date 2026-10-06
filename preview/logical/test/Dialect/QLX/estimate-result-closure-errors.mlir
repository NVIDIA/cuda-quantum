// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s 2>&1 | FileCheck %s

fabric.protocol @work : () -> () {
  fabric.protocol_return
}

qlx.estimate_result @detached {
  assumptions = [], data = {}, evidence = [@work],
  metadata = {producer = "fixture", producer_version = "1"},
  root = @work, schema = "qlx.fabric-counts/v1", tier = "static"
}

// CHECK: 'qlx.estimate_result' op static tier requires a device and no lower-tier result
