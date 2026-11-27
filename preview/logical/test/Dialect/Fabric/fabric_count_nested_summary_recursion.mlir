// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt
// RUN: not qlx-opt %s --fabric-count='root=entry' 2>&1 | FileCheck %s

// Summary probes must retain the entire active call stack even before any
// of the summaries in this indirect cycle have been inserted into the cache.
fabric.gadget @first() {
  fabric.call @second() : () -> ()
  fabric.return
}
fabric.gadget @second() {
  fabric.call @first() : () -> ()
  fabric.return
}
fabric.gadget @entry() {
  fabric.call @first() : () -> ()
  fabric.return
}
// CHECK: fabric-count rejects recursive executable call through @first
