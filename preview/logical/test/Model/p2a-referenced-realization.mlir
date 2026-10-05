// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// RUN: qlx-opt %s | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: qlx-opt --fabric-inline-realisations %s | FileCheck %s --check-prefix=INLINE

module attributes {qlx.profiles = ["p2a"]} {
  fabric.circuit @flip_body(%value: i1) -> i1 {
    %true = arith.constant true
    %flipped = arith.xori %value, %true : i1
    fabric.return %flipped : i1
  }

  fabric.gadget @flip(%value: i1) -> i1 realization @flip_body
}

// ROUNDTRIP: fabric.circuit @flip_body
// ROUNDTRIP: fabric.gadget @flip(%{{.*}}: i1) -> i1 realization @flip_body
// ROUNDTRIP-NOT: arith.xori

// INLINE: fabric.circuit @flip_body
// INLINE-LABEL: fabric.gadget @flip
// INLINE-NOT: realization @flip_body
// INLINE: arith.xori
// INLINE: fabric.return
