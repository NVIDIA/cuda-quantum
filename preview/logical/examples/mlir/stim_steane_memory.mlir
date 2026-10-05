// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// One round of CSS syndrome extraction on the Steane [[7,1,3]] code, followed
// by a destructive data readout.
//
// The gadget takes its encoded patch as an entry argument rather than
// allocating one: Stim emission projects a realization, it does not prepare an
// encoded state, so the data carriers must already be initialized. Only the
// ancilla partitions are reset here.

fabric.code @Steane {
  distance = 3 : i64,
  n = 7 : i64,
  k = 1 : i64,
  r = 0 : i64,
  hx = [array<i64: 0, 1, 2, 3>, array<i64: 0, 1, 4, 5>, array<i64: 0, 2, 4, 6>],
  hz = [array<i64: 0, 1, 2, 3>, array<i64: 0, 1, 4, 5>, array<i64: 0, 2, 4, 6>],
  partitions = {data = 7 : i64, sx = 3 : i64, sz = 3 : i64}
}

fabric.machine @qec {
  fabric.region @memory {
    code = @Steane,
    floorplan = #fabric.floorplan<direct, [1]>,
    role = #fabric.role<memory>
  }
}

fabric.gadget @steane_memory {entry} on @qec(%patch: !fabric.patch<@Steane>)
    -> tensor<7xi1> {
  // X-type extraction: reset the sx ancillas, conjugate with H, couple them to
  // the data carriers named by each hx row, then undo the conjugation.
  %0 = fabric.reset %patch sx : !fabric.patch<@Steane>
  %1 = fabric.h %0 sx : !fabric.patch<@Steane>
  %2 = fabric.cx %1 sx -> data
      {pairs = "0:0,0:1,0:2,0:3,1:0,1:1,1:4,1:5,2:0,2:2,2:4,2:6"} : <@Steane>
  %3 = fabric.h %2 sx : !fabric.patch<@Steane>

  // Z-type extraction: reset the sz ancillas and couple data into them.
  %4 = fabric.reset %3 sz : !fabric.patch<@Steane>
  %5 = fabric.cx %4 data -> sz
      {pairs = "0:0,1:0,2:0,3:0,0:1,1:1,4:1,5:1,0:2,2:2,4:2,6:2"} : <@Steane>

  // Read the syndrome, then measure the data block destructively.
  %out, %syndrome = fabric.read_syndrome_ancillas %5 {record = "syndrome0"}
      : <@Steane> -> <@Steane>
  %out2, %bits = fabric.mz %out data {record = "mz0"}
      : !fabric.patch<@Steane> -> tensor<7xi1>
  fabric.dealloc %out2 : <@Steane>
  fabric.return %bits : tensor<7xi1>
}
