// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// Two independent Bell pairs plus one cross-pair interaction -- the running
// example from `LoweringQubitsToRegions.md`.
//
//   q0 --H--*------------
//           |
//   q1 -----X------*-----
//                  |
//   q2 --H--*------X-----
//           |
//   q3 -----X------------
//
// On a 2x2 QPU the two Bell pairs land in separate regions and run in
// parallel (depth 2, not 4); the cross-pair gate then costs one inter-region
// move.

quake.wire_set @wires[4]

func.func @bell_cross() attributes {"cudaq-entrypoint"} {
  %q0 = quake.borrow_wire @wires[0] : !quake.wire
  %q1 = quake.borrow_wire @wires[1] : !quake.wire
  %q2 = quake.borrow_wire @wires[2] : !quake.wire
  %q3 = quake.borrow_wire @wires[3] : !quake.wire

  %h0 = quake.h %q0 : (!quake.wire) -> !quake.wire
  %bell0:2 = quake.x [%h0] %q1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)

  %h1 = quake.h %q2 : (!quake.wire) -> !quake.wire
  %bell1:2 = quake.x [%h1] %q3 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)

  %cross:2 = quake.x [%bell0#1] %bell1#0 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)

  quake.return_wire %bell0#0 : !quake.wire
  quake.return_wire %cross#0 : !quake.wire
  quake.return_wire %cross#1 : !quake.wire
  quake.return_wire %bell1#1 : !quake.wire
  return
}
