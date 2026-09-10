// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// Four qubits in one region, ending with a gate between the two qubits placed
// furthest apart. On an all-to-all region this costs nothing; on a `line`
// region it forces intra-region hops.

quake.wire_set @wires[4]

func.func @line_chain() attributes {"cudaq-entrypoint"} {
  %q0 = quake.borrow_wire @wires[0] : !quake.wire
  %q1 = quake.borrow_wire @wires[1] : !quake.wire
  %q2 = quake.borrow_wire @wires[2] : !quake.wire
  %q3 = quake.borrow_wire @wires[3] : !quake.wire

  %a0 = quake.h %q0 : (!quake.wire) -> !quake.wire
  %a:2 = quake.x [%a0] %q1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
  %b:2 = quake.x [%q2] %q3 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)

  // Ends of the line: vq0 at slot 0 and vq3 at slot 3.
  %c:2 = quake.x [%a#0] %b#1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)

  %m:2 = quake.mz %c#0 : (!quake.wire) -> (!quake.measure, !quake.wire)

  quake.return_wire %a#1 : !quake.wire
  quake.return_wire %b#0 : !quake.wire
  quake.return_wire %m#1 : !quake.wire
  return
}
