// ========================================================================== //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        //
// All rights reserved.                                                       //
//                                                                            //
// This source code and the accompanying materials are made available under   //
// the terms of the Apache License 2.0 which accompanies this distribution.   //
// ========================================================================== //

// REQUIRES: qlx-opt, cudaq-quake
// RUN: qlx-opt --phys-to-quake=entry=bit %s | FileCheck --check-prefix=BIT \
// RUN:   --implicit-check-not=phys. --implicit-check-not=qlx. %s
// RUN: qlx-opt --phys-to-quake=entry=flag %s | FileCheck --check-prefix=FLAG \
// RUN:   --implicit-check-not=phys. --implicit-check-not=qlx. %s
// RUN: qlx-opt --phys-to-quake=entry=none %s | FileCheck --check-prefix=NONE \
// RUN:   --implicit-check-not=phys. --implicit-check-not=qlx. %s
// RUN: not qlx-opt --phys-to-quake %s 2>&1 | FileCheck --check-prefix=EMPTY %s
// RUN: not qlx-opt --phys-to-quake=entry=missing %s 2>&1 | \
// RUN:   FileCheck --check-prefix=MISSING %s
// RUN: not qlx-opt --phys-to-quake=entry=machine %s 2>&1 | \
// RUN:   FileCheck --check-prefix=MACHINE %s

// The input marks no graph as the entry point. Each run picks one with the
// pass option `entry=<graph>`. @bit returns one Boolean as an `i1`, @flag
// takes an `i1` argument, and @none returns nothing. The pass erases the
// other graphs untranslated, including @unsupported, whose `!phys.state`
// result it would reject. It also removes Logical's `qlx.*` module
// attributes. The last three runs fail because `entry` must name a
// phys.graph: they omit it, name a missing symbol, and name the phys.machine.
module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  phys.machine @machine {
    phys.resource_class @qubits {count = 3 : i64, kind = "qubit", native_actions = [], native_instruments = [@mz]}
  }
  phys.instrument @mz {arity = 1 : i64, kind = "measure", preserves_inputs, process = "{\"kind\":\"builtin\",\"name\":\"measure_z\",\"parameters\":{}}", record_schema = "bit"}
  phys.resource @a {architecture = @machine, index = 0 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @b {architecture = @machine, index = 1 : i64, kind = "qubit", resource_class = @qubits}
  phys.resource @c {architecture = @machine, index = 2 : i64, kind = "qubit", resource_class = @qubits}
  phys.graph @bit on @machine : () -> !phys.record<@bit> {
    %0 = phys.acquire [@a] {event_id = "bit0"} : !phys.state<@a>
    %record = phys.measure @mz(%0) {destructive, event_id = "bit1", record_id = "r"} : (!phys.state<@a>) -> !phys.record<@bit>
    phys.return %record : !phys.record<@bit>
  }
  phys.graph @flag on @machine : (i1) -> i1 {
  ^bb0(%arg0: i1):
    %0 = phys.condition %arg0 {event_id = "flag0", expected = false} : i1 -> i1
    phys.return %0 : i1
  }
  phys.graph @none on @machine : () -> () {
    %0 = phys.acquire [@b] {event_id = "none0"} : !phys.state<@b>
    phys.release %0 {event_id = "none1"} : !phys.state<@b>
    phys.return
  }
  phys.graph @unsupported on @machine : () -> !phys.state<@c> {
    %0 = phys.acquire [@c] {event_id = "unsupported0"} : !phys.state<@c>
    phys.return %0 : !phys.state<@c>
  }
}

// BIT:         module attributes {quake.noOptimization} {
// BIT-LABEL:   func.func @__nvqpp__mlirgen__bit() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// BIT:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// BIT:           %[[VAL_0:.*]], %[[MZ_0:.*]] = quake.mz %[[NULL_WIRE_0]] : (!quake.wire) -> (!cc.measure_handle, !quake.wire)
// BIT:           quake.sink %[[MZ_0]] : !quake.wire
// BIT:           %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[VAL_0]] : (!cc.measure_handle) -> i1
// BIT:           return %[[DISCRIMINATE_0]] : i1
// BIT:         }

// FLAG:         module attributes {quake.noOptimization} {
// FLAG-LABEL:   func.func @__nvqpp__mlirgen__flag(
// FLAG-SAME:      %[[ARG0:.*]]: i1) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// FLAG:           %[[CONSTANT_0:.*]] = arith.constant true
// FLAG:           %[[XORI_0:.*]] = arith.xori %[[ARG0]], %[[CONSTANT_0]] : i1
// FLAG:           return %[[XORI_0]] : i1
// FLAG:         }

// NONE:         module attributes {quake.noOptimization} {
// NONE-LABEL:   func.func @__nvqpp__mlirgen__none() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// NONE:           %[[NULL_WIRE_0:.*]] = quake.null_wire
// NONE:           quake.sink %[[NULL_WIRE_0]] : !quake.wire
// NONE:           return
// NONE:         }

// EMPTY: error: `phys-to-quake` needs the entry option to name a phys.graph, but '' names none
// MISSING: error: `phys-to-quake` needs the entry option to name a phys.graph, but 'missing' names none
// MACHINE: error: `phys-to-quake` needs the entry option to name a phys.graph, but 'machine' names none
