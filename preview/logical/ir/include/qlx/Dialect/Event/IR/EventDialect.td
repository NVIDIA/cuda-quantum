/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef EVENT_DIALECT_TD
#define EVENT_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Event Dialect
//===----------------------------------------------------------------------===//

def Event_Dialect : Dialect {
  let name = "event";
  let cppNamespace = "qlx::event";

  let summary = "asynchronous delivery: event handles, their observers, and their consumers";
  let description = [{
    This dialect represents asynchronous delivery. An `!event.handle`
    stands for one thing that has been requested but is not yet in hand --
    a resource a factory is still producing, a runtime verdict, a
    controller signal, the completion of a protocol -- and the operations
    here either observe that handle or consume it to take what it
    delivers.

    Waiting is the same contract at every semantic tier, so it is defined
    once here and embedded directly into each tier's bodies instead of
    being restated per tier. Only the *producer* is tier-local: a tier's
    own `resource_request`/`request` operation names the kind, stream, or
    provider it draws on and returns an `!event.handle` whose payload is
    that tier's resource type. The payload type is the only part of an
    asynchronous wait that differs between tiers, which is why this
    dialect depends on no other dialect and no other dialect's types
    appear in its operations.

    The operations that take a handle fall into two disjoint classes, and
    that split is why the handle type separates readiness from payload
    ownership:

    * `test`, `poll` and `select_ready` observe. They never move
      ownership, may run any number of times, and leave the handle
      usable.
    * `await`, `try_take` and `cancel` consume. Each takes the handle's
      ownership, and at most one of them runs for a given handle, so a
      payload declared linear cannot acquire a second owner however often
      the program branched on readiness first. SSA permits a value to
      have several uses, so that last part is a contract this dialect
      states and does not check; the linear-use analysis enforcing it
      runs outside the dialect.

    None of those observers is `Pure`, deliberately: a handle's state
    depends on when it is read, so two polls of one handle may
    legitimately disagree and CSE must not merge them.

    Observed state is named, not numeric. The portable states are
    `pending`, `ready`, `failed`, `cancelled` and `exhausted`;
    `event.poll` returns one as an `i8` whose encoding this dialect
    deliberately leaves unspecified, and `event.is` names the state a
    program or an analysis is asking about. `event.is` takes no handle and
    is pure -- the status it reads is a reading already taken -- and
    nothing here checks that the `i8` it reads came from a poll at all.

    Two things this dialect does not model. Duration: a wait is a
    dependence on a delivery, orders the continuation that uses the
    payload, and names no time. Fulfilment: how a handle comes to resolve
    is the producing tier's business, and a handle resolving at most once
    is a contract on producers that no verifier here enforces.

    `event.fence` and `event.selection` hold no handle. They are here
    because every tier needs them in exactly one form, and because they
    are declarations about the surrounding program rather than operations
    on an outstanding delivery: a fence orders by effect class, a
    selection states which Boolean decides acceptance.

    Every operation except `event.yield` accepts an optional `event_id`,
    a stable identity for tiers that place these operations into a
    schedule or an event graph, where an out-of-band sidecar must be able
    to name one occurrence. It is absent by default and nothing here
    reads it; the tier that requires it -- P3, where every schedulable
    operation in a graph carries one and the graph verifier uniques them
    -- checks it itself.
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // EVENT_DIALECT_TD
