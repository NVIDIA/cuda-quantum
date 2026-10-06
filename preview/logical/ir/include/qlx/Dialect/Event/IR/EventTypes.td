/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
//   !event.handle<payload, ownership, stream?> - one outstanding
//     asynchronous delivery of a typed payload.
//
//===----------------------------------------------------------------------===//

#ifndef EVENT_TYPES_TD
#define EVENT_TYPES_TD

include "qlx/Dialect/Event/IR/EventDialect.td"
include "qlx/Dialect/Event/IR/EventInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

def Event_HandleType : TypeDef<Event_Dialect, "Handle"> {
  let mnemonic = "handle";
  let summary = "one outstanding asynchronous delivery of a typed payload";
  let description = [{
    One outstanding asynchronous delivery: something requested whose
    result is not available yet. A handle is an ordinary value and not a
    location -- nothing is stored through it, and the operations that
    consume it take its ownership rather than mutating it. It is what
    every tier's `resource_request`/`request` producer returns and what
    every `event.*` operation observes or consumes.

    `payload` is what a successful delivery yields: the type `event.await`
    returns, and the type `event.try_take`'s `ready` region receives. Each
    tier instantiates it with its own resource type.

    `ownership` states how the delivered payload may be used. `"linear"`
    means it has exactly one owner and must be consumed exactly once;
    `"shared"` means it may be copied once ready. The parameter is an open
    string, checked only for non-emptiness, because which disciplines
    exist is a tier's decision rather than this dialect's. `event.await`,
    `event.try_take` and `event.cancel` each reject anything but
    `"linear"`, so a `"shared"` handle can be observed but has no
    consuming operation here today.

    `stream` optionally names the machine stream or provider the request
    was placed against, for payload types that do not already name it.
    Carrying it on the type keeps the binding with the value: a consumer
    reads it without locating the producer, and a handle passed across a
    region boundary does not lose it. No verifier reads it back.

    Example:
    ```mlir
    !event.handle<!qlx.logical_resource<"t_state">, "linear">
    !event.handle<!lvm.logical_resource<"t_state", @m::@s>, "linear", @m::@s>
    ```
  }];
  let parameters = (ins
    "mlir::Type":$payload,
    StringRefParameter<"ownership discipline of the handle">:$ownership,
    OptionalParameter<"mlir::SymbolRefAttr">:$stream
  );
  let assemblyFormat = "`<` $payload `,` $ownership (`,` $stream^)? `>`";
  let genVerifyDecl = 1;
}

#endif // EVENT_TYPES_TD
