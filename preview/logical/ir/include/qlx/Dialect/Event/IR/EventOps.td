/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef EVENT_OPS_TD
#define EVENT_OPS_TD

include "qlx/Dialect/Event/IR/EventDialect.td"
include "qlx/Dialect/Event/IR/EventTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Event_Op<string mnemonic, list<Trait> traits = []> :
    Op<Event_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TestOp
//===----------------------------------------------------------------------===//

def Event_TestOp : Event_Op<"test", []> {
  let summary = "read whether an event's payload can be taken now";
  let description = [{
    Nonblocking, non-consuming readiness read: `ready` is true exactly
    when `event`'s payload can be taken, which is the `ready` state and
    nothing else.

    Pending, failed, cancelled and exhausted all read false, so a false
    result does not distinguish "not yet" from "never", and a retry loop
    driven by this operation alone cannot learn that it should stop
    retrying. Use `event.poll` with `event.is` where that difference
    decides control flow, and this operation where only the single bit
    matters.
  }];
  let arguments = (ins Event_HandleType:$event, OptionalAttr<StrAttr>:$event_id);
  let results = (outs I1:$ready);
  let assemblyFormat = "$event attr-dict `:` qualified(type($event)) `->` type($ready)";
}

//===----------------------------------------------------------------------===//
// PollOp
//===----------------------------------------------------------------------===//

def Event_PollOp : Event_Op<"poll", []> {
  let summary = "read an event's complete state, without consuming it";
  let description = [{
    Nonblocking, non-consuming read of `event`'s complete state as an
    `i8`: one of `pending`, `ready`, `failed`, `cancelled` or `exhausted`.

    The numeric encoding is deliberately unspecified. `event.is` names the
    states, and a program that compares the `i8` directly has pinned an
    encoding this dialect does not promise. The result is an ordinary
    copyable value that may be tested any number of times; it is a reading
    taken at this point in the program, not a live view of the handle.
  }];
  let arguments = (ins Event_HandleType:$event, OptionalAttr<StrAttr>:$event_id);
  let results = (outs I8:$status);
  let assemblyFormat = "$event attr-dict `:` qualified(type($event)) `->` type($status)";
}

//===----------------------------------------------------------------------===//
// IsOp
//===----------------------------------------------------------------------===//

def Event_IsOp : Event_Op<"is", [Pure]> {
  let summary = "test a polled status against one named state";
  let description = [{
    True when `status`, a state as produced by `event.poll`, denotes
    `state`, which must be one of `pending`, `ready`, `failed`, `cancelled`
    or `exhausted`. Any other spelling fails verification.

    Naming the state keeps `event.poll`'s encoding out of programs and
    analyses, and gives the vocabulary somewhere to be checked: a mistyped
    state is rejected here, where an integer comparison against a
    hand-written constant would have compiled and silently never matched.

    `status` is any `i8`. Nothing checks that it came from an `event.poll`,
    or from a poll of the handle the surrounding code reasons about.
  }];
  let arguments = (ins I8:$status, StrAttr:$state, OptionalAttr<StrAttr>:$event_id);
  let results = (outs I1:$matches);
  let assemblyFormat = "$status $state attr-dict `:` type($status) `->` type($matches)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SelectReadyOp
//===----------------------------------------------------------------------===//

def Event_SelectReadyOp : Event_Op<"select_ready", [SameTypeOperands]> {
  let summary = "choose one ready event among several, without consuming any";
  let description = [{
    Nonblocking, non-consuming choice among `events`: `selected` is the
    operand index of a handle that is ready, or the number of operands when
    none of them is. That count is how "nothing is ready" is reported, and
    it is why the result is an index rather than a handle -- returning a
    handle would hand back an owner the caller already holds.

    `policy` is part of what the program means rather than an
    implementation detail, because with several requests outstanding the
    choice decides which producer gets drained: `priority` takes the
    earliest ready operand, operand order being the only priority this
    operation carries; `deterministic` requires the same readiness pattern
    to always select the same operand; `fair` forbids starving one operand
    across repeated selections. Verification checks only that the name is
    one of the three -- honouring the policy is the consuming tier's
    obligation.

    All operands share one handle type, so a caller can act on the selected
    index without a type test. Choosing among differently typed requests
    means polling them separately.
  }];
  let arguments = (ins
    Variadic<Event_HandleType>:$events,
    DefaultValuedAttr<StrAttr, "\"priority\"">:$policy,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Index:$selected);
  let assemblyFormat = "`(` $events `)` attr-dict `:` qualified(type($events)) `->` type($selected)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TryTakeOp
//===----------------------------------------------------------------------===//

def Event_TryTakeOp : Event_Op<"try_take", [
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "take an event's payload if it is ready, branching on the outcome";
  let description = [{
    Nonblocking counterpart to `event.await`, shaped as a three-way
    ownership match. `event` is consumed, and exactly one region runs,
    chosen by the handle's state:

    * `ready` receives the payload. The handle is gone and the delivery is
      the region's to own.
    * `pending` receives a handle of the same type: a successor of the one
      consumed, still outstanding, so a caller can retry later. Nothing is
      lost by the attempt, but the retry goes through the new owner rather
      than the operand staying live alongside it.
    * `failed` receives the `i8` status, read with `event.is`, for the
      states no retry resolves.

    Each region's remaining block arguments are `carries`, in order and by
    type, and each terminates with `event.yield` whose operands have
    exactly this operation's result types -- which are exactly the carry
    types. Every path out of the match therefore leaves it owning the same
    things, and that is what makes a branch on readiness usable for a
    linear payload at all. The obvious alternative shape, one operation
    returning a readiness bit beside an optional payload, would produce a
    payload that is valid on one path while the handle stays live on the
    other, which is two candidate owners of one delivery. Handling `failed`
    explicitly, rather than as another flavour of "not ready", falls out of
    the same shape.

    `event`'s ownership must be `"linear"`. A carry type implementing
    `UniqueCarryOwnerInterface` is rejected when another carry reports an
    equal owner key; a carry type that does not implement it is not
    checked.
  }];
  let arguments = (ins
    Event_HandleType:$event,
    Variadic<AnyType>:$carries,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$ready,
                         SizedRegion<1>:$pending,
                         SizedRegion<1>:$failed);
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CancelOp
//===----------------------------------------------------------------------===//

def Event_CancelOp : Event_Op<"cancel", []> {
  let summary = "withdraw an outstanding event and consume its ownership";
  let description = [{
    Withdraws the request `event` stands for and consumes the handle, whose
    ownership must be `"linear"`. The payload is never delivered to the
    caller: a handle the producer had already resolved is consumed all the
    same, and its delivery is dropped with it.

    That is why cancelling reports a state rather than nothing. `status`
    uses `event.poll`'s encoding, so a caller can distinguish a
    cancellation that took effect from one that lost the race with its
    producer, and can account for a delivery that was dropped.

    `reason` records a human-readable cause for diagnostics and provenance.
    It is never interpreted and never verified.
  }];
  let arguments = (ins
    Event_HandleType:$event,
    OptionalAttr<StrAttr>:$reason,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs I8:$status);
  let assemblyFormat = "$event attr-dict `:` qualified(type($event)) `->` type($status)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

def Event_AwaitOp : Event_Op<"await", [
    TypesMatchWith<"result type equals the event payload type",
      "event", "payload",
      "::llvm::cast<::qlx::event::HandleType>($_self).getPayload()">
  ]> {
  let summary = "block until an event resolves, then consume it for its payload";
  let description = [{
    Blocks until `event` resolves, consumes it, and yields the delivered
    payload. The result type is the handle's declared payload type, and
    `event`'s ownership must be `"linear"`.

    There is no failure path. Awaiting commits the program to a delivery,
    so a handle that fails, is cancelled or is exhausted has no outcome
    this operation can report. A program that must react to those states
    polls first, or uses `event.try_take`, whose `failed` region exists for
    exactly this case.

    Blocking is a dependence, not a duration: the wait orders the
    continuation that uses the payload, work that does not depend on the
    payload is not ordered against it, and nothing here names how long the
    wait takes.
  }];
  let arguments = (ins Event_HandleType:$event, OptionalAttr<StrAttr>:$event_id);
  let results = (outs AnyType:$payload);
  let assemblyFormat = "$event attr-dict `:` qualified(type($event)) `->` type($payload)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FenceOp
//===----------------------------------------------------------------------===//

def Event_FenceOp : Event_Op<"fence", []> {
  let summary = "order the operations that touch declared effect classes";
  let description = [{
    Orders operations by what they touch. Operations whose effects
    intersect `effects` may not move across this one; operations that touch
    none of the declared classes still reorder freely past it. `effects`
    draws from `quantum`, `classical`, `resource`, `event`, `frame`,
    `outcome` and `selection`, or the single entry `all` for every class.
    At least one entry is required, entries must be unique, and `all` may
    not be combined with another.

    Naming the classes, instead of fencing everything, keeps the constraint
    as narrow as the dependence the program actually has: a fence over
    `resource` and `event` says nothing about classical bookkeeping, and a
    tier lowering it must not widen it into a device barrier. This
    operation has no duration, is not a delay, and does not synchronize
    clock or resource domains.

    No operation in this dialect declares memory effects, so a generic MLIR
    transform sees a fence as having unknown effects and treats it
    conservatively rather than honouring `effects` precisely. A pass that
    wants the narrow ordering reads the attribute itself.
  }];
  let arguments = (ins StrArrayAttr:$effects, OptionalAttr<StrAttr>:$event_id);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SelectionOp
//===----------------------------------------------------------------------===//

def Event_SelectionOp : Event_Op<"selection", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "declare which Boolean decides acceptance, and how to apply it";
  let description = [{
    Declares that `predicate` is the acceptance decision for the
    surrounding attempt or experiment, and how that decision is meant to be
    applied:

    * `require` -- only accepted runs are meaningful; a rejected run is
      discarded.
    * `condition_results` -- accepted and rejected runs are both kept, and
      reported statistics are conditioned on acceptance.
    * `abort_on` -- `predicate` names the failure condition, and acceptance
      is its negation.

    `accept_when` is the polarity the mode implies, true for `require` and
    `condition_results` and false for `abort_on`, and is verified to agree
    with it. The redundancy is deliberate: a consumer reads the polarity
    from one attribute without carrying the mode vocabulary.

    Nothing about this operation runs. It has no result, gates no
    operation, and neither retries nor aborts anything; a tier that acts on
    acceptance does so in its own operations, such as a bounded retry over
    a success predicate, or after the run in postprocessing. It is in the
    IR because acceptance changes what the surrounding numbers mean: an
    error rate measured under postselection is a conditional rate, and a
    consumer that never saw this declaration would report it as an
    unconditional one.

    `attempt` optionally names the realization attempt whose success
    predicate this is, and `profile` the analysis that defines that
    predicate; `profile` requires `attempt`. Verification here only checks
    that whichever are present resolve to some symbol, because this dialect
    cannot know what kind of symbol a tier's attempts and analyses are. A
    tier that does know is the only place that can check that `predicate`
    really is that attempt's success predicate.
  }];
  let arguments = (ins
    I1:$predicate,
    StrAttr:$mode,
    DefaultValuedAttr<BoolAttr, "true">:$accept_when,
    OptionalAttr<FlatSymbolRefAttr>:$attempt,
    OptionalAttr<FlatSymbolRefAttr>:$profile,
    OptionalAttr<StrAttr>:$event_id
  );
  let assemblyFormat = "$predicate attr-dict `:` type($predicate)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Event_YieldOp : Event_Op<"yield", [
    Pure, ReturnLike, Terminator,
    HasParent<"TryTakeOp">
  ]> {
  let summary = "terminate one try_take region, yielding its results";
  let description = [{
    Terminates one of `event.try_take`'s `ready`, `pending` and `failed`
    regions, handing `operands` to the parent as that branch's results.
    Operand types are exactly the parent's result types, so every branch of
    the match leaves it owning the same things.

    Alone among this dialect's operations it takes no `event_id`: it is the
    boundary of a region, not an action a schedule can place.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

#endif // EVENT_OPS_TD
