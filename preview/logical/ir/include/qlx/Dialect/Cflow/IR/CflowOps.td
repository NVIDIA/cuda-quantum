/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CFLOW_OPS_TD
#define CFLOW_OPS_TD

include "qlx/Dialect/Cflow/IR/CflowDialect.td"
include "mlir/IR/OpBase.td"

class Cflow_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cflow_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

def Cflow_RepeatOp : Cflow_Op<"repeat", [
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Repeat a body N times with loop-carried values";
  let description = [{
    Executes its body region `count` times, threading values through iterations
    via iter-args. Each iteration receives the previous iteration's yielded
    values as block arguments; the op's results are the values yielded by the
    final iteration.

    `count` is a compile-time constant, not an SSA value: there is no runtime
    trip count to evaluate, so this op stays folded rather than materialized.
    A pass computing resource cost or schedule length may multiply through it
    analytically instead of unrolling every iteration.

    `event_id` is an optional schedule/idle-event join key, set only by a caller
    that needs to correlate this op with a physical schedule entry; most callers
    leave it absent.

    Example:
    ```mlir
    %q_out = cflow.repeat 64 iter(%qi : !some.type = %q_in) {
      %qn = "some.op"(%qi) : (!some.type) -> !some.type
      cflow.yield %qn : !some.type
    }
    ```
  }];

  let arguments = (ins
    ConfinedAttr<I64Attr, [IntNonNegative]>:$count,
    Variadic<AnyType>:$inits,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Cflow_YieldOp : Cflow_Op<"yield", [
    Terminator,
    ParentOneOf<["RepeatOp", "IfOp", "WhileOp"]>
  ]> {
  let summary = "Yield values from a cflow.repeat/if/while body";
  let description = [{
    Terminates the body of `cflow.repeat`, one branch of `cflow.if`, or the
    `after_region` of `cflow.while`, yielding values back to the parent op.

    A dialect that also has an `event_try_take` op terminates it with a
    separate, dialect-local yield operation of its own: this op does not accept
    `event_try_take` as a parent and is never that terminator.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::ValueRange{});
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// WhileConditionOp
//===----------------------------------------------------------------------===//

def Cflow_WhileConditionOp : Cflow_Op<"while_condition", [
    Terminator, HasParent<"WhileOp">
  ]> {
  let summary = "Decide whether a dynamic loop continues";
  let description = [{
    Terminates the `before` region of `cflow.while`. `condition` controls entry
    into the `after` body and `forwarded` supplies the body's complete carried
    state. The forwarded values have exactly the while's result types, so every
    linear owner has one successor on both the continue and exit edges.
  }];
  let arguments = (ins I1:$condition, Variadic<AnyType>:$forwarded);
  let assemblyFormat =
    "$condition (`,` $forwarded^ `:` type($forwarded))? attr-dict";
}

//===----------------------------------------------------------------------===//
// WhileOp -- dynamic structured loop
//===----------------------------------------------------------------------===//

def Cflow_WhileOp : Cflow_Op<"while", [SingleBlock]> {
  let summary = "Dynamic loop with typed linear carries";
  let description = [{
    A two-region loop: the `before` region receives the current carries and ends
    in `cflow.while_condition`; when true, its forwarded values enter the
    `after` body. The body ends in `cflow.yield` and feeds the next condition
    evaluation. `max_iterations` is an optional semantic bound, not an
    instruction to unroll the loop.

    Unlike `repeat`, this is genuinely dynamic control flow: the number of
    iterations is not known at compile time, and the condition is re-evaluated
    at runtime on each pass (for example, a repeat-until-success protocol that
    loops until a measurement succeeds).

    `event_id` is an optional schedule/idle-event join key, set only by a caller
    that needs to correlate this op with a physical schedule entry; most callers
    leave it absent.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inits,
    OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$max_iterations,
    OptionalAttr<StrAttr>:$event_id
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$before_region,
    SizedRegion<1>:$after_region
  );
  let assemblyFormat = [{
    `(` $inits `)` (`max_iterations` `=` $max_iterations^)?
    (`event_id` `=` $event_id^)? attr-dict `:`
    functional-type($inits, $results)
    $before_region `do` $after_region
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp -- conditional execution
//===----------------------------------------------------------------------===//

def Cflow_IfOp : Cflow_Op<"if", [
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Conditional execution on a classical bit";
  let description = [{
    Executes one of two single-block regions depending on `condition`. The op
    returns the values yielded by the selected branch.

    This is the shape used for measurement-dependent feed-forward (e.g. Pauli
    byproduct corrections) -- genuinely dynamic control flow: exactly one
    branch's operations physically execute, decided at runtime by whatever
    produced `condition`.

    Branch regions may reference values defined outside the op. When a branch
    updates a linear value, the updated value should be yielded and threaded
    through the op result -- exactly one dynamic consume per execution, since
    only one branch ever runs.

    `event_id` is an optional schedule/idle-event join key, set only by a caller
    that needs to correlate this op with a physical schedule entry; most callers
    leave it absent.
  }];

  let arguments = (ins I1:$condition, OptionalAttr<StrAttr>:$event_id);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$then_region,
    SizedRegion<1>:$else_region
  );

  let assemblyFormat = [{
    $condition (`event_id` `=` $event_id^)? (`->` type($results)^)?
    attr-dict-with-keyword $then_region `else` $else_region
  }];
  let hasRegionVerifier = 1;
}

#endif // CFLOW_OPS_TD
