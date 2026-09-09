/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_COMMON_TRAITS
#define CUDAQ_OPTIMIZER_DIALECT_COMMON_TRAITS

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Quantum operation traits
//===----------------------------------------------------------------------===//

def QuantumGate : NativeOpTrait<"QuantumGate"> {
  let cppNamespace = "::cudaq";
}

def Hermitian : NativeOpTrait<"Hermitian"> {
  let cppNamespace = "::cudaq";
}

def Rotation : NativeOpTrait<"Rotation"> {
  let cppNamespace = "::cudaq";
}

def QuantumMeasure : NativeOpTrait<"QuantumMeasure"> {
  let cppNamespace = "::cudaq";
}

// Marks an op that is not itself a QuantumGate or QuantumMeasure but that
// implicitly dereferences a `!quake.ref`/`!quake.veq` operand and modifies
// the wire(s) it refers to (typically by macro-expanding to, or invoking,
// one or more quantum operators), e.g. quake.apply, quake.compute_action,
// quake.apply_noise, quake.call_by_ref, quake.evince.
def QuantumSideEffects : NativeOpTrait<"QuantumSideEffects"> {
  let cppNamespace = "::cudaq";
}

//===----------------------------------------------------------------------===//
// Classical compute operation traits
//===----------------------------------------------------------------------===//

def JumpWithUnwind : NativeOpTrait<"JumpWithUnwind"> {
  let cppNamespace = "::cudaq";
}

//===----------------------------------------------------------------------===//
// Common op constraints
//===----------------------------------------------------------------------===//

// Note: In the following it seems that `size_t` is not acceptable as a type.
// Hence this is the reason why we use `int` instead.
// TODO: investigate further
//
// A negative `n` means the count is unconstrained (variadic); the predicate
// is trivially satisfied so callers can always instantiate `NumParameters<n>`
// / `NumTargets<n>` uniformly without conditionally omitting the trait.
class NumParameters<int n> : StructuralOpTrait,
    PredOpTrait<"the number of parameters is equal to " # n,
                CPred<!if(!ge(n, 0), "$parameters.size() == " # n, "true")>>;

class NumTargets<int n> : StructuralOpTrait,
    PredOpTrait<"the number of targets is equal to " # n,
                CPred<!if(!ge(n, 0), "$targets.size() == " # n, "true")>>;


#endif // CUDAQ_OPTIMIZER_DIALECT_COMMON_TRAITS

