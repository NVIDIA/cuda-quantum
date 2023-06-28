/***********************************************************-*- tablegen -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
class NumParameters<int n> : StructuralOpTrait,
    PredOpTrait<"the number of parameters is equal to " # n,
                CPred<"$parameters.size() == " # n>>;

class NumTargets<int n> : StructuralOpTrait,
    PredOpTrait<"the number of targets is equal to " # n,
                CPred<"$targets.size() == " # n>>;


#endif // CUDAQ_OPTIMIZER_DIALECT_COMMON_TRAITS

