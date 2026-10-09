/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_LOGICAL_INTERFACES_SEMANTICINTERFACES_TD
#define CUDAQ_LOGICAL_INTERFACES_SEMANTICINTERFACES_TD

include "mlir/IR/OpBase.td"

def CUDAQLogical_SemanticRootOpInterface
    : OpInterface<"SemanticRootOpInterface"> {
  let cppNamespace = "::cudaq::logical";
  let description = [{
    Identifies a canonical executable root and reports the semantic commitment
    represented by that root. The answers are intrinsic properties of the
    operation class. They are not attributes to serialize alongside the root.
  }];
  let methods = [
    InterfaceMethod<
      "Return the P0--P4 commitment represented by this root.",
      "::cudaq::logical::Stage", "getSemanticStage"
    >,
    InterfaceMethod<
      "Return the root's semantic role without inspecting its dialect name.",
      "::cudaq::logical::RootKind", "getSemanticRootKind"
    >
  ];
}

#endif // CUDAQ_LOGICAL_INTERFACES_SEMANTICINTERFACES_TD
