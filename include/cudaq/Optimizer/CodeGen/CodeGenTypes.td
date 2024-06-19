/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_CODEGEN_TYPES
#define CUDAQ_OPTIMIZER_CODEGEN_TYPES

include "cudaq/Optimizer/CodeGen/CodeGenDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// BaseType
//===----------------------------------------------------------------------===//

class CodeGenType<string name, string typeMnemonic, list<Trait> traits = [],
             string base = "mlir::Type">
    : TypeDef<CodeGenDialect, name, traits, base> {
  let mnemonic = typeMnemonic;
}

// There are no codegen dialect types, but the dialect tablegen generates type
// boilerplate for the dialect anyway.

def codegen_DoNotUseType : CodeGenType<"DoNotUse", "dummy"> {
  let summary = "";
  let description = [{ }];
}

#endif // CUDAQ_OPTIMIZER_CODEGEN_TYPES
