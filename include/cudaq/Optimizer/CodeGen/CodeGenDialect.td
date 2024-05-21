/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_CODEGEN_DIALECT
#define CUDAQ_OPTIMIZER_CODEGEN_DIALECT

include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def CodeGenDialect : Dialect {
  let name = "codegen";
  let summary = "Code generation helpers";
  let description = [{
    Do not use this dialect outside of code generation.
  }];

  let cppNamespace = "cudaq::codegen";
  let useDefaultTypePrinterParser = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;

  let extraClassDeclaration = [{
    void registerTypes(); // register at least a bogo type.
  }];
}

#endif
