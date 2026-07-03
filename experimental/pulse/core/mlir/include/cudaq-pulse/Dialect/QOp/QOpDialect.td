/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_QOP
#define CUDAQ_PULSE_DIALECT_QOP

include "mlir/IR/OpBase.td"

def QOpDialect : Dialect {
  let name = "qop";
  let summary = "Backend-agnostic quantum operator algebra dialect";
  let description = [{
    The `qop` dialect captures the algebraic structure of quantum operators
    (Hamiltonians, Lindbladians, observables) as first-class MLIR values.

    Key concepts:
    - **Handlers**: Leaf operators (Pauli, bosonic, fermionic, matrix)
    - **Products**: Tensor products of handlers with scalar coefficients
    - **Sums**: Linear combinations of products (the general operator)
    - **Super-operators**: Left/right action pairs for master equations
  }];
  let cppNamespace = "::qop";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // CUDAQ_PULSE_DIALECT_QOP
