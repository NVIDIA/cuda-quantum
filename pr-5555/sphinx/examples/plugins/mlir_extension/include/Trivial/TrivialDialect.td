/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef TRIVIAL_DIALECT
#define TRIVIAL_DIALECT

include "mlir/IR/OpBase.td"

def Trivial_Dialect : Dialect {
  let name = "trivial";
  let summary = "A minimal out-of-tree dialect used to validate the devel wheel.";
  let cppNamespace = "::trivial";
}

class Trivial_Op<string mnemonic, list<Trait> traits = []>
    : Op<Trivial_Dialect, mnemonic, traits>;

#endif // TRIVIAL_DIALECT
