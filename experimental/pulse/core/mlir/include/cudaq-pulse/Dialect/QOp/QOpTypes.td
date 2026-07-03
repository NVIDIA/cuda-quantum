/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_QOP_TYPES
#define CUDAQ_PULSE_DIALECT_QOP_TYPES

include "cudaq-pulse/Dialect/QOp/QOpDialect.td"
include "mlir/IR/AttrTypeBase.td"

class QOpType<string name, string typeMnemonic, list<Trait> traits = [],
              string baseCppClass = "mlir::Type">
    : TypeDef<QOpDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def QOP_ScalarType : QOpType<"Scalar", "scalar"> {
  let summary = "Scalar coefficient (constant or time/parameter-dependent)";
  let genStorageClass = 0;
}

def QOP_HandlerType : QOpType<"Handler", "handler"> {
  let summary = "Leaf operator acting on specific degrees of freedom";
  let genStorageClass = 0;
}

def QOP_ProductType : QOpType<"Product", "product"> {
  let summary = "Tensor product of handlers with a scalar coefficient";
  let genStorageClass = 0;
}

def QOP_OpType : QOpType<"Op", "op"> {
  let summary = "Sum of products -- the general quantum operator";
  let genStorageClass = 0;
}

def QOP_SuperOpType : QOpType<"SuperOp", "super_op"> {
  let summary = "Super-operator for density matrix evolution";
  let genStorageClass = 0;
}

#endif // CUDAQ_PULSE_DIALECT_QOP_TYPES
