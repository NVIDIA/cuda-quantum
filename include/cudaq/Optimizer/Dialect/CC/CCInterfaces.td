/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_CC_IR_INTERFACES
#define CUDAQ_OPTIMIZER_DIALECT_CC_IR_INTERFACES

include "mlir/IR/OpBase.td"

def LinearTypeArgsTrait : NativeOpTrait<"LinearTypeArgsTrait"> {
  let cppNamespace = "::cudaq::cc";
}

#endif // CUDAQ_OPTIMIZER_DIALECT_CC_IR_INTERFACES
