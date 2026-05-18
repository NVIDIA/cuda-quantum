/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QEC_IR_QEC
#define CUDAQ_OPTIMIZER_DIALECT_QEC_IR_QEC

include "mlir/IR/OpBase.td"

def QECDialect : Dialect {
  let name = "qec";
  let summary = "QEC operations for CUDA-Q (detectors, logical observables).";
  let description = [{
    The QEC dialect houses quantum error correction operations such as detector
    declarations and logical observable declarations. These are separate from
    the Quake dialect to keep Quake focused on circuit semantics while giving
    QEC annotations a dedicated namespace.
  }];
  let cppNamespace = "cudaq::qec";
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QEC_IR_QEC
