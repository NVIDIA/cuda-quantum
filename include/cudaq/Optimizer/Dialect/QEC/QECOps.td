/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
#define CUDAQ_OPTIMIZER_DIALECT_QEC_OPS

include "cudaq/Optimizer/Dialect/CC/CCTypes.td"
include "cudaq/Optimizer/Dialect/QEC/QECDialect.td"
include "mlir/IR/OpBase.td"

class QECOp<string mnemonic, list<Trait> traits = []>
    : Op<QECDialect, mnemonic, traits>;

def QEC_DetectorOp : QECOp<"detector"> {
  let summary = "Declare a detector over measurement results.";
  let description = [{
    Declares a parity constraint over one or more measurement results.
    Under noise-free execution the XOR of the referenced measurements
    is deterministic. Operands may be individual `!cc.measure_handle`
    values or a `!cc.stdvec<!cc.measure_handle>` collection. This is a
    side-effecting declaration that informs the backend.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[cc_MeasureHandleType,
                        StdvecOf<[cc_MeasureHandleType]>]>>:$measurements
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` attr-dict `:` `(` type($measurements) `)`
  }];
}

def QEC_DetectorsOp : QECOp<"detectors"> {
  let summary = "Declare N detectors by pairing previous and current measurement vectors.";
  let description = [{
    Given two `!cc.stdvec<!cc.measure_handle>` collections of measurement
    results, declares N detectors where detector `i` is the parity of
    `prev[i]` and `curr[i]`. `!cc.stdvec` carries no static size, so size
    agreement between `prev` and `curr` is checked at runtime.
  }];
  let arguments = (ins
    StdvecOf<[cc_MeasureHandleType]>:$prev,
    StdvecOf<[cc_MeasureHandleType]>:$curr
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $prev `,` $curr `)` attr-dict `:`
    `(` qualified(type($prev)) `,` qualified(type($curr)) `)`
  }];
}

def QEC_ObservableOp : QECOp<"observable"> {
  let summary = "Declare a logical observable over measurement results.";
  let description = [{
    Declares a logical observable as a binary sum of measurement results.
    Operands may be individual `!cc.measure_handle` values or a
    `!cc.stdvec<!cc.measure_handle>` collection.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[cc_MeasureHandleType,
                        StdvecOf<[cc_MeasureHandleType]>]>>:$measurements,
    DefaultValuedOptionalAttr<I64Attr, "0">:$observableIndex
  );
  let results = (outs);
  let assemblyFormat = [{
    `(` $measurements `)` (`index` $observableIndex^)? attr-dict `:` `(` type($measurements) `)`
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QEC_OPS
