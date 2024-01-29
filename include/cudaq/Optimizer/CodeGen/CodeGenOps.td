/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_CODEGEN_OPS
#define CUDAQ_OPTIMIZER_CODEGEN_OPS

include "cudaq/Optimizer/CodeGen/CodeGenDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "cudaq/Optimizer/Dialect/Common/Traits.td"
include "cudaq/Optimizer/Dialect/CC/CCTypes.td"
include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.td"

//===----------------------------------------------------------------------===//
// The codegen quake dialect is a transitory set of operations used exclusively
// during codegen. The ops defined here make the process of converting the Quake
// code to a another target dialect (e.g. LLVM-IR) easier. They should not be
// used outside of the codegen passes.
//===----------------------------------------------------------------------===//

class CGQOp<string mnemonic, list<Trait> traits = []>
    : Op<CodeGenDialect, mnemonic, traits>;

def cgq_RAIIOp : CGQOp<"qmem_raii", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Combine allocation and initialization of set of qubits.";
  let description = [{
    Used only in QIR code generation.
  }];

  let arguments = (ins
    cc_PointerType:$initState,
    TypeAttr:$allocType,
    Optional<AnySignlessIntegerOrIndex>:$allocSize
  );
  let results = (outs VeqType);

  let assemblyFormat = [{
    $initState `(` $allocType ( `[` $allocSize^ `]` )? `)` `:`
    functional-type(operands, results) attr-dict
  }];
}

#endif
