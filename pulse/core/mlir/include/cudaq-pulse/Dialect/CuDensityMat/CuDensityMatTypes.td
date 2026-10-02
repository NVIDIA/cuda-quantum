/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_CUDENSITYMAT_TYPES
#define CUDAQ_PULSE_DIALECT_CUDENSITYMAT_TYPES

include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.td"
include "mlir/IR/AttrTypeBase.td"

class CuDensityMatType<string name, string typeMnemonic,
                       list<Trait> traits = [],
                       string baseCppClass = "mlir::Type">
    : TypeDef<CuDensityMatDialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
  let genStorageClass = 0;
}

def CUDM_HandleType : CuDensityMatType<"Handle", "handle"> {
  let summary = "cuDensityMat library context handle";
}

def CUDM_WorkspaceType : CuDensityMatType<"Workspace", "workspace"> {
  let summary = "Workspace buffer descriptor";
}

def CUDM_StateType : CuDensityMatType<"State", "state"> {
  let summary = "Quantum state (pure or mixed, optionally batched)";
}

def CUDM_ElementaryOpType : CuDensityMatType<"ElementaryOp", "elementary_op"> {
  let summary = "Elementary tensor operator acting on specific modes";
}

def CUDM_MatrixOpType : CuDensityMatType<"MatrixOp", "matrix_op"> {
  let summary = "Full matrix operator acting on all modes";
}

def CUDM_OpTermType : CuDensityMatType<"OpTerm", "op_term"> {
  let summary = "Operator term: sum of products of elementary/matrix operators";
}

def CUDM_OperatorType : CuDensityMatType<"Operator", "operator"> {
  let summary = "Composite operator / super-operator";
}

def CUDM_OpActionType : CuDensityMatType<"OpAction", "op_action"> {
  let summary = "Aggregate operator action for coupled ODE systems";
}

def CUDM_ExpectationType : CuDensityMatType<"Expectation", "expectation"> {
  let summary = "Expectation value computation handle";
}

def CUDM_SpectrumType : CuDensityMatType<"Spectrum", "spectrum"> {
  let summary = "Operator eigenspectrum computation handle";
}

def CUDM_DecoherenceModelType
    : CuDensityMatType<"DecoherenceModel", "decoherence_model"> {
  let summary = "T1/T2 decoherence parameters per qubit mode";
}

def CUDM_TrajectoryResultType
    : CuDensityMatType<"TrajectoryResult", "trajectory_result"> {
  let summary = "Ensemble results from the SSE quantum trajectory solver";
}

#endif // CUDAQ_PULSE_DIALECT_CUDENSITYMAT_TYPES
