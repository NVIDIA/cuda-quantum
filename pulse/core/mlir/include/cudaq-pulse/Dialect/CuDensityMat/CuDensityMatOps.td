/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_CUDENSITYMAT_OPS
#define CUDAQ_PULSE_DIALECT_CUDENSITYMAT_OPS

include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.td"
include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.td"
include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class CuDensityMatOp<string mnemonic, list<Trait> traits = []> :
    Op<CuDensityMatDialect, mnemonic, traits>;

// ---- Context Management ----

def CUDM_InitHandleOp : CuDensityMatOp<"init_handle", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create cuDensityMat library context";
  let results = (outs CUDM_HandleType:$handle);
  let assemblyFormat = "attr-dict `:` type($handle)";
}

def CUDM_DestroyHandleOp : CuDensityMatOp<"destroy_handle", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy cuDensityMat library context";
  let arguments = (ins CUDM_HandleType:$handle);
  let assemblyFormat = "$handle attr-dict `:` type($handle)";
}

// ---- Workspace Management ----

def CUDM_CreateWorkspaceOp : CuDensityMatOp<"create_workspace", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create a workspace descriptor";
  let arguments = (ins CUDM_HandleType:$handle);
  let results = (outs CUDM_WorkspaceType:$workspace);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, $workspace)";
}

def CUDM_DestroyWorkspaceOp : CuDensityMatOp<"destroy_workspace", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy a workspace descriptor";
  let arguments = (ins CUDM_WorkspaceType:$workspace);
  let assemblyFormat = "$workspace attr-dict `:` type($workspace)";
}

// ---- Quantum State ----

def CUDM_CreateStateOp : CuDensityMatOp<"create_state", [MemoryEffects<[MemWrite]>]> {
  let summary = "Define an empty quantum state of given purity and shape";
  let arguments = (ins
    CUDM_HandleType:$handle,
    CUDM_StatePurityAttr:$purity,
    CUDM_ComputeTypeAttr:$data_type,
    DenseI64ArrayAttr:$mode_extents,
    DefaultValuedAttr<I64Attr, "0">:$batch_size,
    DefaultValuedAttr<BoolAttr, "false">:$distributed
  );
  let results = (outs CUDM_StateType:$state);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, $state)";
  let hasVerifier = 1;
}

def CUDM_DestroyStateOp : CuDensityMatOp<"destroy_state", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy a quantum state";
  let arguments = (ins CUDM_StateType:$state);
  let assemblyFormat = "$state attr-dict `:` type($state)";
}

// ---- Elementary Operator ----

def CUDM_CreateElementaryOpOp
    : CuDensityMatOp<"create_elementary_op", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create an elementary tensor operator";
  let arguments = (ins
    CUDM_HandleType:$handle,
    AnyType:$tensor_data,
    CUDM_SparsityAttr:$sparsity,
    CUDM_ComputeTypeAttr:$data_type,
    DenseI64ArrayAttr:$mode_extents,
    OptionalAttr<FlatSymbolRefAttr>:$callback
  );
  let results = (outs CUDM_ElementaryOpType:$elem_op);
  let assemblyFormat = "$handle `,` $tensor_data attr-dict `:` type($handle) `,` type($tensor_data) `->` type($elem_op)";
  let hasVerifier = 1;
}

def CUDM_DestroyElementaryOpOp
    : CuDensityMatOp<"destroy_elementary_op", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy an elementary tensor operator";
  let arguments = (ins CUDM_ElementaryOpType:$elem_op);
  let assemblyFormat = "$elem_op attr-dict `:` type($elem_op)";
}

// ---- Operator Term ----

def CUDM_CreateOpTermOp : CuDensityMatOp<"create_op_term", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create an empty operator term";
  let arguments = (ins CUDM_HandleType:$handle, DenseI64ArrayAttr:$mode_extents);
  let results = (outs CUDM_OpTermType:$op_term);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, $op_term)";
}

def CUDM_AppendElementaryProductOp
    : CuDensityMatOp<"append_elementary_product", [MemoryEffects<[MemWrite]>]> {
  let summary = "Append a product of elementary operators to a term";
  let arguments = (ins
    CUDM_HandleType:$handle,
    CUDM_OpTermType:$op_term,
    Variadic<CUDM_ElementaryOpType>:$elem_ops,
    DenseI32ArrayAttr:$modes_acted_on,
    DenseI32ArrayAttr:$duality,
    F64Attr:$coeff_real,
    F64Attr:$coeff_imag,
    OptionalAttr<FlatSymbolRefAttr>:$callback
  );
  let assemblyFormat = "$handle `,` $op_term `,` $elem_ops attr-dict `:` type($handle) `,` type($op_term) `,` type($elem_ops)";
  let hasVerifier = 1;
}

def CUDM_DestroyOpTermOp : CuDensityMatOp<"destroy_op_term", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy an operator term";
  let arguments = (ins CUDM_OpTermType:$op_term);
  let assemblyFormat = "$op_term attr-dict `:` type($op_term)";
}

// ---- Composite Operator ----

def CUDM_CreateOperatorOp : CuDensityMatOp<"create_operator", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create an empty composite operator (super-operator)";
  let arguments = (ins CUDM_HandleType:$handle, DenseI64ArrayAttr:$mode_extents);
  let results = (outs CUDM_OperatorType:$op);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, $op)";
}

def CUDM_OperatorAppendTermOp
    : CuDensityMatOp<"operator_append_term", [MemoryEffects<[MemWrite]>]> {
  let summary = "Append an operator term to a composite operator";
  let arguments = (ins
    CUDM_HandleType:$handle,
    CUDM_OperatorType:$op,
    CUDM_OpTermType:$term,
    I32Attr:$duality,
    F64Attr:$coeff_real,
    F64Attr:$coeff_imag,
    OptionalAttr<FlatSymbolRefAttr>:$callback
  );
  let assemblyFormat = "$handle `,` $op `,` $term attr-dict `:` type($handle) `,` type($op) `,` type($term)";
  let hasVerifier = 1;
}

def CUDM_DestroyOperatorOp : CuDensityMatOp<"destroy_operator", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy a composite operator";
  let arguments = (ins CUDM_OperatorType:$op);
  let assemblyFormat = "$op attr-dict `:` type($op)";
}

// ---- Operator Action ----

def CUDM_OperatorPrepareActionOp
    : CuDensityMatOp<"operator_prepare_action", [MemoryEffects<[MemWrite]>]> {
  let summary = "Prepare operator for action on quantum state";
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_OperatorType:$op,
    CUDM_StateType:$state_in, CUDM_StateType:$state_out,
    CUDM_ComputeTypeAttr:$compute_type,
    I64Attr:$workspace_limit,
    CUDM_WorkspaceType:$workspace
  );
  let assemblyFormat = "$handle `,` $op `,` $state_in `,` $state_out `,` $workspace attr-dict `:` type($handle) `,` type($op) `,` type($state_in) `,` type($state_out) `,` type($workspace)";
  let hasVerifier = 1;
}

def CUDM_OperatorComputeActionOp
    : CuDensityMatOp<"operator_compute_action", [MemoryEffects<[MemWrite]>]> {
  let summary = "Compute operator action: state_out += Op(t, params) * state_in";
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_OperatorType:$op,
    F64:$time, I64:$batch_size, I32:$num_params, AnyType:$params,
    CUDM_StateType:$state_in, CUDM_StateType:$state_out,
    CUDM_WorkspaceType:$workspace
  );
  let assemblyFormat = "$handle `,` $op `,` $time `,` $batch_size `,` $num_params `,` $params `,` $state_in `,` $state_out `,` $workspace attr-dict `:` type($handle) `,` type($op) `,` type($params) `,` type($state_in) `,` type($state_out) `,` type($workspace)";
  let hasVerifier = 1;
}

// ---- Expectation ----

def CUDM_CreateExpectationOp
    : CuDensityMatOp<"create_expectation", [MemoryEffects<[MemWrite]>]> {
  let summary = "Create expectation value computation object";
  let arguments = (ins CUDM_HandleType:$handle, CUDM_OperatorType:$op);
  let results = (outs CUDM_ExpectationType:$expectation);
  let assemblyFormat = "$handle `,` $op attr-dict `:` type($handle) `,` type($op) `->` type($expectation)";
}

def CUDM_ExpectationComputeOp
    : CuDensityMatOp<"expectation_compute", [MemoryEffects<[MemWrite]>]> {
  let summary = "Compute operator expectation value(s)";
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_ExpectationType:$expectation,
    F64:$time, I64:$batch_size, I32:$num_params, AnyType:$params,
    CUDM_StateType:$state, CUDM_WorkspaceType:$workspace
  );
  let results = (outs AnyType:$value);
  let assemblyFormat = "$handle `,` $expectation `,` $time `,` $batch_size `,` $num_params `,` $params `,` $state `,` $workspace attr-dict `:` type($handle) `,` type($expectation) `,` type($params) `,` type($state) `,` type($workspace) `->` type($value)";
  let hasVerifier = 1;
}

def CUDM_DestroyExpectationOp
    : CuDensityMatOp<"destroy_expectation", [MemoryEffects<[MemWrite]>]> {
  let summary = "Destroy expectation computation object";
  let arguments = (ins CUDM_ExpectationType:$expectation);
  let assemblyFormat = "$expectation attr-dict `:` type($expectation)";
}

// ---- Tier 2: Integration / Solver ops ----

def CUDM_EvolveOp : CuDensityMatOp<"evolve", [MemoryEffects<[MemWrite]>]> {
  let summary = "Time-evolve a quantum state under an operator";
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_OperatorType:$op,
    CUDM_StateType:$state_in, CUDM_StateType:$state_out,
    CUDM_WorkspaceType:$workspace,
    CUDM_IntegratorKindAttr:$integrator,
    F64Attr:$t_start, F64Attr:$t_end, I64Attr:$num_steps,
    OptionalAttr<CUDM_ComputeTypeAttr>:$compute_type
  );
  let results = (outs CUDM_StateType:$result);
  let assemblyFormat = "$handle `,` $op `,` $state_in `,` $state_out `,` $workspace attr-dict `:` type($handle) `,` type($op) `,` type($state_in) `,` type($state_out) `,` type($workspace) `->` type($result)";
}

def CUDM_ApplyDecoherenceOp : CuDensityMatOp<"apply_decoherence", [MemoryEffects<[MemWrite]>]> {
  let summary = "Apply Lindblad T1/T2 decoherence for one time step";
  let arguments = (ins
    CUDM_HandleType:$handle, CUDM_StateType:$state,
    CUDM_DecoherenceModelType:$model, F64:$dt
  );
  let assemblyFormat = "$handle `,` $state `,` $model `,` $dt attr-dict `:` type($handle) `,` type($state) `,` type($model)";
}

def CUDM_SSEEvolveOp : CuDensityMatOp<"sse_evolve", [MemoryEffects<[MemWrite]>]> {
  let summary = "Stochastic Schrodinger equation ensemble evolution";
  let arguments = (ins
    CUDM_HandleType:$handle,
    CUDM_OperatorType:$hamiltonian,
    Variadic<CUDM_ElementaryOpType>:$collapse_ops,
    CUDM_StateType:$initial_state,
    CUDM_WorkspaceType:$workspace,
    CUDM_IntegratorKindAttr:$integrator,
    I64Attr:$num_trajectories,
    F64Attr:$t_start, F64Attr:$t_end, I64Attr:$num_steps,
    I64Attr:$seed
  );
  let results = (outs CUDM_TrajectoryResultType:$result);
  let assemblyFormat = "$handle `,` $hamiltonian `,` `(` $collapse_ops `)` `,` $initial_state `,` $workspace attr-dict `:` type($handle) `,` type($hamiltonian) `,` `(` type($collapse_ops) `)` `,` type($initial_state) `,` type($workspace) `->` type($result)";
  let hasVerifier = 1;
}

#endif // CUDAQ_PULSE_DIALECT_CUDENSITYMAT_OPS
