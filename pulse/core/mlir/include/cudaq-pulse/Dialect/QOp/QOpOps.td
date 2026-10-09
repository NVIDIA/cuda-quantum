/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_QOP_OPS
#define CUDAQ_PULSE_DIALECT_QOP_OPS

include "cudaq-pulse/Dialect/QOp/QOpDialect.td"
include "cudaq-pulse/Dialect/QOp/QOpTypes.td"
include "cudaq-pulse/Dialect/QOp/QOpAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class QOpOp<string mnemonic, list<Trait> traits = []> :
    Op<QOpDialect, mnemonic, traits>;

// ---- Leaf operator construction ----

def QOP_SpinOp : QOpOp<"spin", [Pure]> {
  let summary = "Create a Pauli spin operator on a target qubit";
  let arguments = (ins I64:$target, QOP_HandlerKindAttr:$kind);
  let results = (outs QOP_HandlerType:$result);
  let assemblyFormat = "`(` $target `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QOP_BosonOp : QOpOp<"boson", [Pure]> {
  let summary = "Create a bosonic operator on a target mode";
  let arguments = (ins I64:$target, I64Attr:$dimension, QOP_HandlerKindAttr:$kind);
  let results = (outs QOP_HandlerType:$result);
  let assemblyFormat = "`(` $target `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QOP_FermionOp : QOpOp<"fermion", [Pure]> {
  let summary = "Create a fermionic operator on a target mode";
  let arguments = (ins I64:$target, QOP_HandlerKindAttr:$kind);
  let results = (outs QOP_HandlerType:$result);
  let assemblyFormat = "`(` $target `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QOP_MatrixLeafOp : QOpOp<"matrix_leaf", [Pure]> {
  let summary = "Create a user-defined matrix leaf operator";
  let arguments = (ins
    DenseI64ArrayAttr:$targets,
    AnyType:$data,
    DenseI64ArrayAttr:$dimensions
  );
  let results = (outs QOP_HandlerType:$result);
  let assemblyFormat = "`(` $data `)` attr-dict `:` type($data) `->` type($result)";
  let hasVerifier = 1;
}

// ---- Scalar coefficient construction ----

def QOP_ConstScalarOp : QOpOp<"const_scalar", [Pure]> {
  let summary = "Create a constant complex scalar coefficient";
  let arguments = (ins F64Attr:$real, F64Attr:$imag);
  let results = (outs QOP_ScalarType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def QOP_CallbackScalarOp : QOpOp<"callback_scalar", [Pure]> {
  let summary = "Create a time/parameter-dependent scalar coefficient";
  let arguments = (ins FlatSymbolRefAttr:$callback);
  let results = (outs QOP_ScalarType:$result);
  let assemblyFormat = "$callback attr-dict `:` type($result)";
  let hasVerifier = 1;
}

// ---- Operator composition ----

def QOP_MakeProductOp : QOpOp<"make_product", [Pure]> {
  let summary = "Create a tensor product of handlers with a coefficient";
  let arguments = (ins QOP_ScalarType:$coefficient, Variadic<QOP_HandlerType>:$factors);
  let results = (outs QOP_ProductType:$result);
  let assemblyFormat = "`(` $coefficient `,` $factors `)` attr-dict `:` type($result)";
}

def QOP_MakeSumOp : QOpOp<"make_sum", [Pure]> {
  let summary = "Create a sum of product terms";
  let arguments = (ins Variadic<QOP_ProductType>:$terms);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "`(` $terms `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def QOP_AddOp : QOpOp<"add", [Pure]> {
  let summary = "Add two operators";
  let arguments = (ins QOP_OpType:$lhs, QOP_OpType:$rhs);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def QOP_ScaleOp : QOpOp<"scale", [Pure]> {
  let summary = "Multiply an operator by a scalar coefficient";
  let arguments = (ins QOP_ScalarType:$coefficient, QOP_OpType:$operand);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$coefficient `,` $operand attr-dict `:` type($result)";
}

def QOP_DaggerOp : QOpOp<"dagger", [Pure]> {
  let summary = "Hermitian conjugate of an operator";
  let arguments = (ins QOP_OpType:$operand);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def QOP_CommutatorOp : QOpOp<"commutator", [Pure]> {
  let summary = "Compute [A, B] = AB - BA";
  let arguments = (ins QOP_OpType:$lhs, QOP_OpType:$rhs);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def QOP_AnticommutatorOp : QOpOp<"anticommutator", [Pure]> {
  let summary = "Compute {A, B} = AB + BA";
  let arguments = (ins QOP_OpType:$lhs, QOP_OpType:$rhs);
  let results = (outs QOP_OpType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// ---- Lindblad / super-operator construction ----

def QOP_LindbladOp : QOpOp<"lindblad", [Pure]> {
  let summary = "Construct Lindblad super-operator from H and collapse ops";
  let arguments = (ins QOP_OpType:$hamiltonian, Variadic<QOP_OpType>:$collapse_ops);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "`(` $hamiltonian `,` $collapse_ops `)` attr-dict `:` type($result)";
}

def QOP_SuperLeftOp : QOpOp<"super_left", [Pure]> {
  let summary = "Super-operator for left multiplication: S[rho] = A * rho";
  let arguments = (ins QOP_OpType:$operand);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def QOP_SuperRightOp : QOpOp<"super_right", [Pure]> {
  let summary = "Super-operator for right multiplication: S[rho] = rho * A";
  let arguments = (ins QOP_OpType:$operand);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def QOP_SuperBothOp : QOpOp<"super_both", [Pure]> {
  let summary = "Super-operator: S[rho] = A * rho * B";
  let arguments = (ins QOP_OpType:$left, QOP_OpType:$right);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "$left `,` $right attr-dict `:` type($result)";
}

def QOP_SuperAddOp : QOpOp<"super_add", [Pure]> {
  let summary = "Add two super-operators";
  let arguments = (ins QOP_SuperOpType:$lhs, QOP_SuperOpType:$rhs);
  let results = (outs QOP_SuperOpType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// ---- Evaluation / materialization ----

def QOP_ToMatrixOp : QOpOp<"to_matrix", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Materialize operator to dense matrix at given time/params";
  let arguments = (ins
    QOP_OpType:$operand,
    DenseI64ArrayAttr:$dimensions,
    F64:$time,
    AnyType:$params
  );
  let results = (outs AnyType:$matrix);
  let assemblyFormat = "$operand `,` $time `,` $params attr-dict `:` type($operand) `,` type($params) `->` type($matrix)";
  let hasVerifier = 1;
}

def QOP_DegreesOp : QOpOp<"degrees", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Query which degrees of freedom an operator acts on";
  let arguments = (ins QOP_OpType:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
  let hasVerifier = 1;
}

#endif // CUDAQ_PULSE_DIALECT_QOP_OPS
