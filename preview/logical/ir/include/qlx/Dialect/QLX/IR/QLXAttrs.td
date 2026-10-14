/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// This file defines the attributes for the QLX dialect: enum attributes for
// built-in P0 actions, instruments, and Pauli bases, plus the composite
// CliffordAction attribute the target-manifest and P0 strata share.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_ATTRS_TD
#define QLX_ATTRS_TD

include "qlx/Dialect/QLX/IR/QLXDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// QLX Enum Attributes
//===----------------------------------------------------------------------===//

// Portable built-in logical actions are semantic values, not module-level
// declarations.  Custom and composite actions still use qlx.action symbols.
def QLX_BuiltinAction_H             : I32EnumAttrCase<"h", 0>;
def QLX_BuiltinAction_S             : I32EnumAttrCase<"s", 1>;
def QLX_BuiltinAction_Sdg           : I32EnumAttrCase<"sdg", 2>;
def QLX_BuiltinAction_X             : I32EnumAttrCase<"x", 3>;
def QLX_BuiltinAction_Y             : I32EnumAttrCase<"y", 4>;
def QLX_BuiltinAction_Z             : I32EnumAttrCase<"z", 5>;
def QLX_BuiltinAction_T             : I32EnumAttrCase<"t", 6>;
def QLX_BuiltinAction_Tdg           : I32EnumAttrCase<"tdg", 7>;
def QLX_BuiltinAction_CX            : I32EnumAttrCase<"cx", 8>;
def QLX_BuiltinAction_CZ            : I32EnumAttrCase<"cz", 9>;
def QLX_BuiltinAction_CCZ           : I32EnumAttrCase<"ccz", 10>;
def QLX_BuiltinAction_Idle          : I32EnumAttrCase<"idle", 11>;
def QLX_BuiltinAction_PauliRotation : I32EnumAttrCase<"pauli_rotation", 12>;
def QLX_BuiltinAction_CCX           : I32EnumAttrCase<"ccx", 13>;

def QLX_BuiltinActionEnum : I32EnumAttr<"BuiltinAction",
    "Canonical built-in unplaced logical action",
    [QLX_BuiltinAction_H, QLX_BuiltinAction_S, QLX_BuiltinAction_Sdg,
     QLX_BuiltinAction_X, QLX_BuiltinAction_Y, QLX_BuiltinAction_Z,
     QLX_BuiltinAction_T, QLX_BuiltinAction_Tdg, QLX_BuiltinAction_CX,
     QLX_BuiltinAction_CZ, QLX_BuiltinAction_CCZ, QLX_BuiltinAction_Idle,
     QLX_BuiltinAction_PauliRotation, QLX_BuiltinAction_CCX]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_BuiltinActionAttr : EnumAttr<QLX_Dialect, QLX_BuiltinActionEnum,
                                      "action"> {
  let assemblyFormat = "`<` $value `>`";
}

// General instruments remain extensible symbols.  This attribute covers only
// built-ins whose complete semantics are owned by the QLX dialect.
def QLX_BuiltinInstrument_MPP : I32EnumAttrCase<"mpp", 0>;

def QLX_BuiltinInstrumentEnum : I32EnumAttr<"BuiltinInstrument",
    "Canonical built-in unplaced logical instrument",
    [QLX_BuiltinInstrument_MPP]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_BuiltinInstrumentAttr
    : EnumAttr<QLX_Dialect, QLX_BuiltinInstrumentEnum, "instrument"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Pauli basis: X, Y, Z
//===----------------------------------------------------------------------===//

def QLX_Pauli_X : I32EnumAttrCase<"X", 0>;
def QLX_Pauli_Y : I32EnumAttrCase<"Y", 1>;
def QLX_Pauli_Z : I32EnumAttrCase<"Z", 2>;

def QLX_PauliEnum : I32EnumAttr<"Pauli",
    "Pauli operator basis for measurements",
    [QLX_Pauli_X, QLX_Pauli_Y, QLX_Pauli_Z]> {
  let cppNamespace = "qlx";
  let genSpecializedAttr = 0;
}

def QLX_PauliAttr : EnumAttr<QLX_Dialect, QLX_PauliEnum, "pauli"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// QLX Composite Attributes
//===----------------------------------------------------------------------===//

def QLX_CliffordActionAttr : AttrDef<QLX_Dialect, "CliffordAction"> {
  let mnemonic = "clifford_action";
  let summary = "complete signed binary-symplectic logical action";
  let description = [{
    Stores a flattened row-major 2k-by-2k binary symplectic matrix, one
    binary sign phase per generator image, and the ordered logical port
    signature.  Matrix rows are the images of X_0..X_(k-1), Z_0..Z_(k-1);
    columns are x_0..x_(k-1), z_0..z_(k-1).
  }];
  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$matrix,
    "::mlir::DenseI64ArrayAttr":$phases,
    "::mlir::ArrayAttr":$ports
  );
  let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<` `matrix` `=` $matrix `,` `phases` `=` $phases `,`
        `ports` `=` $ports `>`
  }];
}

#endif // QLX_ATTRS_TD
