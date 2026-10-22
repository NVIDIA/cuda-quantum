/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_QOP_ATTRS
#define CUDAQ_PULSE_DIALECT_QOP_ATTRS

include "cudaq-pulse/Dialect/QOp/QOpDialect.td"
include "mlir/IR/EnumAttr.td"

def QOP_HandlerKind_SpinI            : I32EnumAttrCase<"SpinI", 0, "spin_i">;
def QOP_HandlerKind_SpinX            : I32EnumAttrCase<"SpinX", 1, "spin_x">;
def QOP_HandlerKind_SpinY            : I32EnumAttrCase<"SpinY", 2, "spin_y">;
def QOP_HandlerKind_SpinZ            : I32EnumAttrCase<"SpinZ", 3, "spin_z">;
def QOP_HandlerKind_BosonIdentity    : I32EnumAttrCase<"BosonIdentity", 10, "boson_i">;
def QOP_HandlerKind_BosonCreate      : I32EnumAttrCase<"BosonCreate", 11, "boson_create">;
def QOP_HandlerKind_BosonAnnihilate  : I32EnumAttrCase<"BosonAnnihilate", 12, "boson_annihilate">;
def QOP_HandlerKind_BosonNumber      : I32EnumAttrCase<"BosonNumber", 13, "boson_number">;
def QOP_HandlerKind_FermionIdentity  : I32EnumAttrCase<"FermionIdentity", 20, "fermion_i">;
def QOP_HandlerKind_FermionCreate    : I32EnumAttrCase<"FermionCreate", 21, "fermion_create">;
def QOP_HandlerKind_FermionAnnihilate: I32EnumAttrCase<"FermionAnnihilate", 22, "fermion_annihilate">;
def QOP_HandlerKind_FermionNumber    : I32EnumAttrCase<"FermionNumber", 23, "fermion_number">;
def QOP_HandlerKind_SpinLowering     : I32EnumAttrCase<"SpinLowering", 4, "spin_lowering">;
def QOP_HandlerKind_SpinRaising      : I32EnumAttrCase<"SpinRaising", 5, "spin_raising">;
def QOP_HandlerKind_Matrix           : I32EnumAttrCase<"Matrix", 30, "matrix">;

def QOP_HandlerKind : I32EnumAttr<"HandlerKind",
    "Kind of leaf quantum operator",
    [
      QOP_HandlerKind_SpinI, QOP_HandlerKind_SpinX,
      QOP_HandlerKind_SpinY, QOP_HandlerKind_SpinZ,
      QOP_HandlerKind_SpinLowering, QOP_HandlerKind_SpinRaising,
      QOP_HandlerKind_BosonIdentity, QOP_HandlerKind_BosonCreate,
      QOP_HandlerKind_BosonAnnihilate, QOP_HandlerKind_BosonNumber,
      QOP_HandlerKind_FermionIdentity, QOP_HandlerKind_FermionCreate,
      QOP_HandlerKind_FermionAnnihilate, QOP_HandlerKind_FermionNumber,
      QOP_HandlerKind_Matrix
    ]> {
  let cppNamespace = "::qop";
  let genSpecializedAttr = 0;
}

def QOP_HandlerKindAttr : EnumAttr<QOpDialect, QOP_HandlerKind, "handler_kind">;

#endif // CUDAQ_PULSE_DIALECT_QOP_ATTRS
