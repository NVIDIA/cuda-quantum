/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#ifndef CUDAQ_PULSE_DIALECT_CUDENSITYMAT_ATTRS
#define CUDAQ_PULSE_DIALECT_CUDENSITYMAT_ATTRS

include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.td"
include "mlir/IR/EnumAttr.td"

def CUDM_StatePurity : I32EnumAttr<"StatePurity",
    "Quantum state purity",
    [
      I32EnumAttrCase<"Pure", 0, "pure">,
      I32EnumAttrCase<"Mixed", 1, "mixed">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_StatePurityAttr : EnumAttr<CuDensityMatDialect, CUDM_StatePurity, "purity">;

def CUDM_ComputeType : I32EnumAttr<"ComputeType",
    "Floating-point compute precision",
    [
      I32EnumAttrCase<"F32", 4, "f32">,
      I32EnumAttrCase<"F64", 16, "f64">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_ComputeTypeAttr : EnumAttr<CuDensityMatDialect, CUDM_ComputeType, "compute_type">;

def CUDM_Memspace : I32EnumAttr<"Memspace",
    "Memory space for workspace allocation",
    [
      I32EnumAttrCase<"Device", 0, "device">,
      I32EnumAttrCase<"Host", 1, "host">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_MemspaceAttr : EnumAttr<CuDensityMatDialect, CUDM_Memspace, "memspace">;

def CUDM_WorkspaceKind : I32EnumAttr<"WorkspaceKind",
    "Kind of workspace memory buffer",
    [
      I32EnumAttrCase<"Scratch", 0, "scratch">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_WorkspaceKindAttr : EnumAttr<CuDensityMatDialect, CUDM_WorkspaceKind, "ws_kind">;

def CUDM_Sparsity : I32EnumAttr<"Sparsity",
    "Elementary operator sparsity pattern",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"Multidiagonal", 1, "multidiagonal">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_SparsityAttr : EnumAttr<CuDensityMatDialect, CUDM_Sparsity, "sparsity">;

def CUDM_SpectrumKind : I32EnumAttr<"SpectrumKind",
    "Which extreme eigenvalues to compute",
    [
      I32EnumAttrCase<"Largest", 0, "largest">,
      I32EnumAttrCase<"Smallest", 1, "smallest">,
      I32EnumAttrCase<"LargestReal", 2, "largest_real">,
      I32EnumAttrCase<"SmallestReal", 3, "smallest_real">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_SpectrumKindAttr : EnumAttr<CuDensityMatDialect, CUDM_SpectrumKind, "spectrum_kind">;

def CUDM_DistProvider : I32EnumAttr<"DistributedProvider",
    "Distributed communication provider",
    [
      I32EnumAttrCase<"DistNone", 0, "none">,
      I32EnumAttrCase<"MPI", 1, "mpi">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_DistProviderAttr : EnumAttr<CuDensityMatDialect, CUDM_DistProvider, "dist_provider">;

def CUDM_IntegratorKind : I32EnumAttr<"IntegratorKind",
    "Time-integration strategy",
    [
      I32EnumAttrCase<"MagnusCF4", 0, "magnus_cf4">,
      I32EnumAttrCase<"Lanczos", 1, "lanczos">,
      I32EnumAttrCase<"RungeKutta1", 2, "rk1">,
      I32EnumAttrCase<"RungeKutta2", 3, "rk2">,
      I32EnumAttrCase<"RungeKutta4", 4, "rk4">,
      I32EnumAttrCase<"SSE", 5, "sse">
    ]> {
  let cppNamespace = "::cudm";
  let genSpecializedAttr = 0;
}
def CUDM_IntegratorKindAttr : EnumAttr<CuDensityMatDialect, CUDM_IntegratorKind, "integrator">;

#endif // CUDAQ_PULSE_DIALECT_CUDENSITYMAT_ATTRS
