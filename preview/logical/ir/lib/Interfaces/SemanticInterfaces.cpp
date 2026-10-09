/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CUDAQLogical/Interfaces/SemanticInterfaces.h"

#include "llvm/Support/ErrorHandling.h"

using namespace cudaq::logical;

llvm::StringRef cudaq::logical::stringifyStage(Stage stage) {
  switch (stage) {
  case Stage::P0:
    return "p0";
  case Stage::P1:
    return "p1";
  case Stage::P2:
    return "p2";
  case Stage::P3:
    return "p3";
  case Stage::P4:
    return "p4";
  }
  llvm_unreachable("unknown CUDA-Q Logical stage");
}

llvm::StringRef cudaq::logical::stringifyRootKind(RootKind kind) {
  switch (kind) {
  case RootKind::LogicalProgram:
    return "logical_program";
  case RootKind::PlacedKernel:
    return "placed_kernel";
  case RootKind::QECCircuit:
    return "qec_circuit";
  case RootKind::QECGadget:
    return "qec_gadget";
  case RootKind::QECProtocol:
    return "qec_protocol";
  case RootKind::PhysicalGraph:
    return "physical_graph";
  case RootKind::RealtimePlan:
    return "realtime_plan";
  }
  llvm_unreachable("unknown CUDA-Q Logical root kind");
}

#include "CUDAQLogical/Interfaces/SemanticInterfaces.cpp.inc"
