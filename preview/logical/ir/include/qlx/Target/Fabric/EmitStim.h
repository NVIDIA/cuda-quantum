//===- EmitStim.h - Fabric to Stim translation -----------------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_TRANSLATE_FABRIC_EMITSTIM_H
#define QLX_TRANSLATE_FABRIC_EMITSTIM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace qlx::fabric {

/// Translate a Fabric module to Stim circuit text.
///
/// Walks the entry gadget, resolves patch partitions to absolute qubit
/// indices, tracks raw measurement records, and writes standards-compatible
/// circuit instructions to \p os.
mlir::LogicalResult emitStim(mlir::ModuleOp module, llvm::raw_ostream &os);

/// Register `--fabric-to-stim` with MLIR's translation registry.
void registerFabricToStimTranslation();

} // namespace qlx::fabric

#endif // QLX_TRANSLATE_FABRIC_EMITSTIM_H
