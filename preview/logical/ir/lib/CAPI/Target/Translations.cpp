/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx-c/Target/Translations.h"

#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

#include "qlx/Target/Fabric/EmitStim.h"

#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

// Wrap an MlirStringCallback as a tiny llvm::raw_string_ostream that buffers
// in std::string and ships the whole result through the callback at flush.
//
// We can't drive the callback incrementally because emit*() implementations
// expect an llvm::raw_ostream and we want a single call to land at the
// Python side; allocating one std::string and shipping it once is fine for
// these translations (output is typically a few KB to a few hundred KB).
inline void shipString(const std::string &s, MlirStringCallback callback,
                       void *userData) {
  callback(MlirStringRef{s.data(), s.size()}, userData);
}

} // namespace

MlirLogicalResult qlxTranslateFabricToStim(MlirModule module,
                                           MlirStringCallback callback,
                                           void *userData) {
  auto mod = unwrap(module);
  std::string out;
  llvm::raw_string_ostream os(out);
  if (mlir::failed(qlx::fabric::emitStim(mod, os)))
    return mlirLogicalResultFailure();
  shipString(out, callback, userData);
  return mlirLogicalResultSuccess();
}
