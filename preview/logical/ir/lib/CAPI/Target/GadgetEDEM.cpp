/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 *******************************************************************************/

#include "qlx-c/Target/Translations.h"

#include "qlx/Analysis/GadgetEDEM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include <new>
#include <stdexcept>

MlirLogicalResult qlxDeriveGadgetEDEM(MlirModule module, MlirStringRef profile,
                                      MlirStringRef faults,
                                      cudaq_gf2_owned_buffer *output) {
  if (!output || output->data || output->size || output->deallocate ||
      output->context)
    return mlirLogicalResultFailure();
  try {
    auto result = cudaq::logical::deriveGadgetEDEM(
        unwrap(module), unwrap(profile), unwrap(faults));
    if (mlir::failed(result))
      return mlirLogicalResultFailure();
    *output = result->release();
    return mlirLogicalResultSuccess();
  } catch (const std::bad_alloc &) {
    // Reporting a diagnostic can itself allocate. Preserve output and return
    // failure without allocating another error message at this C boundary.
    return mlirLogicalResultFailure();
  } catch (const std::length_error &) {
    return mlirLogicalResultFailure();
  }
}
