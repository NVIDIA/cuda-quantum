/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_LOGICAL_ANALYSIS_GADGETEDEM_H
#define CUDAQ_LOGICAL_ANALYSIS_GADGETEDEM_H

#include "cudaq/edem/owned.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace cudaq::logical {

/// Internal move-only ownership convenience. Affine offsets remain in the
/// authored contract. Only the released C ownership record and packed bytes
/// cross the native-to-Python boundary; this C++ type is not an exchange ABI.
using PackedGadgetEDEM = cudaq::edem::PackedEDEM;

/// Derive an EDEM from a verified, selected gadget detector contract and an
/// explicit finite fault set (cudaq.logical.edem-faults/v1 JSON). An empty
/// profile name selects the unique profile. Reads the module without mutation.
/// Requires canonical stabilizer-code algebra, proves ideal contract
/// determinism and completeness, and delegates packing to the EDEM C API.
mlir::FailureOr<PackedGadgetEDEM> deriveGadgetEDEM(mlir::ModuleOp module,
                                                   llvm::StringRef profile,
                                                   llvm::StringRef faultJson);

/// Checks a selected gadget detector contract, and optionally the gadget's
/// declared logical action, against one Choi-state probe of its realization.
/// `actionJson` = {"source":"gadget_spec"} resolves supported signed Clifford,
/// preparation and measurement intent, operand bindings and affine outcome maps
/// from the selected compiled GadgetSpec's typed objective closure.
/// Alternatively, `actionJson` is {arity, matrix, phases} (a signed symplectic
/// Clifford action), or empty/null to skip the action check.
/// This explicit matrix form checks supplied rows only; it does not establish
/// equivalence to the module's declared objective. Zero-arity and nonsymplectic
/// matrices are invalid. Semantic violations are reported in the returned
/// cudaq.logical.gadget-verification/v2
/// JSON; structural problems fail with diagnostics. Reads the module without
/// mutation.
mlir::FailureOr<std::string> verifyGadgetContract(mlir::ModuleOp module,
                                                  llvm::StringRef profile,
                                                  llvm::StringRef actionJson);

} // namespace cudaq::logical

#endif // CUDAQ_LOGICAL_ANALYSIS_GADGETEDEM_H
