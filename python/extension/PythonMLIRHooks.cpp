/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq_internal/compiler/TracePassInstrumentation.h"
#include "runtime/cudaq/platform/PythonSignalCheck.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/IR.h"

#include <nanobind/nanobind.h>

// FIXME: Declare this in a header file!
// Forward-declare the Python-aware helper so this translation unit does not
// pull in headers from python/. The symbol is defined in
// python/runtime/cudaq/platform/PythonSignalCheck.cpp, which is linked into
// the same Python extension.
namespace cudaq {
mlir::LogicalResult runPassManagerReleasingGIL(mlir::PassManager &pm,
                                               mlir::Operation *op);
}

namespace nb = nanobind;

static mlir::LogicalResult pythonRunPassManager(mlir::PassManager &pm,
                                                mlir::Operation *op) {
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  cudaq::addPythonSignalInstrumentation(pm);
  cudaq_internal::compiler::configurePassManagerFromEnv(pm);
  return cudaq::runPassManagerReleasingGIL(pm, op);
}

static void pythonRegisterDialects(mlir::DialectRegistry &registry) {
  if (!Py_IsInitialized())
    return;
  nb::gil_scoped_acquire gil;
  nb::object libs = nb::module_::import_("cudaq.mlir._mlir_libs");
  nb::object pyRegistry = libs.attr("get_dialect_registry")();
  MlirDialectRegistry handle = mlirPythonCapsuleToDialectRegistry(
      pyRegistry.attr(MLIR_PYTHON_CAPI_PTR_ATTR).ptr());
  if (!mlirDialectRegistryIsNull(handle))
    unwrap(handle)->appendTo(registry);
}

namespace cudaq_internal::compiler {
void installPythonMLIRHooks() {
  setRunPassManagerHook(&pythonRunPassManager);
  setDialectRegistrationHook(&pythonRegisterDialects);
}
} // namespace cudaq_internal::compiler
