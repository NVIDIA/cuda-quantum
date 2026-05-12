/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compiled with -fno-rtti -fno-exceptions to match LLVM build configuration
// (LLVM_ENABLE_RTTI=OFF, LLVM_ENABLE_EH=OFF). Uses Python C API directly.

#include "PythonSignalCheck.h"
#include "common/Environment.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include <Python.h>

using namespace mlir;

namespace {
class PythonSignalCheckInstrumentation : public PassInstrumentation {
  bool signalDetected = false;

public:
  void runBeforePipeline(std::optional<OperationName>,
                         const PipelineParentInfo &) override {
    signalDetected = false;
  }

  void runAfterPass(Pass *, Operation *op) override {
    if (signalDetected)
      return;
    auto gstate = PyGILState_Ensure();
    if (PyErr_CheckSignals() != 0) {
      signalDetected = true;
      PyErr_Clear();
      op->emitError("compilation interrupted by Python signal");
    }
    PyGILState_Release(gstate);
  }
};
} // namespace

void cudaq::addPythonSignalInstrumentation(PassManager &pm) {
  if (!PyGILState_Check())
    return;
  pm.addInstrumentation(std::make_unique<PythonSignalCheckInstrumentation>());
}

LogicalResult cudaq::runPassManagerReleasingGIL(PassManager &pm,
                                                Operation *op) {
  auto *ctx = pm.getContext();
  auto enablePrintEachPass =
      cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);
  auto disableThreading =
      cudaq::getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", false);
  if (enablePrintEachPass || disableThreading)
    ctx->disableMultithreading();
  if (enablePrintEachPass)
    pm.enableIRPrinting();
  if (!PyGILState_Check())
    return pm.run(op);
  PyThreadState *save = PyEval_SaveThread();
  auto result = pm.run(op);
  PyEval_RestoreThread(save);
  return result;
}
