/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

// Minimal out-of-tree CUDA-Q MLIR Python extension. It drives the pure MLIR
// dialect + pass defined under lib/, so importing and calling into it exercises
// MLIR/LLVM symbol resolution against `cudaq::MLIR` at load time.

#include <nanobind/nanobind.h>

#include "Trivial/TrivialDialect.h"
#include "Trivial/TrivialPasses.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

NB_MODULE(_mlirExtension, m) {
  m.doc() = "Minimal out-of-tree CUDA-Q MLIR Python extension used to validate "
            "the cudaq-devel wheel.";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    unwrap(registry)->insert<trivial::TrivialDialect>();
  });

  m.def("run_trivial_pass", []() {
    MLIRContext context;
    context.getOrLoadDialect<trivial::TrivialDialect>();

    OpBuilder builder(&context);
    OwningOpRef<ModuleOp> module =
        ModuleOp::create(builder, builder.getUnknownLoc());

    PassManager pm(&context);
    pm.addPass(trivial::createTrivialPass());
    if (failed(pm.run(module.get())))
      return false;

    return module.get()->hasAttr("trivial.visited");
  });
}
