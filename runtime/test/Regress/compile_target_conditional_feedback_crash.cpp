/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Regression test that unsupported conditional feedback on measurement results
// terminates compilation. Exceptions thrown from the runtime library are not
// reliably catchable in this test binary (RTTI disabled), so this runs in its
// own process and expects a crash.

// RUN: not --crash compile_target_conditional_feedback_crash

#include "common/KernelArgs.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq/Target/CompileTarget.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

using namespace cudaq_internal::compiler;

/// Kernel that branches on a measurement result (conditional feedback).
static const char *conditionalFeedbackKernel = R"#(
func.func @__nvqpp__mlirgen__feedback() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  %0 = quake.alloca !quake.ref
  %1 = quake.alloca !quake.ref
  quake.h %0 : (!quake.ref) -> ()
  %measOut = quake.mz %0 name "m1" : (!quake.ref) -> !quake.measure
  %2 = quake.discriminate %measOut : (!quake.measure) -> i1
  cc.if(%2) {
    quake.x %1 : (!quake.ref) -> ()
  }
  return
}
)#";

int main() {
  auto context = cudaq_internal::compiler::getOwningMLIRContext();
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(conditionalFeedbackKernel,
                                                     context.get());
  if (!mod)
    return 1;

  auto target = cudaq::CompileTarget{};
  target.supportConditionalsOnMeasureResults = false;

  Compiler compiler(std::move(target), cudaq::CompileOptions{});
  compiler.runPassPipeline("feedback", mod.release().getAsOpaquePointer(),
                           cudaq::KernelArgs{}, /*isEntryPoint=*/true);
  return 0;
}
