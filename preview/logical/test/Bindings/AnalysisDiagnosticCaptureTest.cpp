/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 *******************************************************************************/

#include "AnalysisDiagnosticCapture.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"

#include <atomic>
#include <barrier>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace {
void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
    std::exit(1);
  }
}
} // namespace

int main() {
  mlir::MLIRContext context;
  auto location = mlir::UnknownLoc::get(&context);
  std::atomic<unsigned> foreign{0}, forwarded{0};
  mlir::ScopedDiagnosticHandler fallback(&context, [&](mlir::Diagnostic &diag) {
    if (diag.str() == "foreign diagnostic")
      ++foreign;
    else
      ++forwarded;
    return mlir::success();
  });

  qlx::python::AnalysisDiagnosticCapture capture(&context);
  std::barrier start(2);
  std::thread worker([&] {
    start.arrive_and_wait();
    for (unsigned i = 0; i < 256; ++i)
      mlir::emitError(location, "foreign diagnostic");
  });
  start.arrive_and_wait();
  mlir::emitError(location, "analysis diagnostic");
  auto text = capture.takeDiagnostics();
  worker.join();

  require(text.find("analysis diagnostic") != std::string::npos,
          "capture the analysis error");
  require(text.find("foreign diagnostic") == std::string::npos,
          "leave other threads' diagnostics untouched");
  require(foreign == 256, "forward every foreign diagnostic");
  require(forwarded == 0, "consume the captured analysis diagnostic");

  mlir::emitError(location, "after capture");
  require(forwarded == 1, "unregister before returning captured text");
  {
    qlx::python::AnalysisDiagnosticCapture discarded(&context);
    mlir::emitError(location, "discarded diagnostic");
  }
  mlir::emitError(location, "after destruction");
  require(forwarded == 2, "unregister when capture is discarded");
}
