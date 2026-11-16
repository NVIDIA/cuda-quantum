/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 *******************************************************************************/

#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"

#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace qlx::python {

/// Synchronous analyses emit diagnostics on their calling thread. A handler
/// belongs to the whole MLIR context, so leave other threads' diagnostics for
/// their own handlers even while the analysis has released the Python GIL.
class AnalysisDiagnosticCapture {
public:
  explicit AnalysisDiagnosticCapture(mlir::MLIRContext *context) {
    handler.emplace(context, [this](mlir::Diagnostic &diagnostic) {
      if (std::this_thread::get_id() != thread ||
          diagnostic.getSeverity() != mlir::DiagnosticSeverity::Error)
        return mlir::failure();
      llvm::raw_string_ostream stream(text);
      stream << diagnostic.getLocation() << ": ";
      diagnostic.print(stream);
      stream << '\n';
      return mlir::success();
    });
  }

  AnalysisDiagnosticCapture(const AnalysisDiagnosticCapture &) = delete;
  AnalysisDiagnosticCapture &
  operator=(const AnalysisDiagnosticCapture &) = delete;

  std::string takeDiagnostics() {
    // Unregistration waits for any diagnostic callback in progress. No handler
    // can access the captured text while the caller reads or moves it.
    handler.reset();
    return std::move(text);
  }

private:
  const std::thread::id thread = std::this_thread::get_id();
  std::string text;
  // Destroy the handler before the text, including on exceptional exits.
  std::optional<mlir::ScopedDiagnosticHandler> handler;
};

} // namespace qlx::python
