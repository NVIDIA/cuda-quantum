/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_BINDINGS_PYTHON_PASSDEBUG_H
#define QLX_BINDINGS_PYTHON_PASSDEBUG_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"

namespace qlx::python {

inline std::string lowerEnvValue(const char *value) {
  std::string result(value ? value : "");
  std::transform(
      result.begin(), result.end(), result.begin(),
      [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return result;
}

inline bool envIsEnabled(const char *value) {
  if (!value || !value[0])
    return false;
  std::string lowered = lowerEnvValue(value);
  return lowered != "0" && lowered != "false" && lowered != "off" &&
         lowered != "no";
}

inline bool irPrintingRequested() {
  return envIsEnabled(std::getenv("QLX_MLIR_PRINT_IR"));
}

inline void configureIRPrinting(mlir::PassManager &pm) {
  const char *rawMode = std::getenv("QLX_MLIR_PRINT_IR");
  if (!envIsEnabled(rawMode))
    return;

  // MLIR's IR-printing instrumentation requires a single-threaded context.
  // Command-line tools do this when `--mlir-print-ir-*` is used; embedded
  // pass managers must do it explicitly before installing the instrumentation.
  pm.getContext()->disableMultithreading();

  std::string mode = lowerEnvValue(rawMode);
  bool printBefore = false;
  bool printAfter = true;
  bool printAfterOnlyOnChange = false;
  bool printAfterOnlyOnFailure = false;

  if (mode == "before" || mode == "before-all") {
    printBefore = true;
    printAfter = false;
  } else if (mode == "all" || mode == "both" || mode == "before-after" ||
             mode == "before,after") {
    printBefore = true;
    printAfter = true;
  } else if (mode == "changed" || mode == "after-change" ||
             mode == "after-changed") {
    printAfterOnlyOnChange = true;
  } else if (mode == "failure" || mode == "failed" || mode == "after-failure") {
    printAfterOnlyOnFailure = true;
  }

  pm.enableIRPrinting(
      [printBefore](mlir::Pass *, mlir::Operation *) { return printBefore; },
      [printAfter](mlir::Pass *, mlir::Operation *) { return printAfter; },
      /*printModuleScope=*/true,
      /*printAfterOnlyOnChange=*/printAfterOnlyOnChange,
      /*printAfterOnlyOnFailure=*/printAfterOnlyOnFailure,
      /*out=*/llvm::errs(), mlir::OpPrintingFlags());
}

} // namespace qlx::python

#endif // QLX_BINDINGS_PYTHON_PASSDEBUG_H
