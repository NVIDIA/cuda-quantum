/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include <cstdlib>

namespace cudaq {

/// The emitFatalError() function is used when the compiler reaches a point that
/// it cannot continue and produce valid output code. This is very much like an
/// assertion, but it will not be removed if assertions are disabled.
[[noreturn]] inline void emitFatalError(mlir::Location loc,
                                        const llvm::Twine &message) {
  mlir::emitError(loc, message);
  llvm::report_fatal_error("fatal error, aborting.");
}

} // namespace cudaq

//===----------------------------------------------------------------------===//

#undef TODOQUOTE
#define TODOQUOTE(X) #X
#undef TODO_DEFN3
#undef TODO_DEFN4

#if defined(NDEBUG) || defined(CUDAQ_NOTRACEBACKS)
// Release build: print the message and exit.

#define TODO_DEFN3(ToDoMsg, ToDoFile, ToDoLine)                                \
  do {                                                                         \
    llvm::errs() << ToDoFile << ':' << ToDoLine                                \
                 << ": not yet implemented: " << ToDoMsg << '\n';              \
    std::exit(1);                                                              \
  } while (false)

#define TODO_DEFN4(MlirLoc, ToDoMsg, ToDoFile, ToDoLine)                       \
  do {                                                                         \
    mlir::emitError(MlirLoc, llvm::Twine(ToDoFile ":" TODOQUOTE(               \
                                 ToDoLine) ": not yet implemented: ") +        \
                                 ToDoMsg);                                     \
    std::exit(1);                                                              \
  } while (false)

#else
// Debug build: print the message and exit with traceback. A traceback is useful
// to compiler developers, but not expected by users of the compiler.

#define TODO_DEFN3(ToDoMsg, ToDoFile, ToDoLine)                                \
  do {                                                                         \
    llvm::report_fatal_error(llvm::Twine(ToDoFile ":" TODOQUOTE(               \
                                 ToDoLine) ": not yet implemented: ") +        \
                             ToDoMsg);                                         \
  } while (false)

#define TODO_DEFN4(MlirLoc, ToDoMsg, ToDoFile, ToDoLine)                       \
  do {                                                                         \
    cudaq::emitFatalError(MlirLoc, llvm::Twine(ToDoFile ":" TODOQUOTE(         \
                                       ToDoLine) ": not yet implemented: ") +  \
                                       ToDoMsg);                               \
  } while (false)
#endif

//===----------------------------------------------------------------------===//

/// The TODO macro is to be used to mark parts of the compiler that are not yet
/// implemented. Instead of the compiler crashing or generating bad code for
/// features that have not been implemented, we want the compiler to generate a
/// NYI error message and exit.
#undef TODO
#define TODO(ToDoMsg) TODO_DEFN3(ToDoMsg, __FILE__, __LINE__)

/// The TODO_loc macro is to be used to mark parts of the compiler that are not
/// yet implemented. Instead of the compiler crashing or generating bad code for
/// features that have not been implemented, we want the compiler to generate a
/// NYI error message and exit. It will include source location information in
/// the output, indicating the location the compiler aborted.
#undef TODO_loc
#define TODO_loc(MlirLoc, ToDoMsg)                                             \
  TODO_DEFN4(MlirLoc, ToDoMsg, __FILE__, __LINE__)
