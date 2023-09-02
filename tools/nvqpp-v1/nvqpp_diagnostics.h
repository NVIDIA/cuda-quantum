
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "nvqpp_options.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/ChainedDiagnosticConsumer.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

// TODO: documentation
namespace cudaq {
using DiagnosticsEngine = clang::DiagnosticsEngine;

template <typename T>
using LlvmCntPtr = llvm::IntrusiveRefCntPtr<T>;

//
// diagnostics options
//
struct DiagnosticsOptions : LlvmCntPtr<clang::DiagnosticOptions> {
  using Base = LlvmCntPtr<clang::DiagnosticOptions>;
  DiagnosticsOptions(ArgvT argv)
      : Base(clang::CreateAndPopulateDiagOpts(argv)) {}
};

//
// diagnostics buffer
//
struct DiagnosticsBuffer {
  using Base = clang::TextDiagnosticBuffer;
  explicit DiagnosticsBuffer() : buffer(new Base()) {}
  Base *get() { return buffer.get(); }
  void flush(DiagnosticsEngine &engine) { buffer->FlushDiagnostics(engine); }
  std::unique_ptr<Base> buffer;
};

//
// Diagnostics printer
//
struct DiagnosticsPrinter {
  using Base = clang::TextDiagnosticPrinter;
  explicit DiagnosticsPrinter(DiagnosticsOptions &opts, const std::string &path)
      : printer(new Base(llvm::errs(), opts.get())) {}
  Base *get() { return printer.get(); }
  std::unique_ptr<Base> printer;
};

//
// diagnostics
//
static LlvmCntPtr<clang::DiagnosticIDs> makeIds() {
  return new clang::DiagnosticIDs();
}

struct BufferedDiagnostics {
  explicit BufferedDiagnostics(llvm::ArrayRef<const char *> argv)
      : opts(argv), buffer(), engine(makeIds(), opts, buffer.get(), false) {
    clang::ProcessWarningOptions(engine, *opts, /* ReportDiags */ false);
  }
  void flush() { buffer.flush(engine); }
  DiagnosticsOptions opts;
  DiagnosticsBuffer buffer;
  DiagnosticsEngine engine;
};

struct ErrorsDiagnostics {
  explicit ErrorsDiagnostics(llvm::ArrayRef<const char *> argv,
                             const std::string &path)
      : opts(argv), printer(opts, path),
        engine(makeIds(), opts, printer.get(), false) {
    if (!opts->DiagnosticSerializationFile.empty()) {
      auto consumer = clang::serialized_diags::create(
          opts->DiagnosticSerializationFile, opts.get(),
          /* MergeChildRecords */ true);

      engine.setClient(new clang::ChainedDiagnosticConsumer(
          printer.get(), std::move(consumer)));
    }

    clang::ProcessWarningOptions(engine, *opts, /* ReportDiags */ false);
  }

  void finish() { engine.getClient()->finish(); }

  DiagnosticsOptions opts;
  DiagnosticsPrinter printer;
  DiagnosticsEngine engine;
};
} // namespace cudaq
