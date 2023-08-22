
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/ChainedDiagnosticConsumer.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

#include "nvqpp_args.h"

namespace cudaq {
using diagnostics_engine = clang::DiagnosticsEngine;

template <typename T>
using llvm_cnt_ptr = llvm::IntrusiveRefCntPtr<T>;

//
// diagnostics options
//
struct diagnostics_options : llvm_cnt_ptr<clang::DiagnosticOptions> {
  using base = llvm_cnt_ptr<clang::DiagnosticOptions>;

  diagnostics_options(argv_t argv)
      : base(clang::CreateAndPopulateDiagOpts(argv)) {}
};

//
// diagnostics buffer
//
struct diagnostics_buffer {
  using base = clang::TextDiagnosticBuffer;

  explicit diagnostics_buffer() : buffer(new base()) {}

  base *get() { return buffer.get(); }

  void flush(diagnostics_engine &engine) { buffer->FlushDiagnostics(engine); }

  std::unique_ptr<base> buffer;
};

struct diagnostics_printer {
  using printer_base = clang::TextDiagnosticPrinter;

  explicit diagnostics_printer(diagnostics_options &opts,
                               const std::string &path)
      : printer(new printer_base(llvm::errs(), opts.get())) {}

  printer_base *get() { return printer.get(); }

  std::unique_ptr<printer_base> printer;
};

//
// diagnostics
//
using ids = llvm_cnt_ptr<clang::DiagnosticIDs>;

static ids make_ids() { return new clang::DiagnosticIDs(); }

struct buffered_diagnostics {

  explicit buffered_diagnostics(llvm::ArrayRef<const char *> argv)
      : opts(argv), buffer(), engine(make_ids(), opts, buffer.get(), false) {
    clang::ProcessWarningOptions(engine, *opts, /* ReportDiags */ false);
  }

  void flush() { buffer.flush(engine); }

  diagnostics_options opts;
  diagnostics_buffer buffer;
  diagnostics_engine engine;
};

struct errs_diagnostics {

  explicit errs_diagnostics(llvm::ArrayRef<const char *> argv,
                            const std::string &path)
      : opts(argv), printer(opts, path),
        engine(make_ids(), opts, printer.get(), false) {
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

  diagnostics_options opts;
  diagnostics_printer printer;
  diagnostics_engine engine;
};

} // namespace cudaq
