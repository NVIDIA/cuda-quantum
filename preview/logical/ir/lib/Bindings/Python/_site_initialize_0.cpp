//===- _site_initialize_0.cpp - Dialect auto-registration --------------===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// MLIR's Python bindings load every module named `_site_initialize_<i>`
// found under `_mlir_libs/` at Context construction time and call its
// `register_dialects` function on the per-Context MlirDialectRegistry.
// This is the canonical extension point for downstream projects to make
// their dialects automatically available -- equivalent to what
// MlirContext.append_dialect_registry would do, but transparent to user
// code.
//
// Because QLX is self-contained (no separate _mlirRegisterEverything
// extension), we register the standard MLIR dialects we care about
// (func / arith / cf / llvm) from here too.  A new Context() in this
// package has everything a typical QLX program will use without any
// explicit `load_all_available_dialects` call.
//
// Also eagerly registers every QLX-side pass at module import: the
// pass registry is global, and having it populated once means every
// later parsePassPipeline call (from _qlxRuntime / _qlxJit / or user
// code driving mlir.passmanager.PassManager) resolves correctly.
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

// NanobindAdaptors brings in the type casters that let an
// MlirDialectRegistry parameter accept an mlir.ir.DialectRegistry
// Python object (and similar).
#include "mlir/Bindings/Python/NanobindAdaptors.h"

// Plugin dialect registration: unwrap MlirDialectRegistry → C++ reference.
#include "mlir/CAPI/IR.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

// Standard MLIR dialects auto-registered on every Context.
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/ControlFlow.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/LLVM.h"

// Upstream MLIR transform passes (provides --symbol-dce etc.).  We
// register them from here because the JIT pipelines reference them.
#include "mlir-c/Transforms.h"

// QLX-side dialects.
#include "qlx-c/Dialect/Fabric.h"
#include "qlx-c/Dialect/LVM.h"
#include "qlx-c/Dialect/QLX.h"
#include "qlx-c/Passes.h"

namespace nb = nanobind;

#define QLX_REGISTER_DIALECT(NAME)                                             \
  do {                                                                         \
    MlirDialectHandle handle = mlirGetDialectHandle__##NAME##__();             \
    mlirDialectHandleInsertDialect(handle, registry);                          \
  } while (0)

NB_MODULE(_site_initialize_0, m) {
  m.doc() = "QLX dialect auto-registration for mlir.ir.Context";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    // Standard MLIR dialects our programs and lowerings use.
    QLX_REGISTER_DIALECT(arith);
    QLX_REGISTER_DIALECT(cf);
    QLX_REGISTER_DIALECT(func);
    QLX_REGISTER_DIALECT(llvm);

    // QLX-side dialects.
    QLX_REGISTER_DIALECT(qlx);
    QLX_REGISTER_DIALECT(lvm);
    QLX_REGISTER_DIALECT(fabric);

    // Dynamically-loaded plugin dialects: iterate the paths stored by
    // _qlxRuntime.load_plugin() and register each plugin's dialect
    // extension callbacks so their ops parse in this new context.
    // _qlxRuntime is always loaded before any Context is created (importing
    // qlx loads it), so import_ here is a fast sys.modules lookup.
    try {
      auto rt = nb::module_::import_("cudaq.logical._mlir_libs._qlxRuntime");
      auto paths =
          nb::cast<std::vector<std::string>>(rt.attr("get_plugin_paths")());
      mlir::DialectRegistry &reg = *unwrap(registry);
      for (const auto &path : paths) {
        auto plugin = mlir::DialectPlugin::load(path);
        if (plugin)
          plugin->registerDialectRegistryCallbacks(reg);
      }
    } catch (...) {
      // _qlxRuntime not yet available (e.g. during qlx's own import).
      // Skip: plugin dialects are injected via _native.load_plugin instead.
    }
  });

  // Pass registration.  Eager (at module import) because the pass
  // registry is process-global and we never want a parsePassPipeline
  // call to miss a pass just because the user happened to import
  // _qlxJit or _qlxRuntime before _site_initialize_0 fired.
  m.def("register_passes", []() {
    mlirRegisterTransformsPasses();
    qlxRegisterAllPasses();
  });

  // Fire the pass registration once at import time.  Idempotent.
  mlirRegisterTransformsPasses();
  qlxRegisterAllPasses();
}

#undef QLX_REGISTER_DIALECT
