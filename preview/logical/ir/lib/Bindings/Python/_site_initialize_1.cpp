//===- _site_initialize_1.cpp - Dialect auto-registration --------------===//
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
// CUDA-Q owns `_site_initialize_0`. Index allocation is unmanaged: MLIR's
// loader counts up from 0 and stops at the first gap, so a third
// distribution taking index 2 would only load while cudaq-logical is
// installed.
//
// arith / cf / func / llvm are already registered by CUDA-Q's
// `_site_initialize_0` (`cudaqRegisterAllDialects`). Re-inserting them
// from this DSO would emit a second TypeID (Python extensions are
// RTLD_LOCAL) and trip StorageUniquer. Only QLX / LVM / Fabric are
// added here, via C API handles from libQLXPythonCAPI.
//
// Pass registration is *not* done at module import. `import cudaq` loads
// this initializer via `_quakeDialects`; registering the QLX pass set
// there would make every CUDA-Q import pay for it. The `register_passes`
// hook is invoked by `_qlxRuntime` on first use instead.
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "mlir-c/Transforms.h"

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

NB_MODULE(_site_initialize_1, m) {
  m.doc() = "QLX dialect auto-registration for cudaq.mlir.ir.Context";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    QLX_REGISTER_DIALECT(qlx);
    QLX_REGISTER_DIALECT(lvm);
    QLX_REGISTER_DIALECT(fabric);

    // Dynamically-loaded plugin dialects: only consult _qlxRuntime if it
    // is already imported. `import_` would otherwise load it (and its
    // pass registration) during `import cudaq`.
    try {
      nb::dict sysModules =
          nb::cast<nb::dict>(nb::module_::import_("sys").attr("modules"));
      if (!sysModules.contains("cudaq.mlir._mlir_libs._qlxRuntime"))
        return;
      auto rt = nb::module_::import_("cudaq.mlir._mlir_libs._qlxRuntime");
      auto paths =
          nb::cast<std::vector<std::string>>(rt.attr("get_plugin_paths")());
      mlir::DialectRegistry &reg = *unwrap(registry);
      for (const auto &path : paths) {
        auto plugin = mlir::DialectPlugin::load(path);
        if (plugin)
          plugin->registerDialectRegistryCallbacks(reg);
      }
    } catch (...) {
      // _qlxRuntime not yet available. Plugin dialects are injected via
      // _native.load_plugin instead.
    }
  });

  // Pass registration. Invoked by _qlxRuntime on first use rather than at
  // module import, so a bare `import cudaq` does not populate the QLX pass
  // set. The pass registry is process-global and these calls are
  // idempotent on a Release build; a Debug build asserts if a pass is
  // registered more than once, which is why this is no longer eager.
  m.def("register_passes", []() {
    mlirRegisterTransformsPasses();
    qlxRegisterAllPasses();
  });
}

#undef QLX_REGISTER_DIALECT
