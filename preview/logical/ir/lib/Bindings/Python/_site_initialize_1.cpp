/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// MLIR's Python bindings load every module named `_site_initialize_<i>`
// found under `_mlir_libs/` at Context construction time and call its
// `register_dialects` function on the per-Context MlirDialectRegistry.
//
// - `_site_initialize_0` is owned by CUDA-Q.
// - `_site_initialize_1` is owned by cudaq-logical.
//
// Note that this setup is a bit brittle, as if we ever need to create a
// `_site_initialize_2` module, it would only get loaded if both `_0` and `_1`
// are present (and so that would only work whenever `cudaq-logical` is
// installed.)
//
// Dialects are registered at every `MlirContext` creation time. Passes and
// pipelines are registered once globally at module init time.
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"

#include "mlir-c/Transforms.h"

#include "qlx-c/Dialect/Cflow.h"
#include "qlx-c/Dialect/Event.h"
#include "qlx-c/Dialect/Fabric.h"
#include "qlx-c/Dialect/LVM.h"
#include "qlx-c/Dialect/Phys.h"
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
    QLX_REGISTER_DIALECT(phys);
    QLX_REGISTER_DIALECT(fabric);
    QLX_REGISTER_DIALECT(cflow);
    QLX_REGISTER_DIALECT(event);

    // Make dialects contributed by separately installed providers available
    // in every CUDA-Q MLIR context. The runtime owns the process-global list;
    // this hook only applies each provider's registry callbacks.
    try {
      auto runtime = nb::module_::import_("cudaq.mlir._mlir_libs._qlxRuntime");
      auto paths = nb::cast<std::vector<std::string>>(
          runtime.attr("get_plugin_paths")());
      mlir::DialectRegistry &cppRegistry = *unwrap(registry);
      for (const auto &path : paths) {
        auto plugin = mlir::DialectPlugin::load(path);
        if (plugin)
          plugin->registerDialectRegistryCallbacks(cppRegistry);
      }
    } catch (...) {
      // During initial package import the runtime may not yet be available.
      // Explicit plugin loading still registers into the live registry.
    }
  });

  m.def("register_passes", []() {
    mlirRegisterTransformsPasses();
    qlxRegisterAllPasses();
  });

  mlirRegisterTransformsPasses();
  qlxRegisterAllPasses();
}

#undef QLX_REGISTER_DIALECT
