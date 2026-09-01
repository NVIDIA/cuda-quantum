//===- _qlx_ext.cpp - Main QLX Python extension ----------------------===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// Single NB_MODULE for the QLX dialect surface.  The actual bindings
// are defined in DialectQLX.cpp (qlx submodule) and DialectFabric.cpp
// (fabric submodule); this file just calls their populate* hooks.
//
// Mirrors MLIR upstream's MainModule.cpp pattern, where IRCore.cpp /
// IRTypes.cpp / IRAttributes.cpp contribute chunks to a single _mlir
// extension module via populate* calls.
//
// Runtime helpers (parse / verify / lower / translate / run_pass) live
// in QLXRuntime.cpp (module `_qlxRuntime`) and JIT helpers live in
// QLXJit.cpp (module `_qlxJit`).  Those are functionally separate
// surfaces; only dialect type/attr bindings belong here.
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>

#include "PopulateSubmodules.h"

namespace nb = nanobind;

NB_MODULE(_qlx_ext, m) {
  m.doc() = "QLX Native Extension (dialect type/attr bindings)";

  auto qlxMod = m.def_submodule("qlx", "QLX dialect bindings");
  qlx::python::populateQLXSubmodule(qlxMod);

  auto fabricMod = m.def_submodule("fabric", "Fabric dialect bindings");
  qlx::python::populateFabricSubmodule(fabricMod);
}
