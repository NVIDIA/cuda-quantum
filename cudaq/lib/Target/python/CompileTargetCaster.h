/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Nanobind type caster between `cudaq::CompileTarget` and the `CompileTarget`
/// dataclass of `cudaq.core._target`. Values are deep-copied when converted.
///
/// Include this header in every translation unit whose nanobind bindings
/// mention `cudaq::CompileTarget`.

#include "cudaq/Target/CompileTarget.h"
#include <nanobind/nanobind.h>

namespace nanobind::detail {

template <>
struct type_caster<cudaq::CompileTarget> {
  NB_TYPE_CASTER(cudaq::CompileTarget, const_name("CompileTarget"))

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept;
  static handle from_cpp(const cudaq::CompileTarget &value, rv_policy policy,
                         cleanup_list *cleanup) noexcept;
};

} // namespace nanobind::detail
