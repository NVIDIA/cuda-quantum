/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/CompileTarget.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/platform/default/DefaultQPU.h"
#include <memory>
#include <nanobind/nanobind.h>

namespace cudaq {

void bindQPUHelperTypes(nanobind::module_ &mod);

/// Wrapper QPU type for a Python object implementing SupportsSampleQPU and/or
/// SupportsObserveQPU.
class PyDynamicQPU : public DefaultQPU {
  nanobind::object pyObject;

public:
  PyDynamicQPU() = default;
  PyDynamicQPU(PyDynamicQPU &&) = default;
  PyDynamicQPU &operator=(PyDynamicQPU &&) = default;
  ~PyDynamicQPU() override;

  /// Build a PyDynamicQPU from a Python object implementing
  /// SupportsSampleQPU and/or SupportsObserveQPU.
  static PyDynamicQPU fromPythonObject(nanobind::object obj) noexcept;

  std::unique_ptr<CompileTarget>
  getCompileTarget(const sample_policy &policy) override;
  std::unique_ptr<CompileTarget>
  getCompileTarget(const observe_policy &policy) override;

  sample_result launchKernel(const sample_policy &policy,
                             const CompiledModule &module,
                             KernelArgs args) override;
  observe_result launchKernel(const observe_policy &policy,
                              const CompiledModule &module,
                              KernelArgs args) override;
};
} // namespace cudaq

namespace nanobind::detail {
template <>
struct type_caster<cudaq::PyDynamicQPU> {
  NB_TYPE_CASTER(cudaq::PyDynamicQPU, const_name("PyDynamicQPU"))

  bool from_python(handle src, uint8_t /*flags*/,
                   cleanup_list * /*cleanup*/) noexcept {
    if (!src.is_valid())
      return false;

    value = cudaq::PyDynamicQPU::fromPythonObject(
        nanobind::borrow<nanobind::object>(src));
    return true;
  }

  static handle from_cpp(std::unique_ptr<cudaq::PyDynamicQPU> &&, rv_policy,
                         cleanup_list *) {
    throw std::runtime_error(
        "PyDynamicQPU cannot be converted from C++ to Python.");
  }
};
} // namespace nanobind::detail
