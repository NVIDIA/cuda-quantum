/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_sample.h"
#include "cudaq/algorithms/launch.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/algorithms/sample/policy.h"
#include <nanobind/stl/string.h>

using namespace cudaq;

static void
construct_sample_policy(sample_policy *self, const std::string &kernelName,
                        std::size_t shots, bool explicitMeasurements,
                        std::size_t qpuId, std::size_t batchIteration,
                        std::size_t totalIterations) {
  new (self) sample_policy();
  self->kernelName = kernelName;
  self->options.shots = shots;
  self->options.explicit_measurements = explicitMeasurements;
  self->noiseModel = get_platform().get_noise(qpuId);
  self->batchIteration = batchIteration;
  self->totalIterations = totalIterations;
}

static sample_result launch_sample(const sample_policy &policy,
                                   std::size_t qpuId,
                                   nanobind::callable callable) {
  auto &platform = get_platform();
  if (platform.is_remote()) {
    async_sample_policy async_policy;
    async_policy.inner = policy;
    auto res =
        detail::launch(async_policy, qpuId, platform, [&]() { callable(); });
    return res.get();
  }
  return detail::launch(policy, qpuId, platform, [&]() { callable(); });
}

void cudaq::bindPySample(nanobind::module_ &mod) {
  nanobind::class_<sample_policy>(mod, "SamplePolicy")
      .def("__init__", construct_sample_policy, nanobind::arg("kernel_name"),
           nanobind::arg("shots"), nanobind::arg("explicit_measurements"),
           nanobind::arg("qpu_id") = 0, nanobind::arg("batch_iteration") = 0,
           nanobind::arg("total_iterations") = 0)
      .def_prop_ro(
          "kernel_name",
          [](const sample_policy &policy) { return policy.kernelName; },
          "The kernel name.")
      .def_prop_ro(
          "shots",
          [](const sample_policy &policy) { return policy.options.shots; },
          "The number of shots.")
      .def_prop_ro("explicit_measurements",
                   [](const sample_policy &policy) {
                     return policy.options.explicit_measurements;
                   })
      .def("__repr__", [](const sample_policy &p) {
        return cudaq_fmt::format("SamplePolicy(kernel_name='{}', shots={}, "
                                 "explicit_measurements={})",
                                 p.kernelName, p.options.shots,
                                 p.options.explicit_measurements);
      });

  mod.def("launch_sample", launch_sample, "Policy based sample launch.",
          nanobind::arg("policy"), nanobind::arg("qpu_id"),
          nanobind::arg("callable"));
}
