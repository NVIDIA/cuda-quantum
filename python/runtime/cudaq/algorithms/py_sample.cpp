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

static void construct_sample_policy(sample_policy *self, ExecutionContext &ctx,
                                    const std::string &kernelName,
                                    bool explicitMeasurements) {
  new (self) sample_policy();
  self->kernelName = kernelName;
  self->options.shots = ctx.shots;
  self->options.explicit_measurements = explicitMeasurements;
  self->noiseModel = get_platform().get_noise(ctx.qpuId);
  // For now the noise model has to be duplicated on the policy unfortunately.
  // get_noise() still expects it on the execution context.
  // TODO: Store the noise on the platform or QPU instead.
  ctx.noiseModel = self->noiseModel;
}

static sample_result launch_sample(const sample_policy &policy,
                                   ExecutionContext &ctx,
                                   nanobind::callable callable) {
  auto &platform = get_platform();
  if (platform.is_remote()) {
    async_sample_policy async_policy;
    async_policy.inner = policy;
    auto res = detail::launch(async_policy, ctx.qpuId, ctx, platform,
                              [&]() { callable(); });
    return res.get();
  }
  return detail::launch(policy, ctx.qpuId, ctx, get_platform(),
                        [&]() { callable(); });
}

void cudaq::bindPySample(nanobind::module_ &mod) {
  nanobind::class_<sample_policy>(mod, "SamplePolicy")
      .def("__init__", construct_sample_policy, nanobind::arg("ctx"),
           nanobind::arg("kernel_name"), nanobind::arg("explicit_measurements"))
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
          nanobind::arg("policy"), nanobind::arg("ctx"),
          nanobind::arg("callable"));
}
