/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_observe.h"
#include "cudaq/algorithms/launch.h"
#include "cudaq/algorithms/observe/policy.h"
#include <nanobind/stl/string.h>

using namespace cudaq;

static void construct_observe_policy(observe_policy *self,
                                     ExecutionContext &ctx,
                                     const std::string &kernelName,
                                     const spin_op &H) {
  new (self) observe_policy();
  self->kernelName = kernelName;
  self->options.shots = static_cast<int>(ctx.shots);
  self->spin = cudaq::spin_op::canonicalize(H);
  if (ctx.numberTrajectories > 0)
    self->options.num_trajectories = ctx.numberTrajectories;
  self->noiseModel = get_platform().get_noise(ctx.qpuId);
  // For now the noise model has to be duplicated on the policy unfortunately.
  // get_noise() still expects it on the execution context.
  // TODO: Store the noise on the platform or QPU instead.
  ctx.noiseModel = self->noiseModel;
  ctx.spin = self->spin;
}

static observe_result launch_observe(const observe_policy &policy,
                                     ExecutionContext &ctx,
                                     nanobind::callable callable) {
  auto &platform = get_platform();
  if (platform.is_remote()) {
    async_observe_policy async_policy;
    async_policy.inner = policy;
    auto res = detail::launch(async_policy, ctx.qpuId, ctx, platform,
                              [&]() { callable(); });
    return res.get();
  }
  return detail::launch(policy, ctx.qpuId, ctx, platform,
                        [&]() { callable(); });
}

void cudaq::bindPyObserve(nanobind::module_ &mod) {
  nanobind::class_<observe_policy>(mod, "ObservePolicy")
      .def("__init__", construct_observe_policy, nanobind::arg("ctx"),
           nanobind::arg("kernel_name"), nanobind::arg("spin_operator"))
      .def_prop_ro(
          "kernel_name",
          [](const observe_policy &policy) { return policy.kernelName; },
          "The kernel name.")
      .def_prop_ro(
          "shots",
          [](const observe_policy &policy) { return policy.options.shots; },
          "The number of shots.")
      .def("__repr__", [](const observe_policy &p) {
        return cudaq_fmt::format("ObservePolicy(kernel_name='{}', shots={})",
                                 p.kernelName, p.options.shots);
      });

  mod.def("launch_observe", launch_observe, "Policy based observe launch.",
          nanobind::arg("policy"), nanobind::arg("ctx"),
          nanobind::arg("callable"));
}
