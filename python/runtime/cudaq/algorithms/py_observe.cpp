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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

using namespace cudaq;

static void construct_observe_policy(
    observe_policy *self, const std::string &kernelName, const spin_op &H,
    int shots, std::size_t qpuId, std::optional<std::size_t> numTrajectories,
    std::size_t batchIteration, std::size_t totalIterations) {
  new (self) observe_policy();
  self->kernelName = kernelName;
  self->options.shots = shots;
  self->spin = cudaq::spin_op::canonicalize(H);
  if (numTrajectories > 0)
    self->options.num_trajectories = numTrajectories;
  self->noiseModel = get_platform().get_noise(qpuId);
  self->batchIteration = batchIteration;
  self->totalIterations = totalIterations;
}

static observe_result launch_observe(const observe_policy &policy,
                                     std::size_t qpuId,
                                     nanobind::callable callable) {
  auto &platform = get_platform();
  if (platform.is_remote()) {
    async_observe_policy async_policy;
    async_policy.inner = policy;
    auto res =
        detail::launch(async_policy, qpuId, platform, [&]() { callable(); });
    return res.get();
  }
  return detail::launch(policy, qpuId, platform, [&]() { callable(); });
}

void cudaq::bindPyObserve(nanobind::module_ &mod) {
  nanobind::class_<observe_policy>(mod, "ObservePolicy")
      .def("__init__", construct_observe_policy, nanobind::arg("kernel_name"),
           nanobind::arg("spin_operator"), nanobind::arg("shots"),
           nanobind::arg("qpu_id") = 0,
           nanobind::arg("num_trajectories") = nanobind::none(),
           nanobind::arg("batch_iteration") = 0,
           nanobind::arg("total_iterations") = 0)
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
          nanobind::arg("policy"), nanobind::arg("qpu_id"),
          nanobind::arg("callable"));
}
