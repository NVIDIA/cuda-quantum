/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_resource_count.h"
#include "common/Resources.h"
#include "common/cudaq_json.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/JsonNanobindAdaptors.h"
#include "utils/OpaqueArguments.h"
#include "cudaq/algorithms/estimate/policy.h"
#include "cudaq/algorithms/launch.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <string>

using namespace cudaq;

static estimate_result
estimate_impl(const std::string &kernelName, MlirModule kernelMod,
              std::optional<std::function<bool()>> choice,
              nanobind::args args) {
  auto &platform = cudaq::get_platform();
  args = simplifiedValidateInputArguments(args);

  ExecutionContext ctx("resource-count", 1);
  ctx.kernelName = kernelName;
  // Indicate that this is not an async exec
  ctx.asyncExec = false;

  // Set the choice function for the simulator
  if (!choice) {
    auto seed = cudaq::get_random_seed();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> rand(0, 1);
    choice = [gen = std::move(gen), rand = std::move(rand)]() mutable {
      return rand(gen);
    };
  }

  estimate_policy policy{
      .kernelName = kernelName,
      .choice = *std::move(choice),
  };
  return detail::launch(policy, 0, ctx, platform, [&]() {
    // Pass nullptr for the compiled slot to disable JIT-artifact caching:
    // the resource-counter hooks are installed in the scope above and only
    // fire while the kernel is freshly JIT-compiled. A cached binary would
    // bypass them.
    [[maybe_unused]] auto result =
        cudaq::marshal_and_launch_module(kernelName, kernelMod, args);
  });
}

static Resources
estimate_resources_impl(const std::string &kernelName, MlirModule kernelMod,
                        std::optional<std::function<bool()>> choice,
                        nanobind::args args) {
  return estimate_impl(kernelName, kernelMod, choice, args).get_resources();
}

void cudaq::bindCountResources(nanobind::module_ &mod) {
  nanobind::class_<estimate_result>(
      mod, "EstimateResult",
      "A data-type containing the results of a call to :func:`cudaq.estimate`.")
      .def(
          "__init__",
          [](estimate_result *self, const Resources &resources,
             const nlohmann::json &annotations) {
            new (self) estimate_result(resources, cudaq_json(annotations));
          },
          nanobind::arg("resources") = Resources{},
          nanobind::arg("annotations") = nlohmann::json::object(),
          R"#(Construct an EstimateResult.

Args:
  resources (:class:`Resources`, optional): The gate counts. Defaults to an
    empty `Resources`.
  annotations (dict, optional): Metadata dict for anything the fixed
    `Resources` fields cannot express.)#")
      .def_prop_ro(
          "resources",
          [](estimate_result &self) -> const Resources & {
            return self.get_resources();
          },
          nanobind::rv_policy::reference_internal,
          "The :class:`Resources` gate counts for the estimated kernel.")
      .def_prop_ro(
          "annotations",
          [](estimate_result &self) -> const nlohmann::json & {
            return self.get_annotations().get();
          },
          nanobind::rv_policy::reference_internal,
          "Additional metadata dict set by backends.")
      .def(
          "__repr__",
          [](estimate_result &self) {
            const auto resourcesRepr = nanobind::cast<std::string>(
                nanobind::repr(nanobind::cast(self.get_resources())));

            const auto &annotations = self.get_annotations().get();
            if (annotations.empty())
              return "EstimateResult(" + resourcesRepr + ")";

            const auto annotationsRepr = nanobind::cast<std::string>(
                nanobind::repr(nanobind::cast(annotations)));
            return "EstimateResult(" + resourcesRepr +
                   ", annotations=" + annotationsRepr + ")";
          },
          "A Pythonic representation of EstimateResult.");

  mod.def("estimate_impl", estimate_impl, nanobind::arg("kernel_name"),
          nanobind::arg("kernel_mod"), nanobind::arg("choice").none(),
          nanobind::arg("args"), "See python documentation for estimate.");
  mod.def("estimate_resources_impl", estimate_resources_impl,
          nanobind::arg("kernel_name"), nanobind::arg("kernel_mod"),
          nanobind::arg("choice").none(), nanobind::arg("args"),
          "See python documentation for estimate_resources.");
}
