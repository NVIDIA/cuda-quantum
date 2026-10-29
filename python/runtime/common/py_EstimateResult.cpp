/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_EstimateResult.h"
#include "common/EstimateResult.h"
#include "common/Resources.h"
#include "common/cudaq_json.h"
#include "utils/JsonNanobindAdaptors.h"
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

using namespace cudaq;

namespace {
using InstructionCounts =
    std::map<std::pair<std::string, std::size_t>, std::size_t>;
} // namespace

void cudaq::bindEstimateResult(nanobind::module_ &mod) {
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
          R"#(Construct an EstimateResult from a `Resources`.

Args:
  resources (:class:`Resources`, optional): The gate counts. Defaults to an
    empty `Resources`.
  annotations (dict, optional): Metadata dict for anything the fixed
    `Resources` fields cannot express.)#")
      .def(
          "__init__",
          [](estimate_result *self, const InstructionCounts &instructions,
             std::size_t num_qubits, const nlohmann::json &annotations) {
            std::vector<estimate_result::InstructionCount> entries;
            entries.reserve(instructions.size());
            for (const auto &[key, count] : instructions)
              entries.push_back({key.first, key.second, count});
            new (self)
                estimate_result(entries, num_qubits, cudaq_json(annotations));
          },
          nanobind::arg("instructions"), nanobind::arg("num_qubits") = 0,
          nanobind::arg("annotations") = nlohmann::json::object(),
          R"#(Construct an EstimateResult from instruction counts.

Args:
  instructions (dict): Maps ``(gate_name, num_controls)`` to the number of
    occurrences, used to build the `Resources`. The entries are treated as one
    serial sequence on shared qubits, so `Resources.depth` and
    `Resources.t_depth` are upper bounds (the sum over the entries). Use this
    when the producer has no per-qubit dependency information.
  num_qubits (int, optional): The number of qubits to report.
  annotations (dict, optional): Metadata dict for anything the fixed
    `Resources` fields cannot express.)#")
      .def_prop_ro(
          "resources",
          [](estimate_result &self) -> const Resources & {
            return self.get_resources();
          },
          nanobind::rv_policy::reference_internal,
          "The :class:`Resources` for the estimated kernel. For logical "
          "targets these\n"
          "are the logical resources only; see `annotations` for other tiers.")
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
}
