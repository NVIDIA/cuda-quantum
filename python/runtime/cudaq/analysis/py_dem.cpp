/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "nlohmann/json.hpp"
#include "py_dem.h"
#include "utils/NanobindAdaptors.h"
#include "cudaq/algorithms/dem.h"
#include "cudaq/platform.h"
#include <stdexcept>
#include <string>

using namespace cudaq;

/// @brief Parse a Python dict of DEM options into a `cudaq::dem_options`
/// struct.
///
/// Recognised keys (all optional):
///   - decompose_errors (bool)
///   - fold_loops (bool)
///   - allow_gauge_detectors (bool)
///   - approximate_disjoint_errors_threshold (float)
///   - ignore_decomposition_failures (bool)
///   - block_decomposition_from_introducing_remnant_edges (bool)
///   - return_measurement_matrices (bool)
static cudaq::dem_options parseDemOptions(const nanobind::dict &d) {
  cudaq::dem_options opts;
  for (auto [k, v] : d) {
    std::string key = nanobind::cast<std::string>(k);
    if (key == "decompose_errors")
      opts.decompose_errors = nanobind::cast<bool>(v);
    else if (key == "fold_loops")
      opts.fold_loops = nanobind::cast<bool>(v);
    else if (key == "allow_gauge_detectors")
      opts.allow_gauge_detectors = nanobind::cast<bool>(v);
    else if (key == "approximate_disjoint_errors_threshold")
      opts.approximate_disjoint_errors_threshold = nanobind::cast<double>(v);
    else if (key == "ignore_decomposition_failures")
      opts.ignore_decomposition_failures = nanobind::cast<bool>(v);
    else if (key == "block_decomposition_from_introducing_remnant_edges")
      opts.block_decomposition_from_introducing_remnant_edges =
          nanobind::cast<bool>(v);
    else if (key == "return_measurement_matrices")
      opts.return_measurement_matrices = nanobind::cast<bool>(v);
    else
      throw std::invalid_argument("dem_options: unknown key '" + key + "'");
  }
  return opts;
}

static void construct_dem_policy(dem_policy *self,
                                 const std::string &kernelName,
                                 const noise_model *noise,
                                 const nanobind::dict &options) {
  new (self) dem_policy();
  self->kernelName = kernelName;
  self->noiseModel = noise;
  self->options = parseDemOptions(options);
}

static dem_result launch_dem(const dem_policy &policy,
                             nanobind::callable callable) {
  return cudaq::detail::launchDem(policy, cudaq::get_platform(),
                                  [&]() { callable(); });
}

void cudaq::bindDemFromKernel(nanobind::module_ &mod) {
  nanobind::class_<dem_policy>(mod, "DemPolicy")
      .def("__init__", construct_dem_policy, nanobind::arg("kernel_name"),
           nanobind::arg("noise_model").none(), nanobind::arg("options"),
           nanobind::keep_alive<1, 3>())
      .def_prop_ro("kernel_name",
                   [](const dem_policy &policy) { return policy.kernelName; })
      .def_prop_ro("return_measurement_matrices", [](const dem_policy &policy) {
        return policy.options.return_measurement_matrices;
      });

  nanobind::class_<dem_result>(mod, "DEMResult", nanobind::dynamic_attr())
      .def(
          "__init__",
          [](dem_result *self, const std::string &dem,
             const std::vector<std::vector<std::size_t>> &m2d,
             const std::vector<std::vector<std::size_t>> &m2o,
             std::size_t num_detectors, std::size_t num_observables,
             std::size_t num_measurements, const nlohmann::json &annotations) {
            cudaq::M2DSparseMatrix m2d_mat;
            m2d_mat.rows = m2d;
            cudaq::M2OSparseMatrix m2o_mat;
            m2o_mat.rows = m2o;
            bool matrices_computed = !m2d.empty() || !m2o.empty();
            new (self)
                dem_result(dem, std::move(m2d_mat), std::move(m2o_mat),
                           num_detectors, num_observables, num_measurements,
                           matrices_computed, cudaq_json(annotations));
          },
          nanobind::arg("dem"),
          nanobind::arg("m2d") = std::vector<std::vector<std::size_t>>{},
          nanobind::arg("m2o") = std::vector<std::vector<std::size_t>>{},
          nanobind::arg("num_detectors") = std::size_t(0),
          nanobind::arg("num_observables") = std::size_t(0),
          nanobind::arg("num_measurements") = std::size_t(0),
          nanobind::arg("annotations") = nlohmann::json::object())
      .def_prop_ro(
          "dem",
          [](const dem_result &self) -> const std::string & {
            return self.get_dem();
          },
          "DEM text in Stim's ``.dem`` format.")
      .def_prop_ro(
          "m2d",
          [](const dem_result &self)
              -> const std::vector<std::vector<std::size_t>> & {
            return self.get_m2d().rows;
          },
          "Measurement-to-detector row lists (neutral C++ form).")
      .def_prop_ro(
          "m2o",
          [](const dem_result &self)
              -> const std::vector<std::vector<std::size_t>> & {
            return self.get_m2o().rows;
          },
          "Measurement-to-observable row lists (neutral C++ form).")
      .def_prop_ro("num_detectors", &dem_result::get_num_detectors)
      .def_prop_ro("num_observables", &dem_result::get_num_observables)
      .def_prop_ro("num_measurements", &dem_result::get_num_measurements)
      .def_prop_ro("matrices_computed", &dem_result::get_matrices_computed,
                   "True when m2d / m2o were populated.")
      .def_prop_ro(
          "annotations",
          [](dem_result &self) -> const nlohmann::json & {
            auto &j = self.get_annotations().get();
            if (j.is_null())
              j = nlohmann::json::object();
            return j;
          },
          "Read-only metadata dict set by backends.")
      .def("__str__", [](const dem_result &self) { return self.get_dem(); })
      .def("__repr__",
           [](const dem_result &self) {
             return "DEMResult(detectors=" +
                    std::to_string(self.get_num_detectors()) +
                    ", observables=" +
                    std::to_string(self.get_num_observables()) +
                    ", measurements=" +
                    std::to_string(self.get_num_measurements()) + ")";
           })
      .def_prop_ro(
          "m2d_matrix",
          [](const dem_result &self) -> nanobind::object {
            if (!self.get_matrices_computed())
              return nanobind::none();
            auto self_obj = nanobind::find(self);
            if (self_obj.is_valid() &&
                nanobind::hasattr(self_obj, "_m2d_cache"))
              return nanobind::getattr(self_obj, "_m2d_cache");
            auto dem_mod = nanobind::module_::import_("cudaq.runtime.dem");
            auto result = dem_mod.attr("_make_csr")(
                nanobind::cast(self.get_m2d().rows),
                nanobind::cast(self.get_num_measurements()));
            if (self_obj.is_valid())
              nanobind::setattr(self_obj, "_m2d_cache", result);
            return result;
          },
          "scipy CSR matrix (num_detectors x num_measurements), or None when "
          "matrices were not requested.")
      .def_prop_ro(
          "m2o_matrix",
          [](const dem_result &self) -> nanobind::object {
            if (!self.get_matrices_computed())
              return nanobind::none();
            auto self_obj = nanobind::find(self);
            if (self_obj.is_valid() &&
                nanobind::hasattr(self_obj, "_m2o_cache"))
              return nanobind::getattr(self_obj, "_m2o_cache");
            auto dem_mod = nanobind::module_::import_("cudaq.runtime.dem");
            auto result = dem_mod.attr("_make_csr")(
                nanobind::cast(self.get_m2o().rows),
                nanobind::cast(self.get_num_measurements()));
            if (self_obj.is_valid())
              nanobind::setattr(self_obj, "_m2o_cache", result);
            return result;
          },
          "scipy CSR matrix (num_observables x num_measurements), or None "
          "when matrices were not requested.")
      .def_static(
          "from_matrices",
          [](const std::string &dem, nanobind::object m2d_csr,
             nanobind::object m2o_csr, std::size_t num_detectors,
             std::size_t num_observables, nanobind::object num_measurements,
             nanobind::object annotations) -> dem_result {
            auto get_shape = [](nanobind::object matrix,
                                const char *matrix_name) {
              auto shape = nanobind::cast<std::vector<std::size_t>>(
                  matrix.attr("shape"));
              if (shape.size() != 2)
                throw std::invalid_argument(std::string(matrix_name) +
                                            " must be a two-dimensional CSR "
                                            "matrix.");
              return shape;
            };
            auto m2d_shape = get_shape(m2d_csr, "m2d_csr");
            auto m2o_shape = get_shape(m2o_csr, "m2o_csr");
            const auto m2d_width = m2d_shape[1];
            const auto m2o_width = m2o_shape[1];
            if (m2d_width != m2o_width)
              throw std::invalid_argument(
                  "m2d_csr and m2o_csr must have the same number of "
                  "columns (measurements).");

            const auto resolved_num_measurements =
                num_measurements.is_none()
                    ? m2d_width
                    : nanobind::cast<std::size_t>(num_measurements);
            if (resolved_num_measurements != m2d_width)
              throw std::invalid_argument(
                  "num_measurements must match the number of columns in "
                  "m2d_csr and m2o_csr.");

            auto dem_mod = nanobind::module_::import_("cudaq.runtime.dem");
            auto csr_to_rows = dem_mod.attr("_csr_to_rows");
            auto m2d_rows =
                nanobind::cast<std::vector<std::vector<std::size_t>>>(
                    csr_to_rows(m2d_csr));
            auto m2o_rows =
                nanobind::cast<std::vector<std::vector<std::size_t>>>(
                    csr_to_rows(m2o_csr));

            cudaq::M2DSparseMatrix m2d_mat;
            m2d_mat.rows = std::move(m2d_rows);
            cudaq::M2OSparseMatrix m2o_mat;
            m2o_mat.rows = std::move(m2o_rows);
            bool matrices_computed =
                !m2d_mat.rows.empty() || !m2o_mat.rows.empty();
            cudaq_json ann;
            if (!annotations.is_none())
              ann = cudaq_json(nanobind::cast<nlohmann::json>(annotations));
            return dem_result(dem, std::move(m2d_mat), std::move(m2o_mat),
                              num_detectors, num_observables,
                              resolved_num_measurements, matrices_computed,
                              std::move(ann));
          },
          nanobind::arg("dem"), nanobind::arg("m2d_csr"),
          nanobind::arg("m2o_csr"),
          nanobind::arg("num_detectors") = std::size_t(0),
          nanobind::arg("num_observables") = std::size_t(0),
          nanobind::arg("num_measurements") = nanobind::none(),
          nanobind::arg("annotations") = nanobind::none(),
          "Build a DEMResult from scipy CSR matrices.");

  mod.def("launch_dem", launch_dem, "Policy based DEM launch.",
          nanobind::arg("policy"), nanobind::arg("callable"));
}
