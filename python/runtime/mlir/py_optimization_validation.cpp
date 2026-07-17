/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_optimization_validation.h"
#include "common/Resources.h"
#include "cudaq/Optimizer/Analysis/CircuitValidation.h"
#include "cudaq/Optimizer/Transforms/ResourceCount.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <optional>

namespace nb = nanobind;

/// Resolve the single kernel to validate in \p module. When \p name is set, the
/// symbol of that name is used. Otherwise the module must contain exactly one
/// function with a body. On failure a null FuncOp is returned and \p err is
/// set.
static mlir::func::FuncOp findKernel(mlir::ModuleOp module,
                                     const std::optional<std::string> &name,
                                     std::string &err) {
  if (name) {
    auto func =
        mlir::dyn_cast_or_null<mlir::func::FuncOp>(module.lookupSymbol(*name));
    if (!func || func.empty())
      err = "kernel '" + *name + "' not found or has no body";
    return func;
  }
  mlir::func::FuncOp found;
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    if (func.empty())
      continue;
    if (found) {
      err = "module has multiple kernels; specify kernel_name";
      return {};
    }
    found = func;
  }
  if (!found)
    err = "module has no kernel with a body";
  return found;
}

/// Classify a Quake module against the bounded-unitary validation domain.
static nb::dict preflight_bounded_unitary(MlirModule module,
                                          unsigned exactQubitBound) {
  auto status =
      cudaq::opt::checkBoundedUnitaryDomain(unwrap(module), exactQubitBound);
  nb::dict result;
  result["supported"] = status.supported;
  result["max_qubits"] = status.maxQubits;
  nb::list rejections;
  for (const auto &r : status.rejections) {
    nb::dict entry;
    entry["kind"] = std::string(cudaq::opt::toString(r.kind));
    entry["kernel"] = r.kernel;
    entry["detail"] = r.detail;
    rejections.append(entry);
  }
  result["rejections"] = rejections;
  return result;
}

/// Compare the unitaries of two modules exactly, at their current checkpoint.
static nb::dict compare_unitaries(MlirModule baseline, MlirModule candidate,
                                  std::optional<std::string> kernelName,
                                  double rtol, double atol) {
  nb::dict result;
  std::string err;
  auto baseFunc = findKernel(unwrap(baseline), kernelName, err);
  if (!baseFunc) {
    result["computed"] = false;
    result["error"] = "baseline: " + err;
    return result;
  }
  auto candFunc = findKernel(unwrap(candidate), kernelName, err);
  if (!candFunc) {
    result["computed"] = false;
    result["error"] = "candidate: " + err;
    return result;
  }

  auto cmp = cudaq::opt::compareUnitaries(baseFunc, candFunc, rtol, atol);
  result["computed"] = cmp.computed;
  result["strict_equal"] = cmp.strictEqual;
  result["equal_up_to_global_phase"] = cmp.equalUpToGlobalPhase;
  result["phase"] = cmp.phase;
  result["phase_is_zero"] = cmp.phaseIsZero;
  result["error"] = cmp.error;
  result["kernel"] = baseFunc.getSymName().str();
  return result;
}

/// Count resources from a Quake module at an arbitrary compiler checkpoint.
static nb::dict count_resources_checkpoint(MlirModule module) {
  nb::dict result;
  // countResourcesFromIR mutates its input (it erases counted gates), so count
  // on a clone and let the OwningOpRef destroy it when we return.
  mlir::OwningOpRef<mlir::ModuleOp> cloned(unwrap(module).clone());
  auto counts = cudaq::opt::countResourcesFromIR(cloned.get());
  if (mlir::failed(counts)) {
    result["computed"] = false;
    result["error"] =
        "resource counting failed (e.g. a dynamically-sized register)";
    return result;
  }

  result["computed"] = true;
  result["gate_count"] = counts->count();
  result["depth"] = counts->getCircuitDepth();
  result["num_qubits"] = counts->getNumQubits();
  result["two_qubit_count"] = counts->getGateCountByArity(2);
  result["multi_qubit_count"] = counts->getMultiQubitGateCount();
  nb::dict perGate;
  for (const auto &[name, count] : counts->gateCounts())
    perGate[name.c_str()] = count;
  result["per_gate"] = perGate;
  return result;
}

void cudaq::bindOptimizationValidation(nanobind::module_ &mod) {
  mod.def("preflight_bounded_unitary", &preflight_bounded_unitary,
          nb::arg("module"),
          nb::arg("exact_qubit_bound") = cudaq::opt::kDefaultExactQubitBound,
          "Classify a Quake module against the bounded-unitary validation "
          "domain. Returns {supported, max_qubits, rejections[]}.");
  mod.def(
      "compare_unitaries", &compare_unitaries, nb::arg("baseline"),
      nb::arg("candidate"), nb::arg("kernel_name").none() = nb::none(),
      nb::arg("rtol") = 1e-5, nb::arg("atol") = 1e-8,
      "Compare two Quake modules' unitaries exactly (no simulator). Returns "
      "{computed, strict_equal, equal_up_to_global_phase, phase, "
      "phase_is_zero, error, kernel}.");
  mod.def("count_resources_checkpoint", &count_resources_checkpoint,
          nb::arg("module"),
          "Count resources from a Quake module at its current checkpoint "
          "(operates on a clone). Returns {computed, gate_count, depth, "
          "num_qubits, two_qubit_count, multi_qubit_count, per_gate{}}.");
}
