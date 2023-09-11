/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "py_qubit_qis.h"
#include "cudaq/qis/qubit_qis.h"
#include <fmt/core.h>
#include <pybind11/stl.h>

namespace cudaq {

namespace details {

/// @brief Analyze the incoming arguments and map them to a vector
/// of qubit indices.
std::vector<std::size_t> analyzeAndHandlePyArgs(py::args &args) {
  if (args.size() == 1) {
    // Is this arg a qubit or a qvector/qview?
    if (py::isinstance<qubit &>(args[0])) {
      return {args[0].cast<qubit &>().id()};
    }

    // This has to be a broadcast over a register
    std::vector<std::size_t> mapped;
    if (py::isinstance<qvector<> &>(args[0])) {
      auto &casted = args[0].cast<qvector<> &>();
      std::transform(casted.begin(), casted.end(), std::back_inserter(mapped),
                     [](auto &&el) { return el.id(); });
    } else if (py::isinstance<qview<> &>(args[0])) {
      auto &casted = args[0].cast<qview<> &>();
      std::transform(casted.begin(), casted.end(), std::back_inserter(mapped),
                     [](auto &&el) { return el.id(); });
    }

    return mapped;
  }

  // There are multiple args here
  std::vector<std::size_t> mapped;
  for (auto &arg : args) {
    if (py::isinstance<qubit &>(arg))
      mapped.push_back(arg.cast<qubit &>().id());
    else if (py::isinstance<qvector<> &>(arg)) {
      auto &casted = arg.cast<qvector<> &>();
      std::transform(casted.begin(), casted.end(), std::back_inserter(mapped),
                     [](auto &&el) { return el.id(); });
    } else if (py::isinstance<qview<> &>(arg)) {
      auto &casted = arg.cast<qview<> &>();
      std::transform(casted.begin(), casted.end(), std::back_inserter(mapped),
                     [](auto &&el) { return el.id(); });
    } else
      throw std::runtime_error("Invalid type passed to a quantum operation.");
  }

  return mapped;
}

/// @brief Analyze the input arguments for a controlled operation
/// and map them to a tuple of qubit indices and possible qubit control
/// negations.
std::tuple<std::vector<std::size_t>, std::vector<bool>>
analyzeAndHandlePyArgsCtrl(py::args &args) {
  // Handle the following situations
  // 1. op.ctrl(qubits...)
  // 2. op.ctrl([qvector/qview], qubit)
  // Last arg must be a qubit type
  if (!py::isinstance<qubit &>(args[args.size() - 1]))
    throw std::runtime_error("Last argument on .ctrl must be of qubit type.");

  std::vector<std::size_t> mapped;
  std::vector<bool> isNegated;
  for (auto &arg : args) {
    if (py::isinstance<qubit &>(arg)) {
      mapped.push_back(arg.cast<qubit &>().id());
      isNegated.emplace_back(arg.cast<qubit &>().is_negative());
      if (isNegated.back())
        arg.cast<qubit &>().negate();
    } else if (py::isinstance<qvector<> &>(arg)) {
      auto &casted = arg.cast<qvector<> &>();
      std::transform(casted.begin(), casted.end(), std::back_inserter(mapped),
                     [&](auto &&el) {
                       isNegated.emplace_back(
                           arg.cast<qubit &>().is_negative());
                       if (isNegated.back())
                         arg.cast<qubit &>().negate();
                       return el.id();
                     });
    } else if (py::isinstance<qview<> &>(arg)) {
      auto &casted = arg.cast<qview<> &>();
      std::transform(casted.begin(), casted.end(), std::back_inserter(mapped),
                     [&](auto &&el) {
                       isNegated.emplace_back(el.is_negative());
                       if (isNegated.back())
                         el.negate();
                       return el.id();
                     });
    }
  }
  return std::make_tuple(mapped, isNegated);
}

/// @brief Bind the quantum operation of given type. This function
/// is for all single qubit operations.
template <typename QuantumOp>
void bindQuantumOperation(py::module &mod) {
  QuantumOp op;
  py::class_<QuantumOp>(mod, op.name.c_str(), "")
      .def(py::init<>())
      .def_static(
          "__call__", [&](py::args &args) { op(analyzeAndHandlePyArgs(args)); },
          fmt::format(
              "Apply the {} operation on the given input quantum arguments",
              op.name)
              .c_str())
      .def_static(
          "ctrl",
          [&](py::args &args) {
            auto [mapped, isNegated] = analyzeAndHandlePyArgsCtrl(args);
            op.ctrl(mapped, isNegated);
          },
          fmt::format(
              "Apply a controlled version of the {} operation. The last "
              "argument "
              "must be of qubit type. The first N-1 arguments can be of qubit, "
              "qvector, or qview types and represent any control qubits.",
              op.name)
              .c_str())
      // Only allow .adj on single qubit operations
      .def_static(
          "adj", [&](py::args &args) { op.adj(analyzeAndHandlePyArgs(args)); },
          fmt::format("Apply the adjoint of {}.", op.name).c_str());
}

/// @brief Bind the quantum operation of given type. This function
/// is for all single qubit, parameterized operations.
template <typename QuantumOp>
void bindQuantumOperationWithParameter(py::module &mod) {
  QuantumOp op;
  py::class_<QuantumOp>(mod, op.name.c_str(), "")
      .def(py::init<>())
      .def_static(
          "__call__",
          [&](double angle, py::args &args) {
            // Should handle
            auto mapped = analyzeAndHandlePyArgs(args);
            op(angle, mapped);
          },
          fmt::format("Apply the parameterized {} operation on the given input "
                      "quantum arguments.",
                      op.name)
              .c_str())
      .def_static(
          "ctrl",
          [&](double angle, py::args &args) {
            auto [mapped, isNegated] = analyzeAndHandlePyArgsCtrl(args);
            op.ctrl(angle, mapped, isNegated);
          },
          "Apply a controlled version of the parameterized {} operation. The "
          "last argument must be of qubit type. The first N-1 arguments are "
          "the parameter and the control qubits.")
      .def_static(
          "adj", [&](double angle, qubit &qubit) { op(-angle, {qubit.id()}); },
          fmt::format("Apply the adjoint of {}.", op.name).c_str());
}
} // namespace details

void bindQIS(py::module &mod) {

  py::class_<qubit>(
      mod, "qubit",
      "The qubit is the primary unit of information in a quantum computer. "
      "Qubits can be created individually or as part of larger registers.")
      .def(py::init<>())
      .def(
          "__invert__", [](qubit &self) -> qubit & { return !self; },
          "Negate the control qubit.")
      .def(
          "id", [](qubit &self) { return self.id(); },
          "Return a unique integer identifier for this qubit.");

  py::class_<qview<>>(mod, "qview",
                      "A non-owning view on a register of qubits.")
      .def(
          "size", [](qview<> &self) { return self.size(); },
          "Return the number of qubits in this view.")
      .def(
          "front", [](qview<> &self) -> qubit & { return self.front(); },
          "Return first qubit in this view.")
      .def(
          "front",
          [](qview<> &self, std::size_t count) { return self.front(count); },
          "Return first `count` qubits in this view.")
      .def(
          "back", [](qview<> &self) -> qubit & { return self.back(); },
          "Return the last qubit in this view.")
      .def(
          "back",
          [](qview<> &self, std::size_t count) { return self.back(count); },
          "Return the last `count` qubits in this view.")
      .def(
          "__iter__",
          [](qview<> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>())
      .def(
          "slice",
          [](qview<> &self, std::size_t start, std::size_t count) {
            return self.slice(start, count);
          },
          "Return the `[start, start+count]` qudits as a non-owning qview.")
      .def("__getitem__", &qview<>::operator[],
           py::return_value_policy::reference,
           "Return the qubit at the given index.");

  py::class_<qvector<>>(
      mod, "qvector",
      "An owning, dynamically sized container for qubits. The semantics of the "
      "`qvector` follows that of a `std::vector` or `list` for qubits.")
      .def(py::init<std::size_t>())
      .def(
          "size", [](qvector<> &self) { return self.size(); },
          "Return the number of qubits in this `qvector`.")
      .def(
          "front",
          [](qvector<> &self, std::size_t count) { return self.front(count); },
          "Return first `count` qubits in this `qvector` as a non-owning view.")
      .def(
          "front", [](qvector<> &self) -> qubit & { return self.front(); },
          py::return_value_policy::reference,
          "Return first qubit in this `qvector`.")
      .def(
          "back", [](qvector<> &self) -> qubit & { return self.back(); },
          py::return_value_policy::reference,
          "Return the last qubit in this `qvector`.")
      .def(
          "back",
          [](qvector<> &self, std::size_t count) { return self.back(count); },
          "Return the last `count` qubits in this `qvector` as a non-owning "
          "view.")
      .def(
          "__iter__",
          [](qvector<> &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>())
      .def(
          "slice",
          [](qvector<> &self, std::size_t start, std::size_t count) {
            return self.slice(start, count);
          },
          "Return the `[start, start+count]` qudits as a non-owning qview.")
      .def("__getitem__", &qvector<2>::operator[],
           py::return_value_policy::reference,
           "Return the qubit at the given index.");

  // Bind the quantum operations
  details::bindQuantumOperation<cudaq::types::h>(mod);
  details::bindQuantumOperation<cudaq::types::x>(mod);
  details::bindQuantumOperation<cudaq::types::y>(mod);
  details::bindQuantumOperation<cudaq::types::z>(mod);
  details::bindQuantumOperation<cudaq::types::t>(mod);
  details::bindQuantumOperation<cudaq::types::s>(mod);
  details::bindQuantumOperationWithParameter<cudaq::types::rx>(mod);
  details::bindQuantumOperationWithParameter<cudaq::types::ry>(mod);
  details::bindQuantumOperationWithParameter<cudaq::types::rz>(mod);
  details::bindQuantumOperationWithParameter<cudaq::types::r1>(mod);

  py::class_<cudaq::types::swap>(mod, "swap", "")
      .def(py::init<>())
      .def_static(
          "__call__",
          [&](py::args &args) {
            auto vec = details::analyzeAndHandlePyArgs(args);
            if (vec.size() != 2)
              throw std::runtime_error("Invalid arguments passed to swap.");
            cudaq::types::swap()(vec[0], vec[1]);
          },
          "Apply the swap operation on the given input quantum arguments")
      .def_static(
          "ctrl",
          [&](py::args &args) {
            cudaq::types::swap swapOp;
            auto [mapped, isNegated] =
                details::analyzeAndHandlePyArgsCtrl(args);
            swapOp.ctrl(mapped, isNegated);
          },
          "Apply a controlled version of the swap operation. The last "
          "argument "
          "must be of qubit type. The first N-1 arguments can be of qubit, "
          "qvector, or qview types and represent any control qubits.");

  mod.def(
      "reset", [](qubit &q) { reset(q); }, "Reset the state of the qubit.");

  // Bind measure operations
  mod.def(
      "mx",
      [](qubit &q, std::string register_name) { return mx(q, register_name); },
      py::arg("target"), py::arg("register_name") = "",
      "Measure the given qubit along the `x-axis` and return a boolean.");
  mod.def(
      "my",
      [](qubit &q, std::string register_name) { return my(q, register_name); },
      py::arg("target"), py::arg("register_name") = "",
      "Measure the given qubit along the `y-axis` and return a boolean.");
  mod.def(
      "mz",
      [](qubit &q, std::string register_name) { return mz(q, register_name); },
      py::arg("target"), py::arg("register_name") = "",
      "Measure the given qubit along the `z-axis` and return a boolean.");

  mod.def(
      "mz", [](qvector<> &q) { return mz(q); }, py::arg("target"), "");
  mod.def(
      "mz", [](qview<> &q) { return mz(q); }, py::arg("target"), "");

  mod.def(
      "adjoint",
      [](py::object kernel, py::args &args) {
        cudaq::adjoint([&]() { kernel(*args); });
      },
      "Return the adjoint of the input kernel.");

  mod.def(
      "control",
      [](py::object kernel, py::list &controlQubits, py::args &args) {
        std::vector<std::size_t> controlIds;
        for (std::size_t i = 0; i < controlQubits.size(); i++)
          controlIds.push_back(
              controlQubits[i].attr("id")().cast<std::size_t>());
        cudaq::control([&]() { kernel(*args); }, controlIds);
      },
      "Return the general multi-controlled version of the input kernel.");

  mod.def("compute_action", [](py::object compute, py::object action) {
    cudaq::compute_action([&]() { compute(); }, [&]() { action(); });
  });
}
} // namespace cudaq