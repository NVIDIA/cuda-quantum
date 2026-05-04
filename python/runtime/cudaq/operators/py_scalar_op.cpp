/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <functional>
#include <unordered_map>

#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_helpers.h"
#include "py_scalar_op.h"

namespace cudaq {

namespace {

std::pair<std::unordered_map<std::string, std::string>, bool>
introspectCallable(const nanobind::callable &func) {
  nanobind::module_ inspect = nanobind::module_::import_("inspect");
  nanobind::object argSpec = inspect.attr("getfullargspec")(func);

  if (!argSpec.attr("varargs").is_none())
    throw std::invalid_argument(
        "the function defining a scalar operator must not take *args");

  nanobind::module_ helpers =
      nanobind::module_::import_("cudaq.operators.helpers");
  nanobind::object paramDocsFn = helpers.attr("_parameter_docs");
  nanobind::object docstring = func.attr("__doc__");

  std::unordered_map<std::string, std::string> paramDesc;
  for (nanobind::handle name : argSpec.attr("args")) {
    std::string n = nanobind::cast<std::string>(name);
    std::string doc = nanobind::cast<std::string>(
        paramDocsFn(nanobind::str(n.c_str()), docstring));
    paramDesc[n] = doc;
  }
  for (nanobind::handle name : argSpec.attr("kwonlyargs")) {
    std::string n = nanobind::cast<std::string>(name);
    std::string doc = nanobind::cast<std::string>(
        paramDocsFn(nanobind::str(n.c_str()), docstring));
    paramDesc[n] = doc;
  }

  bool acceptsKwargs = !argSpec.attr("varkw").is_none();
  return {std::move(paramDesc), acceptsKwargs};
}

scalar_callback wrapPythonCallable(nanobind::callable func,
                                   const std::vector<std::string> &paramNames,
                                   bool acceptsKwargs) {
  return [func = std::move(func), paramNames,
          acceptsKwargs](const parameter_map &params) -> std::complex<double> {
    nanobind::gil_scoped_acquire guard;
    nanobind::dict pyKwargs;
    if (acceptsKwargs) {
      for (const auto &[k, v] : params)
        pyKwargs[k.c_str()] = nanobind::cast(v);
    } else {
      for (const auto &name : paramNames) {
        auto it = params.find(name);
        if (it != params.end())
          pyKwargs[name.c_str()] = nanobind::cast(it->second);
      }
    }
    nanobind::object result = func(**pyKwargs);
    return nanobind::cast<std::complex<double>>(result);
  };
}

} // anonymous namespace

void bindScalarOperator(nanobind::module_ &mod) {

  nanobind::class_<scalar_operator>(mod, "ScalarOperator")

      // properties

      .def_prop_ro("parameters", &scalar_operator::get_parameter_descriptions,
                   "Returns a dictionary that maps each parameter "
                   "name to its description.")

      // constructors

      .def(nanobind::init<>(),
           "Creates a scalar operator with constant value 1.")
      .def(nanobind::init<double>(),
           "Creates a scalar operator with the given constant value.")
      .def(nanobind::init<std::complex<double>>(),
           "Creates a scalar operator with the given constant value.")
      .def(
          "__init__",
          [](scalar_operator *self, nanobind::callable func) {
            auto [paramDesc, acceptsKwargs] = introspectCallable(func);
            std::vector<std::string> paramNames;
            for (const auto &[k, v] : paramDesc)
              paramNames.push_back(k);
            auto callback =
                wrapPythonCallable(std::move(func), paramNames, acceptsKwargs);
            new (self)
                scalar_operator(std::move(callback), std::move(paramDesc));
          },
          nanobind::arg("generator"),
          "Creates a scalar operator from a callable. Parameter names are "
          "introspected from the function signature.")
      .def(
          "__init__",
          [](scalar_operator *self, nanobind::callable func,
             const nanobind::kwargs &kwargs) {
            auto [introspected, acceptsKwargs] = introspectCallable(func);
            auto paramDesc = details::kwargs_to_param_description(kwargs);
            std::vector<std::string> paramNames;
            for (const auto &[k, v] : paramDesc)
              paramNames.push_back(k);
            auto callback =
                wrapPythonCallable(std::move(func), paramNames, acceptsKwargs);
            new (self)
                scalar_operator(std::move(callback), std::move(paramDesc));
          },
          "Creates a scalar operator from a callable with keyword argument "
          "parameter descriptions.")
      .def(nanobind::init<const scalar_operator &>(), "Copy constructor.")

      // evaluations

      .def(
          "evaluate",
          [](const scalar_operator &self, const nanobind::kwargs &kwargs) {
            return self.evaluate(details::kwargs_to_param_map(kwargs));
          },
          "Evaluated value of the operator.")

      // comparisons

      .def("__eq__", &scalar_operator::operator==, nanobind::is_operator())

      // general utility functions

      .def("is_constant", &scalar_operator::is_constant,
           "Returns true if the scalar is a constant value.")
      .def("__str__", &scalar_operator::to_string,
           "Returns the string representation of the operator.");
}

void bindScalarWrapper(nanobind::module_ &mod) {
  bindScalarOperator(mod);
  nanobind::implicitly_convertible<double, scalar_operator>();
  nanobind::implicitly_convertible<std::complex<double>, scalar_operator>();
}

} // namespace cudaq
