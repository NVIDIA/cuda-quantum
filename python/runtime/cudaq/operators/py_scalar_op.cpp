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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "cudaq/operators.h"
#include "cudaq/operators/serialization.h"
#include "py_helpers.h"
#include "py_scalar_op.h"

namespace cudaq {

void bindScalarOperator(py::module_ &mod) {
  using scalar_callback =
      std::function<std::complex<double>(const parameter_map &)>;

  py::class_<scalar_operator>(mod, "ScalarOperator")

      // properties

      .def_prop_ro("parameters", &scalar_operator::get_parameter_descriptions,
                   "Returns a dictionary that maps each parameter "
                   "name to its description.")

      // constructors

      .def(py::init<>(), "Creates a scalar operator with constant value 1.")
      .def(py::init<double>(),
           "Creates a scalar operator with the given constant value.")
      .def(py::init<std::complex<double>>(),
           "Creates a scalar operator with the given constant value.")
      // Callable + positional dict of parameter descriptions.
      // Used by _compose: ScalarOperator(generator, param_dict)
      .def(
          "__init__",
          [](scalar_operator *self, py::object func, py::dict param_info) {
            if (!PyCallable_Check(func.ptr()) ||
                py::isinstance<scalar_operator>(func))
              throw py::next_overload();

            auto helpers = py::module_::import_("cudaq.operators.helpers");
            auto eval_gen = helpers.attr("_evaluate_generator");

            std::unordered_map<std::string, std::string> param_desc;
            for (auto [keyPy, valuePy] : param_info) {
              param_desc[py::cast<std::string>(keyPy)] =
                  py::cast<std::string>(valuePy);
            }

            scalar_callback wrapper =
                [func_ref = py::object(func), eval_fn = py::object(eval_gen)](
                    const parameter_map &params) -> std::complex<double> {
              py::dict pydict;
              for (const auto &[k, v] : params)
                pydict[py::str(k.c_str())] = py::cast(v);
              return py::cast<std::complex<double>>(eval_fn(func_ref, pydict));
            };

            new (self)
                scalar_operator(std::move(wrapper), std::move(param_desc));
          },
          "Creates a scalar operator from a callable with parameter "
          "descriptions dict.")
      // Callable + kwargs for parameter descriptions (or auto-introspect).
      // Used by user code: ScalarOperator(lambda x: x*x)
      // or: ScalarOperator(callback, x="doc for x")
      .def(
          "__init__",
          [](scalar_operator *self, py::object func, const py::kwargs &kwargs) {
            if (!PyCallable_Check(func.ptr()) ||
                py::isinstance<scalar_operator>(func))
              throw py::next_overload();

            auto helpers = py::module_::import_("cudaq.operators.helpers");
            auto eval_gen = helpers.attr("_evaluate_generator");

            std::unordered_map<std::string, std::string> param_desc;
            if (kwargs.size() > 0) {
              param_desc = details::kwargs_to_param_description(kwargs);
            } else {
              // Introspect the function to discover parameters
              auto inspect = py::module_::import_("inspect");
              auto param_docs_fn = helpers.attr("_parameter_docs");
              auto arg_spec = inspect.attr("getfullargspec")(func);

              if (!arg_spec.attr("varargs").is_none())
                throw py::value_error("the function defining a scalar "
                                      "operator must not take *args");

              py::list args = py::cast<py::list>(arg_spec.attr("args"));
              py::list kwonlyargs =
                  py::cast<py::list>(arg_spec.attr("kwonlyargs"));
              py::object doc = func.attr("__doc__");

              for (size_t i = 0; i < args.size(); ++i) {
                std::string name = py::cast<std::string>(args[i]);
                param_desc[name] =
                    py::cast<std::string>(param_docs_fn(name, doc));
              }
              for (size_t i = 0; i < kwonlyargs.size(); ++i) {
                std::string name = py::cast<std::string>(kwonlyargs[i]);
                param_desc[name] =
                    py::cast<std::string>(param_docs_fn(name, doc));
              }
            }

            scalar_callback wrapper =
                [func_ref = py::object(func), eval_fn = py::object(eval_gen)](
                    const parameter_map &params) -> std::complex<double> {
              py::dict pydict;
              for (const auto &[k, v] : params)
                pydict[py::str(k.c_str())] = py::cast(v);
              return py::cast<std::complex<double>>(eval_fn(func_ref, pydict));
            };

            new (self)
                scalar_operator(std::move(wrapper), std::move(param_desc));
          },
          "Creates a scalar operator where the given callback function is "
          "invoked during evaluation.")
      .def(py::init<const scalar_operator &>(), "Copy constructor.")

      // evaluations

      .def(
          "evaluate",
          [](const scalar_operator &self, const py::kwargs &kwargs) {
            return self.evaluate(details::kwargs_to_param_map(kwargs));
          },
          "Evaluated value of the operator.")

      // comparisons

      .def("__eq__", &scalar_operator::operator==, py::is_operator())

      // general utility functions

      .def("is_constant", &scalar_operator::is_constant,
           "Returns true if the scalar is a constant value.")
      .def("__str__", &scalar_operator::to_string,
           "Returns the string representation of the operator.");
}

void bindScalarWrapper(py::module_ &mod) {
  bindScalarOperator(mod);
  py::implicitly_convertible<double, scalar_operator>();
  py::implicitly_convertible<std::complex<double>, scalar_operator>();
}

} // namespace cudaq
