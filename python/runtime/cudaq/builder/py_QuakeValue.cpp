/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "py_QuakeValue.h"
#include "cudaq/builder/QuakeValue.h"

namespace cudaq {

void bindQuakeValue(py::module &mod) {
  py::class_<QuakeValue>(
      mod, "QuakeValue",
      "A :class:`QuakeValue` represents a handle to an individual function "
      "argument of a :class:`Kernel`, or a return value from an operation "
      "within it. As documented in :func:`make_kernel`, a :class:`QuakeValue` "
      "can hold values of the following types: int, float, list/List, "
      ":class:`qubit`, or :class:`qreg`. The :class:`QuakeValue` can also hold "
      "kernel operations such as qubit allocations and measurements.\n")
      /// @brief Bind the indexing operator for cudaq::QuakeValue
      .def("__getitem__", &QuakeValue::operator[], py::arg("index"),
           "Return the element of `self` at the provided `index`.\n"
           "\nNote:\n"
           "  Only `list` or :class:`qreg` type :class:`QuakeValue`'s "
           "may be indexed.\n"
           "\nArgs:\n"
           "  index (int): The element of `self` that you'd like to return.\n"
           "\nReturns:\n"
           "  :class:`QuakeValue` : Returns a new :class:`QuakeValue` for the "
           "`index`-th element of `self`."
           "\nRaises:\n"
           "  RuntimeError: if `self` is a non-subscriptable "
           ":class:`QuakeValue`.\n")
      /// @brief Bind the binary operators on `QuakeValue` class. Note:
      /// these are incompatible with the pybind11 built-in
      /// binary operators (see: cudaq::spin_op bindings). Instead,
      /// we bind the functions manually.
      /// Multiplication overloads:
      .def("__mul__", py::overload_cast<const double>(&QuakeValue::operator*),
           py::arg("other"),
           "Return the product of `self` (:class:`QuakeValue`) with `other` "
           "(float).\n"
           "\nRaises:\n"
           "  RuntimeError: if the underlying :class:`QuakeValue` type is not "
           "a float.\n"
           "\n.. code-block:: python\n\n"
           "  # Example:\n"
           "  kernel, value = cudaq.make_kernel(float)\n"
           "  new_value: QuakeValue = value * 5.0\n")
      .def("__mul__", py::overload_cast<QuakeValue>(&QuakeValue::operator*),
           py::arg("other"),
           "Return the product of `self` (:class:`QuakeValue`) with `other` "
           "(:class:`QuakeValue`).\n"
           "\nRaises:\n"
           "  RuntimeError: if the underlying :class:`QuakeValue` type is not "
           "a float.\n"
           "\n.. code-block:: python\n\n"
           "  # Example:\n"
           "  kernel, values = cudaq.make_kernel(list)\n"
           "  new_value: QuakeValue = values[0] * values[1]\n")
      .def(
          "__rmul__",
          [](QuakeValue &self, double other) { return other * self; },
          py::arg("other"),
          "Return the product of `other` (float) with `self` "
          "(:class:`QuakeValue`).\n"
          "\nRaises:\n"
          "  RuntimeError: if the underlying :class:`QuakeValue` type is not a "
          "float.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel, value = cudaq.make_kernel(float)\n"
          "  new_value: QuakeValue = 5.0 * value\n")
      // Addition overloads:
      .def("__add__", py::overload_cast<const double>(&QuakeValue::operator+),
           py::arg("other"),
           "Return the sum of `self` (:class:`QuakeValue`) and `other` "
           "(float).\n"
           "\nRaises:\n"
           "  RuntimeError: if the underlying :class:`QuakeValue` type is not "
           "a float.\n"
           "\n.. code-block:: python\n\n"
           "  # Example:\n"
           "  kernel, value = cudaq.make_kernel(float)\n"
           "  new_value: QuakeValue = value + 5.0\n")
      .def("__add__", py::overload_cast<QuakeValue>(&QuakeValue::operator+),
           py::arg("other"),
           "Return the sum of `self` (:class:`QuakeValue`) and `other` "
           "(:class:`QuakeValue`).\n"
           "\nRaises:\n"
           "  RuntimeError: if the underlying :class:`QuakeValue` type is not "
           "a float.\n"
           "\n.. code-block:: python\n\n"
           "  # Example:\n"
           "  kernel, values = cudaq.make_kernel(list)\n"
           "  new_value: QuakeValue = values[0] + values[1]\n")
      .def(
          "__radd__",
          [](QuakeValue &self, double other) { return other + self; },
          py::arg("other"),
          "Return the sum of `other` (float) and `self` "
          "(:class:`QuakeValue`).\n"
          "\nRaises:\n"
          "  RuntimeError: if the underlying :class:`QuakeValue` type is not a "
          "float.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel, value = cudaq.make_kernel(float)\n"
          "  new_value: QuakeValue = 5.0 + value\n")
      // Subtraction overloads:
      .def("__sub__", py::overload_cast<const double>(&QuakeValue::operator-),
           py::arg("other"),
           "Return the difference of `self` (:class:`QuakeValue`) and `other` "
           "(float).\n"
           "\nRaises:\n"
           "  RuntimeError: if the underlying :class:`QuakeValue` type is not "
           "a float.\n"
           "\n.. code-block:: python\n\n"
           "  # Example:\n"
           "  kernel, value = cudaq.make_kernel(float)\n"
           "  new_value: QuakeValue = value - 5.0\n")
      .def("__sub__", py::overload_cast<QuakeValue>(&QuakeValue::operator-),
           py::arg("other"),
           "Return the difference of `self` (:class:`QuakeValue`) and `other` "
           "(:class:`QuakeValue`).\n"
           "\nRaises:\n"
           "  RuntimeError: if the underlying :class:`QuakeValue` type is not "
           "a float.\n"
           "\n.. code-block:: python\n\n"
           "  # Example:\n"
           "  kernel, values = cudaq.make_kernel(list)\n"
           "  new_value: QuakeValue = values[0] - values[1]\n")
      .def(
          "__rsub__",
          [](QuakeValue &self, double other) { return other - self; },
          py::arg("other"),
          "Return the difference of `other` (float) and `self` "
          "(:class:`QuakeValue`).\n"
          "\nRaises:\n"
          "  RuntimeError: if the underlying :class:`QuakeValue` type is not a "
          "float.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel, value = cudaq.make_kernel(float)\n"
          "  new_value: QuakeValue = 5.0 - value\n")
      .def(
          "__neg__", [](QuakeValue &self) { return -self; },
          "Return the negation of `self` (:class:`QuakeValue`).\n"
          "\nRaises:\n"
          "  RuntimeError: if the underlying :class:`QuakeValue` type is not a "
          "float.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel, value = cudaq.make_kernel(float)\n"
          "  new_value: QuakeValue = -value\n");
}

} // namespace cudaq
