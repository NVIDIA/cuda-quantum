/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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
      .def(
          "__getitem__",
          [](QuakeValue &self, std::size_t idx) { return self[idx]; },
          py::arg("index"),
          R"#(Return the element of `self` at the provided `index`.

Note:
	Only `list` or :class:`qreg` type :class:`QuakeValue`'s may be indexed.

Args:
	index (int): The element of `self` that you'd like to return.

Returns:
	:class:`QuakeValue`: 
	A new :class:`QuakeValue` for the `index`-th element of `self`.

Raises:
	RuntimeError: if `self` is a non-subscriptable :class:`QuakeValue`.)#")
      .def(
          "__getitem__",
          [](QuakeValue &self, QuakeValue &idx) { return self[idx]; },
          py::arg("index"),
          R"#(Return the element of `self` at the provided `index`.

Note:
	Only `list` or :class:`qreg` type :class:`QuakeValue`'s may be indexed.

Args:
	index (QuakeValue): The element of `self` that you'd like to return.

Returns:
	:class:`QuakeValue`: 
	A new :class:`QuakeValue` for the `index`-th element of `self`.

Raises:
	RuntimeError: if `self` is a non-subscriptable :class:`QuakeValue`.)#")
      .def(
          "slice", &QuakeValue::slice, py::arg("start"), py::arg("count"),
          R"#(Return a slice of the given :class:`QuakeValue` as a new :class:`QuakeValue`. 

Note:
  The underlying :class:`QuakeValue` must be a `list` or `veq`.
     
Args:
  start (int): The index to begin the slice from.
  count (int): The number of elements to extract after the `start` index.
     
Returns:
  :class:`QuakeValue`: A new `QuakeValue` containing a slice of `self` 
  from the `start`-th element to the `start + count`-th element.)#")
      /// @brief Bind the binary operators on `QuakeValue` class. Note:
      /// these are incompatible with the pybind11 built-in
      /// binary operators (see: cudaq::spin_op bindings). Instead,
      /// we bind the functions manually.
      /// Multiplication overloads:
      .def(
          "__mul__", py::overload_cast<const double>(&QuakeValue::operator*),
          py::arg("other"),
          R"#(Return the product of `self` (:class:`QuakeValue`) with `other` (float).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

.. code-block:: python

	# Example:
	kernel, value = cudaq.make_kernel(float)
	new_value: QuakeValue = value * 5.0)#")
      .def("__mul__", py::overload_cast<QuakeValue>(&QuakeValue::operator*),
           py::arg("other"),
           R"#(Return the product of `self` (:class:`QuakeValue`) with `other`
(:class:`QuakeValue`).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

.. code-block:: python

	# Example:
	kernel, values = cudaq.make_kernel(list)
	new_value: QuakeValue = values[0] * values[1])#")
      .def(
          "__rmul__",
          [](QuakeValue &self, double other) { return other * self; },
          py::arg("other"),
          R"#(Return the product of `other` (float) with `self` (:class:`QuakeValue`).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

.. code-block:: python

	# Example:
	kernel, value = cudaq.make_kernel(float)
	new_value: QuakeValue = 5.0 * value)#")
      // Addition overloads:
      .def(
          "__add__", py::overload_cast<const double>(&QuakeValue::operator+),
          py::arg("other"),
          R"#(Return the sum of `self` (:class:`QuakeValue`) and `other` (float).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

.. code-block:: python

	# Example:
	kernel, value = cudaq.make_kernel(float)
	new_value: QuakeValue = value + 5.0)#")
      .def(
          "__add__", py::overload_cast<QuakeValue>(&QuakeValue::operator+),
          py::arg("other"),
          R"#(Return the sum of `self` (:class:`QuakeValue`) and `other` (:class:`QuakeValue`).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not the same type as `self`.

.. code-block:: python

	# Example:
	kernel, values = cudaq.make_kernel(list)
	new_value: QuakeValue = values[0] + values[1])#")
      .def("__add__", py::overload_cast<int>(&QuakeValue::operator+),
           py::arg("other"),
           R"#(Return the sum of `self` (:class:`QuakeValue`) and `other` (int).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not an `int`.

.. code-block:: python

	# Example:
	kernel, values = cudaq.make_kernel(list)
	new_value: QuakeValue = values[0] + 2)#")
      .def(
          "__radd__",
          [](QuakeValue &self, double other) { return other + self; },
          py::arg("other"),
          R"#(Return the sum of `other` (float) and `self` (:class:`QuakeValue`).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

.. code-block:: python

	# Example:
	kernel, value = cudaq.make_kernel(float)
	new_value: QuakeValue = 5.0 + value)#")
      // Subtraction overloads:
      .def(
          "__sub__", py::overload_cast<const double>(&QuakeValue::operator-),
          py::arg("other"),
          R"#(Return the difference of `self` (:class:`QuakeValue`) and `other` (float).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

.. code-block:: python

	# Example:
	kernel, value = cudaq.make_kernel(float)
	new_value: QuakeValue = value - 5.0)#")
      .def(
          "__sub__", py::overload_cast<QuakeValue>(&QuakeValue::operator-),
          py::arg("other"),
          R"#(Return the difference of `self` (:class:`QuakeValue`) and `other` (:class:`QuakeValue`).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not the same type as `self`.

.. code-block:: python

	# Example:
	kernel, values = cudaq.make_kernel(list)
	new_value: QuakeValue = values[0] - values[1])#")
      .def(
          "__sub__", py::overload_cast<int>(&QuakeValue::operator-),
          py::arg("other"),
          R"#(Return the difference of `self` (:class:`QuakeValue`) and `other` (int).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a int.

.. code-block:: python
	
	# Example:
	kernel, values = cudaq.make_kernel(list)
	new_value: QuakeValue = values[0] - 2)#")
      .def(
          "__rsub__",
          [](QuakeValue &self, double other) { return other - self; },
          py::arg("other"),
          R"#(Return the difference of `other` (float) and `self` (:class:`QuakeValue`).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

.. code-block:: python

	# Example:
	kernel, value = cudaq.make_kernel(float)
	new_value: QuakeValue = 5.0 - value)#")
      .def(
          "__neg__", [](QuakeValue &self) { return -self; },
          R"#(Return the negation of `self` (:class:`QuakeValue`).

Raises:
	RuntimeError: if the underlying :class:`QuakeValue` type is not a float.

.. code-block:: python

	# Example:
	kernel, value = cudaq.make_kernel(float)
	new_value: QuakeValue = -value)#");
}

} // namespace cudaq
