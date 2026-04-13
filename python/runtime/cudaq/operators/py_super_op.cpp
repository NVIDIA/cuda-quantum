/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cudaq/operators.h"
#include "py_helpers.h"
#include "py_super_op.h"

namespace cudaq {

void bindSuperOperatorWrapper(nb::module_ &mod) {
  auto super_op_class = nb::class_<super_op>(mod, "SuperOperator");

  super_op_class
      .def(nb::init<>(), "Creates a default instantiated super-operator. A "
                         "default instantiated "
                         "super-operator means a no action linear map.")
      .def_static(
          "left_multiply",
          nb::overload_cast<const cudaq::product_op<cudaq::matrix_handler> &>(
              &super_op::left_multiply),
          "Creates a super-operator representing a left "
          "multiplication of the operator to the density matrix.")
      .def_static(
          "right_multiply",
          nb::overload_cast<const cudaq::product_op<cudaq::matrix_handler> &>(
              &super_op::right_multiply),
          "Creates a super-operator representing a right "
          "multiplication of the operator to the density matrix.")
      .def_static(
          "left_right_multiply",
          nb::overload_cast<const cudaq::product_op<cudaq::matrix_handler> &,
                            const cudaq::product_op<cudaq::matrix_handler> &>(
              &super_op::left_right_multiply),
          "Creates a super-operator representing a simultaneous left "
          "multiplication of the first operator operand and right "
          "multiplication of the second operator operand to the "
          "density matrix.")

      .def_static(
          "left_multiply",
          nb::overload_cast<const cudaq::sum_op<cudaq::matrix_handler> &>(
              &super_op::left_multiply),
          "Creates a super-operator representing a left "
          "multiplication of the operator to the density matrix. The sum is "
          "distributed into a linear combination of super-operator actions.")
      .def_static(
          "right_multiply",
          nb::overload_cast<const cudaq::sum_op<cudaq::matrix_handler> &>(
              &super_op::right_multiply),
          "Creates a super-operator representing a right "
          "multiplication of the operator to the density matrix. The sum is "
          "distributed into a linear combination of super-operator actions.")
      .def_static(
          "left_right_multiply",
          nb::overload_cast<const cudaq::sum_op<cudaq::matrix_handler> &,
                            const cudaq::sum_op<cudaq::matrix_handler> &>(
              &super_op::left_right_multiply),
          "Creates a super-operator representing a simultaneous left "
          "multiplication of the first operator operand and right "
          "multiplication of the second operator operand to the "
          "density matrix. The sum is distributed into a linear combination of "
          "super-operator actions.")
      .def(
          "__iter__",
          [](super_op &self) {
            return nb::make_iterator(nb::type<super_op>(), "iterator",
                                     self.begin(), self.end());
          },
          nb::keep_alive<0, 1>(),
          "Loop through each term of the super-operator.")
      .def(nb::self += nb::self, nb::is_operator());
}

} // namespace cudaq
