/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cudaq/operators.h"
#include "py_helpers.h"
#include "py_super_op.h"

namespace cudaq {

void bindSuperOperatorWrapper(nanobind::module_ &mod) {
  auto super_op_class = nanobind::class_<super_op>(mod, "SuperOperator");

  super_op_class
      .def(nanobind::init<>(),
           "Creates a default instantiated super-operator. A "
           "default instantiated "
           "super-operator means a no action linear map.")
      .def_static("left_multiply",
                  nanobind::overload_cast<
                      const cudaq::product_op<cudaq::matrix_handler> &>(
                      &super_op::left_multiply),
                  "Creates a super-operator representing a left "
                  "multiplication of the operator to the density matrix.")
      .def_static("right_multiply",
                  nanobind::overload_cast<
                      const cudaq::product_op<cudaq::matrix_handler> &>(
                      &super_op::right_multiply),
                  "Creates a super-operator representing a right "
                  "multiplication of the operator to the density matrix.")
      .def_static("left_right_multiply",
                  nanobind::overload_cast<
                      const cudaq::product_op<cudaq::matrix_handler> &,
                      const cudaq::product_op<cudaq::matrix_handler> &>(
                      &super_op::left_right_multiply),
                  "Creates a super-operator representing a simultaneous left "
                  "multiplication of the first operator operand and right "
                  "multiplication of the second operator operand to the "
                  "density matrix.")

      .def_static(
          "left_multiply",
          nanobind::overload_cast<const cudaq::sum_op<cudaq::matrix_handler> &>(
              &super_op::left_multiply),
          "Creates a super-operator representing a left "
          "multiplication of the operator to the density matrix. The sum is "
          "distributed into a linear combination of super-operator actions.")
      .def_static(
          "right_multiply",
          nanobind::overload_cast<const cudaq::sum_op<cudaq::matrix_handler> &>(
              &super_op::right_multiply),
          "Creates a super-operator representing a right "
          "multiplication of the operator to the density matrix. The sum is "
          "distributed into a linear combination of super-operator actions.")
      .def_static(
          "left_right_multiply",
          nanobind::overload_cast<const cudaq::sum_op<cudaq::matrix_handler> &,
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
            nanobind::list items;
            for (auto it = self.begin(); it != self.end(); ++it)
              items.append(nanobind::cast(*it));
            return items.attr("__iter__")();
          },
          "Loop through each term of the super-operator.")
      .def(nanobind::self += nanobind::self, nanobind::is_operator());
}

} // namespace cudaq
