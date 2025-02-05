/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "utils.h"
#include "cudaq/operators.h"
#include <gtest/gtest.h>
#include "cudaq/dynamics/spin_operators.h"

TEST(OperatorExpressions, checkPreBuiltSpinOps) {

  // Keeping this fixed throughout.
  int degree_index = 0;
  auto id = utils::id_matrix(2);

  // Identity operator.
  {
    auto op = cudaq::spin_operator::i(degree_index);
    auto got = op.to_matrix();
    auto want = utils::id_matrix(2);
    utils::checkEqual(want, got);
    utils::checkEqual(id, (op * op).to_matrix());
  }

  // Z operator.
  {
    auto op = cudaq::spin_operator::z(degree_index);
    auto got = op.to_matrix();
    auto want = utils::PauliZ_matrix();
    utils::checkEqual(want, got);
    utils::checkEqual(id, (op * op).to_matrix());
  }

  // X operator.
  {
    auto op = cudaq::spin_operator::x(degree_index);
    auto got = op.to_matrix();
    auto want = utils::PauliX_matrix();
    utils::checkEqual(want, got);
    utils::checkEqual(id, (op * op).to_matrix());
  }

  // Y operator.
  {
    auto op = cudaq::spin_operator::y(degree_index);
    auto got = op.to_matrix();
    auto want = 1.0j * utils::PauliX_matrix() * utils::PauliZ_matrix();
    utils::checkEqual(want, got);
    utils::checkEqual(id, (op * op).to_matrix());
  }
}

