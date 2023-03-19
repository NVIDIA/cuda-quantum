/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once
#include <cudaq/algorithm.h>
#include <cudaq/builder.h>

namespace cudaq {

// Define a function that applies a general SO(4) rotation to
// the builder on the provided qubits with the provided parameters.
// Note we keep this qubit and parameter arguments as auto as these
// will default to taking the cudaq::builder::qubit (private inner class)
// and cudaq::Parameter<std::vector<T>>.
template <typename ScalarParamT>
void so4(cudaq::builder &builder, builder::qubit &q, builder::qubit &r,
         Parameter<std::vector<ScalarParamT>> &parameters) {
  builder.ry(parameters[0], q);
  builder.ry(parameters[1], r);

  builder.h(r);
  builder.x<cudaq::ctrl>(q, r);
  builder.h(r);

  builder.ry(parameters[2], q);
  builder.ry(parameters[3], r);

  builder.h(r);
  builder.x<cudaq::ctrl>(q, r);
  builder.h(r);

  builder.ry(parameters[4], q);
  builder.ry(parameters[5], r);

  builder.h(r);
  builder.x<cudaq::ctrl>(q, r);
  builder.h(r);
}

} // namespace cudaq
