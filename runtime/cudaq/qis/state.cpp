/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "state.h"
#include "common/EigenDense.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include <iostream>

namespace cudaq {

void state::dump() { dump(std::cout); }
void state::dump(std::ostream &os) { internal->dump(os); }

std::vector<std::size_t> state::get_shape() const {
  return internal->getDataShape();
}

std::complex<double> state::operator[](std::size_t idx) {
  if (internal->getDataShape().size() != 1)
    throw std::runtime_error("Cannot request 1-d index into density matrix. "
                             "Must be a state vector.");
  // FIXME We should update this in the future to return the
  // same amplitude ordering as the internal runtime. For now,
  // leaving this to retain backwards compatibility.
  std::size_t numQubits = internal->getNumQubits();
  std::size_t newIdx = 0;
  for (std::size_t i = 0; i < numQubits; ++i)
    if (idx & (1ULL << i))
      newIdx |= (1ULL << ((numQubits - 1) - i));
  return internal->vectorElement(newIdx);
}

std::complex<double> state::operator()(std::size_t idx, std::size_t jdx) {
  return internal->matrixElement(idx, jdx);
}

double state::overlap(state &other) {
  return internal->overlap(*other.internal.get());
}

double state::overlap(const std::vector<complex> &hostData) {
  return internal->overlap(hostData);
}

double state::overlap(const std::vector<std::complex<float>> &hostData) {
  return internal->overlap(hostData);
}

double state::overlap(void *deviceOrHostPointer) {
  return internal->overlap(deviceOrHostPointer);
}

} // namespace cudaq
