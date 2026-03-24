/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogRemoteRESTQPU.h"

namespace {

/// @brief The `QuEraRemoteRESTQPU` is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Program via a REST Client.
class QuEraRemoteRESTQPU : public cudaq::AnalogRemoteRESTQPU {
public:
  QuEraRemoteRESTQPU() : AnalogRemoteRESTQPU() {}
  QuEraRemoteRESTQPU(QuEraRemoteRESTQPU &&) = delete;
  virtual ~QuEraRemoteRESTQPU() = default;
};
} // namespace

#ifdef CUDAQ_PYTHON_EXTENSION
extern "C" void cudaq_add_qpu_node(void *node_ptr);

namespace {
struct QuEraQPURegistration {
  llvm::SimpleRegistryEntry<cudaq::QPU> entry;
  llvm::Registry<cudaq::QPU>::node node;
  QuEraQPURegistration()
      : entry("quera", "", &QuEraQPURegistration::ctorFn), node(entry) {
    cudaq_add_qpu_node(&node);
  }
  static std::unique_ptr<cudaq::QPU> ctorFn() {
    return std::make_unique<QuEraRemoteRESTQPU>();
  }
};
static QuEraQPURegistration s_queraQPURegistration;
} // namespace
#else
CUDAQ_REGISTER_TYPE(cudaq::QPU, QuEraRemoteRESTQPU, quera)
#endif
