/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogRemoteRESTQPU.h"

namespace {

/// @brief The `PasqalRemoteRESTQPU` is a subtype of QPU that enables the
/// execution of Analog Hamiltonian Program via a REST Client.
class PasqalRemoteRESTQPU : public cudaq::AnalogRemoteRESTQPU {
public:
  PasqalRemoteRESTQPU() : AnalogRemoteRESTQPU() {}
  PasqalRemoteRESTQPU(PasqalRemoteRESTQPU &&) = delete;
  virtual ~PasqalRemoteRESTQPU() = default;
};
} // namespace

#ifdef CUDAQ_PYTHON_EXTENSION
extern "C" void cudaq_add_qpu_node(void *node_ptr);

namespace {
struct PasqalQPURegistration {
  llvm::SimpleRegistryEntry<cudaq::QPU> entry;
  llvm::Registry<cudaq::QPU>::node node;
  PasqalQPURegistration()
      : entry("pasqal", "", &PasqalQPURegistration::ctorFn), node(entry) {
    cudaq_add_qpu_node(&node);
  }
  static std::unique_ptr<cudaq::QPU> ctorFn() {
    return std::make_unique<PasqalRemoteRESTQPU>();
  }
};
static PasqalQPURegistration s_pasqalQPURegistration;
} // namespace
#else
CUDAQ_REGISTER_TYPE(cudaq::QPU, PasqalRemoteRESTQPU, pasqal)
#endif
