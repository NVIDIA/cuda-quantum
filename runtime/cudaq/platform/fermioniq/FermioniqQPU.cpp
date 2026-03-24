/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FermioniqBaseQPU.h"

namespace {

/// @brief The `FermioniqRestQPU` is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on the Fermioniq simulator via a REST Client.
class FermioniqRestQPU : public cudaq::FermioniqBaseQPU {
public:
  /// @brief The constructor
  FermioniqRestQPU() : FermioniqBaseQPU() {}

  FermioniqRestQPU(FermioniqRestQPU &&) = delete;
  virtual ~FermioniqRestQPU() = default;
};
} // namespace

#ifdef CUDAQ_PYTHON_EXTENSION
extern "C" void cudaq_add_qpu_node(void *node_ptr);

namespace {
struct FermioniqQPURegistration {
  llvm::SimpleRegistryEntry<cudaq::QPU> entry;
  llvm::Registry<cudaq::QPU>::node node;
  FermioniqQPURegistration()
      : entry("fermioniq", "", &FermioniqQPURegistration::ctorFn), node(entry) {
    cudaq_add_qpu_node(&node);
  }
  static std::unique_ptr<cudaq::QPU> ctorFn() {
    return std::make_unique<FermioniqRestQPU>();
  }
};
static FermioniqQPURegistration s_fermioniqQPURegistration;
} // namespace
#else
CUDAQ_REGISTER_TYPE(cudaq::QPU, FermioniqRestQPU, fermioniq)
#endif
