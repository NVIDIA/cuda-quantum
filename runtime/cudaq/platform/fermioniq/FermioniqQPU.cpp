/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FermioniqBaseQPU.h"
#include "common/RuntimeMLIR.h"
// #include "common/BaseRemoteRESTQPU.h"

using namespace mlir;

namespace cudaq {
std::string get_quake_by_name(const std::string &);
} // namespace cudaq

namespace {

/// @brief The `FermioniqRestQPU` is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on the Fermioniq simulator via a REST Client.
class FermioniqRestQPU : public cudaq::FermioniqBaseQPU {
protected:
  std::tuple<ModuleOp, std::unique_ptr<MLIRContext>, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {

    CUDAQ_INFO("extract quake code\n");

    std::unique_ptr<MLIRContext> context = cudaq::getOwningMLIRContext();

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(kernelName);
    auto m_module = parseSourceString<ModuleOp>(quakeCode, context.get());
    if (!m_module)
      throw std::runtime_error("module cannot be parsed");

    return std::make_tuple(m_module.release(), std::move(context), data);
  }

public:
  /// @brief The constructor
  FermioniqRestQPU() : FermioniqBaseQPU() {}

  FermioniqRestQPU(FermioniqRestQPU &&) = delete;
  virtual ~FermioniqRestQPU() = default;
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, FermioniqRestQPU, fermioniq)
