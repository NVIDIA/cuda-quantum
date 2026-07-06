/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecCommunicator.h"

#include "CuStateVecError.h"
#include "cudaq/cudaq_mpi.h"
#include "cudaq/runtime/logger/logger.h"

#include <custatevecEx_ext.h>
#include <dlfcn.h>

#include <exception>
#include <stdexcept>

namespace {

custatevecCommunicatorType_t
communicatorType(cudaq::cusv::CommunicatorProvider provider) {
  if (provider == cudaq::cusv::CommunicatorProvider::OpenMPI)
    return CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;
  if (provider == cudaq::cusv::CommunicatorProvider::MPICH)
    return CUSTATEVEC_COMMUNICATOR_TYPE_MPICH;
  return CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL;
}

void checkCommunicatorStatus(custatevecExCommunicatorStatus_t status,
                             const char *operation) {
  if (status != CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuStateVec communicator failed to ") +
                             operation + ".");
}

// Whether the provider is currently initialized.
bool providerInitialized = false;

} // namespace

namespace cudaq::cusv {

static CommunicatorProvider
resolveCommunicatorProvider(CommunicatorPlugin requested, bool cudaqAvailable,
                            bool openMpiLibrary) {
  switch (requested) {
  case CommunicatorPlugin::Auto:
    if (cudaqAvailable)
      return CommunicatorProvider::Cudaq;
    return openMpiLibrary ? CommunicatorProvider::OpenMPI
                          : CommunicatorProvider::MPICH;
  case CommunicatorPlugin::Self:
    if (!cudaqAvailable)
      throw std::runtime_error(
          "CUDAQ_MGPU_COMM_PLUGIN_TYPE=SELF requires activated CUDA-Q MPI.");
    return CommunicatorProvider::Cudaq;
  case CommunicatorPlugin::OpenMPI:
    return CommunicatorProvider::OpenMPI;
  case CommunicatorPlugin::MPICH:
    return CommunicatorProvider::MPICH;
  case CommunicatorPlugin::External:
    return CommunicatorProvider::External;
  }
  throw std::invalid_argument("Invalid cuStateVec communicator provider.");
}

static bool isOpenMpiLibrary(const std::string &library) {
  void *const handle = dlopen(library.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    const char *const error = dlerror();
    throw std::runtime_error("Unable to load MPI library '" + library +
                             "': " + (error ? error : "unknown error"));
  }
  const bool isOpenMpi = dlsym(handle, "ompi_mpi_comm_world") != nullptr;
  dlclose(handle);
  return isOpenMpi;
}

static const char *communicatorProviderName(CommunicatorProvider provider) {
  switch (provider) {
  case CommunicatorProvider::Cudaq:
    return "CUDA-Q";
  case CommunicatorProvider::OpenMPI:
    return "OpenMPI";
  case CommunicatorProvider::MPICH:
    return "MPICH";
  case CommunicatorProvider::External:
    return "external";
  }
  return "unknown";
}

CuStateVecCommunicator::CuStateVecCommunicator(CommunicatorPlugin plugin,
                                               const std::string &mpiLibrary)
    : m_provider([&] {
        const bool checkCudaq = plugin == CommunicatorPlugin::Auto ||
                                plugin == CommunicatorPlugin::Self;
        const bool cudaqAvailable = checkCudaq && cudaq::mpi::available();
        const bool openMpi = plugin == CommunicatorPlugin::Auto &&
                             !cudaqAvailable && isOpenMpiLibrary(mpiLibrary);
        return resolveCommunicatorProvider(plugin, cudaqAvailable, openMpi);
      }()) {
  initializeProvider(mpiLibrary);
  m_active = true;
  try {
    create();
  } catch (...) {
    reset();
    throw;
  }
}

CuStateVecCommunicator::~CuStateVecCommunicator() { reset(); }

void CuStateVecCommunicator::create() {
  HANDLE_CUSTATEVEC_ERROR(
      custatevecExCommunicatorCreate(/*exCommunicator=*/&m_communicator));
}

void CuStateVecCommunicator::destroy() noexcept {
  if (m_communicator)
    custatevecExCommunicatorDestroy(/*exCommunicator=*/m_communicator);
  m_communicator = nullptr;
}

void CuStateVecCommunicator::reset() noexcept {
  if (!m_active)
    return;
  destroy();
  finalizeProvider();
  m_active = false;
}

void CuStateVecCommunicator::initializeProvider(const std::string &mpiLibrary) {
  // Bring up the process-wide provider (MPI-level init). There is exactly one
  // communicator per process, so this runs once, from the constructor.
  custatevecExCommunicatorStatus_t communicatorStatus =
      CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
  const auto type = communicatorType(m_provider);
  const char *const libraryPath =
      m_provider == CommunicatorProvider::Cudaq ? nullptr : mpiLibrary.c_str();
  const custatevecStatus_t status = custatevecExCommunicatorInitialize(
      /*communicatorType=*/type, /*libraryPath=*/libraryPath, /*argc=*/nullptr,
      /*argv=*/nullptr, /*exCommStatus=*/&communicatorStatus);
  // Validate both the API return code and the exCommStatus out-param.
  HANDLE_CUSTATEVEC_ERROR(status);
  if (communicatorStatus != CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS)
    throw std::runtime_error(
        "cuStateVec communicator provider failed to initialize.");
  providerInitialized = true;
}

void CuStateVecCommunicator::finalizeProvider() noexcept {
  // If the provider is not initialized, there is nothing to finalize.
  if (!providerInitialized)
    return;
  providerInitialized = false;
  custatevecExCommunicatorStatus_t communicatorStatus =
      CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS;
  const custatevecStatus_t status =
      custatevecExCommunicatorFinalize(/*exCommStatus=*/&communicatorStatus);
  if (status != CUSTATEVEC_STATUS_SUCCESS)
    CUDAQ_WARN("cuStateVec communicator provider failed to finalize: {}.",
               custatevecGetErrorString(status));
  if (communicatorStatus != CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS)
    CUDAQ_WARN("cuStateVec communicator provider reported a finalize error "
               "(exCommStatus {}).",
               static_cast<int>(communicatorStatus));
}

custatevecExCommunicatorDescriptor_t
CuStateVecCommunicator::communicator() const {
  if (!m_active || !m_communicator)
    throw std::runtime_error("cuStateVec communicator is inactive.");
  return m_communicator;
}

custatevecExCommunicatorDescriptor_t
CuStateVecCommunicator::descriptor() const {
  return communicator();
}

int32_t CuStateVecCommunicator::size() const {
  int32_t value = 0;
  auto *const valueCommunicator = communicator();
  checkCommunicatorStatus(
      valueCommunicator->intf->getSize(valueCommunicator, &value),
      "query size");
  return value;
}

int32_t CuStateVecCommunicator::rank() const {
  int32_t value = 0;
  auto *const valueCommunicator = communicator();
  checkCommunicatorStatus(
      valueCommunicator->intf->getRank(valueCommunicator, &value),
      "query rank");
  return value;
}

void CuStateVecCommunicator::broadcast(void *buffer, int32_t count,
                                       cudaDataType_t dataType,
                                       int32_t root) const {
  auto *const valueCommunicator = communicator();
  checkCommunicatorStatus(valueCommunicator->intf->bcast(
                              valueCommunicator, buffer, count, dataType, root),
                          "broadcast");
}

void CuStateVecCommunicator::allReduce(const void *sendBuffer,
                                       void *receiveBuffer, int32_t count,
                                       cudaDataType_t dataType) const {
  auto *const valueCommunicator = communicator();
  checkCommunicatorStatus(
      valueCommunicator->intf->allreduce(valueCommunicator, sendBuffer,
                                         receiveBuffer, count, dataType),
      "all-reduce");
}

void CuStateVecCommunicator::allGather(const void *sendBuffer,
                                       void *receiveBuffer, int32_t count,
                                       cudaDataType_t dataType) const {
  auto *const valueCommunicator = communicator();
  checkCommunicatorStatus(
      valueCommunicator->intf->allgather(valueCommunicator, sendBuffer,
                                         receiveBuffer, count, dataType),
      "all-gather");
}

void CuStateVecCommunicator::setCommunicator(void *communicator,
                                             std::size_t communicatorSize) {
  if (m_provider != CommunicatorProvider::Cudaq)
    throw std::runtime_error(
        "A custom CUDA-Q communicator requires the CUDA-Q cuStateVecEx "
        "provider selected by AUTO or SELF; the active provider is " +
        std::string(communicatorProviderName(m_provider)) + ".");
  // Resolve the communicator bridge's exported ABI symbols by name (defined in
  // CuStateVecCommunicatorPlugin.cpp).
  using SetCommunicator = custatevecStatus_t (*)(void *, std::size_t);
  using GetCommunicator = custatevecStatus_t (*)(void **, std::size_t *);
  auto *const setSymbol =
      dlsym(RTLD_DEFAULT, "cudaqCustatevecExSetCommunicator");
  auto *const getSymbol =
      dlsym(RTLD_DEFAULT, "cudaqCustatevecExGetCommunicator");
  if (!setSymbol || !getSymbol)
    throw std::runtime_error(
        "The CUDA-Q cuStateVecEx communicator bridge is not loaded.");
  const auto setCommunicator = reinterpret_cast<SetCommunicator>(setSymbol);
  const auto getCommunicator = reinterpret_cast<GetCommunicator>(getSymbol);

  void *previousCommunicator = nullptr;
  std::size_t previousCommunicatorSize = 0;
  HANDLE_CUSTATEVEC_ERROR(
      getCommunicator(&previousCommunicator, &previousCommunicatorSize));
  destroy();
  try {
    HANDLE_CUSTATEVEC_ERROR(setCommunicator(communicator, communicatorSize));
    create();
  } catch (...) {
    const std::exception_ptr error = std::current_exception();
    destroy();
    try {
      HANDLE_CUSTATEVEC_ERROR(
          setCommunicator(previousCommunicator, previousCommunicatorSize));
      create();
    } catch (...) {
      // If rollback also fails, release this provider reference and leave the
      // wrapper observably inactive rather than retaining a null descriptor.
      reset();
    }
    std::rethrow_exception(error);
  }
}

} // namespace cudaq::cusv
