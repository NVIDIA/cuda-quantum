/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CircuitSimulator.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatState.h"
#include "cudaq.h"
#include "cudaq/distributed/mpi_plugin.h"

namespace {
// Hook to query this shared lib file location at runtime.
extern "C" {
void cuDensityMatSimCppFileMarker() { return; }
}
/// @brief Query the full path to the this lib.
static const char *getThisSharedLibFilePath() {
  static thread_local std::string LIB_PATH;
  if (LIB_PATH.empty()) {
    // Use dladdr query this .so file
    void *funcPtrToFind = (void *)(intptr_t)cuDensityMatSimCppFileMarker;
    Dl_info DLInfo;
    int err = dladdr(funcPtrToFind, &DLInfo);
    if (err != 0) {
      char link_path[PATH_MAX];
      // If the filename is a symlink, we need to resolve and return the
      // location of the actual .so file.
      if (realpath(DLInfo.dli_fname, link_path))
        LIB_PATH = link_path;
    }
  }

  return LIB_PATH.c_str();
}

/// @brief Retrieve the path to the plugin implementation
std::string getMpiPluginFilePath() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Failed to retrieve MPI plugin");

  return mpiPlugin->getPluginPath();
}

/// @brief Retrieve the MPI plugin comm interface
static cudaqDistributedInterface_t *getMpiPluginInterface() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Failed to retrieve MPI plugin");
  cudaqDistributedInterface_t *mpiInterface = mpiPlugin->get();
  if (!mpiInterface)
    throw std::runtime_error("Invalid MPI distributed plugin encountered");
  return mpiInterface;
}

/// @brief Retrieve the MPI plugin (type-erased) comm pointer
static cudaqDistributedCommunicator_t *getMpiCommWrapper() {
  auto mpiPlugin = cudaq::mpi::getMpiPlugin();
  if (!mpiPlugin)
    throw std::runtime_error("Failed to retrieve MPI plugin");
  cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
  if (!comm)
    throw std::runtime_error("Invalid MPI distributed plugin encountered");
  return comm;
}

void initCuDensityMatCommLib() {
  // If CUDENSITYMAT_COMM_LIB environment variable is not set,
  // use this builtin plugin shim (redirect MPI calls to CUDA-Q plugin)
  if (std::getenv("CUDENSITYMAT_COMM_LIB") == nullptr) {
    cudaq::info("Enabling cuDensityMat MPI without environment variable "
                "CUDENSITYMAT_COMM_LIB. \nUse the builtin cuDensityMat "
                "communicator lib from '{}' - CUDA-Q MPI plugin {}.",
                getThisSharedLibFilePath(), getMpiPluginFilePath());
    setenv("CUDENSITYMAT_COMM_LIB", getThisSharedLibFilePath(), 0);
  }

  cudaqDistributedInterface_t *mpiInterface = getMpiPluginInterface();
  cudaqDistributedCommunicator_t *comm = getMpiCommWrapper();
  assert(mpiInterface && comm);
  cudaqDistributedCommunicator_t *dupComm = nullptr;
  const auto dupStatus = mpiInterface->CommDup(comm, &dupComm);
  if (dupStatus != 0 || dupComm == nullptr)
    throw std::runtime_error("Failed to duplicate the MPI communicator when "
                             "initializing cuDensityMat MPI");
  HANDLE_CUDM_ERROR(cudensitymatResetDistributedConfiguration(
      cudaq::dynamics::Context::getCurrentContext()->getHandle(),
      CUDENSITYMAT_DISTRIBUTED_PROVIDER_MPI, dupComm->commPtr,
      dupComm->commSize));
}

class CuDensityMatSim : public nvqir::CircuitSimulatorBase<double> {
private:
  static constexpr int INVALID_CUDA_DEVICE = -1;

protected:
  using ScalarType = double;
  using DataType = std::complex<double>;
  using DataVector = std::vector<DataType>;

  using nvqir::CircuitSimulatorBase<ScalarType>::tracker;
  using nvqir::CircuitSimulatorBase<ScalarType>::nQubitsAllocated;
  using nvqir::CircuitSimulatorBase<ScalarType>::stateDimension;
  using nvqir::CircuitSimulatorBase<ScalarType>::calculateStateDim;
  using nvqir::CircuitSimulatorBase<ScalarType>::executionContext;
  using nvqir::CircuitSimulatorBase<ScalarType>::gateToString;
  using nvqir::CircuitSimulatorBase<ScalarType>::x;
  using nvqir::CircuitSimulatorBase<ScalarType>::flushGateQueue;
  using nvqir::CircuitSimulatorBase<ScalarType>::previousStateDimension;
  using nvqir::CircuitSimulatorBase<ScalarType>::shouldObserveFromSampling;
  using nvqir::CircuitSimulatorBase<ScalarType>::summaryData;

public:
  /// @brief The constructor
  CuDensityMatSim() {
    int numDevices{0};
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
    int currentDevice = INVALID_CUDA_DEVICE;
    HANDLE_CUDA_ERROR(cudaGetDevice(&currentDevice));
    const int deviceId = cudaq::mpi::is_initialized()
                             ? cudaq::mpi::rank() % numDevices
                             : currentDevice;
    if (cudaq::mpi::is_initialized())
      initCuDensityMatCommLib();
    HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  }

  /// The destructor
  virtual ~CuDensityMatSim() {}
  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    return std::make_unique<cudaq::CuDensityMatState>();
  }

  void addQubitToState() override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  void deallocateStateImpl() override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  bool measureQubit(const std::size_t qubitIdx) override {
    throw std::runtime_error("[dynamics target] Quantum gate simulation is not "
                             "supported.");
    return false;
  }
  void applyGate(const GateApplicationTask &task) override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  void setToZeroState() override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  void resetQubit(const std::size_t qubitIdx) override {
    throw std::runtime_error(
        "[dynamics target] Quantum gate simulation is not supported.");
  }
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubitIdxs,
                                const int shots) override {
    throw std::runtime_error("[dynamics target] Quantum gate simulation is not "
                             "supported.");
    return cudaq::ExecutionResult();
  }
  std::string name() const override { return "dynamics"; }
  NVQIR_SIMULATOR_CLONE_IMPL(CuDensityMatSim)
};
} // namespace

NVQIR_REGISTER_SIMULATOR(CuDensityMatSim, dynamics)
