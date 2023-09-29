/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1

#include "common/Logger.h"
#ifdef CUDAQ_HAS_CUDA
#include "cuda_runtime_api.h"
#endif
#include "cudaq/platform.h"
#include "cudaq/utils/registry.h"
#include <dlfcn.h>
#include <map>
#include <regex>
#include <signal.h>
#include <string>
#include <vector>

#ifdef CUDAQ_HAS_MPI
#include <mpi.h>

namespace nvqir {
void tearDownBeforeMPIFinalize();
void setRandomSeed(std::size_t);
} // namespace nvqir

namespace cudaq::mpi {

void initialize() {
  int argc{0};
  char **argv = nullptr;
  initialize(argc, argv);
}

void initialize(int argc, char **argv) {
  int pid, np, thread_provided;
  int mpi_error =
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_provided);
  assert(mpi_error == MPI_SUCCESS && "MPI_Init_thread failed");
  assert(thread_provided == MPI_THREAD_MULTIPLE);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  if (pid == 0)
    cudaq::info("MPI Initialized, nRanks = {}", np);
}

int rank() {
  int pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  return pid;
}

int num_ranks() {
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  return np;
}

bool is_initialized() {
  int i;
  auto err = MPI_Initialized(&i);
  assert(err == MPI_SUCCESS && "MPI_Initialized failed.");
  return i == 1;
}

namespace details {

#define CUDAQ_ALL_REDUCE_IMPL(TYPE, MPI_TYPE, BINARY, MPI_OP)                  \
  TYPE allReduce(const TYPE &local, const BINARY<TYPE> &) {                    \
    TYPE result;                                                               \
    MPI_Allreduce(&local, &result, 1, MPI_TYPE, MPI_OP, MPI_COMM_WORLD);       \
    return result;                                                             \
  }

CUDAQ_ALL_REDUCE_IMPL(float, MPI_FLOAT, std::plus, MPI_SUM)
CUDAQ_ALL_REDUCE_IMPL(float, MPI_FLOAT, std::multiplies, MPI_PROD)

CUDAQ_ALL_REDUCE_IMPL(double, MPI_DOUBLE, std::plus, MPI_SUM)
CUDAQ_ALL_REDUCE_IMPL(double, MPI_DOUBLE, std::multiplies, MPI_PROD)

} // namespace details

void all_gather(std::vector<double> &global, std::vector<double> &local) {
  MPI_Allgather(local.data(), local.size(), MPI_DOUBLE, global.data(),
                local.size(), MPI_DOUBLE, MPI_COMM_WORLD);
}

void finalize() {
  if (rank() == 0)
    cudaq::info("Finalizing MPI.");

  // Inform the simulator that we are
  // about to run MPI Finalize
  nvqir::tearDownBeforeMPIFinalize();

  // Check if finalize has been called.
  int isFinalized;
  MPI_Finalized(&isFinalized);
  if (isFinalized)
    return;

  // Finalize
  int mpi_error = MPI_Finalize();
  assert(mpi_error == MPI_SUCCESS && "MPI_Finalize failed.");
}

} // namespace cudaq::mpi
#else
namespace cudaq::mpi {

void initialize() {}

void initialize(int argc, char **argv) {}

bool is_initialized() { return false; }

int rank() { return 0; }

int num_ranks() { return 1; }

namespace details {

#define CUDAQ_ALL_REDUCE_IMPL(TYPE, BINARY)                                    \
  TYPE allReduce(const TYPE &local, const BINARY<TYPE> &) { return TYPE(); }

CUDAQ_ALL_REDUCE_IMPL(float, std::plus)
CUDAQ_ALL_REDUCE_IMPL(float, std::multiplies)

CUDAQ_ALL_REDUCE_IMPL(double, std::plus)
CUDAQ_ALL_REDUCE_IMPL(double, std::multiplies)

} // namespace details

void all_gather(std::vector<double> &global, std::vector<double> &local) {}

void finalize() {}

} // namespace cudaq::mpi
#endif

namespace cudaq::__internal__ {
std::map<std::string, std::string> runtime_registered_mlir;
std::string demangle_kernel(const char *name) {
  return quantum_platform::demangle(name);
}
bool globalFalse = false;
} // namespace cudaq::__internal__

//===----------------------------------------------------------------------===//
// Registry that maps device code keys to strings of device code. The map is
// created at program startup and can be used to find code to be
// compiled/executed at runtime.
//===----------------------------------------------------------------------===//

static std::vector<std::pair<std::string, std::string>> quakeRegistry;

void cudaq::registry::deviceCodeHolderAdd(const char *key, const char *code) {
  quakeRegistry.emplace_back(key, code);
}

//===----------------------------------------------------------------------===//
// Registry of all kernels that have been generated. The vector of kernels is
// created at program startup time. This list can be consulted by the runtime to
// determine if a particular kernel has been processed for kernel execution,
// including adding the trampoline to call the runtime to launch the kernel.
//===----------------------------------------------------------------------===//

static std::vector<std::string> kernelRegistry;

static std::map<std::string, cudaq::KernelArgsCreator> argsCreators;
static std::map<std::string, std::string> lambdaNames;

void cudaq::registry::cudaqRegisterKernelName(const char *kernelName) {
  kernelRegistry.emplace_back(kernelName);
}

void cudaq::registry::cudaqRegisterArgsCreator(const char *name,
                                               char *rawFunctor) {
  argsCreators.insert(
      {std::string(name), reinterpret_cast<KernelArgsCreator>(rawFunctor)});
}

void cudaq::registry::cudaqRegisterLambdaName(const char *name,
                                              const char *value) {
  lambdaNames.insert({std::string(name), std::string(value)});
}

bool cudaq::__internal__::isKernelGenerated(const std::string &kernelName) {
  for (auto regName : kernelRegistry)
    if (kernelName == regName)
      return true;
  return false;
}

bool cudaq::__internal__::isLibraryMode(const std::string &kernelname) {
  return !isKernelGenerated(kernelname);
}

//===----------------------------------------------------------------------===//

namespace nvqir {
void setRandomSeed(std::size_t);
}

namespace cudaq {

void set_target_backend(const char *backend) {
  std::string backendName(backend);
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendName);
}

KernelArgsCreator getArgsCreator(const std::string &kernelName) {
  return argsCreators[kernelName];
}

std::string get_quake_by_name(const std::string &kernelName,
                              bool throwException) {
  // A prefix name has a '.' before the C++ mangled name suffix.
  auto kernelNamePrefix = kernelName + '.';

  // Find the quake code
  std::optional<std::string> result;
  for (auto [k, v] : quakeRegistry) {
    if (k == kernelName) {
      // Exact match. Return the code.
      return v;
    }
    if (k.starts_with(kernelNamePrefix)) {
      // Prefix match. Record it and make sure that it is a unique prefix.
      if (result.has_value()) {
        if (throwException)
          throw std::runtime_error("Quake code for '" + kernelName +
                                   "' has multiple matches.\n");
      } else {
        result = v;
      }
    }
  }
  if (result.has_value())
    return *result;
  auto msg = "Quake code not found for '" + kernelName + "'.\n";
  if (throwException)
    throw std::runtime_error(msg);
  return {};
}

std::string get_quake_by_name(const std::string &kernelName) {
  return get_quake_by_name(kernelName, true);
}

bool kernelHasConditionalFeedback(const std::string &kernelName) {
  auto quakeCode = get_quake_by_name(kernelName, false);
  return !quakeCode.empty() &&
         quakeCode.find("qubitMeasurementFeedback = true") != std::string::npos;
}

// Ignore warnings about deprecations in platform.set_shots and
// platform.clear_shots because the functions that are using them here
// (cudaq::set_shots and cudaq::clear_shots are also deprecated and will be
// removed at the same time.)
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
void set_shots(const std::size_t nShots) {
  auto &platform = cudaq::get_platform();
  platform.set_shots(nShots);
}
void clear_shots(const std::size_t nShots) {
  auto &platform = cudaq::get_platform();
  platform.clear_shots();
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

void set_noise(const cudaq::noise_model &model) {
  auto &platform = cudaq::get_platform();
  platform.set_noise(&model);
}

void unset_noise() {
  auto &platform = cudaq::get_platform();
  platform.set_noise(nullptr);
}

thread_local static std::size_t cudaq_random_seed = 0;

/// @brief Note: a seed value of 0 will cause broadcast operations to use
/// std::random_device (or something similar) as a seed for the PRNGs, so this
/// will not be repeatable for those operations.
void set_random_seed(std::size_t seed) {
  cudaq_random_seed = seed;
  nvqir::setRandomSeed(seed);
}

std::size_t get_random_seed() { return cudaq_random_seed; }

int num_available_gpus() {
  int nDevices = 0;
#ifdef CUDAQ_HAS_CUDA
  cudaGetDeviceCount(&nDevices);
#endif
  return nDevices;
}

namespace __internal__ {
void cudaqCtrlCHandler(int signal) {
  printf(" CTRL-C caught in cudaq runtime.\n");
  std::exit(1);
}

__attribute__((constructor)) void startSigIntHandler() {
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = cudaqCtrlCHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
}
} // namespace __internal__

} // namespace cudaq

namespace cudaq::support {
extern "C" {
void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &result,
                                             char *initList, std::size_t size) {
  // result is a sret return value. Make sure it is default initialized. Takes
  // advantage of default empty vector being all 0s.
  std::memset(&result, 0, sizeof(result));
  // Allocate space.
  result.reserve(size);
  // Copy in the initialization list data.
  char *p = initList;
  for (std::size_t i = 0; i < size; ++i, ++p)
    result.push_back(static_cast<bool>(*p));
  // Free the initialization list, which was stack allocated.
  free(initList);
}
}
} // namespace cudaq::support
