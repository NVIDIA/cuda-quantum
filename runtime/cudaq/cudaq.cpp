/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1

#include "common/FmtCore.h"
#include "common/Logger.h"
#ifdef CUDAQ_HAS_CUDA
#include "cuda_runtime_api.h"
#endif
#include "cudaq/platform.h"
#include "distributed/mpi_plugin.h"
#include <dlfcn.h>
#include <filesystem>
#include <map>
#include <signal.h>
#include <string>
#include <vector>
namespace nvqir {
void tearDownBeforeMPIFinalize();
void setRandomSeed(std::size_t);
} // namespace nvqir

namespace cudaq::mpi {
cudaq::MPIPlugin *getMpiPlugin(bool unsafe) {
  // Locate and load the MPI comm plugin.
  // Rationale: we don't want to explicitly link `libcudaq.so` against any
  // specific MPI implementation for compatibility. Rather, MPI functionalities
  // are encapsulated inside a runtime-loadable plugin.
  static std::unique_ptr<cudaq::MPIPlugin> g_plugin;
  if (g_plugin)
    return g_plugin.get();

  // Search priority:
  //  (1) Environment variable take precedence (e.g., by running the
  // activation script)
  //  (2) Previously-activated custom plugin at its default location
  //  (3) Built-in comm plugin (e.g., docker container or build from source
  // with MPI)
  const char *mpiLibPath = std::getenv("CUDAQ_MPI_COMM_LIB");
  if (mpiLibPath) {
    // The user has set the environment variable.
    CUDAQ_INFO("Load MPI comm plugin from CUDAQ_MPI_COMM_LIB environment "
               "variable at '{}'",
               mpiLibPath);
    g_plugin = std::make_unique<cudaq::MPIPlugin>(mpiLibPath);
  } else {
    // Try locate MPI plugins in the install directory
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    // First, look for the previously-activated plugin in the
    // `distributed_interfaces/` directory.
    const auto distributedInterfacesDir =
        cudaqLibPath.parent_path().parent_path() / "distributed_interfaces";
    // Note: this file name must match the one defined in
    // `activate_custom_mpi.sh`.
    constexpr std::string_view activatedInterfaceLibFilename =
        "libcudaq_distributed_interface_mpi.so";
    const auto activatedInterfaceLibFile =
        distributedInterfacesDir / activatedInterfaceLibFilename;
    if (std::filesystem::exists(activatedInterfaceLibFile)) {
      CUDAQ_INFO("Load MPI comm plugin from '{}'",
                 activatedInterfaceLibFile.c_str());
      g_plugin =
          std::make_unique<cudaq::MPIPlugin>(activatedInterfaceLibFile.c_str());
    } else {
      const auto pluginsPath = cudaqLibPath.parent_path() / "plugins";
#if defined(__APPLE__) && defined(__MACH__)
      constexpr std::string_view libSuffix = "dylib";
#else
      constexpr std::string_view libSuffix = "so";
#endif
      // The builtin (native) plugin if present
      const auto pluginLibFile =
          pluginsPath / fmt::format("libcudaq-comm-plugin.{}", libSuffix);
      if (std::filesystem::exists(pluginLibFile) &&
          cudaq::MPIPlugin::isValidInterfaceLib(pluginLibFile.c_str())) {
        CUDAQ_INFO("Load builtin MPI comm plugin at '{}'",
                   pluginLibFile.c_str());
        g_plugin = std::make_unique<cudaq::MPIPlugin>(pluginLibFile.c_str());
      }
    }
  }
  if (!g_plugin) {
    if (unsafe)
      return nullptr;
    throw std::runtime_error(
        "No MPI support can be found when attempted to use cudaq::mpi APIs. "
        "Please refer to the documentation for instructions to activate MPI "
        "support.");
  }

  return g_plugin.get();
};

bool available() { return getMpiPlugin(/*unsafe=*/true); }

void initialize() {
  auto *commPlugin = getMpiPlugin();
  commPlugin->initialize();
}

void initialize(int argc, char **argv) {
  auto *commPlugin = getMpiPlugin();
  commPlugin->initialize(argc, argv);
  const auto pid = commPlugin->rank();
  const auto np = commPlugin->num_ranks();
  if (pid == 0)
    CUDAQ_INFO("MPI Initialized, nRanks = {}", np);
}

int rank() { return getMpiPlugin()->rank(); }

int num_ranks() { return getMpiPlugin()->num_ranks(); }

bool is_initialized() {
  // Allow to probe is_initialized even without MPI support (hence unsafe =
  // true)
  auto *commPlugin = getMpiPlugin(true);
  // If no MPI plugin is available, returns false (MPI is not initialized)
  if (!commPlugin)
    return false;

  return commPlugin->is_initialized();
}

namespace details {

#define CUDAQ_ALL_REDUCE_IMPL(TYPE, BINARY, REDUCE_OP)                         \
  TYPE allReduce(const TYPE &local, const BINARY<TYPE> &) {                    \
    static_assert(std::is_floating_point<TYPE>::value,                         \
                  "all_reduce argument must be a floating point number");      \
    std::vector<double> result(1);                                             \
    std::vector<double> localVec{static_cast<double>(local)};                  \
    auto *commPlugin = getMpiPlugin();                                         \
    commPlugin->all_reduce(result, localVec, REDUCE_OP);                       \
    return static_cast<TYPE>(result.front());                                  \
  }

CUDAQ_ALL_REDUCE_IMPL(float, std::plus, SUM)
CUDAQ_ALL_REDUCE_IMPL(float, std::multiplies, PROD)

CUDAQ_ALL_REDUCE_IMPL(double, std::plus, SUM)
CUDAQ_ALL_REDUCE_IMPL(double, std::multiplies, PROD)

} // namespace details

#define CUDAQ_ALL_GATHER_IMPL(TYPE)                                            \
  void all_gather(std::vector<TYPE> &global, const std::vector<TYPE> &local) { \
    auto *commPlugin = getMpiPlugin();                                         \
    commPlugin->all_gather(global, local);                                     \
  }

CUDAQ_ALL_GATHER_IMPL(double)
CUDAQ_ALL_GATHER_IMPL(int)

void broadcast(std::vector<double> &data, int rootRank) {
  auto *commPlugin = getMpiPlugin();
  commPlugin->broadcast(data, rootRank);
}

void broadcast(std::string &data, int rootRank) {
  auto *commPlugin = getMpiPlugin();
  commPlugin->broadcast(data, rootRank);
}

std::pair<void *, std::size_t> comm_dup() {
  auto *commPlugin = getMpiPlugin();
  cudaqDistributedCommunicator_t *dupComm = nullptr;
  cudaqDistributedCommunicator_t *comm = commPlugin->getComm();
  const auto dupStatus = commPlugin->get()->CommDup(comm, &dupComm);
  if (dupStatus != 0 || dupComm == nullptr)
    throw std::runtime_error("Failed to duplicate the MPI communicator.");
  return std::make_pair(dupComm->commPtr, dupComm->commSize);
}

void finalize() {
  // Inform the simulator that we are
  // about to run MPI Finalize
  nvqir::tearDownBeforeMPIFinalize();
  auto *commPlugin = getMpiPlugin();
  if (!commPlugin->is_finalized()) {
    if (rank() == 0)
      CUDAQ_INFO("Finalizing MPI.");
    commPlugin->finalize();
  }
}

} // namespace cudaq::mpi

namespace cudaq::__internal__ {
std::map<std::string, std::string> runtime_registered_mlir;
std::string demangle_kernel(const char *name) {
  return quantum_platform::demangle(name);
}
bool globalFalse = false;
} // namespace cudaq::__internal__

//===----------------------------------------------------------------------===//

namespace nvqir {
void setRandomSeed(std::size_t);
}

namespace cudaq {

void set_target_backend(const char *backend) {
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(std::string(backend));
}

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
  auto &platform = cudaq::get_platform();
  // Notify the platform that a new random seed value is set.
  platform.onRandomSeedSet(seed);
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
  // Free the initialization list, which was heap allocated.
  free(initList);
}

/// Construct a block of 0 and 1 bytes that corresponds to the `vector<bool>`
/// values. This gets rid of the bit packing implementation of the
/// `std::vector<bool>` overload. The conversion turns the `std::vector<bool>`
/// into a mock vector structure that looks like `std::vector<char>`. The
/// calling routine must cleanup the buffer allocated by this code.
/// This helper routine may only be called on the host side.
void __nvqpp_vector_bool_to_initializer_list(
    void *outData, const std::vector<bool> &inVec,
    std::vector<char *> **allocations) {
  // The MockVector must be allocated by the caller.
  struct MockVector {
    char *start;
    char *end;
    char *end2;
  };
  MockVector *mockVec = reinterpret_cast<MockVector *>(outData);
  auto outSize = inVec.size();
  // The buffer allocated here must be freed by the caller.
  if (!*allocations)
    *allocations = new std::vector<char *>;
  char *newData = static_cast<char *>(malloc(outSize));
  (*allocations)->push_back(newData);
  mockVec->start = newData;
  mockVec->end2 = mockVec->end = newData + outSize;
  for (unsigned i = 0; i < outSize; ++i)
    newData[i] = static_cast<char>(inVec[i]);
}

/// This helper routine deletes the vector that tracks all the temporaries that
/// were created as well as the temporaries themselves.
/// This routine may only be called on the host side.
void __nvqpp_vector_bool_free_temporary_initlists(
    std::vector<char *> *allocations) {
  for (auto *p : *allocations)
    free(p);
  delete allocations;
}

/// Quasi-portable string helpers for Python (non-C++ frontends).  These library
/// helper functions allow non-C++ front-ends to remain portable with the core
/// layer. As these helpers ought to be built along with the bindings, there
/// should not be a compatibility issue.
const char *__nvqpp_getStringData(const std::string &s) { return s.data(); }
std::uint64_t __nvqpp_getStringSize(const std::string &s) { return s.size(); }
}
} // namespace cudaq::support
