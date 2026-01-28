/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/GPUInfo.h"
#include "common/Registry.h"
#include "common/RemoteKernelExecutor.h"
#include "llvm/Support/CommandLine.h"

#ifdef __linux__
#include <signal.h>
#include <sys/prctl.h>
#endif

// Declare CUDA-Q MPI API that we need since we cannot compile with cudaq.h
// without RTTI (needed to link this tool against LLVMSupport).
namespace cudaq {
namespace mpi {
void initialize();
void finalize();
bool available();
} // namespace mpi
} // namespace cudaq

//===----------------------------------------------------------------------===//
// Command line options.
//===----------------------------------------------------------------------===//

static llvm::cl::opt<int>
    port("port", llvm::cl::desc("TCP/IP port that the server will listen to."),
         llvm::cl::init(3030));
constexpr const char *DEFAULT_SERVER_IMPL = "rest";
static llvm::cl::opt<std::string> serverSubType(
    "type", llvm::cl::desc("HTTP server subtype handling incoming requests."),
    llvm::cl::init(DEFAULT_SERVER_IMPL));
static llvm::cl::opt<bool> printRestPayloadVersion(
    "schema-version",
    llvm::cl::desc(
        "Display the REST request payload version that this server supports."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> printCudaProperties(
    "cuda-properties",
    llvm::cl::desc("Display the CUDA properties of the host and exit."),
    llvm::cl::init(false));

int main(int argc, char **argv) {
#ifdef __linux__
  // Request termination signal when parent process dies.
  // This ensures cleanup when auto-launched by remote-mqpu tests,
  // even if the parent is killed by SIGKILL (e.g., llvm-lit timeout).
  prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif

  // The "fast" instruction selection compilation algorithm is actually very
  // slow for large quantum circuits. Disable that here. Revisit this
  // decision by testing large UCCSD circuits if jitCodeGenOptLevel is changed
  // in the future. Also note that llvm::TargetMachine::setFastIsel() and
  // setO0WantsFastISel() do not retain their values in our current version of
  // LLVM. This use of LLVM command line parameters could be changed if the LLVM
  // JIT ever supports the TargetMachine options in the future.
  std::vector<const char *> extraArgv(
      argc + 2); // +1 for new parameter, +1 for nullptr at end of list
  for (int i = 0; i < argc; i++)
    extraArgv[i] = argv[i];
  extraArgv[argc] = "-fast-isel=0";

  llvm::cl::ParseCommandLineOptions(argc + 1, extraArgv.data(),
                                    "CUDA-Q REST server\n");
  if (printCudaProperties) {
    const auto deviceProps = cudaq::getCudaProperties();
    if (deviceProps.has_value()) {
      nlohmann::json j(*deviceProps);
      printf("%s\n", j.dump().c_str());
    }
    return 0;
  }
  if (cudaq::mpi::available())
    cudaq::mpi::initialize();
  // Check the server type arg is valid.
  if (!cudaq::registry::isRegistered<cudaq::RemoteRuntimeServer>(serverSubType))
    throw std::runtime_error(
        std::string("[cudaq-qpud] Unknown server sub-type requested: ") +
        std::string(serverSubType));
  // Only log if this is not the default locally-hosted Rest server
  // implementation.
  if (serverSubType != std::string(DEFAULT_SERVER_IMPL))
    printf("[cudaq-qpud] Using server subtype: %s\n", serverSubType.c_str());
  auto restServer =
      cudaq::registry::get<cudaq::RemoteRuntimeServer>(serverSubType);

  if (printRestPayloadVersion) {
    printf("\nCUDA-Q REST API version: %d.%d\n", restServer->version().first,
           restServer->version().second);
    return 0;
  }

  restServer->init({{"port", std::to_string(port)}});
  restServer->start();
  if (cudaq::mpi::available())
    cudaq::mpi::finalize();
}
