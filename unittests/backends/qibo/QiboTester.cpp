#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "CUDAQTestUtils.h"
#include "cudaq/platform/quantum_platform.h"
#include "gtest/gtest.h"

std::string mockPort = "62450";
std::string auth_token = "api_key";
std::string backendStringTemplate =
    "qibo;emulate;false;url;http://localhost:{};auth_token;{};";

TEST(QiboTester, checkSimpleCircuit) {
  // Initialize the platform
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, auth_token);
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  // Create a simple circuit
  auto kernel = cudaq::make_kernel();
  auto qubits = kernel.qalloc(2);
  kernel.h(qubits[0]);
  kernel.mz(qubits);

  // Execute the circuit
  auto counts = cudaq::sample(kernel);

  // Check results
  EXPECT_EQ(counts.size(), 2);
  // EXPECT_TRUE(counts.has_key("00"));
  // EXPECT_TRUE(counts.has_key("11"));
}