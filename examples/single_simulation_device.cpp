/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq.h"
#include "cudaq/driver/device.h"
#include "cudaq/qclink/qclink.h"

// clang-format off
// compile and run with 
// (replace /path/to/ with actual path)
// nvq++ single_simulation_device.cpp -I /path/to/libs/qclink/include/ -I /path/to/libs/core/include/ -L /path/to/qclink/lib -lcudaq-qclink  -Wl,-rpath,$PWD/lib
// ./a.out
// clang-format on

__qpu__ int random_bit(int i) {
  cudaq::qvector q(i);
  h(q);
  return mz(q[0]);
}

// helper to get the quake source code and kernel name
namespace cudaq::qclink {
template <typename QuantumKernel>
std::tuple<std::string, std::string> extract_code(QuantumKernel &&kernel) {
  std::string kernelName{
      cudaq::details::getKernelName(std::forward<QuantumKernel>(kernel))};
  auto code = details::removeNonEntrypointFunctionsManual(
      cudaq::get_quake_by_name(kernelName));
  return std::make_tuple(code, kernelName);
}
} // namespace cudaq::qclink

int main() {

  using namespace cudaq;

  // Configure you logical QPU, initialize the system
  std::vector<std::unique_ptr<qclink::device>> devices;
  devices.emplace_back(std::make_unique<qclink::nv_simulation_device>());
  qclink::lqpu cfg(std::move(devices));

  // Initialize the library
  qclink::initialize(&cfg);

  // Create kernel arguments and return pointer
  int numQubits = 2;

  // Get the code for the provided quantum kernel
  auto [code, name] = qclink::extract_code(random_bit);

  // auto devPtr = qclink::malloc(sizeof(int), 3);

  // Load the kernel, kicks off rt_host specific JIT compilation
  auto kernelHandle = qclink::load_kernel(code, name);

  // Launch the kernel, pass the input args, get back the result
  auto retInt = qclink::launch_kernel<int>(kernelHandle, numQubits);

  printf("Qubit Measurement = %d\n", retInt);

  // shutdown
  qclink::shutdown();
}
