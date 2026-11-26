/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Everything lives in this single translation unit, so that linking any entry
// point from the static archive also brings in the runtime ABI check below.

#include "nvqpp_capi.h"
#include "common/DeviceCodeRegistry.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/platform/quantum_platform.h"
#include <cstdlib>
#include <cstring>

// Link-time check that the runtime implements the interface this library
// was built against. See `nvqpp_capi.h`.
[[gnu::used]] static const void *const cudaqRuntimeAbi =
    &__cudaq_runtime_abi_v1;

//===----------------------------------------------------------------------===//
// Kernel launch
//===----------------------------------------------------------------------===//

cudaq::KernelThunkResultType
__cudaq_altLaunchKernel(const char *kernelName, cudaq::KernelThunkType kernel,
                        void *args, std::uint64_t argsSize,
                        std::uint64_t resultOffset) {
  return cudaq::altLaunchKernel(kernelName, kernel, args, argsSize,
                                resultOffset);
}

cudaq::KernelThunkResultType
__cudaq_streamlinedLaunchKernel(const char *kernelName,
                                const std::vector<void *> &rawArgs) {
  return cudaq::streamlinedLaunchKernel(kernelName, rawArgs);
}

cudaq::KernelThunkResultType
__cudaq_hybridLaunchKernel(const char *kernelName,
                           cudaq::KernelThunkType kernel, void *args,
                           std::uint64_t argsSize, std::uint64_t resultOffset,
                           const std::vector<void *> &rawArgs) {
  return cudaq::hybridLaunchKernel(kernelName, kernel, args, argsSize,
                                   resultOffset, rawArgs);
}

//===----------------------------------------------------------------------===//
// Kernel registration
//===----------------------------------------------------------------------===//

void __cudaq_deviceCodeHolderAdd(const char *key, const char *code) {
  cudaq::registry::deviceCodeHolderAdd(key, code);
}

void __cudaq_registerKernelName(const char *kernelName) {
  cudaq::registry::registerKernelName(kernelName);
}

void __cudaq_registerCallbackName(const char *name, void *unmarshalFunc,
                                  void *devFunc) {
  cudaq::registry::registerCallbackName(name, unmarshalFunc, devFunc);
}

void __cudaq_registerArgsCreator(const char *name, char *rawFunctor) {
  cudaq::registry::registerArgsCreator(name, rawFunctor);
}

void __cudaq_registerLambdaName(const char *name, const char *value) {
  cudaq::registry::registerLambdaName(name, value);
}

void __cudaq_registerLinkableKernel(void *hostSideFunc, const char *kernelName,
                                    void *deviceSideFunc) {
  cudaq::registry::registerLinkableKernel(hostSideFunc, kernelName,
                                          deviceSideFunc);
}

void __cudaq_registerRunnableKernel(const char *kernelName,
                                    void *runnableEntry) {
  cudaq::registry::registerRunnableKernel(kernelName, runnableEntry);
}

std::intptr_t __cudaq_getLinkableKernelKey(void *p) {
  return cudaq::registry::getLinkableKernelKey(p);
}

void *__cudaq_getLinkableKernelDeviceFunction(std::intptr_t key) {
  return cudaq::registry::getLinkableKernelDeviceFunction(key);
}

//===----------------------------------------------------------------------===//
// `cudaq::state` helpers
//===----------------------------------------------------------------------===//

std::int64_t __cudaq_stateNumberOfQubits(cudaq::state *obj) {
  return cudaq::detail::stateNumberOfQubits(obj);
}

cudaq::state *__cudaq_stateCreateFromDataF64(double *data, std::size_t size) {
  return cudaq::detail::stateCreateFromDataF64(data, size);
}

cudaq::state *__cudaq_stateCreateFromDataF32(float *data, std::size_t size) {
  return cudaq::detail::stateCreateFromDataF32(data, size);
}

cudaq::state *__cudaq_stateCreateFromDataComplexF64(std::complex<double> *data,
                                                    std::size_t size) {
  return cudaq::detail::stateCreateFromDataComplexF64(data, size);
}

cudaq::state *__cudaq_stateCreateFromDataComplexF32(std::complex<float> *data,
                                                    std::size_t size) {
  return cudaq::detail::stateCreateFromDataComplexF32(data, size);
}

void __cudaq_stateDelete(cudaq::state *obj) { cudaq::detail::stateDelete(obj); }

//===----------------------------------------------------------------------===//
// Host-side helpers
//===----------------------------------------------------------------------===//

void __cudaq_initializerListToVectorBool(std::vector<bool> &result,
                                         char *initList, std::size_t size) {
  // result is a sret return value. Make sure it is default initialized. Takes
  // advantage of default empty vector being all 0s.
  std::memset(static_cast<void *>(&result), 0, sizeof(result));
  // Allocate space.
  result.reserve(size);
  // Copy in the initialization list data.
  char *p = initList;
  for (std::size_t i = 0; i < size; ++i, ++p)
    result.push_back(static_cast<bool>(*p));
  // Free the initialization list, which was heap allocated.
  free(initList);
}

void __cudaq_vectorBoolToInitializerList(void *outData,
                                         const std::vector<bool> &inVec,
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

void __cudaq_vectorBoolDestroy(std::vector<bool> &vec) { vec.~vector(); }

void __cudaq_vectorBoolFreeTemporaryInitLists(
    std::vector<char *> *allocations) {
  for (auto *p : *allocations)
    free(p);
  delete allocations;
}

void __cudaq_hostDeallocate(void *ptr) { ::operator delete(ptr); }

const char *__cudaq_getStringData(const std::string &s) { return s.data(); }

std::uint64_t __cudaq_getStringSize(const std::string &s) { return s.size(); }

cudaq::KernelThunkResultType
__cudaq_deviceCallbackRun(std::int64_t deviceId, const char *name,
                          void *unmarshalFunc, void *buffer,
                          std::int64_t bufferSize, std::int64_t returnOffset,
                          std::int64_t numBlocks, std::int64_t numThreads) {
  auto thunk = reinterpret_cast<cudaq::KernelThunkType>(unmarshalFunc);
  return thunk(buffer, /*isRemote=*/false);
}
