/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
namespace cudaq::runtime {

/// Prefix for all kernel entry functions.
static constexpr const char cudaqGenPrefixName[] = "__nvqpp__mlirgen__";

/// Convenience constant for the length of the kernel entry prefix.
static constexpr unsigned cudaqGenPrefixLength = sizeof(cudaqGenPrefixName) - 1;

/// Name of the callback into the runtime.
/// A kernel entry procedure can either be replaced with a new function at
/// compile time (see `cudaqGenPrefixName`) or it can be rewritten to call back
/// to the runtime library (and be handled at runtime).
static constexpr const char launchKernelFuncName[] = "__cudaq_altLaunchKernel";
static constexpr const char launchKernelStreamlinedFuncName[] =
    "__cudaq_streamlinedLaunchKernel";
static constexpr const char launchKernelHybridFuncName[] =
    "__cudaq_hybridLaunchKernel";

static constexpr const char deviceCodeHolderAdd[] =
    "__cudaq_deviceCodeHolderAdd";

static constexpr const char registerLinkableKernel[] =
    "__cudaq_registerLinkableKernel";
static constexpr const char registerRunnableKernel[] =
    "__cudaq_registerRunnableKernel";
static constexpr const char getLinkableKernelKey[] =
    "__cudaq_getLinkableKernelKey";
static constexpr const char getLinkableKernelDeviceSide[] =
    "__cudaq_getLinkableKernelDeviceFunction";

static constexpr const char CudaqRegisterLambdaName[] =
    "__cudaq_registerLambdaName";
static constexpr const char CudaqRegisterArgsCreator[] =
    "__cudaq_registerArgsCreator";
static constexpr const char CudaqRegisterKernelName[] =
    "__cudaq_registerKernelName";
static constexpr const char CudaqRegisterCallbackName[] =
    "__cudaq_registerCallbackName";
// Same-process dispatch hook used by the generalized (distributed-memory
// reference) `device_call` lowering. See DistributedDeviceCall.cpp.
static constexpr const char callDeviceCallback[] = "__cudaq_deviceCallbackRun";

/// Prefix for an analog kernel entry functions.
static constexpr const char cudaqAHKPrefixName[] =
    "__cudaq_analog_hamiltonian_kernel__";

// Host-side helper functions for working with `cudaq::pauli_word` or a
// `std::string`. These include both fully dynamic and binding time (library
// build time) helper functions.
static constexpr const char getPauliWordSize[] =
    "_ZNK5cudaq10pauli_word11_nvqpp_sizeEv";
static constexpr const char getPauliWordData[] =
    "_ZNK5cudaq10pauli_word11_nvqpp_dataEv";
static constexpr const char bindingGetStringData[] = "__cudaq_getStringData";
static constexpr const char bindingGetStringSize[] = "__cudaq_getStringSize";
static constexpr const char bindingInitializeString[] =
    "__cudaq_initializeStringFromSpan";
static constexpr const char bindingDeconstructString[] =
    "__cudaq_deconstructString";

/// Release the storage of a `std::vector<T>` built by host code. This is a
/// wrapper with C linkage in the runtime around the C++ `operator delete`, to
/// isolate the compiler from the mangled name, which may vary by platform.
static constexpr const char hostDeallocate[] = "__cudaq_hostDeallocate";
// Runtime layer of a `device_call` application based on CUDA-Q Realtime.
static constexpr const char deviceCallAcquireRealtimeFrame[] =
    "__cudaq_device_call_acquire_realtime_frame";
static constexpr const char deviceCallDispatchRealtimeFrame[] =
    "__cudaq_device_call_dispatch_realtime_frame";
static constexpr const char deviceCallSafelyReleaseRealtimeFrame[] =
    "__cudaq_device_call_safely_release_realtime_frame";
static constexpr const char extractDevPtr[] =
    "__nvqpp__device_extract_device_ptr";

// Garbage collection for arrays created during kernel execution.
static constexpr const char cleanupArrays[] = "__nvqir_cleanupArrays";

} // namespace cudaq::runtime
