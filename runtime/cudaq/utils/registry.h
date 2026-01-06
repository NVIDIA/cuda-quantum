/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <string>

namespace cudaq::registry {
extern "C" {
void __cudaq_deviceCodeHolderAdd(const char *, const char *);
void cudaqRegisterKernelName(const char *);
void cudaqRegisterArgsCreator(const char *, char *);
void cudaqRegisterLambdaName(const char *, const char *);

/// Register a kernel with the runtime for kernel runtime stitching.
void __cudaq_registerLinkableKernel(void *, const char *, void *);
/// Register a `runnable` kernel with the runtime.
void __cudaq_registerRunnableKernel(const char *name, void *runnableEntry);

/// Return the kernel key from a `qkernel` object. If \p p is a `nullptr` this
/// will throw a runtime error.
std::intptr_t __cudaq_getLinkableKernelKey(void *p);

/// Given a kernel key value, return the name of the kernel. If the kernel is
/// not registered, throws a runtime error.
const char *__cudaq_getLinkableKernelName(std::intptr_t);

/// Given a kernel key value, return the corresponding device-side kernel
/// function. If the kernel is not registered, throws a runtime error.
void *__cudaq_getLinkableKernelDeviceFunction(std::intptr_t);
}

/// Given a kernel key value, return the name of the kernel. If the kernel is
/// not registered, runs a `nullptr`. Note this function is not exposed to the
/// compiler API as an `extern C` function.
const char *getLinkableKernelNameOrNull(std::intptr_t);

/// Given the address of the host-side kernel function, determine the associated
/// `runnable` kernel entry point.
void *getRunnableKernelOrNull(const std::string &kernelName);
void *__cudaq_getRunnableKernel(const std::string &kernelName);
} // namespace cudaq::registry

namespace cudaq::detail {
/// Is the kernel `kernelName` registered?
bool isKernelGenerated(const std::string &kernelName);
} // namespace cudaq::detail
