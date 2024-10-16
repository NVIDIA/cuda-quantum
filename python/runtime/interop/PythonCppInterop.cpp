/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "PythonCppInterop.h"
#include "cudaq.h"

namespace cudaq::python {

std::string getKernelName(std::string &input) {
  size_t pos = 0;
  std::string result = "";
  while (true) {
    // Find the next occurrence of "func.func @"
    size_t start = input.find("func.func @", pos) + 11;

    if (start == std::string::npos)
      break;

    // Find the position of the first "(" after "func.func @"
    size_t end = input.find("(", start);

    if (end == std::string::npos)
      break;

    // Extract the substring
    result = input.substr(start, end - start);

    // Check if the substring doesn't contain ".thunk"
    if (result.find(".thunk") == std::string::npos)
      break;

    // Move the position to continue searching
    pos = end;
  }
  return result;
}

std::string extractSubstring(const std::string &input,
                             const std::string &startStr,
                             const std::string &endStr) {
  size_t startPos = input.find(startStr);
  if (startPos == std::string::npos) {
    return ""; // Start string not found
  }

  startPos += startStr.length(); // Move to the end of the start string
  size_t endPos = input.find(endStr, startPos);
  if (endPos == std::string::npos) {
    return ""; // End string not found
  }

  return input.substr(startPos, endPos - startPos);
}

std::tuple<std::string, std::string>
getMLIRCodeAndName(const std::string &name, const std::string mangledArgs) {
  auto cppMLIRCode =
      cudaq::get_quake(std::remove_cvref_t<decltype(name)>(name), mangledArgs);
  auto kernelName = cudaq::python::getKernelName(cppMLIRCode);
  cppMLIRCode =
      "module {\nfunc.func @" + kernelName +
      extractSubstring(cppMLIRCode, "func.func @" + kernelName, "func.func") +
      "\n}";
  return std::make_tuple(kernelName, cppMLIRCode);
}

/// Map device kernels represented as mod1.mod2...function to their MLIR
/// representation.
static std::unordered_map<std::string, std::tuple<std::string, std::string>>
    deviceKernelMLIRMap;

__attribute__((visibility("default"))) void
registerDeviceKernel(const std::string &module, const std::string &name,
                     const std::string &mangled) {
  auto key = module + "." + name;
  deviceKernelMLIRMap.insert({key, getMLIRCodeAndName(name, mangled)});
}

bool isRegisteredDeviceModule(const std::string &compositeName) {
  for (auto &[k, v] : deviceKernelMLIRMap) {
    if (k.starts_with(compositeName)) // FIXME is this valid?
      return true;
  }

  return false;
}

std::tuple<std::string, std::string>
getDeviceKernel(const std::string &compositeName) {
  auto iter = deviceKernelMLIRMap.find(compositeName);
  if (iter == deviceKernelMLIRMap.end())
    throw std::runtime_error("Invalid composite name for device kernel map.");
  return iter->second;
}

} // namespace cudaq::python