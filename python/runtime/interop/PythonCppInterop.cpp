/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PythonCppInterop.h"
#include "cudaq.h" // unfortunately, cudaq::get_quake is here at top level
#include "cudaq/utils/cudaq_utils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

cudaq::python::CppPyKernelDecorator::~CppPyKernelDecorator() {
  if (execution_engine) {
    auto *ee = reinterpret_cast<mlir::ExecutionEngine *>(execution_engine);
    delete ee;
    execution_engine = nullptr;
  }
}

std::string cudaq::python::getKernelName(const std::string &input) {
  size_t pos = 0;
  std::string result;
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

std::string cudaq::python::extractSubstring(const std::string &input,
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

// Helper to extract all external calls from the MLIR code.
static std::vector<std::string> getExternalCall(const std::string &mlirCode) {
  std::vector<std::string> externCalls;
  // Split the code into lines
  const auto lines = cudaq::split(mlirCode, '\n');

  for (auto &line : lines) {
    // find these external calls, e.g. `call @malloc` or `device_call
    // @device_kernel`
    auto start = line.find("call @");
    if (start == std::string::npos)
      continue;
    start += 6; // Move to the end of "call @"
    const auto end = line.find("(", start);
    if (end == std::string::npos)
      continue;
    const std::string callFuncName = line.substr(start, end - start);
    externCalls.emplace_back(callFuncName);
  }

  return externCalls;
}

// Helper to find the function declaration in the MLIR code.
static std::string findFuncDecl(const std::string &mlirCode,
                                const std::string &funcName) {
  const auto start = mlirCode.find("func.func private @" + funcName);
  if (start == std::string::npos)
    return "";
  const auto end = mlirCode.find("\n", start);
  if (end == std::string::npos)
    return "";
  return mlirCode.substr(start, end - start);
}

std::tuple<std::string, std::string>
cudaq::python::getMLIRCodeAndName(const std::string &name,
                                  const std::string mangledArgs) {
  const auto originalCppMLIRCode =
      cudaq::get_quake(std::remove_cvref_t<decltype(name)>(name), mangledArgs);
  auto kernelName = cudaq::python::getKernelName(originalCppMLIRCode);
  auto cppMLIRCode = "module {\nfunc.func @" + kernelName +
                     extractSubstring(originalCppMLIRCode,
                                      "func.func @" + kernelName, "func.func") +
                     "\n";
  // If there are external calls, we need to find their declarations
  // and add them to the MLIR code.
  const auto externalCalls = getExternalCall(cppMLIRCode);
  for (const auto &externalCall : externalCalls) {
    cppMLIRCode += findFuncDecl(originalCppMLIRCode, externalCall);
  }
  cppMLIRCode += "\n}";
  return std::make_tuple(kernelName, cppMLIRCode);
}

/// Map device kernels represented as mod1.mod2...function to their MLIR
/// representation.
static std::unordered_map<std::string, std::tuple<std::string, std::string>>
    deviceKernelMLIRMap;

__attribute__((visibility("default"))) void
cudaq::python::registerDeviceKernel(const std::string &module,
                                    const std::string &name,
                                    const std::string &mangled) {
  auto key = module + "." + name;
  deviceKernelMLIRMap[key] = getMLIRCodeAndName(name, mangled);
}

bool cudaq::python::isRegisteredDeviceModule(const std::string &compositeName) {
  for (auto &[k, v] : deviceKernelMLIRMap) {
    if (k.starts_with(compositeName)) // FIXME is this valid?
      return true;
  }

  return false;
}

std::tuple<std::string, std::string>
cudaq::python::getDeviceKernel(const std::string &compositeName) {
  auto iter = deviceKernelMLIRMap.find(compositeName);
  if (iter == deviceKernelMLIRMap.end())
    throw std::runtime_error("Invalid composite name for device kernel map.");
  return iter->second;
}
