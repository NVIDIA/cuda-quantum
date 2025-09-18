/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/qclink/compilers/cudaq.h"
#include "../utils/logger.h"
#include <dlfcn.h>
#include <cassert> 

namespace cudaq::qclink {

bool cudaq_compiler::understands_code(const std::string &code) const {
  // FIXME make this better
  return code.find("module") != std::string::npos &&
         code.find("func.func") != std::string::npos;
}

std::unique_ptr<compiled_kernel>
cudaq_compiler::compile(const std::string &code, const std::string &kernelName,
                        std::size_t num_qcs_devices) {

  assert(num_qcs_devices == 1 && "only support 1 qcs device for now.");

  // Use the cuda-q toolchain to compile to object code
  auto result = cudaq::pipeline(code, "/tmp") | cudaq::opt(passes) |
                cudaq::translate("qir") | llvm::llc("x86_64-unknown-linux-gnu");
  auto linkResult = clang::linker().linkMultiple({result.outputFile});

  // Open the compiled result
  void *handle = dlopen(linkResult.outputFile.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("Failed to load shared library: " +
                             std::string(dlerror()));
  // Need to track the allocated resources, i.e. dlclose when we are done
  std::unordered_map<void *, std::function<void(void *)>> tracked_resources(
      {{handle, [](void *hdl) -> void { dlclose(hdl); }}});

  // Get the quantum device id
  auto qid = 0;
  Logger::log("Compiling {} on device {}", kernelName, qid);

  // FIXME how do we get these symbol strings generically?
  auto *argsCreator = static_cast<std::byte *>(
      dlsym(handle, fmt::format("{}.argsCreator", kernelName).c_str()));
  if (auto *err = dlerror())
    throw std::runtime_error("could not extract argsCreator function.");
  auto *thunk = static_cast<std::byte *>(
      dlsym(handle, fmt::format("{}.thunk", kernelName).c_str()));
  if (auto *err = dlerror())
    throw std::runtime_error("could not extract thunk function.");

  // build up the compiled kernel instance
  qcontrol_program program;
  program.qdevice_id = qid;
  program.binary.resize(2 * sizeof(void *));
  std::memcpy(program.binary.data(), &argsCreator, sizeof(void *));
  std::memcpy(program.binary.data() + sizeof(void *), &thunk, sizeof(void *));
  return std::make_unique<compiled_kernel>(
      kernelName, std::vector<qcontrol_program>{program}, tracked_resources);
}

class default_cudaq_compiler : public cudaq_compiler {
public:
  default_cudaq_compiler() : cudaq_compiler() {
    passes = {
        "--add-dealloc",        "--memtoreg=quantum=0",      "--cse",
        "--canonicalize",       "--apply-op-specialization", "--canonicalize",
        "--device-code-loader", "--kernel-execution"};
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      default_cudaq_compiler, "cudaq",
      static std::unique_ptr<compiler> create() {
        return std::make_unique<default_cudaq_compiler>();
      })
};

CUDAQ_REGISTER_TYPE(default_cudaq_compiler)

} // namespace cudaq::qclink
