/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/Plugin.h"

using namespace llvm;
namespace cudaq {
Expected<Plugin> Plugin::Load(const std::string &fileName) {
  std::string error;
  auto library =
      sys::DynamicLibrary::getPermanentLibrary(fileName.c_str(), &error);
  if (!library.isValid())
    return make_error<StringError>(Twine("Could not load library '") +
                                       fileName + "': " + error,
                                   inconvertibleErrorCode());

  Plugin plugin{fileName, library};
  auto getDetailsFn = library.getAddressOfSymbol("cudaqGetPluginInfo");

  if (!getDetailsFn)
    return make_error<StringError>(Twine("Plugin entry point not found in '") +
                                       fileName + "'. Is this a legacy plugin?",
                                   inconvertibleErrorCode());

  plugin.pluginInfo =
      reinterpret_cast<decltype(cudaqGetPluginInfo) *>(getDetailsFn)();

  return plugin;
}
} // namespace cudaq
