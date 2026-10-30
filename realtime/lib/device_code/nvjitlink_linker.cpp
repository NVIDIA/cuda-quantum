/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvjitlink_linker.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
// Suppress the toolkit-minor wrappers; CUDA 12.6 and 13.0 do not declare the
// generic entry points in this mode, so the function table spells their types.
#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>
#include <string>
#include <vector>

namespace {

struct nvjitlink_api {
  void *library = nullptr;
  decltype(&nvJitLinkVersion) version = nullptr;
  nvJitLinkResult (*create)(nvJitLinkHandle *, std::uint32_t,
                            const char **) = nullptr;
  nvJitLinkResult (*destroy)(nvJitLinkHandle *) = nullptr;
  nvJitLinkResult (*add_data)(nvJitLinkHandle, nvJitLinkInputType, const void *,
                              std::size_t, const char *) = nullptr;
  nvJitLinkResult (*complete)(nvJitLinkHandle) = nullptr;
  nvJitLinkResult (*get_linked_cubin_size)(nvJitLinkHandle,
                                           std::size_t *) = nullptr;
  nvJitLinkResult (*get_linked_cubin)(nvJitLinkHandle, void *) = nullptr;
  nvJitLinkResult (*get_error_log_size)(nvJitLinkHandle,
                                        std::size_t *) = nullptr;
  nvJitLinkResult (*get_error_log)(nvJitLinkHandle, char *) = nullptr;
};

template <typename Function>
bool load_nvjitlink_symbol(void *library, const char *name,
                           Function &function) {
  dlerror();
  void *symbol = dlsym(library, name);
  const char *error = dlerror();
  if (error || !symbol) {
    std::fprintf(stderr,
                 "ERROR: device link: libnvJitLink does not provide %s: %s\n",
                 name, error ? error : "symbol not found");
    return false;
  }
  static_assert(sizeof(function) == sizeof(symbol));
  std::memcpy(&function, &symbol, sizeof(function));
  return true;
}

const nvjitlink_api *nvjitlink_library() {
  static const nvjitlink_api *const api = []() -> const nvjitlink_api * {
    static nvjitlink_api table{};
    const std::string soname =
        "libnvJitLink.so." + std::to_string(CUDART_VERSION / 1000);
    table.library = dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!table.library) {
      const char *error = dlerror();
      std::fprintf(stderr, "ERROR: device link: cannot load %s: %s\n",
                   soname.c_str(), error ? error : "unknown error");
      return nullptr;
    }
    if (!load_nvjitlink_symbol(table.library, "nvJitLinkVersion",
                               table.version) ||
        !load_nvjitlink_symbol(table.library, "nvJitLinkCreate",
                               table.create) ||
        !load_nvjitlink_symbol(table.library, "nvJitLinkDestroy",
                               table.destroy) ||
        !load_nvjitlink_symbol(table.library, "nvJitLinkAddData",
                               table.add_data) ||
        !load_nvjitlink_symbol(table.library, "nvJitLinkComplete",
                               table.complete) ||
        !load_nvjitlink_symbol(table.library, "nvJitLinkGetLinkedCubinSize",
                               table.get_linked_cubin_size) ||
        !load_nvjitlink_symbol(table.library, "nvJitLinkGetLinkedCubin",
                               table.get_linked_cubin) ||
        !load_nvjitlink_symbol(table.library, "nvJitLinkGetErrorLogSize",
                               table.get_error_log_size) ||
        !load_nvjitlink_symbol(table.library, "nvJitLinkGetErrorLog",
                               table.get_error_log))
      return nullptr;
    return &table;
  }();
  return api;
}

struct fatbin_header {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t header_size;
  std::uint64_t payload_size;
};
constexpr std::uint32_t fatbin_magic = 0xBA55ED50u;

// An image holds one `fatbin`, or several back to back with zero padding
// between them, as a binary's __nv_relfatbin section does.  Appends each one
// to `out`; returns false when a header overruns the image or none is found.
bool split_fatbins(const cudaq_device_image_t &image,
                   std::vector<cudaq_device_image_t> &out) {
  const char *begin = static_cast<const char *>(image.data);
  if (!begin)
    return false;
  const std::size_t size = image.size;
  const std::size_t first = out.size();
  std::size_t at = 0;
  while (at + sizeof(fatbin_header) <= size) {
    fatbin_header header;
    std::memcpy(&header, begin + at, sizeof(header));
    if (header.magic != fatbin_magic) {
      ++at;
      continue;
    }
    const std::uint64_t length =
        std::uint64_t{header.header_size} + header.payload_size;
    if (header.header_size < sizeof(fatbin_header) || length > size - at)
      return false;
    out.push_back({begin + at, static_cast<std::size_t>(length)});
    at += static_cast<std::size_t>(length);
  }
  return out.size() > first;
}

void print_link_failure(const nvjitlink_api &api, nvJitLinkHandle handle,
                        const std::string &what, nvJitLinkResult result,
                        bool adding_input, const std::string &arch = {}) {
  unsigned major = 0;
  unsigned minor = 0;
  api.version(&major, &minor);
  std::fprintf(stderr,
               "ERROR: device link: %s failed (nvJitLinkResult %d, nvJitLink "
               "%u.%u)\n",
               what.c_str(), static_cast<int>(result), major, minor);
  std::size_t size = 0;
  if (handle && api.get_error_log_size(handle, &size) == NVJITLINK_SUCCESS &&
      size > 1) {
    std::string log(size, '\0');
    if (api.get_error_log(handle, log.data()) == NVJITLINK_SUCCESS)
      std::fprintf(stderr, "%s\n", log.c_str());
  }
  if (adding_input)
    std::fprintf(stderr,
                 "  images must come from a CUDA toolkit no newer than this "
                 "nvJitLink (%u.%u),\n"
                 "  and this nvJitLink must support %s\n",
                 major, minor, arch.c_str());
}

} // namespace

const cudaq::realtime::detail::driver_library_api *
cudaq::realtime::detail::driver_library() {
  static const driver_library_api *const api =
      []() -> const driver_library_api * {
    static driver_library_api table{};
    const struct {
      const char *name;
      void **fn;
    } entries[] = {
        {"cuLibraryLoadData", reinterpret_cast<void **>(&table.load_data)},
        {"cuLibraryGetKernel", reinterpret_cast<void **>(&table.get_kernel)},
        {"cuLibraryGetGlobal", reinterpret_cast<void **>(&table.get_global)},
        {"cuLibraryUnload", reinterpret_cast<void **>(&table.unload)},
        {"cuLaunchKernel", reinterpret_cast<void **>(&table.launch_kernel)},
    };
    for (const auto &entry : entries) {
      cudaDriverEntryPointQueryResult found =
          cudaDriverEntryPointSymbolNotFound;
      if (cudaGetDriverEntryPointByVersion(entry.name, entry.fn, 12000,
                                           cudaEnableDefault,
                                           &found) != cudaSuccess ||
          found != cudaDriverEntryPointSuccess || !*entry.fn) {
        std::fprintf(stderr,
                     "ERROR: device link: the CUDA driver does not provide "
                     "%s\n",
                     entry.name);
        return nullptr;
      }
    }
    return &table;
  }();
  return api;
}

cudaq_status_t
cudaq::realtime::detail::link_device_code(const device_code_input *inputs,
                                          std::size_t input_count,
                                          CUlibrary *out_library) {
  if (!inputs || !out_library)
    return CUDAQ_ERR_INVALID_ARG;
  const driver_library_api *driver = driver_library();
  if (!driver)
    return CUDAQ_ERR_CUDA;
  const nvjitlink_api *linker = nvjitlink_library();
  if (!linker)
    return CUDAQ_ERR_UNSUPPORTED;

  int device = 0;
  int major = 0;
  int minor = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess) {
    std::fprintf(stderr,
                 "ERROR: device link: cannot query the current device\n");
    return CUDAQ_ERR_CUDA;
  }
  // The generic target only: the images carry generic LTO-IR, PTX or SASS.
  const std::string arch =
      "-arch=sm_" + std::to_string(major) + std::to_string(minor);
  const char *options[] = {arch.c_str(), "-lto"};

  nvJitLinkHandle handle = nullptr;
  nvJitLinkResult result = linker->create(&handle, 2, options);
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(*linker, handle, "creating the linker for " + arch,
                       result, false);
    return CUDAQ_ERR_INTERNAL;
  }

  for (std::size_t i = 0; i < input_count; ++i) {
    const device_code_input &input = inputs[i];
    for (std::size_t j = 0; j < input.count; ++j) {
      const std::string image_name =
          std::string(input.origin) + " image " + std::to_string(j);
      std::vector<cudaq_device_image_t> fatbins;
      if (!split_fatbins(input.images[j], fatbins)) {
        std::fprintf(stderr,
                     "ERROR: device link: the %s is not relocatable device "
                     "code\n",
                     image_name.c_str());
        linker->destroy(&handle);
        return CUDAQ_ERR_INTERNAL;
      }
      for (std::size_t k = 0; k < fatbins.size(); ++k) {
        const std::string name = image_name + " fatbin " + std::to_string(k);
        result =
            linker->add_data(handle, NVJITLINK_INPUT_FATBIN, fatbins[k].data,
                             fatbins[k].size, name.c_str());
        if (result != NVJITLINK_SUCCESS) {
          print_link_failure(*linker, handle, "adding the " + name, result,
                             true, arch);
          linker->destroy(&handle);
          return CUDAQ_ERR_INTERNAL;
        }
      }
    }
  }

  result = linker->complete(handle);
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(*linker, handle, "linking for " + arch, result, false);
    linker->destroy(&handle);
    return CUDAQ_ERR_INTERNAL;
  }

  std::size_t cubin_size = 0;
  std::vector<char> cubin;
  result = linker->get_linked_cubin_size(handle, &cubin_size);
  if (result == NVJITLINK_SUCCESS) {
    cubin.resize(cubin_size);
    result = linker->get_linked_cubin(handle, cubin.data());
  }
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(*linker, handle, "retrieving the linked cubin", result,
                       false);
    linker->destroy(&handle);
    return CUDAQ_ERR_INTERNAL;
  }
  linker->destroy(&handle);

  CUlibrary library = nullptr;
  const CUresult err = driver->load_data(&library, cubin.data(), nullptr,
                                         nullptr, 0, nullptr, nullptr, 0);
  if (err != CUDA_SUCCESS) {
    std::fprintf(stderr,
                 "ERROR: device link: loading the linked module "
                 "failed: CUresult %d\n",
                 static_cast<int>(err));
    return CUDAQ_ERR_CUDA;
  }
  *out_library = library;
  return CUDAQ_OK;
}
