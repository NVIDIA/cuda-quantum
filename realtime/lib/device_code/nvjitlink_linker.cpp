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
#include <nvJitLink.h>
#include <string>
#include <vector>

namespace {

// __nvJitLink<name>_<CUDAQ_REALTIME_NVJITLINK_ABI>, the versioned entry point
// the header's inline nvJitLink<name> wrapper calls.
#define CUDAQ_NVJITLINK_CONCAT_(a, b) a##b
#define CUDAQ_NVJITLINK_CONCAT(a, b) CUDAQ_NVJITLINK_CONCAT_(a, b)
#define CUDAQ_NVJITLINK_STRING_(x) #x
#define CUDAQ_NVJITLINK_STRING(x) CUDAQ_NVJITLINK_STRING_(x)
#define CUDAQ_NVJITLINK_ENTRY(name)                                            \
  CUDAQ_NVJITLINK_CONCAT(__nvJitLink##name##_, CUDAQ_REALTIME_NVJITLINK_ABI)

/// The nvJitLink functions the link uses.  libcudaq-realtime.so must load on
/// hosts without nvJitLink (pure host dispatch needs no device link), so the
/// library is opened, and these looked up, only when a link first needs them.
struct nvjitlink_api {
  decltype(&CUDAQ_NVJITLINK_ENTRY(Create)) create;
  decltype(&CUDAQ_NVJITLINK_ENTRY(Destroy)) destroy;
  decltype(&CUDAQ_NVJITLINK_ENTRY(AddData)) add_data;
  decltype(&CUDAQ_NVJITLINK_ENTRY(Complete)) complete;
  decltype(&CUDAQ_NVJITLINK_ENTRY(GetLinkedCubinSize)) get_linked_cubin_size;
  decltype(&CUDAQ_NVJITLINK_ENTRY(GetLinkedCubin)) get_linked_cubin;
  decltype(&CUDAQ_NVJITLINK_ENTRY(GetErrorLogSize)) get_error_log_size;
  decltype(&CUDAQ_NVJITLINK_ENTRY(GetErrorLog)) get_error_log;
  decltype(&nvJitLinkVersion) version;
};

/// nvJitLink, opened and looked up on the first call.  Null, after printing
/// what is missing, if the library cannot be opened or lacks a function.
const nvjitlink_api *nvjitlink_library() {
  static const nvjitlink_api *const api = []() -> const nvjitlink_api * {
    void *library = dlopen(CUDAQ_REALTIME_NVJITLINK_SONAME, RTLD_NOW);
    if (!library) {
      std::fprintf(stderr, "ERROR: device link: cannot load %s: %s\n",
                   CUDAQ_REALTIME_NVJITLINK_SONAME, dlerror());
      return nullptr;
    }
    static nvjitlink_api table{};
    const struct {
      const char *name;
      void **fn;
    } entries[] = {
#define CUDAQ_NVJITLINK_LOOKUP(name, field)                                    \
  {CUDAQ_NVJITLINK_STRING(CUDAQ_NVJITLINK_ENTRY(name)),                        \
   reinterpret_cast<void **>(&table.field)}
        CUDAQ_NVJITLINK_LOOKUP(Create, create),
        CUDAQ_NVJITLINK_LOOKUP(Destroy, destroy),
        CUDAQ_NVJITLINK_LOOKUP(AddData, add_data),
        CUDAQ_NVJITLINK_LOOKUP(Complete, complete),
        CUDAQ_NVJITLINK_LOOKUP(GetLinkedCubinSize, get_linked_cubin_size),
        CUDAQ_NVJITLINK_LOOKUP(GetLinkedCubin, get_linked_cubin),
        CUDAQ_NVJITLINK_LOOKUP(GetErrorLogSize, get_error_log_size),
        CUDAQ_NVJITLINK_LOOKUP(GetErrorLog, get_error_log),
#undef CUDAQ_NVJITLINK_LOOKUP
        {"nvJitLinkVersion", reinterpret_cast<void **>(&table.version)},
    };
    for (const auto &entry : entries) {
      *entry.fn = dlsym(library, entry.name);
      if (!*entry.fn) {
        std::fprintf(stderr,
                     "ERROR: device link: %s does not provide %s; it must "
                     "be nvJitLink %s or newer\n",
                     CUDAQ_REALTIME_NVJITLINK_SONAME, entry.name,
                     CUDAQ_REALTIME_NVJITLINK_VERSION);
        dlclose(library);
        return nullptr;
      }
    }
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

void print_link_failure(const nvjitlink_api &nvjitlink, nvJitLinkHandle handle,
                        const std::string &what, nvJitLinkResult result,
                        bool adding_input, const std::string &arch = {}) {
  unsigned major = 0;
  unsigned minor = 0;
  nvjitlink.version(&major, &minor);
  std::fprintf(stderr,
               "ERROR: device link: %s failed (nvJitLinkResult %d, nvJitLink "
               "%u.%u)\n",
               what.c_str(), static_cast<int>(result), major, minor);
  std::size_t size = 0;
  if (handle &&
      nvjitlink.get_error_log_size(handle, &size) == NVJITLINK_SUCCESS &&
      size > 1) {
    std::string log(size, '\0');
    if (nvjitlink.get_error_log(handle, log.data()) == NVJITLINK_SUCCESS)
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
  const nvjitlink_api *nvjitlink = nvjitlink_library();
  if (!nvjitlink)
    return CUDAQ_ERR_CUDA;

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
  nvJitLinkResult result = nvjitlink->create(&handle, 2, options);
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(*nvjitlink, handle, "creating the linker for " + arch,
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
        nvjitlink->destroy(&handle);
        return CUDAQ_ERR_INTERNAL;
      }
      for (std::size_t k = 0; k < fatbins.size(); ++k) {
        const std::string name = image_name + " fatbin " + std::to_string(k);
        result =
            nvjitlink->add_data(handle, NVJITLINK_INPUT_FATBIN, fatbins[k].data,
                                fatbins[k].size, name.c_str());
        if (result != NVJITLINK_SUCCESS) {
          print_link_failure(*nvjitlink, handle, "adding the " + name, result,
                             true, arch);
          nvjitlink->destroy(&handle);
          return CUDAQ_ERR_INTERNAL;
        }
      }
    }
  }

  result = nvjitlink->complete(handle);
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(*nvjitlink, handle, "linking for " + arch, result,
                       false);
    nvjitlink->destroy(&handle);
    return CUDAQ_ERR_INTERNAL;
  }

  std::size_t cubin_size = 0;
  std::vector<char> cubin;
  result = nvjitlink->get_linked_cubin_size(handle, &cubin_size);
  if (result == NVJITLINK_SUCCESS) {
    cubin.resize(cubin_size);
    result = nvjitlink->get_linked_cubin(handle, cubin.data());
  }
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(*nvjitlink, handle, "retrieving the linked cubin",
                       result, false);
    nvjitlink->destroy(&handle);
    return CUDAQ_ERR_INTERNAL;
  }
  nvjitlink->destroy(&handle);

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
