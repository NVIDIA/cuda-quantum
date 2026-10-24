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
#include <nvJitLink.h>
#include <string>
#include <vector>

namespace {

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

void print_link_failure(nvJitLinkHandle handle, const std::string &what,
                        nvJitLinkResult result, bool adding_input) {
  unsigned major = 0;
  unsigned minor = 0;
  nvJitLinkVersion(&major, &minor);
  std::fprintf(stderr,
               "ERROR: device link: %s failed (nvJitLinkResult %d, nvJitLink "
               "%u.%u)\n",
               what.c_str(), static_cast<int>(result), major, minor);
  std::size_t size = 0;
  if (handle && nvJitLinkGetErrorLogSize(handle, &size) == NVJITLINK_SUCCESS &&
      size > 1) {
    std::string log(size, '\0');
    if (nvJitLinkGetErrorLog(handle, log.data()) == NVJITLINK_SUCCESS)
      std::fprintf(stderr, "%s\n", log.c_str());
  }
  if (adding_input)
    std::fprintf(stderr,
                 "  images must come from a CUDA toolkit no newer than this "
                 "nvJitLink (%u.%u)\n",
                 major, minor);
}

} // namespace

cudaq_status_t
cudaq::realtime::detail::link_device_code(const device_code_input *inputs,
                                          std::size_t input_count,
                                          cudaLibrary_t *out_library) {
  if (!inputs || !out_library)
    return CUDAQ_ERR_INVALID_ARG;

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
  nvJitLinkResult result = nvJitLinkCreate(&handle, 2, options);
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(handle, "creating the linker for " + arch, result,
                       false);
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
        nvJitLinkDestroy(&handle);
        return CUDAQ_ERR_INTERNAL;
      }
      for (std::size_t k = 0; k < fatbins.size(); ++k) {
        const std::string name = image_name + " fatbin " + std::to_string(k);
        result =
            nvJitLinkAddData(handle, NVJITLINK_INPUT_FATBIN, fatbins[k].data,
                             fatbins[k].size, name.c_str());
        if (result != NVJITLINK_SUCCESS) {
          print_link_failure(handle, "adding the " + name, result, true);
          nvJitLinkDestroy(&handle);
          return CUDAQ_ERR_INTERNAL;
        }
      }
    }
  }

  result = nvJitLinkComplete(handle);
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(handle, "linking for " + arch, result, false);
    nvJitLinkDestroy(&handle);
    return CUDAQ_ERR_INTERNAL;
  }

  std::size_t cubin_size = 0;
  std::vector<char> cubin;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  if (result == NVJITLINK_SUCCESS) {
    cubin.resize(cubin_size);
    result = nvJitLinkGetLinkedCubin(handle, cubin.data());
  }
  if (result != NVJITLINK_SUCCESS) {
    print_link_failure(handle, "retrieving the linked cubin", result, false);
    nvJitLinkDestroy(&handle);
    return CUDAQ_ERR_INTERNAL;
  }
  nvJitLinkDestroy(&handle);

  cudaLibrary_t library = nullptr;
  const cudaError_t err = cudaLibraryLoadData(&library, cubin.data(), nullptr,
                                              nullptr, 0, nullptr, nullptr, 0);
  if (err != cudaSuccess) {
    std::fprintf(stderr,
                 "ERROR: device link: loading the linked module "
                 "failed: %s\n",
                 cudaGetErrorString(err));
    return CUDAQ_ERR_CUDA;
  }
  *out_library = library;
  return CUDAQ_OK;
}
