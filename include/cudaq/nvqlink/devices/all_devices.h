#pragma once

#include "cpu_shmem_device.h"
#include "cuda_device.h"
#include "nv_simulation_device.h"

// Note - we want to enable nvq++ to provide external devices if available.
// Strategy may be to search an ext_devices directory, create this
// preprocessor variable, and then auto-gen a header file that includes
// those device headers and force includes it here (-include ...)

#define CUDAQ_NVQLINK_BUILTIN_DEVICES                                          \
  cpu_shmem_device, cuda_device, nv_simulation_device
namespace cudaq::nvqlink {
#ifdef CUDAQ_NVQLINK_EXTERNAL_DEVICES
using any_device =
    std::variant<CUDAQ_NVQLINK_BUILTIN_DEVICES, CUDAQ_NVQLINK_EXTERNAL_DEVICES>;
#else
using any_device = std::variant<CUDAQ_NVQLINK_BUILTIN_DEVICES>;
#endif
} // namespace cudaq::nvqlink
