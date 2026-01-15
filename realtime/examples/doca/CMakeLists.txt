# DOCA Channel Examples
# Requires NVQLINK_ENABLE_DOCA=ON

if(NOT NVQLINK_ENABLE_DOCA)
  message(STATUS "DOCA examples disabled (requires -DNVQLINK_ENABLE_DOCA=ON)")
  return()
endif()

find_program(NVIDIA_SMI "nvidia-smi")
if(${NVIDIA_SMI} STREQUAL "NVIDIA_SMI-NOTFOUND")
  # No NVIDIA GPU found, we cannot build DOCA example executables,
  # as they require linking against CUDA driver lib (libcuda.so.1),
  # which is not available on systems without NVIDIA GPUs.
  message(STATUS "NVIDIA GPU not found, skipping DOCA examples.")
  return()
endif()

# Fetch nlohmann_json for UDP control server
include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# DOCA Daemon Example (from Plan Phase 5)
# The main file is .cpp for C++20 compatibility with daemon headers.
# GPU functions are in a separate .cu file compiled with nvcc.
add_executable(doca_daemon_example
  doca_daemon_example.cpp
  gpu_functions.cu
)

# CRITICAL: Enable CUDA separable compilation for device function pointers
# to work across compilation units (gpu_functions.cu <-> doca_rpc_kernel.cu)
set_target_properties(doca_daemon_example PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(doca_daemon_example
  PRIVATE
    nvqlink_daemon
    nvqlink_network
    nlohmann_json::nlohmann_json
)

target_include_directories(doca_daemon_example
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# DOCA Client (ibverbs-based, adapted from roce_daemon_client)
add_executable(doca_client
  doca_client.cpp
)

target_link_libraries(doca_client
  PRIVATE
    ibverbs
    nlohmann_json::nlohmann_json
)

target_include_directories(doca_client
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/examples/roce  # For roce_client_common.h
)

message(STATUS "DOCA examples enabled:")
message(STATUS "  - doca_daemon_example")
message(STATUS "  - doca_client")
