/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvjitlink_linker.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatch_loop.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// The DEVICE ring launcher, used when the caller sets none.  Defined in
// libcudaq-realtime-dispatch.a, which the consumer device-links together with
// its handlers; it is not part of this library.  Weak, so a binary that
// never links the archive still loads, and the address is null.  It
// binds to the first copy exported in the process, so a consumer whose copy
// is hidden (--exclude-libs) or not the first names its own launcher instead.
#pragma weak cudaq_launch_dispatch_kernel_regular

// The unified dispatch kernel's LTO-IR fatbin, embedded by
// unified_dispatch_kernel_image.cpp.
extern "C" const char cudaq_unified_dispatch_kernel_image_begin[]
    __attribute__((visibility("hidden")));
extern "C" const char cudaq_unified_dispatch_kernel_image_end[]
    __attribute__((visibility("hidden")));

struct cudaq_dispatch_manager_t {
  int reserved = 0;
};

struct JitDeviceLibrary {
  std::vector<cudaq_device_image_t> bridge_device_code;
  CUlibrary library = nullptr;
  CUkernel dispatch_kernel = nullptr;
  std::vector<cudaq_device_image_t> application_device_code;
  // Handler addresses get_device_handler has handed out; start() accepts no
  // other DEVICE_CALL address.
  std::unordered_set<void *> linked_functions;
};

struct cudaq_dispatcher_t {
  cudaq_dispatcher_config_t config{};
  cudaq_ringbuffer_t ringbuffer{};
  cudaq_function_table_t table{};
  cudaq_dispatch_launch_fn_t launch_fn = nullptr;
  cudaq_unified_launch_fn_t unified_launch_fn = nullptr;
  void *transport_ctx = nullptr;
  volatile int *shutdown_flag = nullptr;
  uint64_t *stats = nullptr;
  cudaStream_t stream = nullptr;
  bool running = false;
  // The unified loop's data-plane (HOST + UNIFIED path), copied in by
  // set_cpu_dataplane or set_bridge; its hooks stay NULL until one of them
  // runs.  The public setter is void* to keep the API transport-agnostic;
  // stored typed here since this TU includes bridge_interface.h.
  cudaq_cpu_dataplane_t cpu_dataplane{};
  // Both HOST dispatch paths (ring and unified) run their loop on this thread:
  // spawned in cudaq_dispatcher_start, joined in stop/destroy.
  std::thread host_thread;
  // The bridge wired by cudaq_dispatcher_set_bridge, if any.
  cudaq_realtime_bridge_handle_t bridge = nullptr;
  // DEVICE + UNIFIED over a GPU data plane: the bridge's hook images, copied
  // by set_bridge, and the module the first handler lookup (or start) links
  // them into, with the handlers' binary's images it was linked with.
  JitDeviceLibrary jit_device_lib;
};

namespace {
// Bridges currently wired to a dispatcher.  A bridge drives exactly one.
std::mutex attached_bridges_mutex;
std::unordered_set<cudaq_realtime_bridge_handle_t> attached_bridges;
} // namespace

// True once set_bridge has wired a GPU data plane: the dispatcher links its
// module, and start() runs it through its own kernel or the application's
// launcher.
static bool has_gpu_dataplane(const cudaq_dispatcher_t *dispatcher) {
  return !dispatcher->jit_device_lib.bridge_device_code.empty();
}

// True for HOST + UNIFIED: the dispatcher runs cudaq_host_unified_loop on its
// own thread over the ring data-plane (HOST_CALL inline).
static bool is_host_unified_dispatcher(const cudaq_dispatcher_t *dispatcher) {
  return dispatcher &&
         dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST &&
         dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED;
}

static bool is_valid_kernel_type(cudaq_kernel_type_t kernel_type) {
  switch (kernel_type) {
  case CUDAQ_KERNEL_REGULAR:
  case CUDAQ_KERNEL_UNIFIED:
    return true;
  default:
    return false;
  }
}

// The DEVICE ring launcher start() calls: the caller's, else the archive's
// for the kernel type.  Null when neither is available.
static cudaq_dispatch_launch_fn_t
ring_launch_fn(const cudaq_dispatcher_t *dispatcher) {
  if (dispatcher->launch_fn)
    return dispatcher->launch_fn;
  switch (dispatcher->config.kernel_type) {
  case CUDAQ_KERNEL_REGULAR:
    return &cudaq_launch_dispatch_kernel_regular;
  default:
    return nullptr;
  }
}

static bool is_valid_dispatch_mode(cudaq_dispatch_mode_t dispatch_mode) {
  switch (dispatch_mode) {
  case CUDAQ_DISPATCH_DEVICE_CALL:
  case CUDAQ_DISPATCH_GRAPH_LAUNCH:
  case CUDAQ_DISPATCH_HOST_CALL:
    return true;
  default:
    return false;
  }
}

static cudaq_status_t validate_dispatcher(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (!is_valid_kernel_type(dispatcher->config.kernel_type))
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->shutdown_flag || !dispatcher->stats)
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->table.entries || dispatcher->table.count == 0)
    return CUDAQ_ERR_INVALID_ARG;

  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    // The host loops serve HOST_CALL entries only.  Under shared_ring_mode a
    // DEVICE_CALL or GRAPH_LAUNCH entry may belong to a peer dispatcher on the
    // same ring; otherwise nothing could serve it and its requests would be
    // dropped unanswered, so refuse the table.
    if (!dispatcher->config.shared_ring_mode)
      for (uint32_t i = 0; i < dispatcher->table.count; ++i)
        if (dispatcher->table.entries[i].dispatch_mode !=
            CUDAQ_DISPATCH_HOST_CALL)
          return CUDAQ_ERR_UNSUPPORTED;
    if (dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED) {
      if (!dispatcher->cpu_dataplane.rx_poll ||
          !dispatcher->cpu_dataplane.tx_publish)
        return CUDAQ_ERR_INVALID_ARG;
      return CUDAQ_OK;
    }
    if (!dispatcher->ringbuffer.rx_flags_host ||
        !dispatcher->ringbuffer.tx_flags_host ||
        !dispatcher->ringbuffer.rx_data_host ||
        !dispatcher->ringbuffer.tx_data_host)
      return CUDAQ_ERR_INVALID_ARG;
    return CUDAQ_OK;
  }

  if (dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED) {
    if (!dispatcher->transport_ctx)
      return CUDAQ_ERR_INVALID_ARG;
    if (dispatcher->config.slot_size == 0)
      return CUDAQ_ERR_INVALID_ARG;
    // One block of >= 1 threads: the block is what every DEVICE_CALL handler
    // runs on.  The application's launcher over a GPU data plane picks its
    // own grid.
    const bool launcher_grid =
        dispatcher->unified_launch_fn && has_gpu_dataplane(dispatcher);
    if ((!launcher_grid && dispatcher->config.num_blocks != 1) ||
        dispatcher->config.threads_per_block == 0)
      return CUDAQ_ERR_INVALID_ARG;
  } else {
    if (!ring_launch_fn(dispatcher))
      return CUDAQ_ERR_INVALID_ARG;
    if (!dispatcher->ringbuffer.rx_flags || !dispatcher->ringbuffer.tx_flags)
      return CUDAQ_ERR_INVALID_ARG;
    // One block of >= 1 threads, as for the unified kernel.
    if (dispatcher->config.num_blocks != 1 ||
        dispatcher->config.threads_per_block == 0 ||
        dispatcher->config.num_slots == 0 || dispatcher->config.slot_size == 0)
      return CUDAQ_ERR_INVALID_ARG;
    if (!is_valid_dispatch_mode(dispatcher->config.dispatch_mode))
      return CUDAQ_ERR_INVALID_ARG;
  }
  return CUDAQ_OK;
}

static void release_bridge_attachment(cudaq_realtime_bridge_handle_t bridge) {
  std::lock_guard<std::mutex> lock(attached_bridges_mutex);
  attached_bridges.erase(bridge);
}

static const char *bridge_shape_name(const cudaq_dispatcher_config_t &config) {
  const bool unified = config.kernel_type == CUDAQ_KERNEL_UNIFIED;
  if (config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST)
    return unified ? "HOST unified" : "HOST ring";
  return unified ? "DEVICE unified" : "DEVICE ring";
}

// Take the geometry a config left at 0 from the bridge, and refuse one that
// disagrees with it.  A provider that reports no geometry leaves the config as
// given, for start() to validate.
static cudaq_status_t
apply_bridge_geometry(cudaq_realtime_bridge_handle_t bridge,
                      cudaq_dispatcher_config_t &config) {
  uint32_t num_slots = 0;
  uint32_t slot_size = 0;
  const cudaq_status_t status =
      cudaq_bridge_get_ring_geometry(bridge, &num_slots, &slot_size);
  if (status == CUDAQ_ERR_UNSUPPORTED)
    return CUDAQ_OK;
  if (status != CUDAQ_OK) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_set_bridge: the bridge failed to report "
            "its ring geometry (status %d)\n",
            status);
    return status;
  }
  if ((config.num_slots != 0 && config.num_slots != num_slots) ||
      (config.slot_size != 0 && config.slot_size != slot_size)) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_set_bridge: the config's geometry "
            "(num_slots=%u slot_size=%u) disagrees with the bridge's "
            "(num_slots=%u slot_size=%u); leave them 0 to take the bridge's\n",
            config.num_slots, config.slot_size, num_slots, slot_size);
    return CUDAQ_ERR_INVALID_ARG;
  }
  if (config.num_slots == 0)
    config.num_slots = num_slots;
  if (config.slot_size == 0)
    config.slot_size = slot_size;
  return CUDAQ_OK;
}

static bool is_device_unified_dispatcher(const cudaq_dispatcher_t *dispatcher) {
  return dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_DEVICE &&
         dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED;
}

cudaq_status_t
cudaq_dispatch_manager_create(cudaq_dispatch_manager_t **out_mgr) {
  if (!out_mgr)
    return CUDAQ_ERR_INVALID_ARG;
  auto *mgr = new (std::nothrow) cudaq_dispatch_manager_t();
  if (!mgr)
    return CUDAQ_ERR_INTERNAL;
  *out_mgr = mgr;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatch_manager_destroy(cudaq_dispatch_manager_t *mgr) {
  if (mgr)
    delete mgr;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_create(cudaq_dispatch_manager_t *,
                                       const cudaq_dispatcher_config_t *config,
                                       cudaq_dispatcher_t **out_dispatcher) {
  if (!config || !out_dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  auto *dispatcher = new (std::nothrow) cudaq_dispatcher_t();
  if (!dispatcher)
    return CUDAQ_ERR_INTERNAL;
  dispatcher->config = *config;
  // A negative device id means device 0.
  if (dispatcher->config.device_id < 0)
    dispatcher->config.device_id = 0;
  *out_dispatcher = dispatcher;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_destroy(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->running) {
    if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
      // `const_cast` drops the flag's `volatile` qualifier (`reinterpret_cast`
      // can't cast away cv-qualifiers) so it can be written as a plain atomic.
      if (dispatcher->shutdown_flag)
        reinterpret_cast<std::atomic<int> *>(
            const_cast<int *>(dispatcher->shutdown_flag))
            ->store(1, std::memory_order_relaxed);
      if (dispatcher->host_thread.joinable())
        dispatcher->host_thread.join();
    }
    dispatcher->running = false;
  }
  if (dispatcher->bridge)
    release_bridge_attachment(dispatcher->bridge);
  // A DEVICE kernel that was never stopped may still run from the module, so
  // the module is left loaded rather than unloaded under it.
  if (dispatcher->jit_device_lib.library && !dispatcher->stream)
    cudaq::realtime::detail::driver_library()->unload(
        dispatcher->jit_device_lib.library);
  delete dispatcher;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_ringbuffer(cudaq_dispatcher_t *dispatcher,
                                const cudaq_ringbuffer_t *ringbuffer) {
  if (!dispatcher || !ringbuffer)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->ringbuffer = *ringbuffer;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_function_table(cudaq_dispatcher_t *dispatcher,
                                    const cudaq_function_table_t *table) {
  if (!dispatcher || !table)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->table = *table;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_set_control(cudaq_dispatcher_t *dispatcher,
                                            volatile int *shutdown_flag,
                                            uint64_t *stats) {
  if (!dispatcher || !shutdown_flag || !stats)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->shutdown_flag = shutdown_flag;
  dispatcher->stats = stats;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_launch_fn(cudaq_dispatcher_t *dispatcher,
                               cudaq_dispatch_launch_fn_t launch_fn) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST &&
      launch_fn != nullptr)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->config.dispatch_path != CUDAQ_DISPATCH_PATH_HOST &&
      !launch_fn)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->launch_fn = launch_fn;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_unified_launch(cudaq_dispatcher_t *dispatcher,
                                    cudaq_unified_launch_fn_t unified_launch_fn,
                                    void *transport_ctx) {
  if (!dispatcher || !unified_launch_fn)
    return CUDAQ_ERR_INVALID_ARG;
  if (has_gpu_dataplane(dispatcher)) {
    // The application's own dispatch loop, over the bridge's context.
    if (transport_ctx || dispatcher->running) {
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_set_unified_launch: over a GPU data "
              "plane, pass a NULL transport context (the bridge's is kept) "
              "and call it before cudaq_dispatcher_start\n");
      return CUDAQ_ERR_INVALID_ARG;
    }
    dispatcher->unified_launch_fn = unified_launch_fn;
    return CUDAQ_OK;
  }
  // Both are required: the dispatcher's own unified kernel runs only over a
  // bridge's GPU data plane, wired by set_bridge.
  if (!transport_ctx)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->bridge) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_set_unified_launch: the dispatcher's "
            "bridge serves no GPU data plane; a launcher runs only over one, "
            "or on a dispatcher without a bridge\n");
    return CUDAQ_ERR_INVALID_ARG;
  }
  dispatcher->unified_launch_fn = unified_launch_fn;
  dispatcher->transport_ctx = transport_ctx;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_cpu_dataplane(cudaq_dispatcher_t *dispatcher,
                                   void *cpu_dataplane) {
  if (!dispatcher || !cpu_dataplane)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->cpu_dataplane =
      *static_cast<const cudaq_cpu_dataplane_t *>(cpu_dataplane);
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_bridge(cudaq_dispatcher_t *dispatcher,
                            cudaq_realtime_bridge_handle_t bridge) {
  if (!dispatcher || !bridge || dispatcher->running || dispatcher->bridge)
    return CUDAQ_ERR_INVALID_ARG;
  if (!is_valid_kernel_type(dispatcher->config.kernel_type))
    return CUDAQ_ERR_INVALID_ARG;

  // Everything is gathered into locals and committed only once the bridge
  // has answered every query, so a refusal leaves the dispatcher untouched.
  cudaq_dispatcher_config_t config = dispatcher->config;
  const char *shape = bridge_shape_name(config);
  const bool host = config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST;
  const bool unified = config.kernel_type == CUDAQ_KERNEL_UNIFIED;

  cudaq_status_t status = apply_bridge_geometry(bridge, config);
  if (status != CUDAQ_OK)
    return status;

  if (config.num_slots == 0) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_set_bridge (%s): the config's geometry "
            "(num_slots=%u) is 0; the bridge must report its geometry\n",
            shape, config.num_slots);
    return CUDAQ_ERR_INVALID_ARG;
  }

  cudaq_ringbuffer_t ring{};
  cudaq_cpu_dataplane_t cpu_dataplane{};
  cudaq_unified_dispatch_ctx_t unified_ctx{};
  cudaq_gpu_dataplane_t gpu_dataplane{};
  if (!unified) {
    status = cudaq_bridge_get_transport_context(bridge, RING_BUFFER, &ring);
    if (status != CUDAQ_OK) {
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge serves no "
              "ring buffer (status %d); a provider started in its unified "
              "mode refuses it\n",
              shape, status);
      return status;
    }
    const bool complete =
        host ? (ring.rx_flags_host && ring.tx_flags_host && ring.rx_data_host &&
                ring.tx_data_host)
             : (ring.rx_flags && ring.tx_flags && ring.rx_data && ring.tx_data);
    if (!complete) {
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge's ring "
              "buffer lacks the %s views this path polls\n",
              shape, host ? "host" : "device");
      return CUDAQ_ERR_INVALID_ARG;
    }
  } else if (host) {
    status = cudaq_bridge_get_cpu_dataplane(bridge, &cpu_dataplane);
    if (status != CUDAQ_OK) {
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge serves no "
              "CPU data plane (status %d); the provider must be started in "
              "its unified mode\n",
              shape, status);
      return status;
    }
    // The unified loop dereferences these without checking them.
    if (!cpu_dataplane.rx_poll || !cpu_dataplane.tx_publish ||
        !cpu_dataplane.ring.rx_data_host || !cpu_dataplane.ring.tx_data_host ||
        cpu_dataplane.ring.rx_stride_sz == 0 ||
        cpu_dataplane.ring.tx_stride_sz == 0) {
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge returned an "
              "incomplete CPU data plane\n",
              shape);
      return CUDAQ_ERR_INVALID_ARG;
    }
  } else {
    // A launch override first: a transport that replaces the dispatch loop
    // outright.  A bridge without one may refuse the UNIFIED context, and the
    // dispatcher's own kernel then runs over the bridge's GPU data plane.
    // Refusing means CUDAQ_ERR_UNSUPPORTED; any other error is a failure.
    const cudaq_status_t override_status =
        cudaq_bridge_get_transport_context(bridge, UNIFIED, &unified_ctx);
    if (override_status == CUDAQ_ERR_UNSUPPORTED) {
      unified_ctx = {};
    } else if (override_status != CUDAQ_OK) {
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge failed to "
              "report its launch override (status %d)\n",
              shape, override_status);
      return override_status;
    }
    if (unified_ctx.launch_fn) {
      if (!unified_ctx.transport_ctx) {
        fprintf(stderr,
                "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge's launch "
                "override comes without a transport context\n",
                shape);
        return CUDAQ_ERR_INVALID_ARG;
      }
    } else {
      status = cudaq_bridge_get_gpu_dataplane(bridge, &gpu_dataplane);
      if (status == CUDAQ_ERR_UNSUPPORTED) {
        // A provider whose unified loop runs on the host answers with no
        // override at all.
        if (override_status == CUDAQ_OK) {
          fprintf(stderr,
                  "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge has no "
                  "device-side data plane; its unified loop runs on the host "
                  "(use CUDAQ_DISPATCH_PATH_HOST)\n",
                  shape);
        } else {
          fprintf(stderr,
                  "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge serves "
                  "neither a launch override (status %d) nor a GPU data "
                  "plane; the provider must be started in its unified mode\n",
                  shape, override_status);
        }
        return status;
      }
      if (status != CUDAQ_OK) {
        fprintf(stderr,
                "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge failed to "
                "serve its GPU data plane (status %d)\n",
                shape, status);
        return status;
      }
      if (!gpu_dataplane.ctx || !gpu_dataplane.device_code ||
          gpu_dataplane.device_code_count == 0) {
        fprintf(stderr,
                "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge returned "
                "an incomplete GPU data plane; it needs a device context and "
                "the hooks' device code\n",
                shape);
        return CUDAQ_ERR_INVALID_ARG;
      }
    }
  }

  {
    std::lock_guard<std::mutex> lock(attached_bridges_mutex);
    if (!attached_bridges.insert(bridge).second) {
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_set_bridge (%s): the bridge already "
              "drives another dispatcher\n",
              shape);
      return CUDAQ_ERR_INVALID_ARG;
    }
  }

  dispatcher->config = config;
  dispatcher->bridge = bridge;
  if (!unified) {
    dispatcher->ringbuffer = ring;
  } else if (host) {
    dispatcher->cpu_dataplane = cpu_dataplane;
  } else if (unified_ctx.launch_fn) {
    dispatcher->unified_launch_fn = unified_ctx.launch_fn;
    dispatcher->transport_ctx = unified_ctx.transport_ctx;
  } else {
    dispatcher->unified_launch_fn = nullptr;
    dispatcher->transport_ctx = gpu_dataplane.ctx;
    dispatcher->jit_device_lib.bridge_device_code.assign(
        gpu_dataplane.device_code,
        gpu_dataplane.device_code + gpu_dataplane.device_code_count);
  }
  return CUDAQ_OK;
}

// Link the unified dispatch kernel, the bridge's device hooks and `images`,
// the device code of the binary that holds the handlers (none when no handler
// was looked up), into the dispatcher's module, and load it.  `caller` names
// the API call in diagnostics.
static cudaq_status_t link_dispatcher(cudaq_dispatcher_t *dispatcher,
                                      const char *caller,
                                      const cudaq_device_image_t *images,
                                      size_t count) {
  if (!is_device_unified_dispatcher(dispatcher)) {
    fprintf(stderr,
            "ERROR: %s: only a DEVICE unified dispatcher links device code\n",
            caller);
    return CUDAQ_ERR_INVALID_ARG;
  }
  if (dispatcher->running) {
    fprintf(stderr, "ERROR: %s: the dispatcher is already running\n", caller);
    return CUDAQ_ERR_INVALID_ARG;
  }
  if (dispatcher->unified_launch_fn && !has_gpu_dataplane(dispatcher)) {
    fprintf(stderr,
            "ERROR: %s: the bridge's launch override replaces the dispatch "
            "kernel; there is nothing to link\n",
            caller);
    return CUDAQ_ERR_INVALID_ARG;
  }
  if (dispatcher->jit_device_lib.bridge_device_code.empty()) {
    fprintf(stderr,
            "ERROR: %s: wire a bridge with cudaq_dispatcher_set_bridge first\n",
            caller);
    return CUDAQ_ERR_INVALID_ARG;
  }
  if (cudaSetDevice(dispatcher->config.device_id) != cudaSuccess)
    return CUDAQ_ERR_CUDA;

  const cudaq_device_image_t kernel_image = {
      cudaq_unified_dispatch_kernel_image_begin,
      static_cast<size_t>(cudaq_unified_dispatch_kernel_image_end -
                          cudaq_unified_dispatch_kernel_image_begin)};
  const cudaq::realtime::detail::device_code_input inputs[] = {
      {"dispatch kernel", &kernel_image, 1},
      {"bridge", dispatcher->jit_device_lib.bridge_device_code.data(),
       dispatcher->jit_device_lib.bridge_device_code.size()},
      {"application", images, count},
  };
  CUlibrary library = nullptr;
  const cudaq_status_t status = cudaq::realtime::detail::link_device_code(
      inputs, sizeof(inputs) / sizeof(inputs[0]), &library);
  if (status != CUDAQ_OK)
    return status;

  const auto *driver = cudaq::realtime::detail::driver_library();
  CUkernel kernel = nullptr;
  const CUresult err =
      driver->get_kernel(&kernel, library, "cudaq_unified_dispatch_kernel");
  if (err != CUDA_SUCCESS) {
    fprintf(stderr,
            "ERROR: %s: the linked module has no "
            "cudaq_unified_dispatch_kernel: CUresult %d\n",
            caller, static_cast<int>(err));
    driver->unload(library);
    return CUDAQ_ERR_CUDA;
  }
  dispatcher->jit_device_lib.library = library;
  dispatcher->jit_device_lib.dispatch_kernel = kernel;
  dispatcher->jit_device_lib.application_device_code.assign(images,
                                                            images + count);
  return CUDAQ_OK;
}

// Whether `images` describe the device code the module was linked with.  Each
// translation unit of a binary passes its own copy of the same images.
static bool is_linked_device_code(const cudaq_dispatcher_t *dispatcher,
                                  const cudaq_device_image_t *images,
                                  size_t count) {
  const auto &linked = dispatcher->jit_device_lib.application_device_code;
  if (linked.size() != count)
    return false;
  for (size_t i = 0; i < count; ++i)
    if (linked[i].data != images[i].data || linked[i].size != images[i].size)
      return false;
  return true;
}

// Launch the unified kernel of the dispatcher's linked module, with the
// arguments of cudaq_unified_launch_fn_t: one block of `threads_per_block`
// threads.
static CUresult launch_linked_unified_kernel(
    CUkernel kernel, void *transport_ctx, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, uint32_t threads_per_block,
    cudaStream_t stream) {
  // Exactly the kernel's parameter types: cudaLaunchKernel copies each
  // argument by the size of the parameter it lands in.
  void *args[] = {&transport_ctx,  &tx_stride_sz, &shutdown_flag,
                  &function_table, &func_count,   &stats};
  return cudaq::realtime::detail::driver_library()->launch_kernel(
      reinterpret_cast<CUfunction>(kernel), 1, 1, 1, threads_per_block, 1, 1, 0,
      stream, args, nullptr);
}

// Address and size of the global `name` in the linked module.
static CUresult find_device_global(cudaq_dispatcher_t *dispatcher,
                                   const char *name, void **out_ptr,
                                   size_t *out_size) {
  CUdeviceptr ptr = 0;
  const CUresult err = cudaq::realtime::detail::driver_library()->get_global(
      &ptr, out_size, dispatcher->jit_device_lib.library, name);
  if (err == CUDA_SUCCESS)
    *out_ptr = reinterpret_cast<void *>(ptr);
  return err;
}

// The first lookup links the module with `images`; later ones must pass the
// same device code.
static cudaq_status_t link_for_lookup(cudaq_dispatcher_t *dispatcher,
                                      const char *caller,
                                      const cudaq_device_image_t *images,
                                      size_t count) {
  if (!dispatcher->jit_device_lib.library)
    return link_dispatcher(dispatcher, caller, images, count);
  if (!is_linked_device_code(dispatcher, images, count)) {
    fprintf(stderr,
            "ERROR: %s: the dispatcher's module was linked with other device "
            "code; all handlers and kernels of one dispatcher come from one "
            "binary, looked up before start\n",
            caller);
    return CUDAQ_ERR_INVALID_ARG;
  }
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_get_device_handler_in(
    cudaq_dispatcher_t *dispatcher, const cudaq_device_image_t *images,
    size_t count, const char *name, void **out_fn) {
  if (!dispatcher || !images || count == 0 || !name || !out_fn)
    return CUDAQ_ERR_INVALID_ARG;
  const cudaq_status_t status = link_for_lookup(
      dispatcher, "cudaq_dispatcher_get_device_handler", images, count);
  if (status != CUDAQ_OK)
    return status;
  if (cudaSetDevice(dispatcher->config.device_id) != cudaSuccess)
    return CUDAQ_ERR_CUDA;
  const std::string global =
      std::string(CUDAQ_REALTIME_DEVICE_FUNCTION_PREFIX) + name;
  void *slot = nullptr;
  size_t size = 0;
  const CUresult found =
      find_device_global(dispatcher, global.c_str(), &slot, &size);
  if (found == CUDA_ERROR_NOT_FOUND) {
    fprintf(
        stderr,
        "ERROR: cudaq_dispatcher_get_device_handler: the linked module "
        "does not export '%s'; add CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(%s)\n",
        name, name);
    return CUDAQ_ERR_INVALID_ARG;
  }
  if (found != CUDA_SUCCESS) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_get_device_handler: looking up '%s' in "
            "the linked module failed: CUresult %d\n",
            name, static_cast<int>(found));
    return CUDAQ_ERR_CUDA;
  }
  if (size != sizeof(void *))
    return CUDAQ_ERR_INVALID_ARG;
  void *fn = nullptr;
  if (cudaMemcpy(&fn, slot, sizeof(fn), cudaMemcpyDeviceToHost) !=
          cudaSuccess ||
      !fn)
    return CUDAQ_ERR_CUDA;
  dispatcher->jit_device_lib.linked_functions.insert(fn);
  *out_fn = fn;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_get_device_kernel_in(
    cudaq_dispatcher_t *dispatcher, const cudaq_device_image_t *images,
    size_t count, const char *name, cudaKernel_t *out_kernel) {
  if (!dispatcher || !images || count == 0 || !name || !out_kernel)
    return CUDAQ_ERR_INVALID_ARG;
  if (std::strcmp(name, "cudaq_unified_dispatch_kernel") == 0) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_get_device_kernel: "
            "cudaq_unified_dispatch_kernel is the dispatcher's own kernel, "
            "which start() launches when no launcher is set\n");
    return CUDAQ_ERR_INVALID_ARG;
  }
  const cudaq_status_t status = link_for_lookup(
      dispatcher, "cudaq_dispatcher_get_device_kernel", images, count);
  if (status != CUDAQ_OK)
    return status;
  CUkernel kernel = nullptr;
  const CUresult found = cudaq::realtime::detail::driver_library()->get_kernel(
      &kernel, dispatcher->jit_device_lib.library, name);
  if (found == CUDA_ERROR_NOT_FOUND) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_get_device_kernel: the linked module has "
            "no kernel '%s'; define it extern \"C\" __global__ and embed its "
            "LTO-IR in the binary's __nv_relfatbin section (see "
            "realtime/docs/device_code_contract.md)\n",
            name);
    return CUDAQ_ERR_INVALID_ARG;
  }
  if (found != CUDA_SUCCESS) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_get_device_kernel: looking up '%s' in "
            "the linked module failed: CUresult %d\n",
            name, static_cast<int>(found));
    return CUDAQ_ERR_CUDA;
  }
  *out_kernel = kernel;
  return CUDAQ_OK;
}

// Raise the shutdown flag, wait for the DEVICE dispatcher's stream and destroy
// it.  If the flag cannot be written, the stream is kept, so destroy() leaves
// the linked module loaded under whatever may still run from it.
static cudaq_status_t stop_device_kernel(cudaq_dispatcher_t *dispatcher) {
  int shutdown = 1;
  if (cudaMemcpy(const_cast<int *>(dispatcher->shutdown_flag), &shutdown,
                 sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDAQ_ERR_CUDA;
  cudaStreamSynchronize(dispatcher->stream);
  cudaStreamDestroy(dispatcher->stream);
  dispatcher->stream = nullptr;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_start(cudaq_dispatcher_t *dispatcher) {
  if (dispatcher &&
      dispatcher->config.dispatch_path != CUDAQ_DISPATCH_PATH_HOST &&
      dispatcher->config.kernel_type != CUDAQ_KERNEL_UNIFIED &&
      !ring_launch_fn(dispatcher)) {
    fprintf(stderr,
            "ERROR: cudaq_dispatcher_start (DEVICE ring): no launch function; "
            "link libcudaq-realtime-dispatch.a into this binary, or name a "
            "launcher with cudaq_dispatcher_set_launch_fn\n");
    return CUDAQ_ERR_INVALID_ARG;
  }
  const bool linked_kernel =
      dispatcher && is_device_unified_dispatcher(dispatcher) &&
      (has_gpu_dataplane(dispatcher) || !dispatcher->unified_launch_fn);
  cudaq_status_t status = validate_dispatcher(dispatcher);
  if (status != CUDAQ_OK)
    return status;
  if (dispatcher->running)
    return CUDAQ_OK;
  if (linked_kernel) {
    // An address taken in the application's own module points at code the
    // linked module does not contain.
    if (cudaSetDevice(dispatcher->config.device_id) != cudaSuccess)
      return CUDAQ_ERR_CUDA;
    std::vector<cudaq_function_entry_t> entries(dispatcher->table.count);
    if (cudaMemcpy(entries.data(), dispatcher->table.entries,
                   entries.size() * sizeof(cudaq_function_entry_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "ERROR: cudaq_dispatcher_start (DEVICE unified): "
                      "cannot read the function table from the device\n");
      return CUDAQ_ERR_CUDA;
    }
    for (size_t i = 0; i < entries.size(); ++i) {
      if (entries[i].dispatch_mode != CUDAQ_DISPATCH_DEVICE_CALL ||
          dispatcher->jit_device_lib.linked_functions.count(
              entries[i].handler.device_fn_ptr))
        continue;
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_start (DEVICE unified): function "
              "table entry %zu (function_id 0x%x) holds a handler address "
              "that is not from the linked module; take handler addresses "
              "from cudaq_dispatcher_get_device_handler\n",
              i, entries[i].function_id);
      return CUDAQ_ERR_INVALID_ARG;
    }
  }
  if (linked_kernel && !dispatcher->jit_device_lib.dispatch_kernel) {
    // No handler was looked up: link the kernel and the bridge's hooks alone.
    status = link_dispatcher(dispatcher, "cudaq_dispatcher_start", nullptr, 0);
    if (status != CUDAQ_OK)
      return status;
  }
  if (dispatcher->bridge) {
    status =
        cudaq_bridge_set_function_table(dispatcher->bridge, &dispatcher->table);
    if (status != CUDAQ_OK) {
      fprintf(stderr,
              "ERROR: cudaq_dispatcher_start: the bridge refused the "
              "function table (status %d)\n",
              status);
      return status;
    }
  }

  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    // The host loops run HOST_CALL handlers only and never touch the CUDA
    // runtime, so no device is selected here; a handler that uses CUDA
    // selects its own.
    volatile int *shutdown_flag = dispatcher->shutdown_flag;
    uint64_t *stats = dispatcher->stats;
    cudaq_function_table_t tbl = dispatcher->table;
    if (is_host_unified_dispatcher(dispatcher)) {
      // stop() and destroy() join the thread before the dispatcher goes away.
      cudaq_cpu_dataplane_t *cpu_dataplane = &dispatcher->cpu_dataplane;
      dispatcher->host_thread =
          std::thread([cpu_dataplane, tbl, shutdown_flag, stats] {
            cudaq_host_unified_loop(cpu_dataplane, &tbl, shutdown_flag, stats);
          });
    } else {
      cudaq_ringbuffer_t rb = dispatcher->ringbuffer;
      cudaq_dispatcher_config_t cfg = dispatcher->config;
      dispatcher->host_thread = std::thread([rb, tbl, cfg, shutdown_flag,
                                             stats] {
        cudaq_host_ring_dispatch_loop(&rb, &tbl, &cfg, shutdown_flag, stats);
      });
    }
    dispatcher->running = true;
    return CUDAQ_OK;
  }

  // create() mapped a negative device_id to 0.
  if (cudaSetDevice(dispatcher->config.device_id) != cudaSuccess)
    return CUDAQ_ERR_CUDA;

  // Non-blocking, so that no legacy-stream copy (stop(), get_processed(), a
  // lookup on another dispatcher) queues behind the resident kernel.  Such a
  // stream no longer waits for the caller's queued legacy-stream work
  // (function table, transport context), so the launch waits for it on an
  // event.
  if (cudaStreamCreateWithFlags(&dispatcher->stream, cudaStreamNonBlocking) !=
      cudaSuccess)
    return CUDAQ_ERR_CUDA;
  // A marker in a stream; it only orders work, so it keeps no timestamp.
  cudaEvent_t caller_writes = nullptr;
  cudaError_t ordering =
      cudaEventCreateWithFlags(&caller_writes, cudaEventDisableTiming);
  // Create a marker that will wait for the caller's default stream work.
  if (ordering == cudaSuccess)
    ordering = cudaEventRecord(caller_writes, cudaStreamPerThread);
  // Drop the marker in the asynchronous stream.
  if (ordering == cudaSuccess)
    ordering = cudaStreamWaitEvent(dispatcher->stream, caller_writes, 0);
  // The wait is registered, so the event can go now.
  if (caller_writes)
    cudaEventDestroy(caller_writes);
  if (ordering != cudaSuccess) {
    cudaStreamDestroy(dispatcher->stream);
    dispatcher->stream = nullptr;
    return CUDAQ_ERR_CUDA;
  }

  // NOTE on config.shared_ring_mode for DEVICE_LOOP:
  //
  // The device dispatch kernel reads shared_ring_mode from a __constant__
  // symbol that lives in libcudaq-realtime-dispatch.a (the static lib).
  // libcudaq-realtime.so does NOT link the static lib (architecturally
  // separate: consumers link the static lib themselves), so we cannot
  // call cudaq_dispatch_kernel_set_shared_ring_mode() from here.
  //
  // Callers that want shared_ring_mode for DEVICE_LOOP must invoke
  // cudaq_dispatch_kernel_set_shared_ring_mode(1) themselves BEFORE
  // cudaq_dispatcher_start().  The HOST_LOOP path reads
  // config.shared_ring_mode directly from this struct (it has no
  // __constant__ indirection) -- nothing needed here.

  if (dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED) {
    // The unified kernel runs from the dispatcher's linked module
    // (linked by the first handler lookup or above).  unified_launch_fn is an
    // override, used only by a transport that replaces the dispatch loop
    // outright, or by the application's own loop over the linked module.
    if (dispatcher->unified_launch_fn) {
      const cudaq_status_t launched = dispatcher->unified_launch_fn(
          dispatcher->transport_ctx, dispatcher->config.slot_size,
          dispatcher->table.entries, dispatcher->table.count,
          dispatcher->shutdown_flag, dispatcher->stats,
          dispatcher->config.threads_per_block, dispatcher->stream);
      if (launched != CUDAQ_OK) {
        fprintf(stderr,
                "ERROR: cudaq_dispatcher_start: the unified launcher failed "
                "(status %d)\n",
                launched);
        // The launcher may have enqueued its kernel before failing.
        stop_device_kernel(dispatcher);
        return launched;
      }
    } else {
      const CUresult launched = launch_linked_unified_kernel(
          dispatcher->jit_device_lib.dispatch_kernel, dispatcher->transport_ctx,
          dispatcher->config.slot_size, dispatcher->table.entries,
          dispatcher->table.count, dispatcher->shutdown_flag, dispatcher->stats,
          dispatcher->config.threads_per_block, dispatcher->stream);
      if (launched != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA error in dispatcher launch: CUresult %d\n",
                static_cast<int>(launched));
        cudaStreamDestroy(dispatcher->stream);
        dispatcher->stream = nullptr;
        return CUDAQ_ERR_CUDA;
      }
    }
  } else {
    ring_launch_fn(dispatcher)(
        dispatcher->ringbuffer.rx_flags, dispatcher->ringbuffer.tx_flags,
        dispatcher->ringbuffer.rx_data, dispatcher->ringbuffer.tx_data,
        dispatcher->ringbuffer.rx_stride_sz,
        dispatcher->ringbuffer.tx_stride_sz, dispatcher->table.entries,
        dispatcher->table.count, dispatcher->shutdown_flag, dispatcher->stats,
        dispatcher->config.num_slots, dispatcher->config.num_blocks,
        dispatcher->config.threads_per_block, dispatcher->stream);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in dispatcher launch: %s (%d)\n",
            cudaGetErrorString(err), err);
    // The launch itself may have succeeded.
    stop_device_kernel(dispatcher);
    return CUDAQ_ERR_CUDA;
  }

  dispatcher->running = true;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_stop(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->running)
    return CUDAQ_OK;

  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    // Both HOST paths (ring + unified) run on host_thread.
    // `const_cast` drops the flag's `volatile` qualifier (`reinterpret_cast`
    // can't cast away cv-qualifiers) so it can be written as a plain atomic.
    if (dispatcher->shutdown_flag)
      reinterpret_cast<std::atomic<int> *>(
          const_cast<int *>(dispatcher->shutdown_flag))
          ->store(1, std::memory_order_relaxed);
    if (dispatcher->host_thread.joinable())
      dispatcher->host_thread.join();
    dispatcher->running = false;
    return CUDAQ_OK;
  }

  const cudaq_status_t stopped = stop_device_kernel(dispatcher);
  if (stopped != CUDAQ_OK)
    return stopped;
  dispatcher->running = false;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_get_processed(cudaq_dispatcher_t *dispatcher,
                                              uint64_t *out_packets) {
  if (!dispatcher || !out_packets || !dispatcher->stats)
    return CUDAQ_ERR_INVALID_ARG;

  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    *out_packets = *dispatcher->stats;
    return CUDAQ_OK;
  }

  if (cudaMemcpy(out_packets, dispatcher->stats, sizeof(uint64_t),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return CUDAQ_ERR_CUDA;

  return CUDAQ_OK;
}

//==============================================================================
// Ring buffer slot helpers
//==============================================================================

static inline uint64_t atomic_load_u64(volatile uint64_t *ptr) {
  auto *ap =
      reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(ptr));
  return ap->load(std::memory_order_acquire);
}

static inline void atomic_store_u64(volatile uint64_t *ptr, uint64_t val) {
  auto *ap =
      reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(ptr));
  ap->store(val, std::memory_order_release);
}

cudaq_status_t cudaq_host_ringbuffer_write_rpc_request(
    const cudaq_ringbuffer_t *rb, uint32_t slot_idx, uint32_t function_id,
    const void *payload, uint32_t payload_len, uint32_t request_id,
    uint64_t ptp_timestamp) {
  if (!rb || !rb->rx_data_host)
    return CUDAQ_ERR_INVALID_ARG;
  if (CUDAQ_RPC_HEADER_SIZE + payload_len > rb->rx_stride_sz)
    return CUDAQ_ERR_INVALID_ARG;

  uint8_t *slot = rb->rx_data_host + slot_idx * rb->rx_stride_sz;
  uint32_t *hdr32 = reinterpret_cast<uint32_t *>(slot);
  hdr32[0] = CUDAQ_RPC_MAGIC_REQUEST;
  hdr32[1] = function_id;
  hdr32[2] = payload_len;
  hdr32[3] = request_id;
  uint64_t *hdr64 = reinterpret_cast<uint64_t *>(slot + 16);
  *hdr64 = ptp_timestamp;

  if (payload && payload_len > 0)
    std::memcpy(slot + CUDAQ_RPC_HEADER_SIZE, payload, payload_len);

  return CUDAQ_OK;
}

void cudaq_host_ringbuffer_signal_slot(const cudaq_ringbuffer_t *rb,
                                       uint32_t slot_idx) {
  uint64_t addr = reinterpret_cast<uint64_t>(rb->rx_data_host +
                                             slot_idx * rb->rx_stride_sz);
  atomic_store_u64(&rb->rx_flags_host[slot_idx], addr);
}

cudaq_tx_status_t cudaq_host_ringbuffer_poll_tx_flag(
    const cudaq_ringbuffer_t *rb, uint32_t slot_idx, int * /*out_cuda_error*/) {
  // A non-zero flag is the TX slot's address, stored once the response is in
  // place; there are no other flag values.
  return atomic_load_u64(&rb->tx_flags_host[slot_idx]) == 0 ? CUDAQ_TX_EMPTY
                                                            : CUDAQ_TX_READY;
}

int cudaq_host_ringbuffer_slot_available(const cudaq_ringbuffer_t *rb,
                                         uint32_t slot_idx) {
  return atomic_load_u64(&rb->rx_flags_host[slot_idx]) == 0 &&
         atomic_load_u64(&rb->tx_flags_host[slot_idx]) == 0;
}

void cudaq_host_ringbuffer_clear_slot(const cudaq_ringbuffer_t *rb,
                                      uint32_t slot_idx) {
  atomic_store_u64(&rb->tx_flags_host[slot_idx], 0);
}
