/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Test-only helper for building small CUDA device-call service shims. This
// emits the service plugin discovery entry point consumed by the runtime.
//
// Example: given a natural CUDA device function:
//
//   extern "C" __device__ int foo(int a, int b) { return a + b; }
//
// The macro pair wraps it so the realtime dispatch kernel can call it through
// the flat byte-buffer realtime RPC ABI (RPCHeader + packed args -> RPCResponse
// + result):
//
//   CUDAQ_DEVICE_CALL_LIBRARY_BEGIN(my_lib)
//   CUDAQ_DEVICE_CALL_EXPORT(foo)
//   CUDAQ_DEVICE_CALL_LIBRARY_END()
//
// At build time this emits a `__global__` init kernel that fills one
// `cudaq_function_entry_t` for `foo`:
//
//   entry.function_id       = fnv1a_hash("foo")   // dispatch key
//   entry.handler.device_fn_ptr = &DeviceCallWrapper<&foo>::call
//   entry.dispatch_mode     = CUDAQ_DISPATCH_DEVICE_CALL
//
// At runtime, when the dispatch kernel receives an RPC request with
// `function_id == fnv1a_hash("foo")`, it calls `DeviceCallWrapper<&foo>::call`
// which adapts the flat byte-buffer RPC ABI to the natural C++ signature:
//   1. Decodes `int a` and `int b` sequentially from the RX byte buffer.
//   2. Calls `foo(a, b)` directly (same device module, zero overhead).
//   3. Writes the returned `int` into the TX byte buffer.

#include "cudaq_internal/device_call/DeviceCallService.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <type_traits>

namespace cudaq_internal::device_call::detail {

// Canonical payload storage for the first prototype ABI. References and
// cv-qualification are erased before packing/unpacking so the wire format uses
// value semantics for plain scalar arguments.
template <typename T>
using scalar_storage_t = std::remove_cvref_t<T>;

// Non-const pointer argument: the caller owns the storage and the device
// function writes results into it (by-ref output array).
template <typename T>
struct is_output_array_pointer
    : std::bool_constant<
          std::is_pointer_v<scalar_storage_t<T>> &&
          !std::is_const_v<std::remove_pointer_t<scalar_storage_t<T>>>> {};

template <typename T>
struct array_element;

template <typename T>
struct array_element<T *> {
  using type = T;
};

template <typename T>
struct array_element<const T *> {
  using type = T;
};

template <typename T>
using array_element_t = typename array_element<scalar_storage_t<T>>::type;

// --- Compile-time helpers ----------------------------------------------------

// Returns the natural alignment for T in the argument buffer. Must match the
// alignment the MLIR marshalling uses when packing scalars on the host side.
template <typename T>
__host__ __device__ constexpr std::uint64_t scalarAlignment() {
  constexpr auto size = sizeof(scalar_storage_t<T>);
  if constexpr (size >= sizeof(std::uint64_t))
    return sizeof(std::uint64_t);
  else if constexpr (size >= sizeof(std::uint32_t))
    return sizeof(std::uint32_t);
  else if constexpr (size >= sizeof(std::uint16_t))
    return sizeof(std::uint16_t);
  else
    return 1;
}

// --- Decode helpers ----------------------------------------------------------

// Caller must have aligned `offset` via alignOffset before calling; the
// direct dereference is safe because the buffer is already T-aligned.
template <typename T>
__device__ scalar_storage_t<T> loadScalar(const std::uint8_t *bytes,
                                          std::uint64_t offset) {
  return *reinterpret_cast<const scalar_storage_t<T> *>(bytes + offset);
}

template <typename T>
__device__ void storeScalar(void *bytes, const T &value) {
  std::memcpy(bytes, &value, sizeof(scalar_storage_t<T>));
}

template <typename T>
__device__ void zeroObject(T &object) {
  std::memset(&object, 0, sizeof(T));
}

// Advances `offset` to the next `alignment` boundary and checks it fits within
// `argLen`. If the MLIR marshalling is correct the advance is always zero (the
// buffer is already aligned); a non-zero advance or a bounds failure indicates
// a marshalling bug on the host side.
__device__ inline bool alignOffset(std::uint64_t &offset,
                                   std::uint64_t alignment,
                                   std::uint64_t argLen) {
  if (alignment <= 1)
    return offset <= argLen;
  std::uint64_t addend = alignment - 1;
  if (offset > ~std::uint64_t{0} - addend)
    return false;
  offset = (offset + addend) & ~addend;
  return offset <= argLen;
}

// Decodes a single scalar of payload type `T` from `args`, advancing `offset`.
// alignOffset should be a no-op if the MLIR marshalling packed this scalar at
// the correct alignment boundary; a non-zero advance signals a marshalling bug.
// Returns 0 on success or -1 if the buffer is malformed.
template <typename T>
__device__ inline std::int32_t
decodeScalar(const std::uint8_t *args, std::uint64_t argLen,
             std::uint64_t &offset, scalar_storage_t<T> &out) {
  using S = scalar_storage_t<T>;
  if (!alignOffset(offset, scalarAlignment<T>(), argLen) ||
      sizeof(S) > argLen - offset)
    return -1; // malformed payload — likely a marshalling mismatch
  out = loadScalar<T>(args, offset);
  offset += sizeof(S);
  return 0;
}

// --- Adapter wrappers --------------------------------------------------------

// Adapter from the realtime handler ABI to the user's natural device function.
// Important: this wrapper directly calls Fn. Therefore the
// shim translation unit must have the natural CUDA implementation available in
// the same CUDA device module, e.g. by including a .cuh implementation body.
// A device function that exists only in an already-linked CUDA shared library
// cannot be resolved by this direct-call wrapper.
template <auto Fn>
struct DeviceCallWrapper;

template <typename R, typename... Args, R (*Fn)(Args...)>
struct DeviceCallWrapper<Fn> {
  __device__ static std::int32_t call(const void *input, void *output,
                                      std::uint32_t argLen,
                                      std::uint32_t maxResultLen,
                                      std::uint32_t *resultLen) {
    // The realtime payload is exactly the canonical device-call argument
    // buffer.
    if (!resultLen || (argLen > 0 && !input))
      return -1; // malformed payload

    const auto *const args = static_cast<const std::uint8_t *>(input);
    return decodeAndInvoke<0>(args, argLen, 0, output, maxResultLen, resultLen,
                              0);
  }

private:
  template <typename... Decoded>
  __device__ static std::int32_t
  invokeDecoded(void *output, std::uint32_t maxResultLen,
                std::uint32_t *resultLen, std::uint32_t byRefResultLen,
                Decoded... decoded) {
    // The realtime ABI always reports an explicit result length. Void
    // functions either produce no bytes or fill one by-ref result array;
    // scalar functions store one packed result.
    if constexpr (std::is_void_v<R>) {
      Fn(decoded...);
      *resultLen = byRefResultLen;
      return 0;
    } else {
      if (!output || maxResultLen < sizeof(scalar_storage_t<R>))
        return -1; // result buffer too small
      const R result = Fn(decoded...);
      storeScalar(output, result);
      *resultLen = sizeof(scalar_storage_t<R>);
      return 0;
    }
  }

  template <std::size_t Index, typename... Decoded>
  __device__ static std::int32_t
  decodeAndInvoke(const std::uint8_t *args, std::uint64_t argLen,
                  std::uint64_t offset, void *output,
                  std::uint32_t maxResultLen, std::uint32_t *resultLen,
                  std::uint32_t byRefResultLen, Decoded... decoded) {
    if constexpr (Index == sizeof...(Args)) {
      // Base case: all arguments decoded; invoke the function.
      if (offset != argLen)
        return -1; // malformed payload
      return invokeDecoded(output, maxResultLen, resultLen, byRefResultLen,
                           decoded...);
    } else {
      using Arg = std::tuple_element_t<Index, std::tuple<Args...>>;
      if constexpr (std::is_pointer_v<scalar_storage_t<Arg>>) {
        // Array argument: wire format is [elementCount: u64][elements...].
        // The next argument (Index+1) is the paired element-count parameter
        // that the callee receives alongside the pointer.
        static_assert(Index + 1 < sizeof...(Args),
                      "cudaq device_call array pointer argument requires a "
                      "following element count");
        using LengthArg = std::tuple_element_t<Index + 1, std::tuple<Args...>>;
        using Element = scalar_storage_t<array_element_t<Arg>>;
        std::uint64_t elementCount = 0;
        if (auto rc =
                decodeScalar<std::uint64_t>(args, argLen, offset, elementCount))
          return rc;
        using Pointer = scalar_storage_t<Arg>;
        Pointer elements = nullptr;
        if constexpr (is_output_array_pointer<Arg>::value) {
          // Output array (non-const pointer): elements are written into the TX
          // result buffer rather than read from the RX argument buffer.
          if (byRefResultLen > maxResultLen ||
              elementCount > (maxResultLen - byRefResultLen) / sizeof(Element))
            return -1; // result buffer too small
          const auto arrayBytes =
              static_cast<std::uint32_t>(elementCount * sizeof(Element));
          if (arrayBytes && !output)
            return -1; // result buffer too small
          if (arrayBytes) {
            auto *outBytes = static_cast<std::uint8_t *>(output);
            elements = reinterpret_cast<Pointer>(outBytes + byRefResultLen);
          }
          return decodeAndInvoke<Index + 2>(
              args, argLen, offset, output, maxResultLen, resultLen,
              byRefResultLen + arrayBytes, decoded..., elements,
              static_cast<scalar_storage_t<LengthArg>>(elementCount));
        } else {
          // Input array (const pointer): elements are read directly from the
          // RX argument buffer with no copy.
          if (elementCount > (argLen - offset) / sizeof(Element))
            return -1; // malformed payload
          const Element *const constElements =
              reinterpret_cast<const Element *>(args + offset);
          offset += elementCount * sizeof(Element);
          if constexpr (std::is_const_v<std::remove_pointer_t<Pointer>>)
            elements = constElements;
          else
            elements = const_cast<Element *>(constElements);
          return decodeAndInvoke<Index + 2>(
              args, argLen, offset, output, maxResultLen, resultLen,
              byRefResultLen, decoded..., elements,
              static_cast<scalar_storage_t<LengthArg>>(elementCount));
        }
      } else {
        // Scalar argument: decode one value from the buffer and recurse.
        scalar_storage_t<Arg> value;
        if (auto rc = decodeScalar<Arg>(args, argLen, offset, value))
          return rc;
        return decodeAndInvoke<Index + 1>(args, argLen, offset, output,
                                          maxResultLen, resultLen,
                                          byRefResultLen, decoded..., value);
      }
    }
  }
};

// Initialises one function-table entry inside the CUDA init kernel.
template <auto Fn>
__device__ void fillEntry(cudaq_function_entry_t &entry,
                          std::uint32_t functionId) {
  zeroObject(entry);
  entry.handler.device_fn_ptr =
      reinterpret_cast<void *>(&DeviceCallWrapper<Fn>::call);
  entry.function_id = functionId;
  entry.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  entry.schema = {};
}

inline cudaError_t reportCudaError(cudaError_t err, const char *expr) {
  if (err == cudaSuccess)
    return cudaSuccess;
  std::fprintf(stderr, "%s failed: %s\n", expr, cudaGetErrorString(err));
  return err;
}

// --- Generated service traits -----------------------------------------------
//
// Common host-side machinery for a generated realtime device_call service.
// Parameterized by the per-TU table-init kernel and entry count, so each
// translation unit gets its own instantiation (and its own dispatch stream).
template <auto InitTableKernel, std::uint32_t Count>
struct GeneratedDeviceCallService : public DeviceCallService {
  static inline cudaStream_t dispatchStream = nullptr;

  static cudaError_t getDispatchStream(cudaStream_t *stream) {
    if (!stream)
      return cudaErrorInvalidValue;
    if (!dispatchStream) {
      const cudaError_t err =
          cudaStreamCreateWithFlags(&dispatchStream, cudaStreamNonBlocking);
      if (err != cudaSuccess)
        return err;
    }
    *stream = dispatchStream;
    return cudaSuccess;
  }

  // GPU dispatch path: delegates entirely to the production persistent dispatch
  // kernel in `realtime/lib/daemon/dispatcher/dispatch_kernel.cu`. This service
  // owns the launch stream (`dispatchStream`) rather than using the one passed
  // by the caller, so that the kernel lifetime is tied to this service
  // instance.
  static void launchDispatchKernel(
      volatile std::uint64_t *rxFlags, volatile std::uint64_t *txFlags,
      std::uint8_t *rxData, std::uint8_t *txData, std::size_t rxStrideSize,
      std::size_t txStrideSize, cudaq_function_entry_t *functionTable,
      std::size_t functionCount, volatile int *shutdownFlag,
      std::uint64_t *stats, std::size_t numSlots, std::uint32_t numBlocks,
      std::uint32_t threadsPerBlock, cudaStream_t /*stream*/) {
    cudaStream_t launchStream = nullptr;
    if (reportCudaError(getDispatchStream(&launchStream),
                        "getDispatchStream") != cudaSuccess)
      return;
    cudaq_launch_dispatch_kernel_regular(
        rxFlags, txFlags, rxData, txData, rxStrideSize, txStrideSize,
        functionTable, functionCount, shutdownFlag, stats, numSlots, numBlocks,
        threadsPerBlock, launchStream);
    reportCudaError(cudaGetLastError(), "cudaq_launch_dispatch_kernel_regular");
  }

  static cudaError_t synchronizeDispatchKernel() {
    if (!dispatchStream)
      return cudaSuccess;
    const cudaError_t syncErr = cudaStreamSynchronize(dispatchStream);
    const cudaError_t destroyErr = cudaStreamDestroy(dispatchStream);
    dispatchStream = nullptr;
    return syncErr != cudaSuccess ? syncErr : destroyErr;
  }

  int create(const void *, std::size_t) override { return 0; }
  int destroy() noexcept override { return 0; }
  std::uint32_t getFunctionCount() const override { return Count; }
  int start() override { return 0; }
  int stop() noexcept override {
    return reportCudaError(synchronizeDispatchKernel(),
                           "synchronizeDispatchKernel") != cudaSuccess;
  }
  int populateTable(cudaq_function_entry_t *entries, std::uint32_t capacity,
                    cudaStream_t stream) override {
    if (!entries || capacity < Count)
      return 1;
    InitTableKernel<<<1, 1, 0, stream>>>(entries);
    return reportCudaError(cudaGetLastError(),
                           "device_call_init_table launch") != cudaSuccess;
  }
  cudaq_dispatch_launch_fn_t getDeviceDispatchLaunch() const override {
    return &launchDispatchKernel;
  }
  DeviceCallDispatchSynchronizeFn
  getDeviceDispatchSynchronize() const override {
    return &synchronizeDispatchKernel;
  }

  static DeviceCallService *getService() {
    static GeneratedDeviceCallService service;
    return &service;
  }
};

// --- Host-dispatch graph table helpers ---------------------------------------

using DeviceCallAbiHandler = std::int32_t (*)(const void *, void *,
                                              std::uint32_t, std::uint32_t,
                                              std::uint32_t *);

// Graph node kernel for the host-dispatch path of device_call. Handles one RPC
// request per graph launch. Follows the same `GraphIOContext` mailbox protocol
// used by `graph_io_ctx_increment_kernel` in
// `realtime/unittests/test_host_dispatcher.cu`; this kernel adds function-table
// dispatch on top of that base pattern.
//
// Launch protocol (performed by `GeneratedHostDispatchService` before each
// `cudaGraphLaunch`):
//   1. Host fills `GraphIOContext { rx_slot, tx_slot, tx_flag, ... }`.
//   2. Host writes `GraphIOContext*` into the pinned mailbox slot.
//   3. Host calls `cudaGraphLaunch(graphExec[i], stream)`.
//   4. This kernel runs, dispatches, writes the response, and signals the host.
__global__ void
deviceCallHostDispatchGraphKernel(void **mailboxSlotPtr,
                                  cudaq_function_entry_t *functionTable,
                                  std::uint32_t functionCount) {
  // Single-thread dispatch: the graph node is always launched with 1 block.
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  // `mailboxSlotPtr` is a device pointer into the pinned mailbox bank. The host
  // writes a `GraphIOContext*` there before each launch (see
  // `GeneratedHostDispatchGraphTable::setup` and `host_dispatcher.cu`).
  auto *const ioContext =
      reinterpret_cast<cudaq::realtime::GraphIOContext *>(*mailboxSlotPtr);
  if (!ioContext || !ioContext->rx_slot || !ioContext->tx_slot)
    return;

  // `rx_slot` carries `RPCHeader` followed immediately by the packed argument
  // bytes. `tx_slot` receives `RPCResponse` followed by the result bytes.
  const auto *const request =
      static_cast<const cudaq::realtime::RPCHeader *>(ioContext->rx_slot);
  auto *const response =
      reinterpret_cast<cudaq::realtime::RPCResponse *>(ioContext->tx_slot);

  std::int32_t status = -1; // default: malformed payload
  std::uint32_t resultLen = 0;
  if (request->magic == cudaq::realtime::RPC_MAGIC_REQUEST &&
      ioContext->tx_stride_sz >= sizeof(cudaq::realtime::RPCResponse)) {
    // Linear scan over the full device table.
    const auto *const entry = [&]() -> const cudaq_function_entry_t * {
      for (std::uint32_t i = 0; i < functionCount; ++i)
        if (functionTable[i].function_id == request->function_id)
          return &functionTable[i];
      return nullptr;
    }();

    if (entry && entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL &&
        entry->handler.device_fn_ptr) {
      const auto handler =
          reinterpret_cast<DeviceCallAbiHandler>(entry->handler.device_fn_ptr);
      // Result bytes are written immediately after the RPCResponse header.
      auto *const result =
          ioContext->tx_slot + sizeof(cudaq::realtime::RPCResponse);
      const auto maxResultLen = static_cast<std::uint32_t>(
          ioContext->tx_stride_sz - sizeof(cudaq::realtime::RPCResponse));
      status = handler(request + 1, result, request->arg_len, maxResultLen,
                       &resultLen);
    }
  }

  response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  response->request_id = request->request_id;
  response->ptp_timestamp = request->ptp_timestamp;
  response->status = status;
  response->result_len = status == 0 ? resultLen : 0;

  // Fence before signalling so the host sees a consistent response buffer.
  __threadfence_system();
  if (ioContext->tx_flag)
    *ioContext->tx_flag = ioContext->tx_flag_value;
}

// Build and instantiate a single-node CUDA graph for one host-dispatch slot.
// The graph is destroyed immediately after instantiation; `cudaGraphExec_t`
// is self-contained and does not hold a reference to the source graph.
inline cudaError_t createHostDispatchGraph(cudaGraphExec_t *execOut,
                                           void **d_mailboxSlot,
                                           cudaq_function_entry_t *d_table,
                                           std::uint32_t functionCount) {
  if (!execOut || !d_mailboxSlot || !d_table)
    return cudaErrorInvalidValue;

  cudaGraph_t graph = nullptr;
  cudaError_t err = cudaGraphCreate(&graph, 0);
  if (err != cudaSuccess)
    return err;

  void *kernelArgs[] = {&d_mailboxSlot, &d_table, &functionCount};
  const cudaKernelNodeParams params = [&] {
    cudaKernelNodeParams result{};
    result.func = reinterpret_cast<void *>(deviceCallHostDispatchGraphKernel);
    result.gridDim = dim3(1, 1, 1);
    result.blockDim = dim3(32, 1, 1);
    result.kernelParams = kernelArgs;
    return result;
  }();

  cudaGraphNode_t node = nullptr;
  err = cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    return err;
  }

  err = cudaGraphInstantiate(execOut, graph, nullptr, nullptr, 0);
  cudaGraphDestroy(graph);
  return err;
}

template <auto InitTableKernel, std::uint32_t Count>
struct GeneratedHostDispatchGraphTable {
  static inline cudaq_function_entry_t h_entries[Count]{};
  static inline cudaq_function_entry_t *d_entries = nullptr;
  static inline void **h_mailbox = nullptr;
  static inline void **d_mailbox = nullptr;
  static inline cudaGraphExec_t graphExecs[Count]{};
  static inline bool initialized = false;

  static void teardown() noexcept {
    for (std::uint32_t i = 0; i < Count; ++i) {
      if (graphExecs[i])
        cudaGraphExecDestroy(graphExecs[i]);
      graphExecs[i] = nullptr;
    }
    if (d_entries)
      cudaFree(d_entries);
    if (h_mailbox)
      cudaFreeHost(h_mailbox);
    d_entries = nullptr;
    h_mailbox = nullptr;
    d_mailbox = nullptr;
    std::memset(h_entries, 0, sizeof(h_entries));
    initialized = false;
  }

  static int setup(std::uint32_t deviceId = 0) {
    if (initialized)
      return 0;
    if (Count == 0)
      return 1;
    if (reportCudaError(cudaSetDevice(deviceId), "cudaSetDevice") !=
        cudaSuccess)
      return 1;

    cudaStream_t stream = nullptr;
    if (reportCudaError(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags") != cudaSuccess)
      return 1;

    // Scope-exit guard: fires on any early return to destroy the stream and
    // undo partial setup. Skips cleanup once `initialized` is set to true.
    struct SetupGuard {
      cudaStream_t &stream;
      ~SetupGuard() {
        if (initialized)
          return;
        if (stream)
          cudaStreamDestroy(stream);
        teardown();
      }
    } guard{stream};

    if (reportCudaError(
            cudaMalloc(&d_entries, Count * sizeof(cudaq_function_entry_t)),
            "cudaMalloc device_call host-dispatch table") != cudaSuccess)
      return 1;

    InitTableKernel<<<1, 1, 0, stream>>>(d_entries);
    if (reportCudaError(cudaGetLastError(), "device_call_init_table launch") !=
            cudaSuccess ||
        reportCudaError(cudaStreamSynchronize(stream),
                        "device_call_init_table synchronize") != cudaSuccess)
      return 1;

    if (reportCudaError(cudaMemcpyAsync(h_entries, d_entries,
                                        Count * sizeof(cudaq_function_entry_t),
                                        cudaMemcpyDeviceToHost, stream),
                        "cudaMemcpyAsync device_call host table") !=
            cudaSuccess ||
        reportCudaError(cudaStreamSynchronize(stream),
                        "cudaMemcpy device_call host table synchronize") !=
            cudaSuccess)
      return 1;

    if (reportCudaError(cudaHostAlloc(&h_mailbox, Count * sizeof(void *),
                                      cudaHostAllocMapped),
                        "cudaHostAlloc device_call host mailbox") !=
        cudaSuccess)
      return 1;
    std::memset(h_mailbox, 0, Count * sizeof(void *));
    if (reportCudaError(
            cudaHostGetDevicePointer(reinterpret_cast<void **>(&d_mailbox),
                                     h_mailbox, 0),
            "cudaHostGetDevicePointer device_call host mailbox") != cudaSuccess)
      return 1;

    for (std::uint32_t i = 0; i < Count; ++i) {
      if (reportCudaError(createHostDispatchGraph(&graphExecs[i],
                                                  d_mailbox + i,
                                                  d_entries, Count),
                          "createHostDispatchGraph") != cudaSuccess)
        return 1;
      h_entries[i].handler.graph_exec = graphExecs[i];
      h_entries[i].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    }

    if (reportCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy") !=
        cudaSuccess) {
      stream = nullptr; // prevent double-destroy in guard destructor
      return 1;
    }
    stream = nullptr;
    initialized = true;
    return 0;
  }
};

template <auto InitTableKernel, std::uint32_t Count>
struct GeneratedHostDispatchService : public DeviceCallService {
  using Table = GeneratedHostDispatchGraphTable<InitTableKernel, Count>;
  int create(const void *, std::size_t) override { return 0; }

  int destroy() noexcept override {
    Table::teardown();
    return 0;
  }

  std::uint32_t getFunctionCount() const override { return Count; }

  int getHostDispatchTable(DeviceCallHostDispatchTable &table) override {
    if (Table::setup() != 0)
      return 1;
    table.entries = Table::h_entries;
    table.count = Count;
    table.deviceId = 0;
    table.mailbox = Table::h_mailbox;
    return 0;
  }

  int stop() noexcept override {
    Table::teardown();
    return 0;
  }

  static DeviceCallService *getService() {
    static GeneratedHostDispatchService service;
    return &service;
  }
};

} // namespace cudaq_internal::device_call::detail

// --- Registration macros -----------------------------------------------------

#define CUDAQ_DEVICE_CALL_CONCAT(a, b) CUDAQ_DEVICE_CALL_CONCAT_INNER(a, b)
#define CUDAQ_DEVICE_CALL_CONCAT_INNER(a, b) a##b

#define CUDAQ_DEVICE_CALL_SERVICE_PLUGIN_INFO_NAME(name)                       \
  CUDAQ_DEVICE_CALL_CONCAT(cudaqGetDeviceCallServicePluginInfo_, name)

// Begin one device-call registration table. The macro opens a private
// namespace and a CUDA kernel body; each CUDAQ_DEVICE_CALL_EXPORT invocation
// appends one entry to that kernel. Use exactly one BEGIN/END pair per shim
// translation unit in this prototype.
#define CUDAQ_DEVICE_CALL_LIBRARY_BEGIN(name)                                  \
  namespace {                                                                  \
  ::cudaq_internal::device_call::DeviceCallService *                           \
  __cudaq_device_call_get_service();                                           \
  }                                                                            \
  extern "C" ::cudaq_internal::device_call::DeviceCallServicePluginInfo        \
  CUDAQ_DEVICE_CALL_SERVICE_PLUGIN_INFO_NAME(name)() {                         \
    return {#name, &__cudaq_device_call_get_service};                          \
  }                                                                            \
  extern "C" ::cudaq_internal::device_call::DeviceCallServicePluginInfo        \
  cudaqGetDeviceCallServicePluginInfo() {                                      \
    return CUDAQ_DEVICE_CALL_SERVICE_PLUGIN_INFO_NAME(name)();                 \
  }                                                                            \
  namespace {                                                                  \
  enum { __cudaq_device_call_counter_begin = __COUNTER__ };                    \
  __global__ void                                                              \
  __cudaq_device_call_init_table(cudaq_function_entry_t *entries) {            \
    if (threadIdx.x != 0 || blockIdx.x != 0)                                   \
      return;

// Register one natural device function. The function name string is the
// logical dispatch key. fnv1a_hash is constexpr host-only;
// --expt-relaxed-constexpr allows it to be constant-folded here without a
// device-runtime call.
#define CUDAQ_DEVICE_CALL_EXPORT(function)                                     \
  ::cudaq_internal::device_call::detail::fillEntry<&function>(                 \
      entries[__COUNTER__ - __cudaq_device_call_counter_begin - 1],            \
      cudaq::realtime::fnv1a_hash(#function));

// Finish the table and export the service plugin info entry point. The CUDA-Q
// runtime owns function table allocation and dispatch session lifecycle.
#define CUDAQ_DEVICE_CALL_LIBRARY_END()                                        \
  } /* close __cudaq_device_call_init_table */                                 \
  constexpr std::uint32_t __cudaq_device_call_count =                          \
      static_cast<std::uint32_t>(__COUNTER__ -                                 \
                                 __cudaq_device_call_counter_begin - 1);       \
  using __cudaq_device_call_service_traits =                                   \
      ::cudaq_internal::device_call::detail::GeneratedDeviceCallService<       \
          &__cudaq_device_call_init_table, __cudaq_device_call_count>;         \
  ::cudaq_internal::device_call::DeviceCallService *                           \
  __cudaq_device_call_get_service() {                                          \
    return __cudaq_device_call_service_traits::getService();                   \
  }                                                                            \
  } /* close anonymous namespace */

#define CUDAQ_DEVICE_CALL_HOST_DISPATCH_LIBRARY_END()                          \
  } /* close __cudaq_device_call_init_table */                                 \
  constexpr std::uint32_t __cudaq_device_call_count =                          \
      static_cast<std::uint32_t>(__COUNTER__ -                                 \
                                 __cudaq_device_call_counter_begin - 1);       \
  using __cudaq_device_call_service_traits =                                   \
      ::cudaq_internal::device_call::detail::GeneratedHostDispatchService<     \
          &__cudaq_device_call_init_table, __cudaq_device_call_count>;         \
  ::cudaq_internal::device_call::DeviceCallService *                           \
  __cudaq_device_call_get_service() {                                          \
    return __cudaq_device_call_service_traits::getService();                   \
  }                                                                            \
  } /* close anonymous namespace */
