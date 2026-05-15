/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/DeviceCallService.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/runtime/logger/logger.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace cudaq_internal::device_call;

constexpr const char GpuDispatchChannelName[] = "device_dispatch";
constexpr const char HostDispatchChannelName[] = "host_dispatch";

// Runtime mode selected by configuration
enum class DeviceCallRuntimeMode : std::int32_t {
  Off = 0,
  ServiceGpuDispatch = 1,
  ServiceHostDispatch = 2,
  ExternalChannel = 3,
};

// Catch C++ exceptions and convert them to C ABI status codes.
template <typename Fn>
std::int32_t runDeviceCallAbi(Fn &&fn) noexcept {
  auto invoke = [&]() -> DeviceCallStatus {
    try {
      fn();
      return DeviceCallStatus::Success;
    } catch (const DeviceCallError &e) {
      return e.status();
    } catch (...) {
      return DeviceCallStatus::RemoteError;
    }
  };
  const auto status = toAbiStatus(invoke());
  if (status != toAbiStatus(DeviceCallStatus::Success))
    CUDAQ_INFO("[device-call] ABI call failed status={}", status);
  return status;
}

// Parsed device_call runtime settings shared by service-backed and external
// channel setup.
struct DeviceCallRuntimeConfig {
  bool enabled = true;
  std::string channelName = GpuDispatchChannelName;
  std::vector<std::string> arguments;
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint64_t slotSize = DefaultSlotSize;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
};

// Extract the channel-specific runtime configuration.
DeviceCallChannelConfig
makeChannelConfig(const DeviceCallRuntimeConfig &config) {
  return {config.numSlots, config.slotSize, config.timeoutMs};
}

inline bool isGpuDispatchChannel(std::string_view name) {
  return name == GpuDispatchChannelName;
}

inline bool isHostDispatchChannel(std::string_view name) {
  return name == HostDispatchChannelName;
}

inline bool isBuiltinDispatchChannel(std::string_view name) {
  return isGpuDispatchChannel(name) || isHostDispatchChannel(name);
}

bool setChannelName(DeviceCallRuntimeConfig &config, const char *value) {
  if (!value || !*value)
    return false;
  if (std::strcmp(value, "off") == 0 || std::strcmp(value, "none") == 0 ||
      std::strcmp(value, "disabled") == 0) {
    config.enabled = false;
    config.channelName.clear();
    return true;
  }

  config.enabled = true;
  if (std::strcmp(value, "shared-memory") == 0 ||
      std::strcmp(value, "shared_memory") == 0)
    config.channelName = GpuDispatchChannelName;
  else if (std::strcmp(value, "host-dispatch") == 0 ||
           std::strcmp(value, "host_dispatch") == 0 ||
           std::strcmp(value, "shared-memory-host") == 0 ||
           std::strcmp(value, "shared_memory_host") == 0)
    config.channelName = HostDispatchChannelName;
  else
    config.channelName = value;
  return true;
}

bool parseUInt(const char *value, std::uint64_t maxValue, std::uint64_t &out) {
  if (!value || !*value)
    return false;
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > maxValue)
    return false;
  out = static_cast<std::uint64_t>(parsed);
  return true;
}

template <typename T>
bool parseEnvUInt(const char *name, T &out, std::uint64_t minValue = 0) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    return true;
  std::uint64_t parsed = 0;
  if (!parseUInt(value,
                 static_cast<std::uint64_t>(std::numeric_limits<T>::max()),
                 parsed) ||
      parsed < minValue)
    return false;
  out = static_cast<T>(parsed);
  return true;
}

// Apply CUDAQ_DEVICE_CALL_* environment overrides before command-line parsing.
bool applyDeviceCallEnvironment(DeviceCallRuntimeConfig &config) {
  if (const char *channel = std::getenv("CUDAQ_DEVICE_CALL_CHANNEL")) {
    if (!setChannelName(config, channel))
      return false;
  }

  return parseEnvUInt("CUDAQ_DEVICE_CALL_SLOTS", config.numSlots, 1) &&
         parseEnvUInt("CUDAQ_DEVICE_CALL_SLOT_SIZE", config.slotSize, 1) &&
         parseEnvUInt("CUDAQ_DEVICE_CALL_TIMEOUT_MS", config.timeoutMs, 1);
}

// Parse runtime options and retain argv for channel implementations that do
// their own option handling.
//
// Recognized command-line options:
//   --cudaq-device-call=<channel>
//   --cudaq-device-call <channel>
//   --cudaq-device-call-channel=<channel>
//   --cudaq-device-call-channel <channel>
//   --cudaq-device-call-slots=<count>
//   --cudaq-device-call-slots <count>
//   --cudaq-device-call-slot-size=<bytes>
//   --cudaq-device-call-slot-size <bytes>
//   --cudaq-device-call-timeout-ms=<milliseconds>
//   --cudaq-device-call-timeout-ms <milliseconds>
//
// Environment overrides are applied before argv parsing:
//   CUDAQ_DEVICE_CALL_CHANNEL
//   CUDAQ_DEVICE_CALL_SLOTS
//   CUDAQ_DEVICE_CALL_SLOT_SIZE
//   CUDAQ_DEVICE_CALL_TIMEOUT_MS
//
// Unknown options are ignored so application and channel-specific arguments can
// co-exist in the same argv.
DeviceCallRuntimeConfig parseDeviceCallArgs(int argc, char **argv) {
  DeviceCallRuntimeConfig config;
  if (argc > 0 && argv) {
    config.arguments.reserve(argc);
    for (int i = 0; i < argc; ++i)
      if (argv[i])
        config.arguments.emplace_back(argv[i]);
  }
  if (!applyDeviceCallEnvironment(config))
    throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                          "invalid CUDA-Q device_call environment");

  if (argc < 0)
    throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                          "invalid CUDA-Q device_call command line");

  auto consumeValue = [&](int &index, const char *current,
                          const char *option) -> const char * {
    if (!current)
      return nullptr;
    if (std::strcmp(current, option) == 0 && index + 1 < argc)
      return argv[++index];
    const std::size_t optionLen = std::strlen(option);
    if (std::strncmp(current, option, optionLen) == 0 &&
        current[optionLen] == '=')
      return current + optionLen + 1;
    return nullptr;
  };

  auto parseUIntOption = [](const char *value, auto &out,
                            std::uint64_t minValue) {
    using T = std::remove_reference_t<decltype(out)>;
    std::uint64_t parsed = 0;
    if (!parseUInt(value,
                   static_cast<std::uint64_t>(std::numeric_limits<T>::max()),
                   parsed) ||
        parsed < minValue)
      return false;
    out = static_cast<T>(parsed);
    return true;
  };

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv ? argv[i] : nullptr;
    if (!arg)
      continue;

    if (const char *value = consumeValue(i, arg, "--cudaq-device-call")) {
      if (!setChannelName(config, value))
        throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                              "invalid CUDA-Q device_call command line");
      continue;
    }
    if (const char *value =
            consumeValue(i, arg, "--cudaq-device-call-channel")) {
      if (!setChannelName(config, value))
        throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                              "invalid CUDA-Q device_call command line");
      continue;
    }
    if (const char *value = consumeValue(i, arg, "--cudaq-device-call-slots")) {
      if (!parseUIntOption(value, config.numSlots, 1))
        throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                              "invalid CUDA-Q device_call command line");
      continue;
    }
    if (const char *value =
            consumeValue(i, arg, "--cudaq-device-call-slot-size")) {
      if (!parseUIntOption(value, config.slotSize, 1))
        throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                              "invalid CUDA-Q device_call command line");
      continue;
    }
    if (const char *value =
            consumeValue(i, arg, "--cudaq-device-call-timeout-ms")) {
      if (!parseUIntOption(value, config.timeoutMs, 1))
        throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                              "invalid CUDA-Q device_call command line");
      continue;
    }
  }

  if (config.enabled && config.channelName.empty())
    throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                          "CUDA-Q device_call channel name is empty");
  CUDAQ_INFO("[device-call] runtime args parsed enabled={} channel='{}' "
             "slots={} slotSize={} timeoutMs={} arguments={}",
             config.enabled, config.channelName, config.numSlots,
             config.slotSize, config.timeoutMs, config.arguments.size());
  return config;
}

// Determine the CUDA device associated with a function table pointer. Host
// pointers are associated with the current CUDA device.
int getPointerDevice(cudaq_function_entry_t *entries) {
  int device = 0;
  cudaGetDevice(&device);

  cudaPointerAttributes attributes{};
  const auto err = cudaPointerGetAttributes(&attributes, entries);
  if (err == cudaSuccess) {
    if (attributes.type == cudaMemoryTypeDevice ||
        attributes.type == cudaMemoryTypeManaged) {
      CUDAQ_DBG("[device-call] function table pointer is on device {}",
                attributes.device);
      return attributes.device;
    }
  } else {
    // If the pointer is not recognized, assume it's a host pointer and use the
    // current device. Clear the error state since this is expected for host
    // pointers.
    cudaGetLastError();
  }

  CUDAQ_DBG("[device-call] using current CUDA device {} for function table",
            device);
  return device;
}

// APIs that predate multi-device registration bind to the default device slot.
constexpr std::uint32_t DefaultDeviceId = 0;

// Register one process-exit cleanup hook after a session/channel is created.
void registerShutdownHandler();

// Resolve a plugin descriptor into the C++ service object that creates
// per-device dispatch sessions.
DeviceCallService *
loadDeviceCallService(DeviceCallServicePluginInfoFn getPluginInfo) {
  if (!getPluginInfo)
    throw DeviceCallError(DeviceCallStatus::NotInitialized,
                          "device_call service plugin is not initialized");

  const DeviceCallServicePluginInfo pluginInfo = getPluginInfo();
  CUDAQ_INFO("[device-call] service plugin '{}' loaded",
             pluginInfo.pluginName ? pluginInfo.pluginName : "<unnamed>");
  if (!pluginInfo.getService)
    throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                          "device_call service plugin is incomplete");

  DeviceCallService *const service = pluginInfo.getService();
  if (!service)
    throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                          "device_call service provider returned null");
  return service;
}

// Per-device realtime dispatch session: a dispatch channel plus any
// service session (function table, etc.) backing it.
class DeviceCallSession {
public:
  DeviceCallSession() = default;
  DeviceCallSession(const DeviceCallSession &) = delete;
  DeviceCallSession &operator=(const DeviceCallSession &) = delete;
  ~DeviceCallSession() { stop(); }

  // Start plugin-owned service sessions after the transport channel exists.
  void startServices() {
    CUDAQ_DBG("[device-call] session start services count={}", services.size());
    for (auto &service : services) {
      try {
        service->start();
      } catch (...) {
        stop();
        throw;
      }
    }
  }

  // Tear down transport and service state when the last shared owner exits.
  void stop() noexcept {
    if (!channel && services.empty())
      return;
    CUDAQ_DBG("[device-call] session stop channel={} services={}",
              static_cast<bool>(channel), services.size());
    if (channel) {
      channel->stop();
      channel.reset();
    }
    for (auto &service : services)
      service->stop();
    services.clear();
  }

  // Introspection used by the driver to reject conflicting setup paths.
  bool hasChannel() const noexcept { return static_cast<bool>(channel); }
  bool hasService() const noexcept { return !services.empty(); }

  // Transport for realtime request/response frames on this device.
  std::unique_ptr<DeviceCallChannel> channel;
  // Plugin-owned service sessions that keep dispatch state active.
  std::vector<std::unique_ptr<DeviceCallServiceSession>> services;
};

// Coordinator for device_call configuration, per-device sessions,
// plugin discovery, and realtime frame routing.
class DeviceCallDriver {
public:
  // Singleton accessor for the driver instance
  static DeviceCallDriver &instance() {
    static DeviceCallDriver driver;
    return driver;
  }

  DeviceCallDriver(const DeviceCallDriver &) = delete;
  DeviceCallDriver &operator=(const DeviceCallDriver &) = delete;
  DeviceCallDriver(DeviceCallDriver &&) = delete;
  DeviceCallDriver &operator=(DeviceCallDriver &&) = delete;

  // Parse runtime configuration and eagerly create an external channel when
  // requested. Built-in channels are created when a service/table is provided.
  void configure(int argc, char **argv) {
    CUDAQ_DBG("[device-call] driver configure argc={}", argc);
    const DeviceCallRuntimeConfig parsed = parseDeviceCallArgs(argc, argv);
    std::lock_guard<std::mutex> lock(mutex);
    sessions.clear();
    config = parsed;
    CUDAQ_DBG("[device-call] driver configured enabled={} channel='{}' "
              "slots={} slotSize={} timeoutMs={}",
              config.enabled, config.channelName, config.numSlots,
              config.slotSize, config.timeoutMs);

    if (config.enabled && !isBuiltinDispatchChannel(config.channelName)) {
      CUDAQ_INFO("[device-call] driver creating external channel '{}'",
                 config.channelName);
      auto args = [&] {
        DeviceCallChannelCreateArgs result;
        result.channelName = config.channelName;
        result.arguments = config.arguments;
        result.channelConfig = makeChannelConfig(config);
        return result;
      }();
      auto session = std::make_shared<DeviceCallSession>();
      session->channel =
          createDeviceCallChannel(config.channelName, std::move(args));
      sessions[DefaultDeviceId] = std::move(session);
      registerShutdownHandler();
    }
  }

  // Classify the current configuration into the runtime action set to run.
  DeviceCallRuntimeMode configuredMode() const {
    std::lock_guard<std::mutex> lock(mutex);
    const auto mode = [&] {
      if (!config.enabled)
        return DeviceCallRuntimeMode::Off;
      if (isGpuDispatchChannel(config.channelName))
        return DeviceCallRuntimeMode::ServiceGpuDispatch;
      if (isHostDispatchChannel(config.channelName))
        return DeviceCallRuntimeMode::ServiceHostDispatch;
      return DeviceCallRuntimeMode::ExternalChannel;
    }();
    CUDAQ_DBG("[device-call] driver configured mode {}",
              static_cast<int>(mode));
    return mode;
  }

  // Default-device convenience wrapper for service-backed session creation.
  void initializeService() { initializeServiceForDevice(DefaultDeviceId); }

  // Resolve the service, create a dispatch channel, and publish the per-device
  // session used by realtime frame acquisition.
  void initializeServiceForDevice(std::uint32_t deviceId) {
    CUDAQ_INFO("[device-call] driver initialize service device={}", deviceId);
    std::lock_guard<std::mutex> lock(mutex);

    // Bail out if a service session is already available for this device.
    const auto existingSession = [&]() -> std::shared_ptr<DeviceCallSession> {
      auto iter = sessions.find(deviceId);
      if (iter == sessions.end())
        return nullptr;
      return iter->second;
    }();
    if (existingSession && existingSession->hasService()) {
      CUDAQ_DBG("[device-call] service already initialized for device {}",
                deviceId);
      return;
    }

    // The configured channel name selects host- vs GPU-dispatch behavior.
    const bool useHostDispatch =
        config.enabled && isHostDispatchChannel(config.channelName);

    // External (non-builtin) channels are created eagerly in configure() and
    // do not have a service plugin behind them. Confirm the channel is there
    // and bail out; otherwise the configuration is incomplete.
    if (config.enabled && !isBuiltinDispatchChannel(config.channelName)) {
      if (existingSession && existingSession->hasChannel()) {
        CUDAQ_DBG("[device-call] external channel already initialized for "
                  "device {}",
                  deviceId);
        return;
      } else {
        throw DeviceCallError(
            DeviceCallStatus::NotInitialized,
            "external device_call channel is not initialized");
      }
    }

    // Locate and load the per-device service plugin (linked-in shim library
    // that exports the dispatch table of device functions).
    const auto pluginInfoFn = resolveServicePluginNoLock(deviceId);
    CUDAQ_DBG("[device-call] service plugin resolution for device {} "
              "found={}",
              deviceId, static_cast<bool>(pluginInfoFn));
    if (!pluginInfoFn)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call service plugin was not found");

    DeviceCallService *const service = loadDeviceCallService(pluginInfoFn);

    // Ask the service for a dispatch session in the requested mode. The
    // session owns a dispatch table whose layout depends on host vs GPU.
    const auto dispatchMode = useHostDispatch ? DeviceCallDispatchMode::Host
                                              : DeviceCallDispatchMode::Gpu;
    CUDAQ_DBG("[device-call] initializing service for {} dispatch",
              useHostDispatch ? "host" : "gpu");
    auto serviceSession = service->createDispatchSession(dispatchMode);
    if (!serviceSession)
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "device_call service returned null dispatch session");

    // Validate the returned table matches our mode and exports the hooks the
    // channel will need at dispatch time.
    const DeviceCallDispatchTable &table = serviceSession->dispatchTable();
    if (table.mode != dispatchMode)
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "device_call service returned unexpected dispatch mode");
    if (!table.entries || table.count == 0)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "device_call service exported no functions");
    if (!useHostDispatch && !table.launchFn)
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "device_call service is missing dispatch launch hook");

    // Translate the dispatch table into channel-create args. Host vs GPU
    // dispatch use different combinations of mailbox/launchFn — see the
    // ternaries below.
    auto session = std::make_shared<DeviceCallSession>();
    auto args = [&] {
      DeviceCallChannelCreateArgs result;
      result.functionTable = table.entries;
      result.functionCount = table.count;
      result.deviceId =
          useHostDispatch ? table.deviceId : getPointerDevice(table.entries);
      result.launchFn = useHostDispatch ? nullptr : table.launchFn;
      result.synchronizeFn = table.synchronizeFn;
      result.mailbox = useHostDispatch ? table.mailbox : nullptr;
      result.channelName =
          useHostDispatch ? HostDispatchChannelName : GpuDispatchChannelName;
      result.arguments = config.arguments;
      result.channelConfig = makeChannelConfig(config);
      return result;
    }();
    CUDAQ_DBG("[device-call] service channel args channel='{}' device={} "
              "functionCount={} hasMailbox={} launchFn={} synchronizeFn={}",
              args.channelName, args.deviceId, args.functionCount,
              args.mailbox != nullptr, static_cast<bool>(args.launchFn),
              static_cast<bool>(args.synchronizeFn));

    // Build the channel, attach the service session, and start the dispatch
    // loop (e.g., persistent kernel on GPU; host-thread on host dispatch).
    const std::string channelName = args.channelName;
    auto channel = createDeviceCallChannel(channelName, std::move(args));
    session->channel = std::move(channel);
    session->services.push_back(std::move(serviceSession));
    session->startServices();

    // Publish under deviceId so subsequent acquireFrame calls find it, and
    // ensure shutdown tears the session down before process exit.
    sessions.insert_or_assign(deviceId, std::move(session));
    CUDAQ_INFO("[device-call] service initialized for device {}", deviceId);
    registerShutdownHandler();
  }

  // Drop all published sessions. Frame handles keep erased sessions alive.
  void shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (sessions.empty())
      return;
    CUDAQ_INFO("[device-call] driver shutdown sessions={}", sessions.size());
    sessions.clear();
  }

  // Opaque ABI lease for one request/response frame. The shared_ptr keeps the
  // backing session alive across dispatch and release.
  struct DeviceCallFrameHandle {
    std::shared_ptr<DeviceCallSession> session;
    DeviceCallChannel::DeviceCallFrame frame;
  };

  // Lease request/response storage from the session channel for lowered code.
  void acquireFrameForDevice(std::uint32_t deviceId, std::uint32_t functionId,
                             std::uint64_t requestBytes,
                             std::uint64_t responseCapacity, void **frameHandle,
                             void **requestPayload, void **responsePayload) {
    CUDAQ_DBG("[device-call] driver acquire frame device={} functionId={} "
              "requestBytes={} responseCapacity={}",
              deviceId, functionId, requestBytes, responseCapacity);
    if (!frameHandle || !requestPayload || !responsePayload)
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "device_call frame output pointers must be non-null");
    *frameHandle = nullptr;
    *requestPayload = nullptr;
    *responsePayload = nullptr;

    const auto session = [&]() -> std::shared_ptr<DeviceCallSession> {
      std::lock_guard<std::mutex> lock(mutex);
      auto iter = sessions.find(deviceId);
      if (iter == sessions.end())
        return nullptr;
      return iter->second;
    }();
    if (!session)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call session is not initialized");

    auto handle = std::make_unique<DeviceCallFrameHandle>();
    handle->session = session;
    if (!handle->session || !handle->session->channel)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call channel is not initialized");

    handle->session->channel->acquireFrame(functionId, requestBytes,
                                           responseCapacity, handle->frame);

    *requestPayload = handle->frame.request.data;
    *responsePayload =
        responseCapacity == 0 ? nullptr : handle->frame.response.data;
    *frameHandle = handle.release();
    CUDAQ_DBG("[device-call] driver acquired frame hasResponse={}",
              *responsePayload != nullptr);
  }

  // Submit the leased frame and return the number of response bytes produced.
  static std::uint64_t dispatchFrame(void *opaqueFrame) {
    CUDAQ_DBG("[device-call] driver dispatch frame hasFrame={}",
              opaqueFrame != nullptr);
    if (!opaqueFrame)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "invalid device_call frame handle");
    auto *const handle = static_cast<DeviceCallFrameHandle *>(opaqueFrame);
    if (!handle->session || !handle->session->channel)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call channel is not initialized");
    const auto responseBytes =
        handle->session->channel->dispatchFrame(handle->frame);
    CUDAQ_DBG("[device-call] driver dispatch frame complete responseBytes={}",
              responseBytes);
    return responseBytes;
  }

  // Return the leased frame resources; null handles are accepted for cleanup.
  static void releaseFrame(void *opaqueFrame) noexcept {
    if (!opaqueFrame)
      return;
    CUDAQ_DBG("[device-call] driver release frame");
    auto *const handle = static_cast<DeviceCallFrameHandle *>(opaqueFrame);
    if (handle->session && handle->session->channel)
      handle->session->channel->releaseFrame(handle->frame);
    delete handle;
    CUDAQ_DBG("[device-call] driver released frame");
  }

private:
  DeviceCallDriver() = default;

  // Resolve the dlsym-discovered service plugin entry point. The caller must
  // hold the driver mutex.
  DeviceCallServicePluginInfoFn
  resolveServicePluginNoLock(std::uint32_t deviceId) const {
    CUDAQ_DBG("[device-call] resolve service plugin device={}", deviceId);

    constexpr std::string_view pluginInfoSymbol =
        "cudaqGetDeviceCallServicePluginInfo";

    const auto resolve =
        [](std::string_view symbolName) -> DeviceCallServicePluginInfoFn {
      CUDAQ_DBG("[device-call] resolving service plugin symbol {}", symbolName);
      (void)::dlerror();
      void *const symbol =
          ::dlsym(RTLD_DEFAULT, std::string(symbolName).c_str());
      if (!symbol) {
        CUDAQ_DBG("[device-call] service plugin symbol {} not found",
                  symbolName);
        return nullptr;
      }
      CUDAQ_DBG("[device-call] service plugin symbol {} resolved", symbolName);
      return reinterpret_cast<DeviceCallServicePluginInfoFn>(symbol);
    };

    return resolve(pluginInfoSymbol);
  }

  // Protects configuration and published sessions.
  mutable std::mutex mutex;
  // Parsed process-wide runtime configuration.
  DeviceCallRuntimeConfig config;
  // Maps deviceId -> DeviceCallSession.
  std::unordered_map<std::uint32_t, std::shared_ptr<DeviceCallSession>>
      sessions;
};

// atexit-compatible wrapper around the process-wide driver shutdown.
void shutdownDriverNoThrow() { DeviceCallDriver::instance().shutdown(); }

// Register process-exit cleanup once the runtime has owned resources.
void registerShutdownHandler() {
  static std::once_flag once;
  // The runtime library is process-resident, so the atexit handler cannot
  // outlive unloaded library text.
  std::call_once(once, [] { std::atexit(shutdownDriverNoThrow); });
}

// Initialization action for modes that need no eager setup.
void initializeNoOp() {}

// Initialization action for built-in service-backed dispatch channels.
void initializeSharedMemoryRuntime() {
  CUDAQ_INFO("[device-call] runtime initialize shared-memory service");
  DeviceCallDriver::instance().initializeService();
}

// Teardown action for modes with no owned runtime resources.
void teardownNoOp() noexcept {}

// Teardown action for modes that publish driver sessions.
void teardownConfiguredRuntime() noexcept {
  CUDAQ_INFO("[device-call] runtime teardown configured runtime");
  DeviceCallDriver::instance().shutdown();
}

// Small vtable for mode-specific initialize/finalize behavior.
struct RuntimeModeActions {
  void (*initialize)();
  void (*teardown)() noexcept;
};

// Map a configured runtime mode to its initialize/finalize callbacks.
RuntimeModeActions getRuntimeModeActions(DeviceCallRuntimeMode mode) {
  switch (mode) {
  case DeviceCallRuntimeMode::Off:
    return {initializeNoOp, teardownNoOp};
  case DeviceCallRuntimeMode::ServiceGpuDispatch:
    return {initializeSharedMemoryRuntime, teardownConfiguredRuntime};
  case DeviceCallRuntimeMode::ServiceHostDispatch:
    return {initializeSharedMemoryRuntime, teardownConfiguredRuntime};
  case DeviceCallRuntimeMode::ExternalChannel:
    return {initializeNoOp, teardownConfiguredRuntime};
  }
  return {initializeNoOp, teardownNoOp};
}

} // namespace

namespace cudaq_internal::device_call {

//  configure and initialize the selected runtime mode.
void initializeDeviceCallRuntime(int argc, char **argv) {
  CUDAQ_DBG("[device-call] initialize runtime API argc={}", argc);
  DeviceCallDriver::instance().configure(argc, argv);
  const auto mode = DeviceCallDriver::instance().configuredMode();
  CUDAQ_INFO("[device-call] initialize runtime API mode={}",
             static_cast<int>(mode));
  getRuntimeModeActions(mode).initialize();
}

//  finalize resources for the selected runtime mode.
void finalizeDeviceCallRuntime() {
  CUDAQ_DBG("[device-call] finalize runtime API");
  const auto mode = DeviceCallDriver::instance().configuredMode();
  CUDAQ_INFO("[device-call] finalize runtime API mode={}",
             static_cast<int>(mode));
  getRuntimeModeActions(mode).teardown();
}

} // namespace cudaq_internal::device_call

//==============================================================================
// Realtime ABI exposed to CUDA-Q runtime, to be used by device_call compiler
// lowered code implementations.
//==============================================================================

// Lease a realtime request/response frame for lowered code.
// This will block until a frame is available or the timeout expires.
extern "C" std::int32_t __cudaq_device_call_acquire_realtime_frame(
    std::uint32_t deviceId, std::uint32_t functionId,
    std::uint64_t requestBytes, std::uint64_t responseCapacity,
    void **frameHandle, void **requestPayload, void **responsePayload) {
  CUDAQ_DBG("[device-call] ABI acquire realtime frame device={} functionId={} "
            "requestBytes={} responseCapacity={}",
            deviceId, functionId, requestBytes, responseCapacity);
  return runDeviceCallAbi([&] {
    DeviceCallDriver::instance().acquireFrameForDevice(
        deviceId, functionId, requestBytes, responseCapacity, frameHandle,
        requestPayload, responsePayload);
  });
}

// Dispatch a leased frame and report the response byte count.
// If the response capacity was zero, the frame has no response buffer,
// responseBytes will be set to zero, this will return instantly
// (fire-and-forget), the caller can release the frame immediately after.
extern "C" std::int32_t
__cudaq_device_call_dispatch_realtime_frame(void *frameHandle,
                                            std::uint64_t *responseBytes) {
  CUDAQ_DBG("[device-call] ABI dispatch realtime frame hasFrame={}",
            frameHandle != nullptr);
  return runDeviceCallAbi([&] {
    if (!responseBytes)
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "device_call response byte pointer must be non-null");
    *responseBytes = DeviceCallDriver::dispatchFrame(frameHandle);
  });
}

// Release or cancel a leased frame; safe to call.
// In the fire-and-forget case, the frame may not be recycled immediately (e.g.,
// for another acquisition), but this will release the caller's ownership and
// allow resources to be reclaimed when possible.
extern "C" void
__cudaq_device_call_safely_release_realtime_frame(void *frameHandle) {
  CUDAQ_DBG("[device-call] ABI release realtime frame hasFrame={}",
            frameHandle != nullptr);
  DeviceCallDriver::releaseFrame(frameHandle);
}
