/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ThunkInterface.h"
#include "cudaq/operators.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/remote_capabilities.h"
#include "cudaq/version2/cudaq/utils/type_erased_registry.h"
#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace cudaq {

class noise_model;
class ExecutionContext;
struct RemoteCapabilities;
class gradient;
class optimizer;

/// Typedefs for defining the connectivity structure of a QPU
using QubitEdge = std::pair<std::size_t, std::size_t>;
using QubitConnectivity = std::vector<QubitEdge>;

namespace details {

/// The Dispatch Table (Manual V-Table)
///
/// This structure provides a type-erased interface to QPU operations.
/// It contains function pointers (thunks) that wrap the concrete QPU type,
/// enabling runtime polymorphism without virtual functions.
///
struct qpu_dispatch_table {
  std::function<std::optional<QubitConnectivity>(void *)> getConnectivity;
  std::function<std::shared_ptr<void>()> create;
  std::function<void(void *, std::size_t)> setId;
  std::function<std::thread::id(const void *)> getExecutionThreadId;
  std::function<void(void *, const noise_model *)> setNoiseModel;
  std::function<const noise_model *(void *)> getNoiseModel;
  std::function<std::size_t(void *)> getNumQubits;
  std::function<bool(void *)> isSimulator;
  std::function<bool(void *)> supportsConditionalFeedback;
  std::function<bool(void *)> supportsExplicitMeasurements;
  std::function<RemoteCapabilities(const void *)> getRemoteCapabilities;
  std::function<void(void *, int)> setShots;
  std::function<void(void *)> clearShots;
  std::function<bool(void *)> isRemote;
  std::function<bool(void *)> isEmulated;
  std::function<void(void *, QuantumTask &)> enqueue;
  std::function<void(void *, ExecutionContext *)> setExecutionContext;
  std::function<void(void *)> resetExecutionContext;
  std::function<void(void *, const std::string &)> setTargetBackend;
  std::function<void(void *, const std::string &, const void *, gradient *,
                     const spin_op &, optimizer &, const int,
                     const std::size_t)>
      launchVQE;
  std::function<KernelThunkResultType(
      void *, const std::string &, KernelThunkType, void *, std::uint64_t,
      std::uint64_t, const std::vector<void *> &)>
      launchKernelThunk;
  std::function<void(void *, const std::string &, const std::vector<void *> &)>
      launchKernelArgs;
  std::function<void(void *, std::size_t)> onRandomSeedSet;

  template <typename T>
  void build() {
    create = []() { return std::make_shared<T>(); };
    setId = [](void *i, std::size_t id) { static_cast<T *>(i)->setId(id); };
    getExecutionThreadId = [](const void *i) {
      return static_cast<const T *>(i)->getExecutionThreadId();
    };
    setNoiseModel = [](void *i, const noise_model *model) {
      static_cast<T *>(i)->setNoiseModel(model);
    };
    getNoiseModel = [](void *i) {
      return static_cast<T *>(i)->getNoiseModel();
    };
    getNumQubits = [](void *i) { return static_cast<T *>(i)->getNumQubits(); };
    getConnectivity = [](void *i) {
      return static_cast<T *>(i)->getConnectivity();
    };
    isSimulator = [](void *i) { return static_cast<T *>(i)->isSimulator(); };
    supportsConditionalFeedback = [](void *i) {
      return static_cast<T *>(i)->supportsConditionalFeedback();
    };
    supportsExplicitMeasurements = [](void *i) {
      return static_cast<T *>(i)->supportsExplicitMeasurements();
    };
    getRemoteCapabilities = [](const void *i) {
      return static_cast<const T *>(i)->getRemoteCapabilities();
    };
    setShots = [](void *i, int shots) { static_cast<T *>(i)->setShots(shots); };
    clearShots = [](void *i) { static_cast<T *>(i)->clearShots(); };
    isRemote = [](void *i) { return static_cast<T *>(i)->isRemote(); };
    isEmulated = [](void *i) { return static_cast<T *>(i)->isEmulated(); };
    enqueue = [](void *i, QuantumTask &task) {
      static_cast<T *>(i)->enqueue(task);
    };
    setExecutionContext = [](void *i, ExecutionContext *context) {
      static_cast<T *>(i)->setExecutionContext(context);
    };
    resetExecutionContext = [](void *i) {
      static_cast<T *>(i)->resetExecutionContext();
    };
    setTargetBackend = [](void *i, const std::string &backend) {
      static_cast<T *>(i)->setTargetBackend(backend);
    };
    launchVQE = [](void *i, const std::string &name, const void *kernelArgs,
                   gradient *grad, const spin_op &H, optimizer &opt,
                   const int n_params, const std::size_t shots) {
      static_cast<T *>(i)->launchVQE(name, kernelArgs, grad, H, opt, n_params,
                                     shots);
    };
    launchKernelThunk = [](void *i, const std::string &name,
                           KernelThunkType kernelFunc, void *args,
                           std::uint64_t argSize, std::uint64_t resultSize,
                           const std::vector<void *> &rawArgs) {
      return static_cast<T *>(i)->launchKernel(name, kernelFunc, args, argSize,
                                               resultSize, rawArgs);
    };
    launchKernelArgs = [](void *i, const std::string &name,
                          const std::vector<void *> &rawArgs) {
      static_cast<T *>(i)->launchKernel(name, rawArgs);
    };
    onRandomSeedSet = [](void *i, std::size_t seed) {
      static_cast<T *>(i)->onRandomSeedSet(seed);
    };
  }
};
} // namespace details

/// The QPU Dynamic Wrapper
///
/// any_qpu provides a type-erased, runtime-polymorphic interface to any
/// registered QPU backend. It uses the QPURegistry to instantiate the
/// concrete QPU type by name, then wraps it with a dispatch table.
///
/// This enables new template based QPU implementations, which do not share a
/// common base class with the QPU class, to be used while all QPU are migrated
/// to the new design.
///
class any_qpu {
  std::shared_ptr<void> m_instance;
  details::qpu_dispatch_table const *const m_vtable;

public:
  any_qpu(std::shared_ptr<void> instance,
          const details::qpu_dispatch_table *vtable)
      : m_instance(instance), m_vtable(vtable) {
    assert(m_vtable && "any_qpu accessed without a dispatch table.");
  }

  void setId(std::size_t id) { m_vtable->setId(m_instance.get(), id); }
  std::thread::id getExecutionThreadId() const {
    return m_vtable->getExecutionThreadId(m_instance.get());
  }
  void setNoiseModel(const noise_model *model) {
    m_vtable->setNoiseModel(m_instance.get(), model);
  }
  const noise_model *getNoiseModel() {
    return m_vtable->getNoiseModel(m_instance.get());
  }
  std::size_t getNumQubits() {
    return m_vtable->getNumQubits(m_instance.get());
  }
  std::optional<QubitConnectivity> getConnectivity() {
    return m_vtable->getConnectivity(m_instance.get());
  }
  bool isSimulator() { return m_vtable->isSimulator(m_instance.get()); }
  bool supportsConditionalFeedback() {
    return m_vtable->supportsConditionalFeedback(m_instance.get());
  }
  bool supportsExplicitMeasurements() {
    return m_vtable->supportsExplicitMeasurements(m_instance.get());
  }
  RemoteCapabilities getRemoteCapabilities() const {
    return m_vtable->getRemoteCapabilities(m_instance.get());
  }
  void setShots(int shots) { m_vtable->setShots(m_instance.get(), shots); }
  void clearShots() { m_vtable->clearShots(m_instance.get()); }
  bool isRemote() { return m_vtable->isRemote(m_instance.get()); }
  bool isEmulated() { return m_vtable->isEmulated(m_instance.get()); }
  void enqueue(QuantumTask &task) { m_vtable->enqueue(m_instance.get(), task); }
  void setExecutionContext(ExecutionContext *context) {
    m_vtable->setExecutionContext(m_instance.get(), context);
  }
  void resetExecutionContext() {
    m_vtable->resetExecutionContext(m_instance.get());
  }
  void setTargetBackend(const std::string &backend) {
    m_vtable->setTargetBackend(m_instance.get(), backend);
  }
  void launchVQE(const std::string &name, const void *kernelArgs,
                 gradient *grad, const spin_op &H, optimizer &opt,
                 const int n_params, const std::size_t shots) {
    m_vtable->launchVQE(m_instance.get(), name, kernelArgs, grad, H, opt,
                        n_params, shots);
  }
  [[nodiscard]] KernelThunkResultType
  launchKernel(const std::string &name, KernelThunkType kernelFunc, void *args,
               std::uint64_t argSize, std::uint64_t resultSize,
               const std::vector<void *> &rawArgs) {
    return m_vtable->launchKernelThunk(m_instance.get(), name, kernelFunc, args,
                                       argSize, resultSize, rawArgs);
  }
  void launchKernel(const std::string &name,
                    const std::vector<void *> &rawArgs) {
    m_vtable->launchKernelArgs(m_instance.get(), name, rawArgs);
  }
  void onRandomSeedSet(std::size_t seed) {
    m_vtable->onRandomSeedSet(m_instance.get(), seed);
  }
};

namespace registry {
using QPURegistry =
    TypeErasedRegistry<any_qpu, cudaq::details::qpu_dispatch_table>;

inline std::unique_ptr<any_qpu> getQPU(const std::string &name) {
  return QPURegistry::get().instantiate(name);
}

inline bool isQPURegistered(const std::string &name) {
  return QPURegistry::get().is_registered(name);
}

} // namespace registry
} // namespace cudaq

#define CUDAQ_REGISTER_QPU_TYPE(TYPE, NAME)                                    \
  static TypeErasedRegistrar<cudaq::registry::QPURegistry, TYPE> CONCAT(       \
      registrar, NAME)(#NAME);
