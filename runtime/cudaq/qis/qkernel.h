/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#ifdef CUDAQ_LIBRARY_MODE

namespace cudaq {

/// In library mode, the quake compiler is not involved. To streamline things,
/// just have `qkernel` alias the std::function template class.
template <typename Sig>
using qkernel = std::function<Sig>;

} // namespace cudaq

#else

namespace cudaq {

namespace details {
class QKernelDummy;

template <typename R, typename... As>
class QKernelInterface {
public:
  virtual ~QKernelInterface() = default;

  virtual R dispatch(As...) = 0;
  virtual void *getEntry() = 0;
};

template <typename R, typename F, typename... As>
class QKernelHolder : public QKernelInterface<R, As...> {
public:
  using EntryType =
      std::conditional_t<std::is_class_v<F>, R (QKernelDummy::*)(As...),
                         R (*)(As...)>;

  QKernelHolder() : entry{nullptr}, callable{} {}
  QKernelHolder(const QKernelHolder &) = default;
  QKernelHolder(QKernelHolder &&) = default;

  explicit QKernelHolder(F &&f) : callable(std::forward<F>(f)) {
    if constexpr (std::is_same_v<F, R (*)(As...)>) {
      entry = f;
    } else {
      setEntry(&F::operator());
    }
  }
  explicit QKernelHolder(const F &f) : callable(f) {
    if constexpr (std::is_same_v<F, R (*)(As...)>) {
      entry = f;
    } else {
      setEntry(&F::operator());
    }
  }

  QKernelHolder &operator=(const QKernelHolder &) = default;
  QKernelHolder &operator=(QKernelHolder &&holder) = default;

  R dispatch(As... as) override { return std::invoke(callable, as...); }

  // Copy the bits of the member function pointer, \p mfp, into the member
  // function pointer, `entry`.  The kernel launcher will use this to convert
  // from host-side to device-side.
  template <typename MFP>
  void setEntry(const MFP &mfp) {
    memcpy(&entry, &mfp, sizeof(EntryType));
  }

  // This will provide a hook value for the runtime to determine which kernel
  // was captured in the C++ host code.
  void *getEntry() override { return static_cast<void *>(&entry); }

  // We keep a (member) function pointer (specialized by the callable) in order
  // to convert from the host-side to the device-side address space.
  EntryType entry;

  // The actual callable to dispatch upon.
  F callable;
};

} // namespace details

template <typename A>
using remove_cvref_t = std::remove_cvref_t<A>;

/// A `qkernel` must be used to wrap `CUDA-Q` kernels (callables annotated
/// with the `__qpu__` attribute) when those kernels are \e referenced other
/// than by a direct call in code outside of quantum kernels proper. Supports
/// free functions, classes with call operators, and lambdas.
///
/// The quake compiler can inspect these wrappers in the C++ code and tweak them
/// to provide information necessary and sufficient for the CUDA-Q runtime to
/// either stitch together execution in a simulation environment and/or JIT
/// compile and re-link these kernels into a cohesive quantum circuit.
template <typename>
class qkernel;

template <typename R, typename... As>
class qkernel<R(As...)> {
public:
  qkernel() {}
  qkernel(std::nullptr_t) {}
  qkernel(const qkernel &) = default;
  qkernel(qkernel &&) = default;

  template <typename FUNC,
            bool SELF = std::is_same_v<remove_cvref_t<FUNC>, qkernel>>
  using DecayType = typename std::enable_if_t<!SELF, std::decay<FUNC>>::type;

  template <typename FUNC, typename DFUNC = DecayType<FUNC>,
            typename RES =
                std::conditional_t<std::is_invocable_r_v<R, DFUNC, As...>,
                                   std::true_type, std::false_type>>
  struct CallableType : RES {};

  template <typename S, typename = std::enable_if_t<CallableType<S>::value>>
  qkernel(S &&f) {
    using PS = remove_cvref_t<S>;
    if constexpr (std::is_same_v<PS, R (*)(As...)> ||
                  std::is_same_v<PS, R(As...)>) {
      kernelCallable =
          std::make_unique<details::QKernelHolder<R, R (*)(As...), As...>>(f);
    } else {
      kernelCallable =
          std::make_unique<details::QKernelHolder<R, PS, As...>>(f);
    }
  }

  R operator()(As... as) const {
    return kernelCallable->dispatch(std::forward<As>(as)...);
  }
  R operator()(As... as) {
    return kernelCallable->dispatch(std::forward<As>(as)...);
  }

  void **get_entry_kernel_from_holder() const {
    return static_cast<void **>(kernelCallable->getEntry());
  }

private:
  std::unique_ptr<details::QKernelInterface<R, As...>> kernelCallable;
};

template <typename>
struct qkernel_deduction_guide_helper {};

template <typename R, typename P, typename... As>
struct qkernel_deduction_guide_helper<R (P::*)(As...)> {
  using type = R(As...);
};
template <typename R, typename P, typename... As>
struct qkernel_deduction_guide_helper<R (P::*)(As...) const> {
  using type = R(As...);
};
template <typename R, typename P, typename... As>
struct qkernel_deduction_guide_helper<R (P::*)(As...) &> {
  using type = R(As...);
};
template <typename R, typename P, typename... As>
struct qkernel_deduction_guide_helper<R (P::*)(As...) const &> {
  using type = R(As...);
};

// Deduction guides for C++20.

template <typename R, typename... As>
qkernel(R (*)(As...)) -> qkernel<R(As...)>;

template <typename F, typename S = typename qkernel_deduction_guide_helper<
                          decltype(&F::operator())>::type>
qkernel(F) -> qkernel<S>;

} // namespace cudaq

#endif // CUDAQ_LIBRARY_MODE
