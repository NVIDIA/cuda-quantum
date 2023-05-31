/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <utility>

#include "cudaq_utils.h"

namespace cudaq {
template <typename T>
cudaq_pimpl<T>::cudaq_pimpl() : m{new T{}} {}

template <typename T>
cudaq_pimpl<T>::cudaq_pimpl(const cudaq_pimpl<T> &other)
    : m(std::make_unique<T>(other)) {}

template <typename T>
template <typename... Args>
cudaq_pimpl<T>::cudaq_pimpl(Args &&...args)
    : m{new T{std::forward<Args>(args)...}} {}

template <typename T>
cudaq_pimpl<T>::~cudaq_pimpl() {}

template <typename T>
T *cudaq_pimpl<T>::operator->() {
  return m.get();
}
template <typename T>
T *cudaq_pimpl<T>::operator->() const {
  return m.get();
}

template <typename T>
T &cudaq_pimpl<T>::operator*() {
  return *m.get();
}
} // namespace cudaq
