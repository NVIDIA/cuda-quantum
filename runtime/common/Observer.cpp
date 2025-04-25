/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022-2025 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "Observer.h"

#include <queue>
#include <typeindex>
#include <vector>

namespace cudaq {
static std::vector<GlobalStateObserver *> observers;

static std::unordered_map<std::size_t, observer_data> firstMessages;

size_t hash_any(const std::any &a) {
  if (a.type() == typeid(std::size_t)) {
    return std::hash<std::size_t>{}(std::any_cast<std::size_t>(a));
  }
  // Add more types as needed
  // Fallback: hash type_index
  return std::hash<std::type_index>{}(std::type_index(a.type()));
}

// Simple hash combine (boost::hash_combine alternative)
inline void hash_combine(std::size_t &seed, std::size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::size_t
hash_observe_data(const std::unordered_map<std::string, std::any> &m) {
  std::size_t seed = 0;
  for (const auto &[key, value] : m) {
    std::size_t key_hash = std::hash<std::string>{}(key);
    std::size_t value_hash = hash_any(value);
    hash_combine(seed, key_hash);
    hash_combine(seed, value_hash);
  }
  return seed;
}

void registerObserver(GlobalStateObserver *obs) {
  if (observers.empty())
    for (auto &[hash, msg] : firstMessages)
      obs->oneWayNotify(msg);
  observers.push_back(obs);
}

void notifyAll(const observer_data &data) {
  if (observers.empty()) {
    auto hsh = hash_observe_data(data);
    if (auto iter = firstMessages.find(hsh); iter == firstMessages.end()) {
      firstMessages.insert({hsh, data});
      return;
    }
  }

  for (auto &obs : observers)
    obs->oneWayNotify(data);
}

observer_data notifyWithResponse(const observer_data &data) {
  for (auto &obs : observers)
    if (auto [success, response] = obs->notifyWithResponse(data); success)
      return response;
  return {};
}

} // namespace cudaq
