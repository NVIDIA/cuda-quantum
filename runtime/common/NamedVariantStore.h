/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace cudaq::detail {

/// A container of named heterogeneous name-value pairs, retrievable by name and
/// value type.
template <typename... Ts>
class NamedVariantStore {
public:
  using Value = std::variant<Ts...>;
  using Entry = std::pair<std::string, Value>;
  using Storage = std::vector<Entry>;
  using const_iterator = Storage::const_iterator;

  /// Get the value with the given name and type `T`.
  template <typename T>
  const T *get(std::string_view name) const {
    for (const auto &[valName, val] : entries) {
      if (valName != name)
        continue;
      if (const auto *typedVal = std::get_if<T>(&val))
        return typedVal;
    }
    return nullptr;
  }

  /// Range over all entries of type `T` in insertion order.
  template <typename T>
  auto getAllOfType() const {
    return std::views::all(entries) |
           std::views::filter([](const Entry &entry) {
             return std::holds_alternative<T>(entry.second);
           }) |
           std::views::transform(
               [](const Entry &entry)
                   -> std::pair<const std::string &, const T &> {
                 return {entry.first, *std::get_if<T>(&entry.second)};
               });
  }

  /// Add an entry with the given name and value.
  ///
  /// Throws a `std::runtime_error` if an entry with the same name and type
  /// already exists.
  void add(std::string name, Value value) {
    const auto duplicate =
        std::ranges::any_of(entries, [&](const Entry &entry) {
          return entry.first == name && entry.second.index() == value.index();
        });
    if (duplicate)
      throw std::runtime_error("Value with same type and name '" + name +
                               "' already exists");
    entries.emplace_back(std::move(name), std::move(value));
  }

  const_iterator begin() const { return entries.begin(); }
  const_iterator end() const { return entries.end(); }

private:
  Storage entries;
};

} // namespace cudaq::detail
