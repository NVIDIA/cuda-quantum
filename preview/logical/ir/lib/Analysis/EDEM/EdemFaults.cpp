/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Analysis/EdemFaults.h"

#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <unordered_set>

namespace cudaq::logical::edem_analysis {
namespace {

llvm::Error invalid(llvm::StringRef message) {
  return llvm::createStringError(message);
}

llvm::Error rejectUnknownKeys(const llvm::json::Object &object,
                              std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &entry : object) {
    llvm::StringRef key = entry.first;
    if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
      return invalid("unknown fault-input field '" + key.str() + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<std::size_t> nonnegativeIndex(const llvm::json::Value &value,
                                             llvm::StringRef field) {
  auto integer = value.getAsInteger();
  if (!integer || *integer < 0)
    return invalid((field + " must be a nonnegative integer").str());
  return static_cast<std::size_t>(*integer);
}

llvm::Expected<PauliFactor> parseFactor(const llvm::json::Value &value) {
  const auto *object = value.getAsObject();
  if (!object)
    return invalid("fault Pauli factor must be an object");
  if (auto error = rejectUnknownKeys(*object, {"carrier", "pauli"}))
    return std::move(error);
  const auto *rawCarrier = object->get("carrier");
  auto rawPauli = object->getString("pauli");
  if (!rawCarrier || !rawPauli)
    return invalid("fault Pauli factor requires carrier and pauli");
  auto carrier = nonnegativeIndex(*rawCarrier, "fault carrier");
  if (!carrier)
    return carrier.takeError();
  Pauli basis;
  if (*rawPauli == "x")
    basis = Pauli::x;
  else if (*rawPauli == "y")
    basis = Pauli::y;
  else if (*rawPauli == "z")
    basis = Pauli::z;
  else
    return invalid("fault pauli must be x, y, or z");
  return PauliFactor{*carrier, basis};
}

llvm::Expected<PauliInsertion> parseInsertion(const llvm::json::Value &value) {
  const auto *object = value.getAsObject();
  if (!object)
    return invalid("fault insertion must be an object");
  if (auto error = rejectUnknownKeys(*object, {"after", "factors"}))
    return std::move(error);
  auto after = object->getString("after");
  const auto *factors = object->getArray("factors");
  if (!after || !factors || factors->empty())
    return invalid("fault insertion requires after and nonempty factors");
  PauliInsertion result;
  result.afterSite = after->str();
  std::unordered_set<std::size_t> carriers;
  result.factors.reserve(factors->size());
  for (const auto &raw : *factors) {
    auto factor = parseFactor(raw);
    if (!factor)
      return factor.takeError();
    if (!carriers.insert(factor->carrier).second)
      return invalid("fault insertion repeats a carrier");
    result.factors.push_back(*factor);
  }
  return result;
}

llvm::Expected<ElementaryFault> parseFault(const llvm::json::Value &value) {
  const auto *object = value.getAsObject();
  if (!object)
    return invalid("elementary fault must be an object");
  if (auto error =
          rejectUnknownKeys(*object, {"id", "insertions", "record_flips"}))
    return std::move(error);
  auto id = object->getString("id");
  if (!id || id->empty())
    return invalid("elementary fault requires a nonempty id");
  ElementaryFault result;
  result.id = id->str();
  for (auto key : {"insertions", "record_flips"})
    if (object->get(key) && !object->getArray(key))
      return invalid("fault effects must be arrays");
  if (const auto *insertions = object->getArray("insertions")) {
    result.insertions.reserve(insertions->size());
    for (const auto &raw : *insertions) {
      auto insertion = parseInsertion(raw);
      if (!insertion)
        return insertion.takeError();
      result.insertions.push_back(std::move(*insertion));
    }
  }
  if (const auto *flips = object->getArray("record_flips")) {
    result.recordFlips.reserve(flips->size());
    for (const auto &raw : *flips) {
      auto record = raw.getAsString();
      if (!record || record->empty())
        return invalid("record flip must name a stable record");
      result.recordFlips.push_back(record->str());
    }
  }
  if (result.insertions.empty() && result.recordFlips.empty())
    return invalid("elementary fault has no effect");
  return result;
}

} // namespace

llvm::Expected<FaultInput> parseFaultInput(llvm::StringRef text) {
  auto document = llvm::json::parse(text);
  if (!document)
    return document.takeError();
  const auto *root = document->getAsObject();
  if (!root || root->getString("schema") != "cudaq.logical.edem-faults/v1")
    return invalid("fault input requires cudaq.logical.edem-faults/v1");
  if (auto error = rejectUnknownKeys(
          *root, {"schema", "faults", "exclusive_groups", "probabilities"}))
    return std::move(error);
  const auto *rawFaults = root->getArray("faults");
  if (!rawFaults)
    return invalid("fault input requires a faults array");
  if (rawFaults->size() > 1000000)
    return invalid("fault input exceeds the source limit");

  FaultInput result;
  result.faults.reserve(rawFaults->size());
  std::unordered_set<std::string> names;
  for (const auto &raw : *rawFaults) {
    auto fault = parseFault(raw);
    if (!fault)
      return fault.takeError();
    if (!names.insert(fault->id).second)
      return invalid("fault source IDs must be unique");
    result.faults.push_back(std::move(*fault));
  }

  if (root->get("exclusive_groups") && !root->getArray("exclusive_groups"))
    return invalid("exclusive_groups must be an array");
  std::vector<bool> claimed(result.faults.size(), false);
  if (const auto *groups = root->getArray("exclusive_groups")) {
    result.exclusiveGroups.reserve(groups->size());
    for (const auto &rawGroup : *groups) {
      const auto *group = rawGroup.getAsArray();
      if (!group || group->size() < 2)
        return invalid("exclusive group needs at least two alternatives");
      auto &indices = result.exclusiveGroups.emplace_back();
      indices.reserve(group->size());
      for (const auto &rawIndex : *group) {
        auto index = nonnegativeIndex(rawIndex, "exclusive group index");
        if (!index)
          return index.takeError();
        if (*index >= result.faults.size() || claimed[*index])
          return invalid("exclusive group index is repeated or out of range");
        claimed[*index] = true;
        indices.push_back(*index);
      }
      if (!std::is_sorted(indices.begin(), indices.end()))
        return invalid("exclusive group indices must be ordered");
    }
  }

  if (const auto *probabilities = root->getArray("probabilities")) {
    if (probabilities->size() != result.faults.size())
      return invalid("probability count must match fault count");
    result.probabilities.reserve(probabilities->size());
    for (const auto &raw : *probabilities) {
      auto p = raw.getAsNumber();
      if (!p || !std::isfinite(*p) || *p < 0.0 || *p > 1.0)
        return invalid("fault probabilities must be finite in [0,1]");
      result.probabilities.push_back(*p);
    }
  } else {
    return invalid("fault input requires a probabilities array");
  }
  // The EDEM library owns exclusive-total validation. Do not duplicate its
  // probability policy or change the supplied probabilities here.
  return result;
}

} // namespace cudaq::logical::edem_analysis
