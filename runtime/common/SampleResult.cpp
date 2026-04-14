/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "SampleResult.h"
#include "cudaq/spin_op.h"

#include <algorithm>
#include <numeric>
#include <string.h>

#include <iostream>
#include <map>
#include <vector>

static std::string longToBitString(int size, long x) {
  std::string s(size, '0');
  int counter = 0;
  do {
    s[counter] = '0' + (x & 1);
    counter++;
  } while (x >>= 1);
  std::reverse(s.begin(), s.end());
  return s;
}

static void
deserializeCounts(std::vector<std::size_t> &data, std::size_t &stride,
                  std::unordered_map<std::string, std::size_t> &localCounts) {
  auto nBs = data[stride];
  stride++;

  for (std::size_t j = stride; j < stride + nBs * 3; j += 3) {
    auto bitstring_as_long = data[j];
    auto size_of_bitstring = data[j + 1];
    auto count = data[j + 2];
    auto bs = longToBitString(size_of_bitstring, bitstring_as_long);
    localCounts.insert({bs, count});
  }
  stride += nBs * 3;
}

static std::string extractNameFromData(std::vector<std::size_t> &data,
                                       std::size_t &stride) {
  auto nChars = data[stride++];

  std::string name(data.begin() + stride, data.begin() + stride + nChars);

  stride += nChars;
  return name;
}

namespace cudaq {

ExecutionResult::ExecutionResult(CountsDictionary c) : counts(c) {}
ExecutionResult::ExecutionResult(std::string name) : registerName(name) {}
ExecutionResult::ExecutionResult(double e) : expectationValue(e) {}

ExecutionResult::ExecutionResult(CountsDictionary c, std::string name)
    : counts(c), registerName(name) {}
ExecutionResult::ExecutionResult(CountsDictionary c, std::string name, double e)
    : counts(c), expectationValue(e), registerName(name) {}

ExecutionResult::ExecutionResult(CountsDictionary c, double e)
    : counts(c), expectationValue(e) {}
ExecutionResult::ExecutionResult(const ExecutionResult &other)
    : counts(other.counts), expectationValue(other.expectationValue),
      registerName(other.registerName), sequentialData(other.sequentialData) {}

ExecutionResult &ExecutionResult::operator=(const ExecutionResult &other) {
  counts = other.counts;
  expectationValue = other.expectationValue;
  registerName = other.registerName;
  sequentialData = other.sequentialData;
  return *this;
}

void ExecutionResult::appendResult(std::string bitString, std::size_t count) {
  auto [iter, inserted] = counts.emplace(std::move(bitString), count);
  if (!inserted)
    iter->second += count;

  sequentialData.insert(sequentialData.end(), count, iter->first);
}

bool ExecutionResult::operator==(const ExecutionResult &result) const {
  return registerName == result.registerName && counts == result.counts;
}

/// @brief  Encoding - 1st element is size of the register name N, then next N
// represent register name, number of bitstrings M, then for each bit string
// {l, bs.length, count}
/// @return
std::vector<std::size_t> ExecutionResult::serialize() const {
  std::vector<std::size_t> retData;

  // Encode the classical register name
  retData.push_back(registerName.length());
  for (std::size_t j = 0; j < registerName.length(); j++) {
    retData.push_back((std::size_t)registerName[j]);
  }

  // Encode the counts data
  retData.push_back(counts.size());
  for (auto &kv : counts) {
    auto bits = kv.first;
    auto count = kv.second;
    auto l = std::stol(bits, NULL, 2);
    retData.push_back(l);
    retData.push_back(bits.length());
    retData.push_back(count);
  }
  return retData;
}

void ExecutionResult::deserialize(std::vector<std::size_t> &data) {
  std::size_t stride = 0;
  while (stride < data.size()) {
    std::string name = extractNameFromData(data, stride);

    std::unordered_map<std::string, std::size_t> localCounts;
    deserializeCounts(data, stride, localCounts);

    for (const auto &entry : localCounts) {
      counts.insert({entry.first, entry.second});
    }
  }
}

//===----------------------------------------------------------------------===//

std::vector<std::size_t> sample_result::serialize() const {
  std::vector<std::size_t> retData;
  for (auto &result : sampleResults) {
    auto serialized = result.second.serialize();
    retData.insert(retData.end(), serialized.begin(), serialized.end());
  }
  return retData;
}

void sample_result::deserialize(std::vector<std::size_t> &data) {
  std::size_t stride = 0;
  totalShots = 0;

  while (stride < data.size()) {
    std::string name = extractNameFromData(data, stride);

    std::unordered_map<std::string, std::size_t> localCounts;
    deserializeCounts(data, stride, localCounts);

    sampleResults.insert({name, ExecutionResult{localCounts, name}});

    if (stride >= data.size()) {
      totalShots = std::accumulate(
          localCounts.begin(), localCounts.end(), 0,
          [](std::size_t sum, const auto &pair) { return sum + pair.second; });
    }
  }
}

sample_result::sample_result(ExecutionResult &&result) {
  auto counts = result.counts;
  sampleResults.insert({result.registerName, std::move(result)});
  for (auto &[bits, count] : counts)
    totalShots += count;
}

sample_result::sample_result(const ExecutionResult &result)
    : sample_result(ExecutionResult{result}) {}

sample_result::sample_result(const std::vector<ExecutionResult> &results) {
  for (auto &result : results)
    sampleResults.insert({result.registerName, result});

  if (results.empty())
    return;
  for (auto &[bits, count] : results[0].counts)
    totalShots += count;
}

sample_result::sample_result(double preComputedExp,
                             const std::vector<ExecutionResult> &results)
    : sample_result(results) {
  // Create a spot for the pre-computed exp val
  sampleResults.emplace(GlobalRegisterName, preComputedExp);
}

void sample_result::append(const ExecutionResult &result, bool concatenate) {
  // If given a result corresponding to the same register name, either a)
  // replace the existing one if concatenate is false, or b) if concatenate is
  // true, stitch the bitstrings from "result" into the existing one.
  auto iter = sampleResults.find(result.registerName);
  if (iter != sampleResults.end()) {
    auto &existingExecResult = iter->second;
    if (concatenate) {
      // Stitch the bitstrings together
      if (this->totalShots == result.sequentialData.size()) {
        existingExecResult.counts.clear();
        for (std::size_t i = 0; i < this->totalShots; i++) {
          std::string newStr =
              existingExecResult.sequentialData[i] + result.sequentialData[i];
          existingExecResult.counts[newStr]++;
          existingExecResult.sequentialData[i] = std::move(newStr);
        }
      }
    } else {
      // Replace the existing one
      existingExecResult = result;
    }
  } else {
    sampleResults.insert({result.registerName, result});
  }
  if (!totalShots)
    for (auto &[bits, count] : result.counts)
      totalShots += count;
}

bool sample_result::operator==(const sample_result &counts) const {
  return sampleResults == counts.sampleResults;
}

sample_result &sample_result::operator+=(const sample_result &other) {

  for (auto &otherResults : other.sampleResults) {
    auto regName = otherResults.first;
    auto foundIter = sampleResults.find(regName);
    if (foundIter == sampleResults.end()) {
      sampleResults.insert({regName, otherResults.second});
    } else {
      // We already have a sample result with this name, so now lets just merge
      // them.
      auto &sr = sampleResults[regName];
      for (auto &[bits, count] : otherResults.second.counts) {
        auto &ourCounts = sr.counts;
        if (ourCounts.count(bits))
          ourCounts[bits] += count;
        else
          ourCounts.insert({bits, count});
      }

      if (!otherResults.second.sequentialData.empty())
        sr.sequentialData.insert(sr.sequentialData.end(),
                                 otherResults.second.sequentialData.begin(),
                                 otherResults.second.sequentialData.end());
    }
    if (regName == GlobalRegisterName)
      totalShots += other.totalShots;
  }
  return *this;
}

std::pair<bool, const ExecutionResult &>
sample_result::try_retrieve_result(const std::string &registerName) const {
  auto iter = sampleResults.find(registerName);
  if (iter == sampleResults.end()) {
    auto invalid_char = registerName.find_first_not_of("XYZI");
    if (invalid_char == std::string::npos) {
      auto spin = spin_op::from_word(registerName);
      iter = sampleResults.find(spin.canonicalize().get_term_id());
      if (iter != sampleResults.end())
        return {true, iter->second};
    }
    return {false, ExecutionResult()};
  }
  return {true, iter->second};
}

const cudaq::ExecutionResult &
sample_result::retrieve_result(const std::string &registerName) const {
  auto [found, result] = try_retrieve_result(registerName);
  if (!found)
    throw std::runtime_error("no results stored for " + registerName);
  return result;
}

cudaq::ExecutionResult &
sample_result::retrieve_result(const std::string &registerName) {
  auto iter = sampleResults.find(registerName);
  if (iter == sampleResults.end()) {
    auto invalid_char = registerName.find_first_not_of("XYZI");
    if (invalid_char == std::string::npos) {
      auto spin = spin_op::from_word(registerName);
      iter = sampleResults.find(spin.canonicalize().get_term_id());
      if (iter != sampleResults.end())
        return iter->second;
    }
    throw std::runtime_error("no results stored for " + registerName);
  }
  return iter->second;
}

std::vector<std::string>
sample_result::sequential_data(const std::string_view registerName) const {
  return retrieve_result(registerName.data()).getSequentialData();
}

CountsDictionary::iterator sample_result::begin() {
  return retrieve_result(GlobalRegisterName).counts.begin();
}

CountsDictionary::iterator sample_result::end() {
  return retrieve_result(GlobalRegisterName).counts.end();
}

CountsDictionary::const_iterator sample_result::cbegin() const {
  return retrieve_result(GlobalRegisterName).counts.cbegin();
}

CountsDictionary::const_iterator sample_result::cend() const {
  return retrieve_result(GlobalRegisterName).counts.cend();
}

std::size_t
sample_result::size(const std::string_view registerName) const noexcept {
  auto [found, result] = try_retrieve_result(registerName.data());
  if (found)
    return result.counts.size();
  else
    return 0;
}

double sample_result::probability(std::string_view bitStr,
                                  const std::string_view registerName) const {
  const auto &result = retrieve_result(registerName.data());
  const auto countIter = result.counts.find(bitStr.data());
  return (countIter == result.counts.end())
             ? 0.0
             : (double)countIter->second / totalShots;
}

std::size_t sample_result::count(std::string_view bitStr,
                                 const std::string_view registerName) const {
  const auto &counts = retrieve_result(registerName.data()).counts;
  auto it = counts.find(bitStr.data());
  if (it == counts.cend())
    return 0;
  else
    return it->second;
}

std::string
sample_result::most_probable(const std::string_view registerName) const {
  const auto &counts = retrieve_result(registerName.data()).counts;
  return std::max_element(counts.begin(), counts.end(),
                          [](const auto &el1, const auto &el2) {
                            return el1.second < el2.second;
                          })
      ->first;
}

bool sample_result::has_expectation(const std::string_view registerName) const {
  auto [found, result] = try_retrieve_result(registerName.data());
  if (found)
    return result.expectationValue.has_value();
  else
    return false;
}

double sample_result::expectation(const std::string_view registerName) const {
  auto [found, result] = try_retrieve_result(registerName.data());
  if (!found)
    return 0.0;

  if (result.expectationValue.has_value())
    return result.expectationValue.value();

  double aver = 0.0;
  const auto &counts = result.counts;
  for (auto &kv : counts) {
    auto par = has_even_parity(kv.first);
    auto p = probability(kv.first, registerName);
    if (!par) {
      p = -p;
    }
    aver += p;
  }

  return aver;
}

std::vector<std::string> sample_result::register_names() const {
  std::vector<std::string> ret;
  for (auto &kv : sampleResults)
    ret.push_back(kv.first);
  std::sort(ret.begin(), ret.end());

  return ret;
}

CountsDictionary
sample_result::to_map(const std::string_view registerName) const {
  return retrieve_result(registerName.data()).counts;
}

sample_result
sample_result::get_marginal(const std::vector<std::size_t> &marginalIndices,
                            const std::string_view registerName) const {
  const auto &counts = retrieve_result(registerName.data()).counts;
  auto mutableIndices = marginalIndices;

  std::sort(mutableIndices.begin(), mutableIndices.end());

  ExecutionResult sr;
  for (auto &[bits, count] : counts) {
    std::string newBits;
    for ([[maybe_unused]] auto &m : mutableIndices)
      newBits += "0";
    for (int counter = 0; auto &index : mutableIndices) {
      if (index > bits.size())
        throw std::runtime_error("Invalid marginal index (" +
                                 std::to_string(index) +
                                 ", size=" + std::to_string(bits.size()));

      newBits[counter++] = bits[index];
    }
    sr.appendResult(newBits, count);
  }

  return sample_result(sr);
}

void sample_result::clear() {
  sampleResults.clear();
  totalShots = 0;
}

/// @brief This is a helper function to sort the keys of an unordered map
/// without making any deep copies.
template <typename T>
std::vector<typename T::const_iterator> sortByKeys(const T &unordered_map) {
  std::vector<typename T::const_iterator> iterators;
  iterators.reserve(unordered_map.size());
  for (auto it = unordered_map.begin(); it != unordered_map.end(); ++it)
    iterators.push_back(it);
  std::sort(iterators.begin(), iterators.end(),
            [](const auto &a, const auto &b) { return a->first < b->first; });
  return iterators;
}

void sample_result::dump(std::ostream &os) const {
  os << "{ ";
  if (sampleResults.size() > 1) {
    os << "\n  ";
    std::size_t counter = 0;
    for (auto &result : sortByKeys(sampleResults)) {
      os << result->first << " : { ";
      for (auto &kv : sortByKeys(result->second.counts)) {
        os << kv->first << ":" << kv->second << " ";
      }
      bool isLast = counter == sampleResults.size() - 1;
      counter++;
      os << "}\n" << (!isLast ? "   " : "");
    }

  } else if (sampleResults.size() == 1) {

    CountsDictionary counts;
    auto [found, result] = try_retrieve_result(GlobalRegisterName);
    if (found)
      counts = result.counts;
    else {
      auto first = sampleResults.begin();
      os << "\n   " << first->first << " : { ";
      counts = sampleResults.begin()->second.counts;
    }

    for (auto &kv : sortByKeys(counts)) {
      os << kv->first << ":" << kv->second << " ";
    }

    if (!found)
      os << "}\n";
  }
  os << "}\n";
}

void sample_result::dump() const { dump(std::cout); }

bool sample_result::has_even_parity(std::string_view bitString) {
  int c = std::count(bitString.begin(), bitString.end(), '1');
  return c % 2 == 0;
}

void sample_result::reorder(const std::vector<std::size_t> &idx,
                            const std::string_view registerName) {
  // First process the counts
  auto &result = retrieve_result(registerName.data());
  CountsDictionary newCounts;
  for (auto [bits, count] : result.counts) {
    if (idx.size() != bits.size())
      throw std::runtime_error("Calling reorder() with invalid parameter idx");

    std::string newBits(bits);
    int i = 0;
    for (auto oldIdx : idx)
      newBits[i++] = bits[oldIdx];
    newCounts[newBits] = count;
  }
  result.counts = newCounts;

  // Now process the sequential data
  for (auto &s : result.sequentialData) {
    std::string newBits(s);
    int i = 0;
    for (auto oldIdx : idx)
      newBits[i++] = s[oldIdx];
    s = newBits;
  }
}
} // namespace cudaq
