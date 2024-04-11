/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "MeasureCounts.h"

#include <algorithm>
#include <numeric>
#include <string.h>

#include <iostream>
#include <map>
#include <vector>

namespace cudaq {
std::string longToBitString(int size, long x) {
  std::string s(size, '0');
  int counter = 0;
  do {
    s[counter] = '0' + (x & 1);
    counter++;
  } while (x >>= 1);
  std::reverse(s.begin(), s.end());
  return s;
}

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
  auto iter = counts.find(bitString);
  if (iter == counts.end())
    counts.insert({bitString, count});
  else
    iter->second += count;

  sequentialData.insert(sequentialData.end(), count, bitString);
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
    auto nChars = data[stride];
    stride++;
    std::string name = "";
    for (std::size_t i = 0; i < nChars; i++)
      name += std::string(1, char(data[stride + i]));

    stride += nChars;
    std::unordered_map<std::string, std::size_t> localCounts;
    auto nBs = data[stride];
    stride++;
    for (std::size_t j = stride; j < stride + nBs * 3; j += 3) {
      auto bitstring_as_long = data[j];
      auto size_of_bitstring = data[j + 1];
      auto count = data[j + 2];
      auto bs = longToBitString(size_of_bitstring, bitstring_as_long);
      counts.insert({bs, count});
    }
    stride += nBs * 3;
  }
}

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
  while (stride < data.size()) {
    auto nChars = data[stride];
    stride++;
    std::string name = "";
    for (std::size_t i = 0; i < nChars; i++)
      name += std::string(1, char(data[stride + i]));

    stride += nChars;
    std::size_t localShots = 0;
    std::unordered_map<std::string, std::size_t> localCounts;
    auto nBs = data[stride];
    stride++;
    for (std::size_t j = stride; j < stride + nBs * 3; j += 3) {
      auto bitstring_as_long = data[j];
      auto size_of_bitstring = data[j + 1];
      auto count = data[j + 2];
      auto bs = longToBitString(size_of_bitstring, bitstring_as_long);
      localShots += count;
      localCounts.insert({bs, count});
    }

    sampleResults.insert({name, ExecutionResult{localCounts, name}});

    stride += nBs * 3;
    totalShots = localShots;
  }
}

sample_result::sample_result(ExecutionResult &&result) {
  sampleResults.insert({result.registerName, result});
  for (auto &[bits, count] : result.counts)
    totalShots += count;
}

sample_result::sample_result(ExecutionResult &result)
    : sample_result(std::move(result)) {}

sample_result::sample_result(std::vector<ExecutionResult> &results) {
  for (auto &result : results) {
    sampleResults.insert({result.registerName, result});
  }
  if (!results.empty())
    for (auto &[bits, count] : results[0].counts)
      totalShots += count;
}

sample_result::sample_result(double preComputedExp,
                             std::vector<ExecutionResult> &results) {
  for (auto &result : results) {
    sampleResults.insert({result.registerName, result});
  }

  // Create a spot for the pre-computed exp val
  sampleResults.emplace(GlobalRegisterName, preComputedExp);

  if (results.empty())
    return;

  for (auto &[bits, count] : results[0].counts)
    totalShots += count;
}

void sample_result::append(ExecutionResult &result) {
  // If given a result corresponding to the same register name,
  // replace the existing one if in the map.
  auto iter = sampleResults.find(result.registerName);
  if (iter != sampleResults.end())
    iter->second = result;
  else
    sampleResults.insert({result.registerName, result});
  if (!totalShots)
    for (auto &[bits, count] : result.counts)
      totalShots += count;
}

sample_result::sample_result(const sample_result &m)
    : sampleResults(m.sampleResults), totalShots(m.totalShots) {}

sample_result &sample_result::operator=(sample_result &counts) {
  sampleResults.clear();
  for (auto &[name, sampleResult] : counts.sampleResults) {
    sampleResults.insert({name, sampleResult});
  }
  totalShots = counts.totalShots;
  return *this;
}
sample_result &sample_result::operator=(const sample_result &counts) {
  sampleResults.clear();
  for (auto &[name, sampleResult] : counts.sampleResults) {
    sampleResults.insert({name, sampleResult});
  }
  totalShots = counts.totalShots;
  return *this;
}

bool sample_result::operator==(const sample_result &counts) const {
  return sampleResults == counts.sampleResults;
}

sample_result &sample_result::operator+=(const sample_result &other) {

  for (auto &otherResults : other.sampleResults) {
    auto regName = otherResults.first;
    auto foundIter = sampleResults.find(regName);
    if (foundIter == sampleResults.end())
      sampleResults.insert({regName, otherResults.second});
    else {

      // we already have a sample result with this name, so
      // now lets just merge them
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
  }
  return *this;
}

std::vector<std::string>
sample_result::sequential_data(const std::string_view registerName) const {
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    throw std::runtime_error(
        "There is no sample result for the given registerName (" +
        std::string{registerName.begin(), registerName.end()} + ")");

  auto data = iter->second.getSequentialData();
  return data;
}

CountsDictionary::iterator sample_result::begin() {
  auto iter = sampleResults.find(GlobalRegisterName);
  if (iter == sampleResults.end()) {
    throw std::runtime_error(
        "There is no global counts dictionary in this sample_result.");
  }

  return iter->second.counts.begin();
}

CountsDictionary::iterator sample_result::end() {
  auto iter = sampleResults.find(GlobalRegisterName);
  if (iter == sampleResults.end()) {
    throw std::runtime_error(
        "There is no global counts dictionary in this sample_result.");
  }

  return iter->second.counts.end();
}

CountsDictionary::const_iterator sample_result::cbegin() const {
  auto iter = sampleResults.find(GlobalRegisterName);
  if (iter == sampleResults.end()) {
    throw std::runtime_error(
        "There is no global counts dictionary in this sample_result.");
  }

  return iter->second.counts.cbegin();
}

CountsDictionary::const_iterator sample_result::cend() const {
  auto iter = sampleResults.find(GlobalRegisterName);
  if (iter == sampleResults.end()) {
    throw std::runtime_error(
        "There is no global counts dictionary in this sample_result.");
  }

  return iter->second.counts.cend();
}

std::size_t sample_result::size(const std::string_view registerName) noexcept {
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return 0;

  return iter->second.counts.size();
}

double sample_result::probability(std::string_view bitStr,
                                  const std::string_view registerName) const {
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return 0.0;

  const auto countIter = iter->second.counts.find(bitStr.data());
  return (countIter == iter->second.counts.end())
             ? 0.0
             : (double)countIter->second / totalShots;
}

std::size_t sample_result::count(std::string_view bitStr,
                                 const std::string_view registerName) {
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return 0;

  return iter->second.counts[bitStr.data()];
}

std::string sample_result::most_probable(const std::string_view registerName) {
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    throw std::runtime_error(
        "[sample_result::most_probable] invalid sample result register name (" +
        std::string(registerName) + ")");
  auto counts = iter->second.counts;
  return std::max_element(counts.begin(), counts.end(),
                          [](const auto &el1, const auto &el2) {
                            return el1.second < el2.second;
                          })
      ->first;
}

bool sample_result::has_expectation(const std::string_view registerName) const {
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return false;

  return iter->second.expectationValue.has_value();
}

double sample_result::expectation(const std::string_view registerName) const {
  double aver = 0.0;
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return 0.0;

  if (iter->second.expectationValue.has_value())
    return iter->second.expectationValue.value();

  auto counts = iter->second.counts;
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

double sample_result::exp_val_z(const std::string_view registerName) {
  double aver = 0.0;
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return 0.0;

  if (iter->second.expectationValue.has_value())
    return iter->second.expectationValue.value();

  auto counts = iter->second.counts;
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

  return ret;
}

CountsDictionary
sample_result::to_map(const std::string_view registerName) const {
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return CountsDictionary();

  return iter->second.counts;
}

sample_result
sample_result::get_marginal(const std::vector<std::size_t> &marginalIndices,
                            const std::string_view registerName) {
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return sample_result();

  auto counts = iter->second.counts;
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

void sample_result::dump(std::ostream &os) const {
  os << "{ ";
  if (sampleResults.size() > 1) {
    os << "\n  ";
    std::size_t counter = 0;
    for (auto &result : sampleResults) {
      os << result.first << " : { ";
      for (auto &kv : result.second.counts) {
        os << kv.first << ":" << kv.second << " ";
      }
      bool isLast = counter == sampleResults.size() - 1;
      counter++;
      os << "}\n" << (!isLast ? "   " : "");
    }

  } else if (sampleResults.size() == 1) {

    CountsDictionary counts;
    auto iter = sampleResults.find(GlobalRegisterName);
    if (iter != sampleResults.end())
      counts = iter->second.counts;
    else {
      auto first = sampleResults.begin();
      os << "\n   " << first->first << " : { ";
      counts = sampleResults.begin()->second.counts;
    }

    for (auto &kv : counts) {
      os << kv.first << ":" << kv.second << " ";
    }

    if (iter == sampleResults.end())
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
  auto iter = sampleResults.find(registerName.data());
  if (iter == sampleResults.end())
    return;

  // First process the counts
  CountsDictionary newCounts;
  for (auto [bits, count] : iter->second.counts) {
    if (idx.size() != bits.size())
      throw std::runtime_error("Calling reorder() with invalid parameter idx");

    std::string newBits(bits);
    int i = 0;
    for (auto oldIdx : idx)
      newBits[i++] = bits[oldIdx];
    newCounts[newBits] = count;
  }
  iter->second.counts = newCounts;

  // Now process the sequential data
  for (auto &s : iter->second.sequentialData) {
    std::string newBits(s);
    int i = 0;
    for (auto oldIdx : idx)
      newBits[i++] = s[oldIdx];
    s = newBits;
  }
}
} // namespace cudaq
