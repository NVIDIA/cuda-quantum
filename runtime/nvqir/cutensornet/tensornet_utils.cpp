/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "tensornet_utils.h"
#include "common/FmtCore.h"
#include "common/Logger.h"

std::vector<double> randomValues(uint64_t num_samples, double max_value,
                                 std::mt19937 &randomEngine) {
  std::vector<double> rs;
  rs.reserve(num_samples);
  std::uniform_real_distribution<double> distr(0.0, max_value);
  for (uint64_t i = 0; i < num_samples; ++i) {
    rs.emplace_back(distr(randomEngine));
  }
  std::sort(rs.begin(), rs.end());
  return rs;
}

ScratchDeviceMem::ScratchDeviceMem() {
  if (auto *scratchSizePercent =
          std::getenv("CUDAQ_TENSORNET_SCRATCH_SIZE_PERCENTAGE")) {
    auto envIntVal = atoi(scratchSizePercent);
    // Bound the allowed value between 5% and 95%.
    // Note: values near these limits, while allowed, may still exhibit
    // instability, e.g., not enough workspace (if the scratch size is too
    // small).
    constexpr int minScratchSizePercent = 5;
    constexpr int maxScratchSizePercent = 95;

    if (envIntVal < minScratchSizePercent || envIntVal > maxScratchSizePercent)
      throw std::runtime_error(fmt::format(
          "Invalid CUDAQ_TENSORNET_SCRATCH_SIZE_PERCENTAGE environment "
          "variable setting. Expecting a "
          "positive integer value between {} and {}, got '{}'.",
          minScratchSizePercent, maxScratchSizePercent, scratchSizePercent));

    freeMemRatio = static_cast<double>(envIntVal) / 100.0;
    CUDAQ_INFO("Setting scratch size ratio to {}.", freeMemRatio);
  }
}

// Compute the scratch size to allocate.
void ScratchDeviceMem::computeScratchSize() {
  // Query the free memory on Device
  std::size_t freeSize{0}, totalSize{0};
  HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
  scratchSize =
      (freeSize - (freeSize % 4096)) *
      freeMemRatio; // use a set proportion available memory with alignment
}

// Allocate scratch device memory based on available memory
void ScratchDeviceMem::allocate() {
  if (d_scratch)
    throw std::runtime_error(
        "Multiple scratch device memory allocations is not allowed.");

  computeScratchSize();
  // Try allocate device memory
  auto errCode = cudaMalloc(&d_scratch, scratchSize);
  if (errCode == cudaErrorMemoryAllocation) {
    // This indicates race condition whereby other GPU code is allocating
    // memory while we are calling cudaMemGetInfo.
    // Attempt to redo the allocation with an updated cudaMemGetInfo data.
    computeScratchSize();
    HANDLE_CUDA_ERROR(cudaMalloc(&d_scratch, scratchSize));
  } else {
    HANDLE_CUDA_ERROR(errCode);
  }
}

ScratchDeviceMem::~ScratchDeviceMem() {
  if (scratchSize > 0)
    HANDLE_CUDA_ERROR(cudaFree(d_scratch));
}
