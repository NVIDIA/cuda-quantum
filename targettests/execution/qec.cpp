/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -Wl,--export-dynamic --target quantinuum --emulate %s -o %t && %t | FileCheck %s

#include <cstdio>
#include <cudaq.h>

extern "C" void reset_decoder(int64_t decoder_id);
extern "C" void enqueue_syndrome(int64_t block_id, int64_t packed_mz_results,
                                 int64_t last_syndrome);
extern "C" int64_t get_correction(int64_t block_id);

void reset_decoder(int64_t decoder_id) {
  // For local emulation, form and send network packet.
  return;
}

__qpu__ void enqueue_syndrome(int64_t block_id, int64_t packed_mz_results,
                      int64_t last_syndrome) {
  // For local emulation, form and send network packet.
  return;
}

__qpu__ int64_t get_correction(int64_t block_id) {
  // For local emulation, form and send network packet.
  return 0l;
}

__qpu__ int64_t to_integer(std::vector<bool> &results) {
  //  int64_t ret = 0;
  //  int i = 0;
  //  for (int i =0; i < results.size(); ++i) {
  //   ret &= (int64_t)(results[i]) << (int64_t)i;
  //   i++;
  //  }
  // return ret;
  return (int64_t)(results[0]) + (int64_t)(results[1]);
}

__qpu__ int64_t test(int num_iterations) {
  cudaq::qvector q(6);
  //cudaq::device_call(reset_decoder, /*decoder_id=*/0l);
  for (int i = 0; i < num_iterations; i++) {
    auto mz_results = mz(q);
    int64_t syndrome = to_integer(mz_results);
    //int64_t syndrome = cudaq::to_integer(mz_results);
    bool last_iter = i == num_iterations - 1;
    enqueue_syndrome( /*block_id=*/0l, syndrome,
                   /*last_syndrome=*/static_cast<int64_t>(last_iter));

    //cudaq::device_call(enqueue_syndrome, /*block_id=*/0l, syndrome,
    //                   /*last_syndrome=*/static_cast<int64_t>(last_iter));
  }
  int64_t correction = get_correction(/*block_id=*/0l);
  //int64_t correction = cudaq::device_call(get_correction, /*block_id=*/0l);
  return correction;
}

int main() {
  auto myresults = cudaq::run(1, test, /*num_iterations=*/3);
  return 0;
}