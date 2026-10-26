
/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CUDAQTestUtils.h"
#include "cudaq/algorithm.h"

#include "cudaq/utils/cudaq_utils.h"
#include <cpr/cpr.h>
#include <fstream>
#include <iostream>
#include <string>

void readConfig(std::string ref_filename, std::vector<std::string> &config) {
  std::ifstream file(ref_filename);

  if (!file) {
    FAIL() << "Reference file not found: " << ref_filename;
  }

  config.clear();
  std::string line;
  for (uint i = 0; i < 3 && std::getline(file, line); i++) {
    config.emplace_back(line);
  }
}

bool configureMockServer(std::string ref_filename) {
  std::vector<std::string> config;
  readConfig(ref_filename, config);
  std::string api_token = "good_access_token";

  cpr::Response r;
  r = cpr::Get(cpr::Url{"http://0.0.0.0:62443/config/qa/qpu"},
               cpr::Parameters{{"qpu", config[0]}}, cpr::Bearer{api_token});
  if (r.status_code != 200) {
    return false;
  }

  if (!config[1].empty()) {
    r = cpr::Get(cpr::Url{"http://0.0.0.0:62443/config/qa/bad-cz-gates"},
                 cpr::Parameters{{"loci_list", config[1]}},
                 cpr::Bearer{api_token});
    if (r.status_code != 200) {
      return false;
    }
  }

  if (!config[2].empty()) {
    r = cpr::Get(cpr::Url{"http://0.0.0.0:62443/config/qa/bad-prx-gates"},
                 cpr::Parameters{{"loci_list", config[2]}},
                 cpr::Bearer{api_token});
    if (r.status_code != 200) {
      return false;
    }
  }

  return true;
}

bool compareDQA(std::string ref_filename, std::string dqa_filename) {
  std::ifstream ref_file(ref_filename);
  std::ifstream dqa_file(dqa_filename);
  std::string line1, line2;
  uint line;

  // Skip the test configuration in the first 3 lines of the reference file.
  for (line = 0; ref_file && line < 3; line++)
    std::getline(ref_file, line1);
  // Skip the first line of the DQA file which contains qc specific info.
  for (line = 0; dqa_file && line < 1; line++)
    std::getline(dqa_file, line2);

  // Rest of both files must be identical.
  while (!ref_file.eof() && !dqa_file.eof()) {
    std::getline(ref_file, line1);
    std::getline(dqa_file, line2);
    if (line1 != line2) {
      return false;
    }
  }
  return ref_file.eof() && dqa_file.eof();
}

void runRefFile(std::string ref_filename) {
  bool status = configureMockServer(ref_filename);
  ASSERT_TRUE(status) << "Configuring the mock server failed";

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  auto counts = cudaq::sample(kernel);
  EXPECT_EQ(counts.size(), 2);

  status = compareDQA(ref_filename, "dqa_received.txt");
  EXPECT_TRUE(status) << "The calculated quantum architecture is different";
}

/* Crystal-20 architecture */

CUDAQ_TEST(IQMTester, crystal_20_split8_12) {
  runRefFile("dqa_crystal-20_split8-12.txt");
}

CUDAQ_TEST(IQMTester, crystal_20_split9_10) {
  runRefFile("dqa_crystal-20_split9-10.txt");
}

CUDAQ_TEST(IQMTester, crystal_20_split10_10) {
  runRefFile("dqa_crystal-20_split10-10.txt");
}

CUDAQ_TEST(IQMTester, crystal_20_split3_8_2_2_5) {
  runRefFile("dqa_crystal-20_split-3-8-2-2-5.txt");
}

CUDAQ_TEST(IQMTester, crystal_20_vsnake17) {
  runRefFile("dqa_crystal-20_snake17.txt");
}

CUDAQ_TEST(IQMTester, crystal_20_hsnake18) {
  runRefFile("dqa_crystal-20_snake18.txt");
}

CUDAQ_TEST(IQMTester, crystal_20_vsnake20) {
  runRefFile("dqa_crystal-20_snake20.txt");
}

/* Crystal-54 architecture */

CUDAQ_TEST(IQMTester, crystal_54_split31_23) {
  runRefFile("dqa_crystal-54_split31-23.txt");
}

CUDAQ_TEST(IQMTester, crystal_54_split23_22) {
  runRefFile("dqa_crystal-54_split23-22.txt");
}

CUDAQ_TEST(IQMTester, crystal_54_split22_22) {
  runRefFile("dqa_crystal-54_split22-22.txt");
}

CUDAQ_TEST(IQMTester, crystal_54_halo32) {
  runRefFile("dqa_crystal-54_halo32.txt");
}

CUDAQ_TEST(IQMTester, crystal_54_halo33) {
  runRefFile("dqa_crystal-54_halo33.txt");
}

CUDAQ_TEST(IQMTester, crystal_54_hsnake50) {
  runRefFile("dqa_crystal-54_hsnake50.txt");
}

CUDAQ_TEST(IQMTester, crystal_54_vsnake49) {
  runRefFile("dqa_crystal-54_vsnake49.txt");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleMock(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
