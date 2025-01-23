/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/RecordLogger.h"
#include <cudaq.h>

CUDAQ_TEST(LoggerTester, checkBoolean) {
  std::vector<bool> results = {true, false, false, true};

  const std::string expectedLog =
      "HEADER\tschema_name\tordered\nHEADER\tschema_version\t1.0\n"
      "START\nOUTPUT\tBOOL\ttrue\nEND\t0\nSTART\nOUTPUT\tBOOL\tfalse\nEND\t0\n"
      "START\nOUTPUT\tBOOL\tfalse\nEND\t0\nSTART\nOUTPUT\tBOOL\ttrue\nEND\t0\n";

  cudaq::RecordLogger logger;
  logger.log(results);
  EXPECT_TRUE(expectedLog == logger.getLog());
}