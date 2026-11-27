/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/qpu_utils.h"
#include <gtest/gtest.h>
#include <stdexcept>

TEST(Base64Tester, decodesBase64) {
  EXPECT_EQ(cudaq::detail::decodeBase64(""), "");
  EXPECT_EQ(cudaq::detail::decodeBase64("Zg=="), "f");
  EXPECT_EQ(cudaq::detail::decodeBase64("Zm8="), "fo");
  EXPECT_EQ(cudaq::detail::decodeBase64("Zm9v"), "foo");
  EXPECT_EQ(cudaq::detail::decodeBase64("Zm9vYg=="), "foob");
  EXPECT_EQ(cudaq::detail::decodeBase64("+/8="), "\xfb\xff");
}

TEST(Base64Tester, rejectsMalformedBase64) {
  for (const char *bad : {"Zg", "Z===", "Zg=A", "Zg==Zg=="})
    EXPECT_THROW(cudaq::detail::decodeBase64(bad), std::runtime_error) << bad;
}
