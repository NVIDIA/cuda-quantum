/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/NamedVariantStore.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

using Store = cudaq::detail::NamedVariantStore<int, double, std::string>;

TEST(NamedVariantStoreTester, GetFindsMatchingTypeAfterSameNamedEntry) {
  Store store;
  store.add("artifact", 3.14);
  store.add("artifact", std::string("payload"));
  store.add("other", 7);

  const auto *stringValue = store.get<std::string>("artifact");
  ASSERT_NE(stringValue, nullptr);
  EXPECT_EQ(*stringValue, "payload");

  const auto *doubleValue = store.get<double>("artifact");
  ASSERT_NE(doubleValue, nullptr);
  EXPECT_DOUBLE_EQ(*doubleValue, 3.14);

  EXPECT_EQ(store.get<int>("artifact"), nullptr);
}

TEST(NamedVariantStoreTester, RejectsDuplicateNameAndTypeButAllowsOthers) {
  Store store;
  store.add("artifact", 7);

  try {
    store.add("artifact", 9);
    FAIL() << "Expected duplicate name/type insertion to throw.";
  } catch (const std::runtime_error &err) {
    EXPECT_NE(std::string(err.what()).find("artifact"), std::string::npos);
  }

  EXPECT_NO_THROW(store.add("artifact", 2.5));
  EXPECT_NO_THROW(store.add("other", 9));
}

TEST(NamedVariantStoreTester, GetAllOfTypePreservesInsertionOrder) {
  Store store;
  store.add("alpha", std::string("a"));
  store.add("skip", 1);
  store.add("beta", std::string("b"));
  store.add("skip-again", 2.0);
  store.add("gamma", std::string("c"));

  std::vector<std::string> names;
  std::vector<std::string> values;
  for (const auto &[name, value] : store.getAllOfType<std::string>()) {
    names.push_back(name);
    values.push_back(value);
  }

  EXPECT_EQ(names, (std::vector<std::string>{"alpha", "beta", "gamma"}));
  EXPECT_EQ(values, (std::vector<std::string>{"a", "b", "c"}));
}
