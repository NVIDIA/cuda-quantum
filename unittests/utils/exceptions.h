/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Additional test API that can catch and verify exceptions in code compiled
/// with `libstdc++` that were thrown in code compiled with `libc++`.
#include "gtest/gtest.h"
#define GTEST_TEST_THROW_MIXED_STDLIB(statement, expected_exception, fail)     \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                \
  if (::testing::internal::TrueWithString gtest_msg{}) {                       \
    bool gtest_caught_expected = false;                                        \
    try {                                                                      \
      GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);               \
    } catch (expected_exception const &) {                                     \
      gtest_caught_expected = true;                                            \
    }                                                                          \
    GTEST_TEST_THROW_CATCH_STD_EXCEPTION_(statement, expected_exception)       \
    catch (...) {                                                              \
      std::string exType = __cxxabiv1::__cxa_current_exception_type()->name(); \
      auto demangledPtr = __cxxabiv1::__cxa_demangle(exType.c_str(), nullptr,  \
                                                     nullptr, nullptr);        \
      if (demangledPtr) {                                                      \
        std::string demangledName(demangledPtr);                               \
        if (demangledName == #expected_exception) {                            \
          gtest_caught_expected = true;                                        \
        } else {                                                               \
          gtest_msg.value =                                                    \
              "Expected: " #statement                                          \
              " throws an exception of type " #expected_exception              \
              ".\n  Actual: it throws a different type: " +                    \
              demangledName;                                                   \
          goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);          \
        }                                                                      \
      } else {                                                                 \
        gtest_msg.value =                                                      \
            "Expected: " #statement                                            \
            " throws an exception of type " #expected_exception                \
            ".\n  Actual (cannot demangle): it throws a different type: " +    \
            exType;                                                            \
        goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);            \
      }                                                                        \
    }                                                                          \
    if (!gtest_caught_expected) {                                              \
      gtest_msg.value = "Expected: " #statement                                \
                        " throws an exception of type " #expected_exception    \
                        ".\n  Actual: it throws nothing.";                     \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);              \
    }                                                                          \
  } else /*NOLINT*/                                                            \
    GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__)                      \
        : fail(gtest_msg.value.c_str())

#define EXPECT_THROW_MIXED_STDLIB(statement, expected_exception)               \
  GTEST_TEST_THROW_MIXED_STDLIB(statement, expected_exception,                 \
                                GTEST_NONFATAL_FAILURE_)
