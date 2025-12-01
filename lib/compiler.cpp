/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/nvqlink/compiler.h"
#include "cudaq/nvqlink/rt_host.h"

#include <regex>
#include <sstream>
#include <string>

INSTANTIATE_REGISTRY_NO_ARGS(cudaq::nvqlink::compiler);
INSTANTIATE_REGISTRY(cudaq::nvqlink::rt_host, cudaq::nvqlink::lqpu&);

namespace cudaq::nvqlink::details {

std::string removeNonEntrypointFunctions(const std::string &mlirCode) {
  std::string result = mlirCode;

  // Pattern to match func.func or llvm.func operations
  // This handles both single-line declarations and multi-line definitions
  std::regex funcPattern(
      R"(((?:func\.func|llvm\.func)\s+[^{]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|(?:\s*\n|$)))");

  std::sregex_iterator iter(result.begin(), result.end(), funcPattern);
  std::sregex_iterator end;

  std::vector<std::pair<size_t, size_t>> toRemove;

  // Find all function matches and identify which ones to remove
  for (auto it = iter; it != end; ++it) {
    std::string match = it->str();

    // Skip if this is the entrypoint function (contains "cudaq-entrypoint")
    if (match.find("\"cudaq-entrypoint\"") != std::string::npos) {
      continue;
    }

    // Mark this function for removal
    toRemove.push_back({it->position(), it->position() + it->length()});
  }

  // Remove functions in reverse order to maintain valid positions
  for (auto rit = toRemove.rbegin(); rit != toRemove.rend(); ++rit) {
    result.erase(rit->first, rit->second - rit->first);
  }

  // Clean up any extra newlines that might be left
  result = std::regex_replace(result, std::regex(R"(\n\s*\n\s*\n)"), "\n\n");

  return result;
}

// Alternative approach using manual parsing for more control
std::string removeNonEntrypointFunctionsManual(const std::string &mlirCode) {
  std::istringstream iss(mlirCode);
  std::ostringstream result;
  std::string line;
  bool inEntrypointFunction = false;
  bool inOtherFunction = false;
  int braceLevel = 0;
  std::string currentFunction;

  while (std::getline(iss, line)) {
    // Check if this line starts a function
    if (line.find("func.func") == 0 || line.find("llvm.func") == 0) {
      // Check if this is the entrypoint function
      if (line.find("\"cudaq-entrypoint\"") != std::string::npos) {
        inEntrypointFunction = true;
        result << line << "\n";
        braceLevel = 0;
        // Count braces in the current line
        for (char c : line) {
          if (c == '{')
            braceLevel++;
          else if (c == '}')
            braceLevel--;
        }
      } else {
        // This is a function we want to remove
        inOtherFunction = true;
        braceLevel = 0;
        // Count braces in the current line
        for (char c : line) {
          if (c == '{')
            braceLevel++;
          else if (c == '}')
            braceLevel--;
        }
        // If it's a single-line declaration (braceLevel == 0), skip it entirely
        if (braceLevel == 0) {
          inOtherFunction = false;
        }
      }
    }
    // Handle lines within functions
    else if (inEntrypointFunction) {
      result << line << "\n";
      // Count braces to know when function ends
      for (char c : line) {
        if (c == '{')
          braceLevel++;
        else if (c == '}')
          braceLevel--;
      }
      if (braceLevel == 0) {
        inEntrypointFunction = false;
      }
    } else if (inOtherFunction) {
      // Count braces to know when function ends
      for (char c : line) {
        if (c == '{')
          braceLevel++;
        else if (c == '}')
          braceLevel--;
      }
      if (braceLevel == 0) {
        inOtherFunction = false;
      }
      // Don't add this line to result
    }
    // Handle lines outside functions (module attributes, etc.)
    else {
      result << line << "\n";
    }
  }

  return result.str();
}

} // namespace cudaq::nvqlink::details
