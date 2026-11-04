/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_LOGICAL_ANALYSIS_EDEMFAULTS_H
#define CUDAQ_LOGICAL_ANALYSIS_EDEMFAULTS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cudaq::logical::edem_analysis {

// This is an external analysis input, not canonical MLIR. A fault column may
// contain several correlated effects; categorical groups remain one site.
enum class Pauli { x, y, z };

struct PauliFactor {
  std::size_t carrier;
  Pauli basis;
};

struct PauliInsertion {
  // Empty means the input boundary. Otherwise identify one flattened typed
  // projection instruction; insertion occurs immediately after that site.
  std::string afterSite;
  std::vector<PauliFactor> factors;
};

struct ElementaryFault {
  std::string id;
  std::vector<PauliInsertion> insertions;
  std::vector<std::string> recordFlips;
};

struct FaultInput {
  std::vector<ElementaryFault> faults;
  std::vector<std::vector<std::size_t>> exclusiveGroups;
  std::vector<double> probabilities;
};

// Versioned transport for native analysis and the Python frontend. Parsing
// validates structure and individual probability bounds. EDEM owns group
// totals.
llvm::Expected<FaultInput> parseFaultInput(llvm::StringRef json);

} // namespace cudaq::logical::edem_analysis

#endif // CUDAQ_LOGICAL_ANALYSIS_EDEMFAULTS_H
