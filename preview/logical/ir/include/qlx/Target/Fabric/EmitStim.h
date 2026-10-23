/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_TRANSLATE_FABRIC_EMITSTIM_H
#define QLX_TRANSLATE_FABRIC_EMITSTIM_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace cudaq::logical {

/// A carrier-level instruction from the selected, fully unrolled Clifford
/// realization. Binary gates store control/target pairs in `carriers`; MPP
/// stores its ordered physical Pauli factors in `paulis`. `measurementStart`
/// is the first absolute record index for a measurement instruction.
struct ProjectionInstruction {
  enum class Kind : std::uint8_t {
    R,
    RX,
    H,
    S,
    SDag,
    X,
    Z,
    CX,
    CZ,
    Swap,
    M,
    MX,
    MPP,
  };
  enum class Pauli : std::uint8_t { X, Y, Z };
  struct PauliFactor {
    unsigned carrier;
    Pauli pauli;
  };

  Kind kind;
  std::vector<unsigned> carriers;
  std::vector<PauliFactor> paulis;
  bool inverted = false;
  unsigned measurementStart = 0;
  /// Qualified call/repeat path, source operation ordinal, and subinstruction.
  std::string siteId;
};

struct ProjectionRecord {
  std::string path;
  unsigned measurementIndex;
};

/// One encoded endpoint in the selected GadgetSpec's port order.
struct ProjectionPort {
  unsigned ordinal;
  std::string name;
  std::string direction;
  unsigned patchId;
  std::vector<std::pair<std::string, std::vector<unsigned>>> partitions;
};

/// One physical measurement, including aliases that do not measure again.
struct ProjectionMeasurement {
  unsigned index;
  std::string record;
  std::string symbol;
  std::vector<std::string> callPath;
  std::string field;
  unsigned lane;
  std::vector<unsigned> carriers;
  std::vector<std::string> aliases;
};

struct Projection {
  std::vector<ProjectionInstruction> instructions;
  /// Includes physical measurement records and non-measuring aliases.
  std::vector<ProjectionRecord> records;
  /// Exactly one stable path per physical measurement in emission order.
  std::vector<std::string> recordOrder;
  /// Semantic input and output patch endpoints, each in GadgetSpec port order.
  std::vector<std::vector<unsigned>> inputDataCarriersByPort;
  std::vector<std::vector<unsigned>> outputDataCarriersByPort;
  std::vector<ProjectionPort> inputPorts;
  std::vector<ProjectionPort> outputPorts;
  std::vector<ProjectionMeasurement> measurements;
  unsigned carrierCount = 0;
  unsigned measurementCount = 0;
};

/// Resolve a selected Fabric gadget to typed carrier-level operations.
/// Uses the same traversal and carrier allocation as `emitStim`; no Stim text
/// is parsed. Unsupported or dynamically controlled realizations fail closed.
/// Requires a linked GadgetSpec and a stable name for every measurement.
/// Site IDs are deterministic for a fixed realization, not across IR rewrites.
/// Does not mutate the module or verify detector-contract correctness.
mlir::FailureOr<Projection> projectClifford(mlir::ModuleOp module,
                                            llvm::StringRef gadgetSymbol);

/// Convenience form requiring exactly one entry gadget, like `emitStim`.
mlir::FailureOr<Projection> projectClifford(mlir::ModuleOp module);

} // namespace cudaq::logical

namespace qlx::fabric {

/// Translate a Fabric module to Stim circuit text.
///
/// Walks the entry gadget, resolves patch partitions to absolute qubit
/// indices, tracks raw measurement records, and writes standards-compatible
/// circuit instructions to \p os.
mlir::LogicalResult emitStim(mlir::ModuleOp module, llvm::raw_ostream &os);

/// Register `--fabric-to-stim` with MLIR's translation registry.
void registerFabricToStimTranslation();

} // namespace qlx::fabric

#endif // QLX_TRANSLATE_FABRIC_EMITSTIM_H
