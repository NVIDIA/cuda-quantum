/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Analysis/GadgetEDEM.h"
#include "qlx/Analysis/CodePresentation.h"
#include "qlx/Analysis/EdemFaults.h"
#include "qlx/Analysis/IdealOutcomes.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/JSON.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "cudaq/edem/core.h"
#include "mlir/IR/Verifier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mlir;

namespace cudaq::logical {
namespace {

using Bits = llvm::SmallBitVector;

struct BoundaryPort {
  CodePresentation boundary;
  std::vector<unsigned> carriers;
};

struct ParityRow {
  Bits records;
  Bits virtualInputs;
  bool constant = false;
};

struct ProfileRows {
  std::vector<ParityRow> detectors;
  std::vector<ParityRow> observables;
  std::vector<ParityRow> success;
  std::vector<ParityRow> outputVirtual;
};

unsigned binaryRank(std::vector<Bits> rows, unsigned columns) {
  unsigned rank = 0;
  for (unsigned column = 0; column < columns && rank < rows.size(); ++column) {
    auto pivot =
        std::find_if(rows.begin() + rank, rows.end(),
                     [column](const Bits &row) { return row.test(column); });
    if (pivot == rows.end())
      continue;
    std::iter_swap(rows.begin() + rank, pivot);
    for (unsigned row = rank + 1; row < rows.size(); ++row)
      if (rows[row].test(column))
        rows[row] ^= rows[rank];
    ++rank;
  }
  return rank;
}

FailureOr<std::vector<BoundaryPort>>
readPorts(qlx::fabric::GadgetContractOp profile,
          std::optional<ArrayAttr> values,
          const std::vector<std::vector<unsigned>> &carriers, StringRef label) {
  if (!values || values->size() != carriers.size()) {
    profile.emitOpError() << "EDEM " << label
                          << " must exactly cover the projected endpoints";
    return failure();
  }
  std::vector<BoundaryPort> result;
  result.reserve(values->size());
  for (auto [ordinal, raw] : llvm::enumerate(*values)) {
    auto entry = dyn_cast<DictionaryAttr>(raw);
    auto index = entry ? entry.getAs<IntegerAttr>("endpoint") : IntegerAttr{};
    auto ref =
        entry ? entry.getAs<FlatSymbolRefAttr>("profile") : FlatSymbolRefAttr{};
    if (!index || index.getInt() != static_cast<int64_t>(ordinal) || !ref) {
      profile.emitOpError()
          << "EDEM " << label << " requires ordered endpoint and code profile";
      return failure();
    }
    auto codeProfile = dyn_cast_or_null<qlx::fabric::CodeProfileOp>(
        SymbolTable::lookupNearestSymbolFrom(profile, ref));
    if (!codeProfile) {
      profile.emitOpError() << "EDEM " << label << " endpoint " << ordinal
                            << " has an unresolved code profile";
      return failure();
    }
    auto boundary = CodePresentation::fromProfile(codeProfile);
    if (failed(boundary))
      return failure();
    if (carriers[ordinal].size() != boundary->physicalQubits()) {
      profile.emitOpError()
          << "EDEM " << label << " endpoint " << ordinal << " has "
          << carriers[ordinal].size() << " projected data carriers, expected "
          << boundary->physicalQubits();
      return failure();
    }
    result.push_back({std::move(*boundary), carriers[ordinal]});
  }
  return result;
}

FailureOr<std::unordered_map<std::string, unsigned>>
readRecordMap(qlx::fabric::GadgetContractOp profile,
              const Projection &projection) {
  auto gadget = dyn_cast_or_null<qlx::fabric::GadgetOp>(
      SymbolTable::lookupNearestSymbolFrom(profile, profile.getGadgetAttr()));
  if (!gadget || !gadget.getSpecAttr()) {
    profile.emitOpError("EDEM requires a linked gadget and GadgetSpec");
    return failure();
  }
  auto spec = dyn_cast_or_null<qlx::fabric::GadgetSpecOp>(
      SymbolTable::lookupNearestSymbolFrom(gadget, gadget.getSpecAttr()));
  if (!spec) {
    profile.emitOpError("EDEM requires a linked GadgetSpec");
    return failure();
  }
  // Public Logical's record_schema includes non-measuring aliases. The
  // verified typed projection, rather than declaration order, owns physical
  // measurement indices. Never allocate columns for those aliases.
  if (projection.recordOrder.size() != projection.measurementCount) {
    profile.emitOpError(
        "EDEM projection has an inconsistent measurement manifest");
    return failure();
  }
  std::unordered_map<std::string, unsigned> result;
  for (const auto &record : projection.records) {
    if (record.path.empty() ||
        record.measurementIndex >= projection.measurementCount) {
      profile.emitOpError("EDEM projection has an invalid stable record");
      return failure();
    }
    auto [it, inserted] = result.emplace(record.path, record.measurementIndex);
    if (!inserted && it->second != record.measurementIndex) {
      profile.emitOpError() << "EDEM record path " << record.path
                            << " has multiple measurement producers";
      return failure();
    }
  }
  for (auto [ordinal, path] : llvm::enumerate(projection.recordOrder))
    if (result.find(path) == result.end() || result[path] != ordinal) {
      profile.emitOpError()
          << "EDEM projection omits stable measurement " << path;
      return failure();
    }
  return result;
}

FailureOr<ParityRow>
readParity(Operation *owner, StringRef gadgetName,
           const std::unordered_map<std::string, unsigned> &recordMap,
           unsigned measurementCount, const std::vector<unsigned> &inputOffsets,
           unsigned inputSyndromes) {
  ParityRow result{Bits(measurementCount), Bits(inputSyndromes), false};
  if (auto rawRecords = owner->getAttrOfType<ArrayAttr>("records")) {
    for (Attribute raw : rawRecords) {
      auto text = dyn_cast<StringAttr>(raw);
      if (!text) {
        owner->emitOpError("EDEM parity record must be a stable string path");
        return failure();
      }
      StringRef path = text.getValue();
      if (!path.consume_front(gadgetName) || !path.consume_front(".")) {
        owner->emitOpError("EDEM parity record belongs to another gadget");
        return failure();
      }
      auto it = recordMap.find(path.str());
      if (it == recordMap.end()) {
        owner->emitOpError() << "EDEM parity record '" << text.getValue()
                             << "' has no exact projected producer";
        return failure();
      }
      result.records.flip(it->second);
    }
  }
  if (auto rawTerms = owner->getAttrOfType<ArrayAttr>("input_syndromes")) {
    for (Attribute raw : rawTerms) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto port =
          entry ? entry.getAs<IntegerAttr>("port_index") : IntegerAttr{};
      auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr{};
      if (!port || !index || port.getInt() < 0 || index.getInt() < 0 ||
          static_cast<std::size_t>(port.getInt()) + 1 >= inputOffsets.size() ||
          inputOffsets[port.getInt()] + index.getInt() >=
              inputOffsets[port.getInt() + 1]) {
        owner->emitOpError("EDEM parity has an invalid input syndrome term");
        return failure();
      }
      result.virtualInputs.flip(inputOffsets[port.getInt()] + index.getInt());
    }
  }
  if (auto constant = owner->getAttrOfType<BoolAttr>("constant"))
    result.constant = constant.getValue();
  return result;
}

FailureOr<ProfileRows>
readProfileRows(qlx::fabric::GadgetContractOp profile,
                const Projection &projection,
                const std::unordered_map<std::string, unsigned> &recordMap,
                const std::vector<BoundaryPort> &inputs,
                const std::vector<BoundaryPort> &outputs) {
  std::vector<unsigned> inputOffsets{0};
  for (const auto &port : inputs)
    inputOffsets.push_back(inputOffsets.back() +
                           port.boundary.effectiveSyndromes());
  unsigned inputSyndromes = inputOffsets.back();
  std::vector<unsigned> outputOffsets{0};
  for (const auto &port : outputs)
    outputOffsets.push_back(outputOffsets.back() +
                            port.boundary.effectiveSyndromes());
  std::vector<std::optional<ParityRow>> assignments(outputOffsets.back());
  ProfileRows result;
  StringRef gadgetName = profile.getGadget();
  for (Operation &operation : profile.getBody().front()) {
    if (!isa<qlx::fabric::DetectorOp, qlx::fabric::ObservableOp,
             qlx::fabric::SuccessOp, qlx::fabric::OutputSyndromeOp>(&operation))
      continue;
    auto parity =
        readParity(&operation, gadgetName, recordMap,
                   projection.measurementCount, inputOffsets, inputSyndromes);
    if (failed(parity))
      return failure();
    if (isa<qlx::fabric::DetectorOp>(&operation))
      result.detectors.push_back(std::move(*parity));
    else if (auto observable =
                 dyn_cast<qlx::fabric::ObservableOp>(&operation)) {
      if (!std::cmp_equal(observable.getIdx(), result.observables.size())) {
        observable.emitOpError("EDEM observable indices must be ordered");
        return failure();
      }
      result.observables.push_back(std::move(*parity));
    } else if (isa<qlx::fabric::SuccessOp>(&operation))
      result.success.push_back(std::move(*parity));
    else if (auto assignment =
                 dyn_cast<qlx::fabric::OutputSyndromeOp>(&operation)) {
      auto port = assignment.getPortIndexAttr().getInt();
      auto index = assignment.getIndexAttr().getInt();
      if (port < 0 || index < 0 ||
          std::cmp_greater_equal(port, outputs.size()) ||
          std::cmp_greater_equal(index, outputOffsets[port + 1] -
                                            outputOffsets[port]) ||
          assignments[outputOffsets[port] + index]) {
        assignment.emitOpError(
            "EDEM output syndrome target is invalid or repeated");
        return failure();
      }
      assignments[outputOffsets[port] + index] = std::move(*parity);
    }
  }
  for (auto &assignment : assignments) {
    if (!assignment) {
      profile.emitOpError("EDEM requires complete output syndrome assignments");
      return failure();
    }
    result.outputVirtual.push_back(std::move(*assignment));
  }
  return result;
}

struct OutcomeRow {
  Bits support;
  bool constant;
};

IdealBoundaryPort idealPort(const BoundaryPort &port) {
  IdealBoundaryPort result;
  result.carriers = port.carriers;
  auto copy = [](auto source, auto &target) {
    target.assign(source.begin(), source.end());
  };
  copy(port.boundary.keptStabilizerBasis(), result.keptStabilizers);
  copy(port.boundary.logicalXBasis(), result.logicalX);
  copy(port.boundary.logicalZBasis(), result.logicalZ);
  return result;
}

/// One violated ideal-contract condition. `kind` names the condition and
/// `message` is the text emitted when the condition aborts EDEM derivation.
struct ContractFailure {
  std::string kind;
  std::optional<unsigned> index;
  std::string message;
};

/// Ranks and violations of the ideal detector-contract check. The outcome code
/// is the space of deterministic parities of the Choi-probed realization;
/// declared rows must be deterministic and, with the observables, span it.
struct ContractReport {
  unsigned outcomeCount = 0;
  unsigned randomBits = 0;
  unsigned outcomeCodeRank = 0;
  unsigned detectorRank = 0;
  unsigned logicalOutcomeRank = 0;
  std::vector<ContractFailure> failures;
  IdealActionReport action;
};

/// Evaluates the profile's declared rows against the ideal outcome analysis of
/// its realization. Structural problems fail with diagnostics; semantic
/// violations are returned in the report so callers choose whether to abort.
FailureOr<ContractReport> evaluateIdealProfile(
    qlx::fabric::GadgetContractOp profile, const Projection &projection,
    const std::vector<BoundaryPort> &inputs,
    const std::vector<BoundaryPort> &outputs, const ProfileRows &rows,
    const IdealLogicalAction *action = nullptr) {
  std::vector<IdealBoundaryPort> idealInputs, idealOutputs;
  idealInputs.reserve(inputs.size());
  idealOutputs.reserve(outputs.size());
  for (const auto &port : inputs)
    idealInputs.push_back(idealPort(port));
  for (const auto &port : outputs)
    idealOutputs.push_back(idealPort(port));
  auto result =
      analyzeIdealOutcomes(projection, idealInputs, idealOutputs, action);
  if (!result) {
    profile.emitOpError() << "EDEM ideal Clifford analysis failed: "
                          << llvm::toString(result.takeError());
    return failure();
  }
  const auto &ideal = *result;
  const unsigned outcomeCount = ideal.outcomes().size();
  const unsigned inputKept = ideal.inputSyndromes().size();
  const unsigned recordOffset = inputKept;
  const unsigned outputOffset = recordOffset + projection.measurementCount;
  if (outcomeCount != outputOffset + ideal.outputSyndromes().size()) {
    profile.emitOpError("EDEM ideal outcome order is inconsistent");
    return failure();
  }

  auto expansions = [](const std::vector<BoundaryPort> &ports, unsigned base,
                       unsigned width) {
    std::vector<OutcomeRow> result;
    unsigned keptOffset = base;
    for (const auto &port : ports) {
      for (auto [index, decomposition] :
           llvm::enumerate(port.boundary.effectiveToKeptDecomposition())) {
        Bits support(width);
        for (int kept : decomposition.set_bits())
          support.set(keptOffset + kept);
        result.push_back(
            {std::move(support), port.boundary.effectiveSigns()[index]});
      }
      keptOffset += port.boundary.keptSyndromes();
    }
    return result;
  };
  const auto inputExpansion = expansions(inputs, 0, outcomeCount);
  const auto outputExpansion = expansions(outputs, outputOffset, outcomeCount);
  auto authored = [&](const ParityRow &row,
                      std::optional<unsigned> outputTarget = std::nullopt) {
    Bits support(outcomeCount);
    bool constant = row.constant;
    for (int input : row.virtualInputs.set_bits()) {
      support ^= inputExpansion[input].support;
      constant ^= inputExpansion[input].constant;
    }
    for (int record : row.records.set_bits())
      support.flip(recordOffset + record);
    if (outputTarget) {
      support ^= outputExpansion[*outputTarget].support;
      constant ^= outputExpansion[*outputTarget].constant;
    }
    return OutcomeRow{std::move(support), constant};
  };

  std::vector<Bits> outcomeRandom;
  outcomeRandom.reserve(outcomeCount);
  for (const auto &bit : ideal.outcomes()) {
    Bits coefficients(ideal.independentRandomBits());
    for (unsigned index = 0; index < ideal.independentRandomBits(); ++index)
      if ((bit.randomWords[index / 64] >> (index % 64)) & 1ULL)
        coefficients.set(index);
    outcomeRandom.push_back(std::move(coefficients));
  }
  auto image = [&](const Bits &support) {
    Bits coefficients(ideal.independentRandomBits());
    for (int index : support.set_bits())
      coefficients ^= outcomeRandom[index];
    return coefficients;
  };
  auto shift = [&](const OutcomeRow &row) {
    bool value = row.constant;
    for (int index : row.support.set_bits())
      value ^= ideal.outcomes()[index].constant;
    return value;
  };

  ContractReport report;
  report.outcomeCount = outcomeCount;
  report.randomBits = ideal.independentRandomBits();
  report.action = ideal.action();
  auto fail = [&](StringRef kind, unsigned index, const llvm::Twine &message) {
    report.failures.push_back({kind.str(), index, message.str()});
  };

  for (auto [index, row] : llvm::enumerate(rows.detectors)) {
    auto parity = authored(row);
    if (image(parity.support).any() || shift(parity))
      fail("detector_not_deterministic", index,
           "EDEM detector " + llvm::Twine(index) +
               " is not deterministically zero in the ideal gadget");
  }
  for (auto [index, row] : llvm::enumerate(rows.success)) {
    auto parity = authored(row);
    if (image(parity.support).any() || shift(parity))
      fail("success_not_deterministic", index,
           "EDEM success row " + llvm::Twine(index) +
               " disagrees with its ideal accepted value");
  }

  std::vector<Bits> declared;
  for (const auto &row : rows.detectors)
    declared.push_back(authored(row).support);
  for (auto [index, row] : llvm::enumerate(rows.outputVirtual)) {
    auto relation = authored(row, index);
    if (image(relation.support).any() || shift(relation))
      fail("output_syndrome_mismatch", index,
           "EDEM output virtual-syndrome relation " + llvm::Twine(index) +
               " disagrees with the ideal output stabilizer");
    declared.push_back(std::move(relation.support));
  }
  for (const auto &row : rows.success)
    declared.push_back(authored(row).support);
  auto declaredRank = [&](const std::vector<Bits> &supports) {
    std::vector<Bits> images;
    for (const auto &row : supports)
      images.push_back(image(row));
    return binaryRank(supports, outcomeCount) -
           binaryRank(std::move(images), ideal.independentRandomBits());
  };
  report.detectorRank = declaredRank(declared);
  // Observables may be random. Their deterministic combinations, modulo the
  // detector/assignment/success rows, complete the ideal outcome code. Take
  // the rank of the union: summing separate ranks double-counts overlap.
  for (const auto &row : rows.observables)
    declared.push_back(authored(row).support);
  const unsigned totalRank = declaredRank(declared);
  report.logicalOutcomeRank = totalRank - report.detectorRank;
  report.outcomeCodeRank = ideal.outcomeDeterministicRelationDimension();
  if (report.failures.empty() && totalRank != report.outcomeCodeRank)
    fail("rank_incomplete", 0,
         "EDEM contract is rank-incomplete: ideal relation rank " +
             llvm::Twine(report.outcomeCodeRank) + ", declared rank " +
             llvm::Twine(totalRank));
  return report;
}

LogicalResult verifyIdealProfile(qlx::fabric::GadgetContractOp profile,
                                 const Projection &projection,
                                 const std::vector<BoundaryPort> &inputs,
                                 const std::vector<BoundaryPort> &outputs,
                                 const ProfileRows &rows) {
  auto report =
      evaluateIdealProfile(profile, projection, inputs, outputs, rows);
  if (failed(report))
    return failure();
  if (!report->failures.empty()) {
    profile.emitOpError() << report->failures.front().message;
    return failure();
  }
  return success();
}

struct SourceLayout {
  std::size_t inputSyndromes;
  std::size_t faultStart;
  std::size_t sourceCount;
};

SourceLayout sourceLayout(const std::vector<BoundaryPort> &inputs,
                          const edem_analysis::FaultInput &faults) {
  std::size_t syndromeCount = 0;
  std::size_t logicalCount = 0;
  for (const auto &port : inputs) {
    syndromeCount += port.boundary.effectiveSyndromes();
    logicalCount += 2 * port.boundary.logicalQubits();
  }
  auto faultStart = 2 * syndromeCount + logicalCount;
  return {syndromeCount, faultStart, faultStart + faults.faults.size()};
}

struct Injection {
  std::size_t source;
  const edem_analysis::PauliInsertion *effect;
};

struct PreparedFaults {
  // Slot zero is the input boundary. Slot i+1 is after instruction i.
  std::vector<std::vector<Injection>> atSlot;
  std::vector<std::vector<std::size_t>> recordFlips;
};

FailureOr<PreparedFaults> prepareFaults(
    qlx::fabric::GadgetContractOp profile, const Projection &projection,
    const std::unordered_map<std::string, unsigned> &recordMap,
    const edem_analysis::FaultInput &faults, const SourceLayout &layout) {
  std::unordered_map<std::string, std::size_t> sites;
  for (auto [ordinal, instruction] : llvm::enumerate(projection.instructions)) {
    if (instruction.siteId.empty() ||
        !sites.emplace(instruction.siteId, ordinal).second) {
      profile.emitOpError(
          "EDEM projection requires unique stable instruction site IDs");
      return failure();
    }
  }
  PreparedFaults prepared;
  prepared.atSlot.resize(projection.instructions.size() + 1);
  prepared.recordFlips.resize(projection.measurementCount);
  StringRef gadgetName = profile.getGadget();
  for (auto [ordinal, fault] : llvm::enumerate(faults.faults)) {
    auto source = layout.faultStart + ordinal;
    for (const auto &insertion : fault.insertions) {
      std::size_t slot = 0;
      if (!insertion.afterSite.empty()) {
        auto site = sites.find(insertion.afterSite);
        if (site == sites.end()) {
          profile.emitOpError()
              << "fault '" << fault.id << "' names unknown insertion site '"
              << insertion.afterSite << "'";
          return failure();
        }
        slot = site->second + 1;
      }
      prepared.atSlot[slot].push_back({source, &insertion});
    }
    for (const auto &rawPath : fault.recordFlips) {
      // A relative record may itself start with the gadget's name. Resolve
      // it exactly before trying a root-qualified spelling. Projection has
      // already rejected aliases that would make those spellings ambiguous.
      auto record = recordMap.find(rawPath);
      if (record == recordMap.end()) {
        StringRef path(rawPath);
        if (path.consume_front((gadgetName + ".").str()))
          record = recordMap.find(path.str());
      }
      if (record == recordMap.end()) {
        profile.emitOpError()
            << "fault '" << fault.id << "' names unknown stable record '"
            << rawPath << "'";
        return failure();
      }
      prepared.recordFlips[record->second].push_back(source);
    }
  }
  return prepared;
}

struct Propagated {
  std::vector<Bits> x;
  std::vector<Bits> z;
  std::vector<Bits> records;
};

FailureOr<Propagated> propagate(qlx::fabric::GadgetContractOp profile,
                                const Projection &projection,
                                const std::vector<BoundaryPort> &inputs,
                                const SourceLayout &layout,
                                const PreparedFaults &prepared) {
  const auto carriers = projection.carrierCount;
  std::vector<Bits> x(carriers, Bits(layout.sourceCount));
  std::vector<Bits> z(carriers, Bits(layout.sourceCount));
  std::vector<bool> live(carriers, false);

  auto applyRepresentative = [&](const BoundaryPort &port, const Bits &pauli,
                                 std::size_t source) {
    const auto n = port.boundary.physicalQubits();
    for (unsigned local = 0; local < n; ++local) {
      const auto carrier = port.carriers[local];
      if (pauli.test(local))
        x[carrier].set(source);
      if (pauli.test(n + local))
        z[carrier].set(source);
    }
  };

  std::size_t syndromeSource = 0;
  for (const auto &port : inputs) {
    for (auto carrier : port.carriers) {
      if (carrier >= carriers || live[carrier]) {
        profile.emitOpError("EDEM input carriers are invalid or duplicated");
        return failure();
      }
      live[carrier] = true;
    }
    for (const auto &pauli : port.boundary.inputSyndromeRepresentatives())
      applyRepresentative(port, pauli, syndromeSource++);
  }
  std::size_t logicalSource = 2 * layout.inputSyndromes;
  for (const auto &port : inputs) {
    for (unsigned logical = 0; logical < port.boundary.logicalQubits();
         ++logical) {
      applyRepresentative(port,
                          port.boundary.inputLogicalXRepresentatives()[logical],
                          logicalSource++);
      applyRepresentative(port,
                          port.boundary.inputLogicalZRepresentatives()[logical],
                          logicalSource++);
    }
  }

  auto applyInsertions = [&](std::size_t slot) -> LogicalResult {
    for (const auto &injection : prepared.atSlot[slot]) {
      for (const auto &factor : injection.effect->factors) {
        if (factor.carrier >= carriers || !live[factor.carrier]) {
          profile.emitOpError()
              << "EDEM fault insertion at site " << slot
              << " targets a non-live carrier " << factor.carrier;
          return failure();
        }
        if (factor.basis != edem_analysis::Pauli::z)
          x[factor.carrier].flip(injection.source);
        if (factor.basis != edem_analysis::Pauli::x)
          z[factor.carrier].flip(injection.source);
      }
    }
    return success();
  };
  if (failed(applyInsertions(0)))
    return failure();

  std::vector<Bits> records;
  records.reserve(projection.measurementCount);
  auto appendRecord = [&](Bits response, unsigned index) -> LogicalResult {
    if (index != records.size() || index >= prepared.recordFlips.size()) {
      profile.emitOpError("EDEM projection measurement order is inconsistent");
      return failure();
    }
    for (auto source : prepared.recordFlips[index])
      response.flip(source);
    records.push_back(std::move(response));
    return success();
  };

  using Kind = ProjectionInstruction::Kind;
  using Basis = ProjectionInstruction::Pauli;
  for (auto [ordinal, instruction] : llvm::enumerate(projection.instructions)) {
    auto requireLive = [&](unsigned carrier) -> LogicalResult {
      if (carrier >= carriers || !live[carrier]) {
        profile.emitOpError()
            << "EDEM operation at site '" << instruction.siteId
            << "' uses a non-live carrier " << carrier;
        return failure();
      }
      return success();
    };
    if (instruction.kind == Kind::R || instruction.kind == Kind::RX) {
      for (unsigned carrier : instruction.carriers) {
        if (carrier >= carriers) {
          profile.emitOpError("EDEM reset carrier is out of range");
          return failure();
        }
        x[carrier].reset();
        z[carrier].reset();
        live[carrier] = true;
      }
    } else if (instruction.kind == Kind::H || instruction.kind == Kind::S ||
               instruction.kind == Kind::SDag || instruction.kind == Kind::X ||
               instruction.kind == Kind::Z) {
      for (unsigned carrier : instruction.carriers) {
        if (failed(requireLive(carrier)))
          return failure();
        if (instruction.kind == Kind::H)
          std::swap(x[carrier], z[carrier]);
        else if (instruction.kind == Kind::S || instruction.kind == Kind::SDag)
          z[carrier] ^= x[carrier];
        // Ideal Pauli conjugations affect only the reference phase, never an
        // error response column. The ideal verifier tracks those phases.
      }
    } else if (instruction.kind == Kind::CX || instruction.kind == Kind::CZ ||
               instruction.kind == Kind::Swap) {
      if (instruction.carriers.size() % 2 != 0) {
        profile.emitOpError("EDEM two-carrier instruction has odd arity");
        return failure();
      }
      for (std::size_t pair = 0; pair < instruction.carriers.size();
           pair += 2) {
        unsigned left = instruction.carriers[pair];
        unsigned right = instruction.carriers[pair + 1];
        if (left == right || failed(requireLive(left)) ||
            failed(requireLive(right)))
          return failure();
        if (instruction.kind == Kind::CX) {
          x[right] ^= x[left];
          z[left] ^= z[right];
        } else if (instruction.kind == Kind::CZ) {
          z[left] ^= x[right];
          z[right] ^= x[left];
        } else {
          std::swap(x[left], x[right]);
          std::swap(z[left], z[right]);
        }
      }
    } else if (instruction.kind == Kind::M || instruction.kind == Kind::MX) {
      for (auto [offset, carrier] : llvm::enumerate(instruction.carriers)) {
        if (failed(requireLive(carrier)))
          return failure();
        Bits response = instruction.kind == Kind::M ? x[carrier] : z[carrier];
        if (failed(appendRecord(std::move(response),
                                instruction.measurementStart + offset)))
          return failure();
        // M/MX retain the measured patch in Fabric, just like MPP. Keep the
        // chosen Pauli frame: a later measurement or gate can still use this
        // carrier. Only a reset discards its prior error response.
      }
    } else if (instruction.kind == Kind::MPP) {
      if (instruction.paulis.empty()) {
        profile.emitOpError("EDEM MPP has no Pauli factors");
        return failure();
      }
      Bits response(layout.sourceCount);
      for (const auto &factor : instruction.paulis) {
        if (failed(requireLive(factor.carrier)))
          return failure();
        if (factor.pauli != Basis::Z)
          response ^= z[factor.carrier];
        if (factor.pauli != Basis::X)
          response ^= x[factor.carrier];
      }
      if (failed(
              appendRecord(std::move(response), instruction.measurementStart)))
        return failure();
    } else {
      profile.emitOpError() << "EDEM has no fault transfer for site '"
                            << instruction.siteId << "'";
      return failure();
    }
    if (failed(applyInsertions(ordinal + 1)))
      return failure();
  }
  if (records.size() != projection.measurementCount) {
    profile.emitOpError("EDEM projection did not produce every stable record");
    return failure();
  }
  return Propagated{std::move(x), std::move(z), std::move(records)};
}

Bits parityResponse(const ParityRow &row, const Propagated &propagated,
                    const SourceLayout &layout) {
  Bits response(layout.sourceCount);
  for (int record : row.records.set_bits())
    response ^= propagated.records[record];
  for (int input : row.virtualInputs.set_bits())
    response.flip(layout.inputSyndromes + input);
  return response;
}

Bits boundaryResponse(const BoundaryPort &port, const Bits &functional,
                      const Propagated &propagated, std::size_t sourceCount) {
  Bits response(sourceCount);
  const auto n = port.boundary.physicalQubits();
  for (int local : functional.set_bits()) {
    auto index = static_cast<unsigned>(local);
    unsigned carrier = port.carriers[index < n ? index : index - n];
    response ^= index < n ? propagated.x[carrier] : propagated.z[carrier];
  }
  return response;
}

struct SourceSlice {
  cudaq_edem_endpoint endpoint;
  unsigned start;
  unsigned width;
};

struct MatrixStorage {
  std::vector<uint64_t> offsets;
  std::vector<uint32_t> indices;
};

FailureOr<PackedGadgetEDEM> assemble(qlx::fabric::GadgetContractOp profile,
                                     const std::vector<BoundaryPort> &inputs,
                                     const std::vector<BoundaryPort> &outputs,
                                     const ProfileRows &profileRows,
                                     const edem_analysis::FaultInput &faults,
                                     const SourceLayout &layout,
                                     const Propagated &propagated) {
  // Source columns are internal propagation coordinates. Construct each block
  // by semantic endpoints; only EDEM owns bundle order and flattened offsets.
  std::vector<SourceSlice> slices;
  if (!faults.faults.empty())
    slices.push_back({{CUDAQ_EDEM_FAULT, 0},
                      unsigned(layout.faultStart),
                      unsigned(faults.faults.size())});
  unsigned syndromeStart = 0, logicalStart = 2 * layout.inputSyndromes;
  for (auto [index, port] : llvm::enumerate(inputs)) {
    unsigned g = port.boundary.effectiveSyndromes();
    unsigned l = 2 * port.boundary.logicalQubits();
    slices.push_back({{CUDAQ_EDEM_VIRTUAL, unsigned(index)},
                      unsigned(layout.inputSyndromes + syndromeStart),
                      g});
    slices.push_back(
        {{CUDAQ_EDEM_SYNDROME, unsigned(index)}, syndromeStart, g});
    slices.push_back({{CUDAQ_EDEM_LOGICAL, unsigned(index)}, logicalStart, l});
    syndromeStart += g;
    logicalStart += l;
  }
  std::vector<unsigned> sourceSlice(layout.sourceCount),
      sourceColumn(layout.sourceCount);
  for (auto [index, slice] : llvm::enumerate(slices))
    for (unsigned column = 0; column < slice.width; ++column) {
      sourceSlice[slice.start + column] = index;
      sourceColumn[slice.start + column] = column;
    }

  std::vector<MatrixStorage> matrices;
  std::vector<cudaq_edem_matrix_block> blocks;
  auto addRows = [&](cudaq_edem_endpoint output,
                     const std::vector<Bits> &rows) {
    // Sparse staging visits only nonzero coefficients, including across large
    // independent fault sets. Row iteration gives sorted CSC row indices.
    std::map<unsigned, std::vector<cudaq_gf2_entry>> byInput;
    for (auto [rowIndex, row] : llvm::enumerate(rows))
      for (int source : row.set_bits())
        byInput[sourceSlice[source]].push_back(
            {unsigned(rowIndex), sourceColumn[source]});
    for (const auto &[inputIndex, entries] : byInput) {
      const auto &slice = slices[inputIndex];
      MatrixStorage storage;
      storage.offsets.resize(slice.width + 1);
      for (auto entry : entries)
        ++storage.offsets[entry.column + 1];
      for (unsigned column = 0; column < slice.width; ++column)
        storage.offsets[column + 1] += storage.offsets[column];
      storage.indices.resize(entries.size());
      auto cursor = storage.offsets;
      for (auto entry : entries)
        storage.indices[cursor[entry.column]++] = entry.row;
      matrices.push_back(std::move(storage));
      auto &owned = matrices.back();
      blocks.push_back({output,
                        slice.endpoint,
                        {unsigned(rows.size()), slice.width, entries.size(),
                         owned.offsets.data(), owned.indices.data()}});
    }
  };
  auto parityRows = [&](const auto &parities) {
    std::vector<Bits> rows;
    for (const auto &row : parities)
      rows.push_back(parityResponse(row, propagated, layout));
    return rows;
  };
  addRows({CUDAQ_EDEM_DETECTOR, 0}, parityRows(profileRows.detectors));
  addRows({CUDAQ_EDEM_OBSERVABLE, 0}, parityRows(profileRows.observables));
  addRows({CUDAQ_EDEM_SUCCESS, 0}, parityRows(profileRows.success));
  unsigned virtualOffset = 0;
  for (auto [index, port] : llvm::enumerate(outputs)) {
    std::vector<Bits> virtualRows, syndromeRows, logicalRows;
    for (const auto &functional : port.boundary.outputSyndromeFunctionals()) {
      virtualRows.push_back(parityResponse(
          profileRows.outputVirtual[virtualOffset++], propagated, layout));
      syndromeRows.push_back(
          boundaryResponse(port, functional, propagated, layout.sourceCount));
    }
    for (unsigned logical = 0; logical < port.boundary.logicalQubits();
         ++logical) {
      logicalRows.push_back(boundaryResponse(
          port, port.boundary.outputLogicalXFunctionals()[logical], propagated,
          layout.sourceCount));
      logicalRows.push_back(boundaryResponse(
          port, port.boundary.outputLogicalZFunctionals()[logical], propagated,
          layout.sourceCount));
    }
    addRows({CUDAQ_EDEM_VIRTUAL, unsigned(index)}, virtualRows);
    addRows({CUDAQ_EDEM_SYNDROME, unsigned(index)}, syndromeRows);
    addRows({CUDAQ_EDEM_LOGICAL, unsigned(index)}, logicalRows);
  }

  std::vector<cudaq_edem_code_presentation> codes;
  std::vector<uint32_t> inputCodes, outputCodes;
  std::unordered_map<std::string, unsigned> codeIds;
  auto addCode = [&](const BoundaryPort &port) {
    const auto &name = port.boundary.name();
    auto [it, added] = codeIds.emplace(name, codes.size());
    // Contracts bind one virtual outcome per selected effective stabilizer.
    if (added)
      codes.push_back({
          .name = name.data(),
          .name_size = name.size(),
          .syndrome_count = port.boundary.effectiveSyndromes(),
          .virtual_outcome_count = port.boundary.effectiveSyndromes(),
          .logical_qubit_count = port.boundary.logicalQubits(),
      });
    return it->second;
  };
  for (const auto &port : inputs)
    inputCodes.push_back(addCode(port));
  for (const auto &port : outputs)
    outputCodes.push_back(addCode(port));

  std::vector<std::vector<uint32_t>> groupIds;
  std::vector<std::vector<double>> probabilities;
  std::vector<bool> claimed(faults.faults.size());
  for (const auto &group : faults.exclusiveGroups) {
    groupIds.emplace_back(group.begin(), group.end());
    for (auto id : group)
      claimed[id] = true;
  }
  std::vector<uint32_t> independent;
  for (unsigned id = 0; id < claimed.size(); ++id)
    if (!claimed[id])
      independent.push_back(id);
  unsigned exclusiveCount = groupIds.size();
  if (!independent.empty())
    groupIds.push_back(std::move(independent));
  std::vector<cudaq_edem_fault_group> groups;
  for (auto [index, ids] : llvm::enumerate(groupIds)) {
    probabilities.emplace_back();
    for (auto id : ids)
      probabilities.back().push_back(faults.probabilities[id]);
    groups.push_back(
        {index < exclusiveCount ? CUDAQ_EDEM_EXCLUSIVE : CUDAQ_EDEM_INDEPENDENT,
         unsigned(ids.size()), ids.data(), probabilities.back().data()});
  }
  cudaq_edem_fault_distribution distribution{
      unsigned(faults.faults.size()), unsigned(groups.size()), groups.data()};
  uint32_t detectorSize = profileRows.detectors.size();
  uint32_t observableSize = profileRows.observables.size();
  uint32_t successSize = profileRows.success.size();
  cudaq_edem_description description{
      .shape = {.input_code_count = unsigned(inputs.size()),
                .output_code_count = unsigned(outputs.size()),
                .fault_ports_count = faults.faults.empty() ? 0u : 1u,
                .detector_ports_count = detectorSize ? 1u : 0u,
                .observable_ports_count = observableSize ? 1u : 0u,
                .success_ports_count = successSize ? 1u : 0u},
      .code_table = {unsigned(codes.size()), codes.data()},
      .input_code_indices = inputCodes.data(),
      .output_code_indices = outputCodes.data(),
      .fault_ports_distributions = &distribution,
      .detector_ports_sizes = &detectorSize,
      .observable_ports_sizes = &observableSize,
      .success_ports_sizes = &successSize,
      .block_count = blocks.size(),
      .blocks = blocks.data(),
  };
  auto check = [&](cudaq_edem_status status) {
    if (status == CUDAQ_EDEM_OK)
      return true;
    profile.emitOpError() << "EDEM assembly failed: "
                          << cudaq_edem_status_string(status);
    return false;
  };
  PackedGadgetEDEM result;
  if (!check(PackedGadgetEDEM::construct(description, result)))
    return failure();
  return result;
}

FailureOr<PackedGadgetEDEM>
buildGadgetEDEM(qlx::fabric::GadgetContractOp profile,
                const Projection &projection,
                const edem_analysis::FaultInput &faults) {
  if (!profile.getBoundaryComplete()) {
    profile.emitOpError("EDEM requires a complete selected boundary contract");
    return failure();
  }
  auto inputs = readPorts(profile, profile.getInputs(),
                          projection.inputDataCarriersByPort, "inputs");
  auto outputs = readPorts(profile, profile.getOutputs(),
                           projection.outputDataCarriersByPort, "outputs");
  if (failed(inputs) || failed(outputs))
    return failure();
  auto records = readRecordMap(profile, projection);
  if (failed(records))
    return failure();
  auto rows = readProfileRows(profile, projection, *records, *inputs, *outputs);
  if (failed(rows))
    return failure();
  if (failed(verifyIdealProfile(profile, projection, *inputs, *outputs, *rows)))
    return failure();
  auto layout = sourceLayout(*inputs, faults);
  if (layout.sourceCount > unsigned(std::numeric_limits<int>::max())) {
    profile.emitOpError("EDEM source count exceeds the analysis index width");
    return failure();
  }
  auto prepared = prepareFaults(profile, projection, *records, faults, layout);
  if (failed(prepared))
    return failure();
  auto propagated = propagate(profile, projection, *inputs, layout, *prepared);
  if (failed(propagated))
    return failure();
  return assemble(profile, *inputs, *outputs, *rows, faults, layout,
                  *propagated);
}

FailureOr<qlx::fabric::GadgetContractOp>
selectContract(ModuleOp module, StringRef contractName) {
  qlx::fabric::GadgetContractOp selected;
  if (!contractName.empty()) {
    selected = module.lookupSymbol<qlx::fabric::GadgetContractOp>(contractName);
  } else {
    for (auto candidate : module.getOps<qlx::fabric::GadgetContractOp>()) {
      if (selected) {
        module.emitError("EDEM requires profile= when multiple gadget "
                         "contracts are present");
        return failure();
      }
      selected = candidate;
    }
  }
  if (!selected) {
    module.emitError("EDEM requires a selected fabric.gadget_contract");
    return failure();
  }
  return selected;
}

/// Expected intent is resolved from the selected, verified module. An unknown
/// objective is not the empty Clifford action: absence of support stays
/// explicit.
struct ObjectiveResolution {
  std::optional<IdealLogicalAction> action;
  std::string reason;
  std::string label;
  bool mismatch = false;
};

IdealLogicalAction identityAction(unsigned arity) {
  IdealLogicalAction action;
  action.arity = arity;
  for (unsigned row = 0; row < 2 * arity; ++row) {
    Bits image(2 * arity);
    image.set(row);
    action.images.push_back(std::move(image));
    action.phases.push_back(false);
  }
  return action;
}

bool conjugateIntent(IdealLogicalAction &action, StringRef name,
                     llvm::ArrayRef<unsigned> wires) {
  if (name == "idle")
    return true;
  const bool single = name == "x" || name == "y" || name == "z" ||
                      name == "h" || name == "s" || name == "sdg";
  if ((!single && name != "cx" && name != "cz") ||
      wires.size() != (single ? 1u : 2u) ||
      llvm::any_of(wires, [&](unsigned q) { return q >= action.arity; }) ||
      (!single && wires[0] == wires[1]))
    return false;
  if (name == "cz") {
    unsigned target = wires[1];
    return conjugateIntent(action, "h", {target}) &&
           conjugateIntent(action, "cx", wires) &&
           conjugateIntent(action, "h", {target});
  }
  const unsigned k = action.arity, q = wires[0];
  for (unsigned row = 0; row < action.images.size(); ++row) {
    Bits &image = action.images[row];
    bool x = image[q], z = image[k + q], phase = action.phases[row];
    if (name == "h") {
      phase ^= x && z;
      image[q] = z;
      image[k + q] = x;
    } else if (name == "s" || name == "sdg") {
      phase ^= x && (name == "s" ? z : !z);
      if (x)
        image.flip(k + q);
    } else if (name == "x")
      phase ^= z;
    else if (name == "z")
      phase ^= x;
    else if (name == "y")
      phase ^= x != z;
    else {
      unsigned target = wires[1];
      bool xt = image[target], zt = image[k + target];
      phase ^= x && zt && (xt ^ z ^ true);
      if (x)
        image.flip(target);
      if (zt)
        image.flip(k + q);
    }
    action.phases[row] = phase;
  }
  return true;
}

bool isQuantum(Type type) { return isa<::qlx::LogicalQubitType>(type); }

struct IntentSignature {
  unsigned inputs = 0, outputs = 0, bits = 0;
  bool supported = true;
};

IntentSignature intentSignature(FunctionType type) {
  IntentSignature result;
  for (Type input : type.getInputs()) {
    if (isQuantum(input))
      ++result.inputs;
    else
      result.supported = false;
  }
  for (Type output : type.getResults()) {
    if (isQuantum(output))
      ++result.outputs;
    else if (output.isInteger(1))
      ++result.bits;
    else
      result.supported = false;
  }
  return result;
}

bool preparationIntent(StringRef state, IdealLogicalAction &action) {
  using Basis = IdealLogicalAction::Basis;
  if (state == "zero" || state == "0" || state == "prepare_zero") {
    action.basis = Basis::Z;
    action.negative = false;
  } else if (state == "one" || state == "1" || state == "prepare_one") {
    action.basis = Basis::Z;
    action.negative = true;
  } else if (state == "plus" || state == "+" || state == "prepare_plus") {
    action.basis = Basis::X;
    action.negative = false;
  } else if (state == "minus" || state == "-" || state == "prepare_minus") {
    action.basis = Basis::X;
    action.negative = true;
  } else
    return false;
  action.kind = IdealLogicalAction::Kind::Preparation;
  return true;
}

/// Interpret only the supported straight-line ideal vocabulary. This follows
/// linked semantics instead of trusting a cached Clifford attribute or a name.
FailureOr<ObjectiveResolution> resolveIntentBody(Operation *declaration,
                                                 Operation *body,
                                                 IntentSignature signature) {
  ObjectiveResolution result;
  IdealLogicalAction action = identityAction(signature.inputs);
  IdealLogicalAction preparedState = identityAction(1);
  llvm::DenseMap<Value, unsigned> wires;
  llvm::DenseSet<Value> liveQuantum;
  struct AffineResult {
    bool measured = false, constant = false;
  };
  llvm::DenseMap<Value, AffineResult> bits;
  llvm::DenseSet<Operation *> active;
  bool prepared = false, measured = false, sawGate = false, discarded = false;
  bool invalid = false;
  std::vector<unsigned> returnedWires;
  std::optional<AffineResult> returnedBit;
  auto unsupported = [&](const Twine &message) {
    result.reason = message.str();
    return failure();
  };
  auto malformed = [&](Operation *owner, const Twine &message) {
    owner->emitError(message);
    invalid = true;
    return failure();
  };
  std::function<LogicalResult(Operation *, ValueRange,
                              SmallVectorImpl<Value> &)>
      walk;
  walk = [&](Operation *current, ValueRange actuals,
             SmallVectorImpl<Value> &returned) -> LogicalResult {
    if (!active.insert(current).second)
      return malformed(current, "recursive ideal objective semantics");
    if (current->getNumRegions() != 1 ||
        !llvm::hasSingleElement(current->getRegion(0)))
      return malformed(current, "ideal objective requires one signature block");
    Block &block = current->getRegion(0).front();
    if (actuals.empty() && current == body) {
      unsigned index = 0;
      for (BlockArgument argument : block.getArguments()) {
        if (!isQuantum(argument.getType()))
          return unsupported("classical or resource inputs to the ideal "
                             "objective are unsupported");
        wires[argument] = index++;
        liveQuantum.insert(argument);
      }
    } else {
      if (actuals.size() != block.getNumArguments())
        return malformed(current,
                         "ideal objective call arity differs from its body");
      for (auto [formal, actual] : llvm::zip(block.getArguments(), actuals)) {
        if (auto wire = wires.find(actual); wire != wires.end()) {
          if (!liveQuantum.erase(actual))
            return malformed(
                current,
                "ideal objective call reuses a consumed logical operand");
          wires[formal] = wire->second;
          liveQuantum.insert(formal);
        } else if (auto bit = bits.find(actual); bit != bits.end())
          bits[formal] = bit->second;
        else
          return malformed(current,
                           "ideal objective call uses an unresolved SSA value");
      }
    }
    for (Operation &operation : block) {
      StringRef name = operation.getName().getStringRef();
      if (name == "qlx.return") {
        llvm::DenseSet<Value> returnedQuantum;
        for (Value value : operation.getOperands())
          if (isQuantum(value.getType()) &&
              (!liveQuantum.contains(value) ||
               !returnedQuantum.insert(value).second))
            return malformed(
                &operation,
                "ideal objective returns a consumed or repeated logical qubit");
        for (Value value : liveQuantum)
          if (value.getParentBlock() == &block &&
              !returnedQuantum.contains(value))
            return malformed(&operation,
                             "ideal objective leaks a live logical qubit");
        returned.append(operation.getOperands().begin(),
                        operation.getOperands().end());
        active.erase(current);
        return success();
      }
      if (name == "arith.constant") {
        auto value = operation.getAttrOfType<IntegerAttr>("value");
        if (!value || !value.getType().isInteger(1) ||
            operation.getNumResults() != 1)
          return unsupported(
              "only Boolean constants are supported in ideal objectives");
        bits[operation.getResult(0)] = {false, !value.getValue().isZero()};
        continue;
      }
      if (name == "qlx.xor" || name == "arith.xori") {
        if (operation.getNumOperands() != 2 || operation.getNumResults() != 1 ||
            !bits.contains(operation.getOperand(0)) ||
            !bits.contains(operation.getOperand(1)))
          return malformed(&operation,
                           "ideal affine outcome has unresolved operands");
        auto a = bits[operation.getOperand(0)],
             b = bits[operation.getOperand(1)];
        bits[operation.getResult(0)] = {a.measured != b.measured,
                                        a.constant != b.constant};
        continue;
      }
      if (name == "qlx.discard") {
        for (Value value : operation.getOperands())
          if (!wires.contains(value) || !liveQuantum.erase(value))
            return malformed(
                &operation,
                "ideal discard has an unresolved or consumed logical operand");
        // Untouched operands may be traced out by a destructive instrument.
        discarded = true;
        continue;
      }
      if (name == "qlx.prepare") {
        if (signature.inputs || prepared || measured ||
            operation.getNumResults() != 1)
          return unsupported(
              "only a single logical X/Z preparation is supported");
        auto state = operation.getAttrOfType<StringAttr>("state");
        if (!state || !preparationIntent(state.getValue(), action))
          return unsupported(
              "the ideal prepared state is not a signed X/Z eigenstate");
        prepared = true;
        preparedState.images[0].reset();
        preparedState.images[0].set(
            action.basis == IdealLogicalAction::Basis::X ? 0 : 1);
        preparedState.phases[0] = action.negative;
        wires[operation.getResult(0)] = 0;
        liveQuantum.insert(operation.getResult(0));
        continue;
      }
      if (name == "qlx.apply" || name == "qlx.instrument" ||
          name == "qlx.call") {
        Attribute calleeAttr =
            name == "qlx.call"
                ? operation.getAttr("callee")
                : operation.getAttr(name == "qlx.apply" ? "action"
                                                        : "instrument");
        if (auto symbol = dyn_cast_or_null<SymbolRefAttr>(calleeAttr)) {
          Operation *callee =
              SymbolTable::lookupNearestSymbolFrom(&operation, symbol);
          // Objective bodies are symbol tables, while the compiler retains
          // their linked logical declarations in the enclosing module.
          if (!callee)
            if (auto module = declaration->getParentOfType<ModuleOp>())
              callee = SymbolTable::lookupSymbolIn(module, symbol);
          if (!callee)
            return malformed(&operation,
                             "ideal objective references an unresolved callee");
          auto calleeTypeAttr =
              callee->getAttrOfType<TypeAttr>("function_type");
          auto calleeType =
              calleeTypeAttr ? dyn_cast<FunctionType>(calleeTypeAttr.getValue())
                             : FunctionType{};
          if (!calleeType ||
              !llvm::equal(calleeType.getInputs(),
                           operation.getOperandTypes()) ||
              !llvm::equal(calleeType.getResults(),
                           operation.getResultTypes()) ||
              (name == "qlx.apply" && !isa<::qlx::ActionOp>(callee)) ||
              (name == "qlx.instrument" &&
               !isa<::qlx::InstrumentDeclOp>(callee)))
            return malformed(
                &operation,
                "ideal objective call disagrees with its typed declaration");
          if (auto semantics =
                  callee->getAttrOfType<SymbolRefAttr>("semantics")) {
            auto *nested =
                SymbolTable::lookupNearestSymbolFrom(callee, semantics);
            auto nestedType =
                nested ? nested->getAttrOfType<TypeAttr>("function_type")
                       : TypeAttr{};
            if (!nested || !isa<::qlx::ObjectiveBodyOp>(nested) ||
                !nestedType || nestedType.getValue() != calleeType)
              return malformed(callee,
                               "ideal objective semantics body is unresolved "
                               "or has an incompatible type");
            SmallVector<Value> yielded;
            if (failed(walk(nested, operation.getOperands(), yielded)))
              return failure();
            if (yielded.size() != operation.getNumResults())
              return malformed(
                  &operation,
                  "ideal objective result arity differs from its body");
            for (auto [output, value] :
                 llvm::zip(operation.getResults(), yielded)) {
              if (wires.contains(value)) {
                if (!liveQuantum.erase(value))
                  return malformed(
                      &operation,
                      "ideal objective call returns a consumed logical qubit");
                wires[output] = wires[value];
                liveQuantum.insert(output);
              } else if (bits.contains(value))
                bits[output] = bits[value];
              else
                return malformed(
                    &operation,
                    "ideal objective returned an unresolved SSA value");
            }
            continue;
          }
          auto kind = callee->getAttrOfType<StringAttr>("kind");
          if (name != "qlx.apply" || !kind)
            return unsupported(
                "the linked ideal instrument has no supported semantics body");
          calleeAttr = ::qlx::BuiltinActionAttr::get(
              operation.getContext(),
              ::qlx::symbolizeBuiltinAction(kind.getValue())
                  .value_or(::qlx::BuiltinAction::t));
        }
        if (auto gate =
                dyn_cast_or_null<::qlx::BuiltinActionAttr>(calleeAttr)) {
          if (measured)
            return unsupported("Clifford gates after ideal measurement require "
                               "general instrument analysis");
          std::vector<unsigned> targets;
          for (Value operand : operation.getOperands()) {
            auto wire = wires.find(operand);
            if (wire == wires.end() || !liveQuantum.erase(operand))
              return malformed(&operation,
                               "ideal Clifford action uses an unresolved or "
                               "consumed logical operand");
            targets.push_back(wire->second);
          }
          if (operation.getNumResults() != targets.size() ||
              !conjugateIntent(prepared ? preparedState : action,
                               ::qlx::stringifyBuiltinAction(gate.getValue()),
                               targets))
            return unsupported("the ideal action contains a non-Clifford or "
                               "unsupported operation");
          for (auto [output, target] :
               llvm::zip(operation.getResults(), targets)) {
            wires[output] = target;
            liveQuantum.insert(output);
          }
          sawGate = true;
          continue;
        }
        auto instrument =
            dyn_cast_or_null<::qlx::BuiltinInstrumentAttr>(calleeAttr);
        if (!instrument ||
            instrument.getValue() != ::qlx::BuiltinInstrument::mpp ||
            prepared || measured || sawGate)
          return unsupported("the ideal instrument is outside signed "
                             "single-logical X/Z measurement");
        auto parameters = operation.getAttrOfType<DictionaryAttr>("parameters");
        auto x = parameters ? parameters.getAs<IntegerAttr>("x_mask")
                            : IntegerAttr{};
        auto z = parameters ? parameters.getAs<IntegerAttr>("z_mask")
                            : IntegerAttr{};
        auto sign =
            parameters ? parameters.getAs<IntegerAttr>("sign") : IntegerAttr{};
        if (!x || !z || !sign || (sign.getInt() != 1 && sign.getInt() != -1))
          return malformed(&operation,
                           "ideal MPP requires canonical Pauli masks and sign");
        bool xBasis = !x.getValue().isZero();
        const APInt &mask = xBasis ? x.getValue() : z.getValue();
        if ((xBasis && !z.getValue().isZero()) || !mask.isPowerOf2() ||
            mask.logBase2() >= operation.getNumOperands() ||
            operation.getNumResults() != operation.getNumOperands() + 1)
          return unsupported("only one nonidentity X or Z factor is supported "
                             "in the ideal MPP");
        Value target = operation.getOperand(mask.logBase2());
        if (!wires.contains(target))
          return malformed(&operation,
                           "ideal MPP has an unresolved logical target");
        action.kind = IdealLogicalAction::Kind::Measurement;
        action.basis = xBasis ? IdealLogicalAction::Basis::X
                              : IdealLogicalAction::Basis::Z;
        action.target = wires[target];
        for (auto [operand, output] : llvm::zip(
                 operation.getOperands(), operation.getResults().drop_back())) {
          if (!wires.contains(operand) || !liveQuantum.erase(operand))
            return malformed(
                &operation,
                "ideal MPP has an unresolved or consumed logical operand");
          wires[output] = wires[operand];
          liveQuantum.insert(output);
        }
        bits[operation.getResult(operation.getNumResults() - 1)] = {
            true, sign.getInt() == -1};
        measured = true;
        continue;
      }
      if (auto measurement = dyn_cast<::qlx::MeasureOp>(operation)) {
        if (prepared || measured || sawGate)
          return unsupported("only one ideal X/Z measurement without preceding "
                             "gates is supported");
        if (measurement.getBasis() != ::qlx::Pauli::X &&
            measurement.getBasis() != ::qlx::Pauli::Z)
          return unsupported("ideal Y measurement is unsupported");
        if (!wires.contains(measurement.getInput()) ||
            !liveQuantum.erase(measurement.getInput()))
          return malformed(&operation, "ideal measurement has an unresolved or "
                                       "consumed logical operand");
        action.kind = IdealLogicalAction::Kind::Measurement;
        action.basis = measurement.getBasis() == ::qlx::Pauli::X
                           ? IdealLogicalAction::Basis::X
                           : IdealLogicalAction::Basis::Z;
        action.target = wires[measurement.getInput()];
        bits[measurement.getResult()] = {true, false};
        measured = true;
        continue;
      }
      return unsupported("unsupported ideal objective operation '" + name +
                         "'");
    }
    return malformed(current, "ideal objective is missing its return");
  };
  SmallVector<Value> returned;
  if (failed(walk(body, {}, returned))) {
    if (invalid)
      return failure();
    return result;
  }
  auto unsupportedResult = [&](const Twine &message) {
    result.reason = message.str();
    return result;
  };
  for (Value value : returned) {
    if (isQuantum(value.getType())) {
      if (!wires.contains(value))
        return declaration->emitError(
                   "ideal objective returns an unresolved logical qubit"),
               failure();
      returnedWires.push_back(wires[value]);
    } else if (value.getType().isInteger(1)) {
      if (returnedBit || !bits.contains(value))
        return unsupportedResult(
            "ideal objective must return exactly one affine "
            "measurement result");
      returnedBit = bits[value];
    } else
      return unsupportedResult("ideal objective returns an unsupported "
                               "classical/resource type");
  }
  auto permutation = returnedWires;
  llvm::sort(permutation);
  if (prepared) {
    if (signature.inputs || returnedWires != std::vector<unsigned>{0} ||
        returnedBit)
      return unsupportedResult("the ideal preparation must produce exactly one "
                               "logical qubit");
    const Bits &stabilizer = preparedState.images[0];
    if (stabilizer[0] == stabilizer[1])
      return unsupportedResult(
          "the ideal prepared state is not a signed X/Z eigenstate");
    action.basis = stabilizer[0] ? IdealLogicalAction::Basis::X
                                 : IdealLogicalAction::Basis::Z;
    action.negative = preparedState.phases[0];
  } else if (measured) {
    if (!returnedBit || !returnedBit->measured || signature.bits != 1)
      return unsupportedResult(
          "ideal measurement result must retain its measured "
          "bit under an affine map");
    action.outcomeConstant = returnedBit->constant;
    action.destructive = returnedWires.empty();
    if (!action.destructive) {
      if (discarded || permutation.size() != signature.inputs ||
          llvm::any_of(llvm::enumerate(permutation),
                       [](auto item) { return item.index() != item.value(); }))
        return unsupportedResult(
            "preserving ideal measurement must return every "
            "logical operand exactly once");
      // Stored temporarily: original input wire -> quantum result position.
      action.outputBindings.resize(signature.inputs);
      for (auto [output, input] : llvm::enumerate(returnedWires))
        action.outputBindings[input] = output;
    }
  } else {
    if (discarded || !signature.inputs || signature.bits ||
        permutation.size() != signature.inputs ||
        llvm::any_of(llvm::enumerate(permutation),
                     [](auto item) { return item.index() != item.value(); }))
      return unsupportedResult("the ideal Clifford objective must preserve a "
                               "nonempty quantum signature");
    for (Bits &row : action.images) {
      Bits remapped(row.size());
      for (auto [output, input] : llvm::enumerate(returnedWires)) {
        remapped[output] = row[input];
        remapped[action.arity + output] = row[action.arity + input];
      }
      row = std::move(remapped);
    }
  }
  result.action = std::move(action);
  return result;
}

FailureOr<std::vector<unsigned>>
bindLogicalSide(qlx::fabric::GadgetSpecOp spec,
                llvm::ArrayRef<ProjectionPort> projected,
                const std::vector<BoundaryPort> &ports,
                llvm::ArrayRef<std::string> formalNames) {
  std::map<std::string, std::vector<unsigned>> aliases;
  std::map<std::string, std::string> requested;
  if (auto binding = spec->getAttrOfType<DictionaryAttr>("logical_ports"))
    for (NamedAttribute item : binding) {
      auto target = dyn_cast<StringAttr>(item.getValue());
      if (!target || !llvm::is_contained(formalNames, item.getName().str())) {
        spec.emitOpError("logical_ports must bind declared quantum operand "
                         "names to logical ports");
        return failure();
      }
      requested[item.getName().str()] = target.str();
    }
  auto specPorts = spec.getPorts();
  unsigned offset = 0;
  if (projected.size() != ports.size()) {
    spec.emitOpError("projected encoded port manifest is incomplete");
    return failure();
  }
  for (auto [index, port] : llvm::enumerate(projected)) {
    auto rawPort = specPorts && port.ordinal < specPorts->size()
                       ? dyn_cast<DictionaryAttr>((*specPorts)[port.ordinal])
                       : DictionaryAttr{};
    auto encodingRef = rawPort ? rawPort.getAs<FlatSymbolRefAttr>("encoding")
                               : FlatSymbolRefAttr{};
    auto encoding =
        encodingRef
            ? dyn_cast_or_null<qlx::fabric::EncodingOp>(
                  SymbolTable::lookupNearestSymbolFrom(spec, encodingRef))
            : qlx::fabric::EncodingOp{};
    unsigned count = ports[index].boundary.logicalQubits();
    if (!encoding || encoding.getLogicalPorts().size() != count) {
      spec.emitOpError("logical binding requires the selected encoding's "
                       "complete logical port table");
      return failure();
    }
    for (auto [logical, rawName] :
         llvm::enumerate(encoding.getLogicalPorts())) {
      auto name = dyn_cast<StringAttr>(rawName);
      if (!name) {
        spec.emitOpError("encoding logical port names must be strings");
        return failure();
      }
      aliases[name.str()].push_back(offset + logical);
      aliases[port.name + "." + name.str()].push_back(offset + logical);
    }
    if (auto local = rawPort.getAs<DictionaryAttr>("logical_port_map"))
      for (NamedAttribute item : local) {
        auto target = dyn_cast<StringAttr>(item.getValue());
        if (!target || !llvm::is_contained(formalNames, item.getName().str())) {
          spec.emitOpError(
              "port logical_port_map names an invalid action operand");
          return failure();
        }
        std::string qualified = port.name + "." + target.str();
        auto [it, inserted] =
            requested.emplace(item.getName().str(), qualified);
        if (!inserted && it->second != qualified &&
            it->second != target.str()) {
          spec.emitOpError(
              "global and endpoint logical-port bindings disagree");
          return failure();
        }
      }
    offset += count;
  }
  std::vector<unsigned> result;
  llvm::DenseSet<unsigned> used;
  for (auto [index, formal] : llvm::enumerate(formalNames)) {
    unsigned target = index;
    if (auto requestedTarget = requested.find(formal);
        requestedTarget != requested.end()) {
      auto found = aliases.find(requestedTarget->second);
      if (found == aliases.end() || found->second.size() != 1) {
        spec.emitOpError(
            "logical-port binding is absent or ambiguous on this boundary: ")
            << requestedTarget->second;
        return failure();
      }
      target = found->second.front();
    }
    if (target >= offset || !used.insert(target).second) {
      spec.emitOpError(
          "logical-port bindings must cover distinct boundary logical qubits");
      return failure();
    }
    result.push_back(target);
  }
  return result;
}

FailureOr<ObjectiveResolution> resolveGadgetObjective(
    ModuleOp module, qlx::fabric::GadgetContractOp profile,
    const Projection &projection, const std::vector<BoundaryPort> &inputs,
    const std::vector<BoundaryPort> &outputs,
    const std::unordered_map<std::string, unsigned> &recordMap) {
  ObjectiveResolution result;
  auto gadget = module.lookupSymbol<qlx::fabric::GadgetOp>(profile.getGadget());
  auto spec = gadget && gadget.getSpecAttr()
                  ? dyn_cast_or_null<qlx::fabric::GadgetSpecOp>(
                        SymbolTable::lookupNearestSymbolFrom(
                            gadget, gadget.getSpecAttr()))
                  : qlx::fabric::GadgetSpecOp{};
  auto objective = spec ? dyn_cast_or_null<qlx::fabric::ObjectiveOp>(
                              SymbolTable::lookupNearestSymbolFrom(
                                  spec, spec.getObjectiveAttr()))
                        : qlx::fabric::ObjectiveOp{};
  if (!spec || !objective) {
    profile.emitOpError("verification requires a linked selected GadgetSpec "
                        "and typed objective");
    return failure();
  }
  if (!objective.getLogicalAttr() || spec.getEntrypoint()) {
    result.reason = "the selected gadget has no declared logical action";
    return result;
  }
  Operation *logical = SymbolTable::lookupNearestSymbolFrom(
      objective, objective.getLogicalAttr());
  if (!logical || (!isa<::qlx::ActionOp, ::qlx::InstrumentDeclOp>(logical))) {
    objective.emitOpError("logical objective must resolve to a typed action or "
                          "instrument declaration");
    return failure();
  }
  auto typeAttr = logical->getAttrOfType<TypeAttr>("function_type");
  auto type =
      typeAttr ? dyn_cast<FunctionType>(typeAttr.getValue()) : FunctionType{};
  if (!type || type != objective.getFunctionType()) {
    objective.emitOpError(
        "logical objective type disagrees with its declaration");
    return failure();
  }
  auto signature = intentSignature(type);
  if (!signature.supported) {
    result.reason = "the ideal objective contains unsupported classical inputs "
                    "or resource types";
    return result;
  }
  auto kindAttr = logical->getAttrOfType<StringAttr>("kind");
  StringRef kind = kindAttr ? kindAttr.getValue() : StringRef{};
  auto semantics = logical->getAttrOfType<SymbolRefAttr>("semantics");
  std::vector<std::string> names;
  auto operands = logical->getAttrOfType<ArrayAttr>("operand_names");
  if (operands)
    for (Attribute item : operands)
      names.push_back(cast<StringAttr>(item).str());
  if (semantics) {
    Operation *body = SymbolTable::lookupNearestSymbolFrom(logical, semantics);
    if (!body || !isa<::qlx::ObjectiveBodyOp>(body)) {
      logical->emitError(
          "logical objective semantics must resolve to a typed objective body");
      return failure();
    }
    auto bodyTypeAttr = body->getAttrOfType<TypeAttr>("function_type");
    if (!bodyTypeAttr || bodyTypeAttr.getValue() != type) {
      logical->emitError(
          "logical objective declaration and semantics signatures disagree");
      return failure();
    }
    auto parsed = resolveIntentBody(logical, body, signature);
    if (failed(parsed))
      return failure();
    result = std::move(*parsed);
    if (!operands)
      if (auto cached = logical->getAttrOfType<::qlx::CliffordActionAttr>(
              "clifford_action"))
        for (Attribute item : cached.getPorts())
          names.push_back(cast<StringAttr>(item).str());
  } else {
    IdealLogicalAction action = identityAction(signature.inputs);
    if (kind == "t" || kind == "tdg" || kind == "ccx" || kind == "ccz" ||
        kind == "prepare_t") {
      result.reason =
          ("the declared logical objective contains non-Clifford action '" +
           kind + "'")
              .str();
      return result;
    }
    if (kind.starts_with("prepare_") && preparationIntent(kind, action)) {
      if (signature.inputs || signature.outputs != 1 || signature.bits)
        result.mismatch = true;
    } else if (kind == "measure_x" || kind == "measure_z") {
      action.kind = IdealLogicalAction::Kind::Measurement;
      action.basis = kind == "measure_x" ? IdealLogicalAction::Basis::X
                                         : IdealLogicalAction::Basis::Z;
      action.destructive = true;
      if (signature.inputs != 1 || signature.outputs || signature.bits != 1)
        result.mismatch = true;
    } else {
      std::vector<unsigned> targets(signature.inputs);
      std::iota(targets.begin(), targets.end(), 0);
      if (!signature.inputs || signature.inputs != signature.outputs ||
          signature.bits || !conjugateIntent(action, kind, targets)) {
        result.reason = "the declared logical objective has no supported "
                        "Clifford or signed X/Z instrument semantics";
        return result;
      }
    }
    if (result.mismatch) {
      result.reason = "the standard objective has an incompatible typed "
                      "quantum/classical signature";
      return result;
    }
    result.action = std::move(action);
    if (!operands) {
      if (kind == "cx")
        names = {"control", "target"};
      else if (kind == "cz")
        names = {"left", "right"};
    }
  }
  result.label = semantics
                     ? cast<StringAttr>(logical->getAttr("sym_name")).str()
                     : ("ql.logical." + kind).str();
  if (!result.action)
    return result;
  unsigned inputCount = 0, outputCount = 0, actualBits = 0;
  for (const auto &port : inputs)
    inputCount += port.boundary.logicalQubits();
  for (const auto &port : outputs)
    outputCount += port.boundary.logicalQubits();
  for (Type output : spec.getFunctionType().getResults())
    actualBits += output.isInteger(1);
  bool partialClifford =
      result.action->kind == IdealLogicalAction::Kind::Clifford &&
      inputCount == outputCount && signature.inputs == signature.outputs &&
      signature.inputs < inputCount;
  if ((!partialClifford &&
       (signature.inputs != inputCount || signature.outputs != outputCount)) ||
      signature.bits != actualBits) {
    result.action.reset();
    result.mismatch = true;
    result.reason = "the declared objective quantum/classical signature "
                    "differs from the gadget boundary";
    return result;
  }
  unsigned formalCount = std::max(signature.inputs, signature.outputs);
  if (names.empty())
    for (unsigned index = 0; index < formalCount; ++index)
      names.push_back("q" + std::to_string(index));
  if (names.size() != formalCount) {
    logical->emitError(
        "logical operand names do not cover the supported quantum objective");
    return failure();
  }
  llvm::DenseSet<llvm::StringRef> seenNames;
  for (const std::string &name : names)
    if (name.empty() || !seenNames.insert(name).second) {
      logical->emitError("logical operand names must be nonempty and unique");
      return failure();
    }
  auto &action = *result.action;
  if (signature.inputs) {
    auto binding = bindLogicalSide(spec, projection.inputPorts, inputs, names);
    if (failed(binding))
      return failure();
    action.inputBindings = std::move(*binding);
  }
  if (signature.outputs) {
    auto binding =
        bindLogicalSide(spec, projection.outputPorts, outputs, names);
    if (failed(binding))
      return failure();
    if (action.kind == IdealLogicalAction::Kind::Measurement &&
        !action.outputBindings.empty()) {
      auto returned = action.outputBindings;
      for (unsigned input = 0; input < returned.size(); ++input)
        action.outputBindings[input] = (*binding)[returned[input]];
    } else
      action.outputBindings = std::move(*binding);
  }
  if (partialClifford) {
    auto selectedInputs = action.inputBindings;
    auto selectedOutputs = action.outputBindings;
    llvm::sort(selectedInputs);
    llvm::sort(selectedOutputs);
    if (selectedInputs != selectedOutputs) {
      result.action.reset();
      result.reason = "a partial Clifford objective must preserve the selected "
                      "boundary logical subset";
      return result;
    }
    // The declared operation acts on selected logicals; every spectator must
    // realize identity. Lift to a full boundary action before the Choi check.
    IdealLogicalAction embedded = identityAction(inputCount);
    for (unsigned row = 0; row < 2 * action.arity; ++row) {
      unsigned input = action.inputBindings[row % action.arity];
      unsigned target = input + (row < action.arity ? 0 : inputCount);
      Bits image(2 * inputCount);
      for (unsigned column = 0; column < action.arity; ++column) {
        image[action.outputBindings[column]] = action.images[row][column];
        image[inputCount + action.outputBindings[column]] =
            action.images[row][action.arity + column];
      }
      embedded.images[target] = std::move(image);
      embedded.phases[target] = action.phases[row];
    }
    action = std::move(embedded);
    action.inputBindings.resize(inputCount);
    action.outputBindings.resize(outputCount);
    std::iota(action.inputBindings.begin(), action.inputBindings.end(), 0);
    std::iota(action.outputBindings.begin(), action.outputBindings.end(), 0);
  }
  if (action.kind == IdealLogicalAction::Kind::Measurement) {
    auto outcome = spec.getOutcomeMap();
    auto records = outcome ? outcome->getAs<ArrayAttr>("records") : ArrayAttr{};
    auto rows = outcome ? outcome->getAs<DenseIntElementsAttr>("rows")
                        : DenseIntElementsAttr{};
    auto constants = outcome ? outcome->getAs<DenseI64ArrayAttr>("constants")
                             : DenseI64ArrayAttr{};
    if (!records || !rows || !constants || rows.getType().getRank() != 2 ||
        rows.getType().getShape()[0] != 1 || constants.size() != 1 ||
        rows.getType().getShape()[1] != static_cast<int64_t>(records.size())) {
      spec.emitOpError("measurement verification requires one complete affine "
                       "outcome-map row");
      return failure();
    }
    action.outcomeConstant ^= constants[0] != 0;
    auto coefficients = rows.getValues<APInt>();
    unsigned column = 0;
    Bits selected(projection.measurementCount);
    for (const APInt &coefficient : coefficients) {
      if (!coefficient.isZero()) {
        auto name = dyn_cast<StringAttr>(records[column]);
        auto found = name ? recordMap.find(name.str()) : recordMap.end();
        if (found == recordMap.end() && name)
          found = recordMap.find(
              (profile.getGadget() + "." + name.getValue()).str());
        if (found == recordMap.end()) {
          spec.emitOpError("measurement outcome map references an unknown "
                           "realization record");
          return failure();
        }
        selected.flip(found->second);
      }
      ++column;
    }
    for (int record : selected.set_bits())
      action.outcomeRecords.push_back(record);
    if (auto syndromeRows = outcome->getAs<ArrayAttr>("input_syndromes")) {
      if (syndromeRows.size() != 1) {
        spec.emitOpError("measurement input-syndrome map must contain one row");
        return failure();
      }
      auto terms = dyn_cast<ArrayAttr>(syndromeRows[0]);
      if (!terms)
        return spec.emitOpError("measurement syndrome row must be an array"),
               failure();
      Bits kept;
      unsigned keptCount = 0;
      for (const auto &port : inputs)
        keptCount += port.boundary.keptSyndromes();
      kept.resize(keptCount);
      for (Attribute rawTerm : terms) {
        auto term = dyn_cast<DictionaryAttr>(rawTerm);
        auto portAttr = term ? term.getAs<IntegerAttr>("port") : IntegerAttr{};
        auto indexAttr =
            term ? term.getAs<IntegerAttr>("index") : IntegerAttr{};
        if (!portAttr || !indexAttr || portAttr.getInt() < 0 ||
            indexAttr.getInt() < 0 ||
            static_cast<uint64_t>(portAttr.getInt()) >= inputs.size())
          return spec.emitOpError(
                     "measurement syndrome term has an invalid endpoint"),
                 failure();
        unsigned port = portAttr.getInt(), index = indexAttr.getInt();
        if (index >= inputs[port].boundary.effectiveSyndromes())
          return spec.emitOpError(
                     "measurement syndrome term has an invalid index"),
                 failure();
        unsigned offset = 0;
        for (unsigned previous = 0; previous < port; ++previous)
          offset += inputs[previous].boundary.keptSyndromes();
        for (int bit : inputs[port]
                           .boundary.effectiveToKeptDecomposition()[index]
                           .set_bits())
          kept.flip(offset + bit);
        action.outcomeConstant ^= inputs[port].boundary.effectiveSigns()[index];
      }
      for (int bit : kept.set_bits())
        action.outcomeInputSyndromes.push_back(bit);
    }
  }
  return result;
}

/// Parses the optional declared logical action (a JSON object with `arity`,
/// `matrix` rows of 0/1, and `phases`); null or empty means none.
FailureOr<std::optional<IdealLogicalAction>> parseAction(ModuleOp module,
                                                         StringRef text) {
  if (text.trim().empty())
    return std::optional<IdealLogicalAction>();
  auto parsed = llvm::json::parse(text);
  if (!parsed) {
    module.emitError() << "invalid logical action: "
                       << llvm::toString(parsed.takeError());
    return failure();
  }
  if (parsed->kind() == llvm::json::Value::Null)
    return std::optional<IdealLogicalAction>();
  auto *object = parsed->getAsObject();
  auto arity = object ? object->getInteger("arity") : std::nullopt;
  auto *matrix = object ? object->getArray("matrix") : nullptr;
  auto *phases = object ? object->getArray("phases") : nullptr;
  if (!arity || *arity <= 0 ||
      static_cast<uint64_t>(*arity) >
          std::numeric_limits<unsigned>::max() / 2 ||
      !matrix || !phases) {
    module.emitError("logical action requires arity, matrix, and phases");
    return failure();
  }
  IdealLogicalAction action;
  action.arity = static_cast<unsigned>(*arity);
  const std::size_t width = 2 * std::size_t{action.arity};
  if (matrix->size() != width || phases->size() != width) {
    module.emitError("logical action matrix and phases must have 2 * arity "
                     "rows");
    return failure();
  }
  for (const llvm::json::Value &rawRow : *matrix) {
    const auto *row = rawRow.getAsArray();
    if (!row || row->size() != width) {
      module.emitError("logical action matrix must be square, 2 * arity wide");
      return failure();
    }
    Bits bits(width);
    for (auto [column, entry] : llvm::enumerate(*row)) {
      auto value = entry.getAsInteger();
      if (!value || (*value != 0 && *value != 1)) {
        module.emitError("logical action matrix entries must be 0 or 1");
        return failure();
      }
      bits[column] = *value;
    }
    action.images.push_back(std::move(bits));
  }
  for (const llvm::json::Value &entry : *phases) {
    auto value = entry.getAsInteger();
    if (!value || (*value != 0 && *value != 1)) {
      module.emitError("logical action phases must be 0 or 1");
      return failure();
    }
    action.phases.push_back(*value);
  }
  for (unsigned left = 0; left < width; ++left)
    for (unsigned right = 0; right < width; ++right) {
      bool parity = false;
      for (unsigned qubit = 0; qubit < action.arity; ++qubit)
        parity ^= (action.images[left][qubit] &&
                   action.images[right][action.arity + qubit]) !=
                  (action.images[left][action.arity + qubit] &&
                   action.images[right][qubit]);
      bool expected = (left < action.arity && right == action.arity + left) ||
                      (right < action.arity && left == action.arity + right);
      if (parity != expected) {
        module.emitError("logical action matrix must be symplectic");
        return failure();
      }
    }
  return std::optional<IdealLogicalAction>(std::move(action));
}

const char *actionRowStatusName(IdealActionRowStatus status) {
  switch (status) {
  case IdealActionRowStatus::Verified:
    return "verified";
  case IdealActionRowStatus::NotStabilized:
    return "not_stabilized";
  case IdealActionRowStatus::SignMismatch:
    return "sign_mismatch";
  case IdealActionRowStatus::RandomSign:
    return "random_sign";
  }
  return "unknown";
}

} // namespace

FailureOr<PackedGadgetEDEM>
deriveGadgetEDEM(ModuleOp module, StringRef contractName, StringRef faultJson) {
  if (failed(verify(module)))
    return failure();
  auto faults = edem_analysis::parseFaultInput(faultJson);
  if (!faults) {
    module.emitError() << "invalid EDEM fault input: "
                       << llvm::toString(faults.takeError());
    return failure();
  }
  auto selected = selectContract(module, contractName);
  if (failed(selected))
    return failure();
  auto projection = projectClifford(module, selected->getGadget());
  if (failed(projection))
    return failure();
  return buildGadgetEDEM(*selected, *projection, *faults);
}

FailureOr<std::string> verifyGadgetContract(ModuleOp module,
                                            StringRef contractName,
                                            StringRef actionJson) {
  if (failed(verify(module)))
    return failure();
  bool fromSpec = false;
  if (!actionJson.trim().empty()) {
    auto request = llvm::json::parse(actionJson);
    if (request) {
      if (auto *object = request->getAsObject();
          object && object->get("source")) {
        if (object->getString("source") != "gadget_spec" ||
            object->size() != 1) {
          module.emitError(
              "logical objective request requires only source=gadget_spec");
          return failure();
        }
        fromSpec = true;
      }
    } else
      llvm::consumeError(request.takeError());
  }
  auto action = parseAction(module, fromSpec ? StringRef{} : actionJson);
  if (failed(action))
    return failure();
  auto selected = selectContract(module, contractName);
  if (failed(selected))
    return failure();
  qlx::fabric::GadgetContractOp profile = *selected;
  auto projection = projectClifford(module, profile.getGadget());
  if (failed(projection))
    return failure();
  if (!profile.getBoundaryComplete()) {
    profile.emitOpError("verification requires a complete selected boundary "
                        "contract");
    return failure();
  }
  auto inputs = readPorts(profile, profile.getInputs(),
                          projection->inputDataCarriersByPort, "inputs");
  auto outputs = readPorts(profile, profile.getOutputs(),
                           projection->outputDataCarriersByPort, "outputs");
  if (failed(inputs) || failed(outputs))
    return failure();
  auto records = readRecordMap(profile, *projection);
  if (failed(records))
    return failure();
  auto rows =
      readProfileRows(profile, *projection, *records, *inputs, *outputs);
  if (failed(rows))
    return failure();
  ObjectiveResolution objective;
  if (fromSpec) {
    auto resolved = resolveGadgetObjective(module, profile, *projection,
                                           *inputs, *outputs, *records);
    if (failed(resolved))
      return failure();
    objective = std::move(*resolved);
    *action = objective.action;
  }
  auto report = evaluateIdealProfile(profile, *projection, *inputs, *outputs,
                                     *rows, *action ? &**action : nullptr);
  if (failed(report))
    return failure();

  llvm::json::Array failures;
  for (const ContractFailure &failure : report->failures) {
    llvm::json::Object entry;
    entry["kind"] = failure.kind;
    if (failure.index)
      entry["index"] = static_cast<int64_t>(*failure.index);
    entry["message"] = failure.message;
    failures.push_back(std::move(entry));
  }
  llvm::json::Object contract;
  contract["ok"] = report->failures.empty();
  contract["outcome_count"] = static_cast<int64_t>(report->outcomeCount);
  contract["random_bits"] = static_cast<int64_t>(report->randomBits);
  contract["outcome_code_rank"] = static_cast<int64_t>(report->outcomeCodeRank);
  contract["detector_rank"] = static_cast<int64_t>(report->detectorRank);
  contract["deterministic_logical_outcome_rank"] =
      static_cast<int64_t>(report->logicalOutcomeRank);
  contract["detector_count"] = static_cast<int64_t>(rows->detectors.size());
  contract["output_relation_count"] =
      static_cast<int64_t>(rows->outputVirtual.size());
  contract["observable_count"] = static_cast<int64_t>(rows->observables.size());
  contract["success_count"] = static_cast<int64_t>(rows->success.size());
  contract["failures"] = std::move(failures);

  llvm::json::Object logical;
  logical["source"] = fromSpec ? "gadget_spec" : "explicit_action";
  if (!objective.label.empty())
    logical["action"] = objective.label;
  if (fromSpec && !objective.action) {
    logical["status"] = objective.mismatch ? "failed" : "unsupported";
    logical["reason"] = objective.reason;
    if (objective.mismatch) {
      logical["rows"] = llvm::json::Array{};
      logical["labels"] = llvm::json::Array{};
      logical["logical_rank"] = 0;
    }
  } else {
    switch (report->action.status) {
    case IdealActionReport::Status::Unchecked:
      logical["status"] = "unchecked";
      break;
    case IdealActionReport::Status::Unsupported:
      logical["status"] = "unsupported";
      logical["reason"] = report->action.reason;
      break;
    case IdealActionReport::Status::Checked: {
      logical["status"] = report->action.verified() ? "verified" : "failed";
      logical["logical_rank"] = static_cast<int64_t>(
          llvm::count(report->action.rows, IdealActionRowStatus::Verified));
      llvm::json::Array statuses;
      for (IdealActionRowStatus row : report->action.rows)
        statuses.push_back(actionRowStatusName(row));
      logical["rows"] = std::move(statuses);
      llvm::json::Array labels;
      for (const auto &label : report->action.labels)
        labels.push_back(label);
      logical["labels"] = std::move(labels);
      break;
    }
    }
  }

  llvm::json::Object result;
  result["schema"] = "cudaq.logical.gadget-verification/v2";
  result["contract"] = profile.getSymName().str();
  result["gadget"] = profile.getGadget().str();
  result["detector_contract"] = std::move(contract);
  result["logical_action"] = std::move(logical);
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << llvm::json::Value(std::move(result));
  return text;
}

} // namespace cudaq::logical
