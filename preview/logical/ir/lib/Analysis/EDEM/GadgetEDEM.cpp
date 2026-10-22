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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "cudaq/edem/core.h"
#include "mlir/IR/Verifier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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
readPorts(qlx::fabric::GadgetProfileOp profile, std::optional<ArrayAttr> values,
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
readRecordMap(qlx::fabric::GadgetProfileOp profile,
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
readProfileRows(qlx::fabric::GadgetProfileOp profile,
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
      if (observable.getIdx() !=
          static_cast<int64_t>(result.observables.size())) {
        observable.emitOpError("EDEM observable indices must be ordered");
        return failure();
      }
      result.observables.push_back(std::move(*parity));
    } else if (isa<qlx::fabric::SuccessOp>(&operation))
      result.success.push_back(std::move(*parity));
    else if (auto assignment =
                 dyn_cast<qlx::fabric::OutputSyndromeOp>(&operation)) {
      auto port = assignment.getPortIndex();
      auto index = assignment.getIndex();
      if (port < 0 || index < 0 ||
          static_cast<std::size_t>(port) + 1 >= outputOffsets.size() ||
          outputOffsets[port] + index >= outputOffsets[port + 1] ||
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

LogicalResult verifyIdealProfile(qlx::fabric::GadgetProfileOp profile,
                                 const Projection &projection,
                                 const std::vector<BoundaryPort> &inputs,
                                 const std::vector<BoundaryPort> &outputs,
                                 const ProfileRows &rows) {
  std::vector<IdealBoundaryPort> idealInputs, idealOutputs;
  idealInputs.reserve(inputs.size());
  idealOutputs.reserve(outputs.size());
  for (const auto &port : inputs)
    idealInputs.push_back(idealPort(port));
  for (const auto &port : outputs)
    idealOutputs.push_back(idealPort(port));
  auto result = analyzeIdealOutcomes(projection, idealInputs, idealOutputs);
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

  for (auto [index, row] : llvm::enumerate(rows.detectors)) {
    auto parity = authored(row);
    if (image(parity.support).any() || shift(parity)) {
      profile.emitOpError()
          << "EDEM detector " << index
          << " is not deterministically zero in the ideal gadget";
      return failure();
    }
  }
  for (auto [index, row] : llvm::enumerate(rows.success)) {
    auto parity = authored(row);
    if (image(parity.support).any() || shift(parity)) {
      profile.emitOpError() << "EDEM success row " << index
                            << " disagrees with its ideal accepted value";
      return failure();
    }
  }

  std::vector<Bits> declared;
  for (const auto &row : rows.detectors)
    declared.push_back(authored(row).support);
  for (auto [index, row] : llvm::enumerate(rows.outputVirtual)) {
    auto relation = authored(row, index);
    if (image(relation.support).any() || shift(relation)) {
      profile.emitOpError() << "EDEM output virtual-syndrome relation " << index
                            << " disagrees with the ideal output stabilizer";
      return failure();
    }
    declared.push_back(std::move(relation.support));
  }
  for (const auto &row : rows.success)
    declared.push_back(authored(row).support);
  // Observables may be random. Their deterministic combinations, modulo the
  // detector/assignment/success rows, complete the ideal outcome code. Take
  // the rank of the union: summing separate ranks double-counts overlap.
  for (const auto &row : rows.observables)
    declared.push_back(authored(row).support);
  std::vector<Bits> images;
  for (const auto &row : declared)
    images.push_back(image(row));
  const unsigned declaredRank =
      binaryRank(declared, outcomeCount) -
      binaryRank(std::move(images), ideal.independentRandomBits());
  const unsigned requiredRank = ideal.outcomeDeterministicRelationDimension();
  if (declaredRank != requiredRank) {
    profile.emitOpError()
        << "EDEM contract is rank-incomplete: ideal relation rank "
        << requiredRank << ", declared rank " << declaredRank;
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
    qlx::fabric::GadgetProfileOp profile, const Projection &projection,
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

FailureOr<Propagated> propagate(qlx::fabric::GadgetProfileOp profile,
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

FailureOr<PackedGadgetEDEM> assemble(qlx::fabric::GadgetProfileOp profile,
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
  std::vector<cudaq_edem_block> blocks;
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
    if (added)
      codes.push_back({name.data(), name.size(),
                       port.boundary.effectiveSyndromes(),
                       port.boundary.logicalQubits()});
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
      {unsigned(codes.size()), unsigned(inputs.size()),
       unsigned(outputs.size()), faults.faults.empty() ? 0u : 1u,
       detectorSize ? 1u : 0u, observableSize ? 1u : 0u, successSize ? 1u : 0u,
       blocks.size()},
      codes.data(),
      inputCodes.data(),
      outputCodes.data(),
      &distribution,
      &detectorSize,
      &observableSize,
      &successSize,
      blocks.data()};
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
buildGadgetEDEM(qlx::fabric::GadgetProfileOp profile,
                const Projection &projection,
                const edem_analysis::FaultInput &faults) {
  if (!profile.getBoundaryComplete()) {
    profile.emitOpError("EDEM requires a complete selected boundary profile");
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

} // namespace

FailureOr<PackedGadgetEDEM>
deriveGadgetEDEM(ModuleOp module, StringRef profileName, StringRef faultJson) {
  if (failed(verify(module)))
    return failure();
  auto faults = edem_analysis::parseFaultInput(faultJson);
  if (!faults) {
    module.emitError() << "invalid EDEM fault input: "
                       << llvm::toString(faults.takeError());
    return failure();
  }
  qlx::fabric::GadgetProfileOp selected;
  if (!profileName.empty()) {
    selected = module.lookupSymbol<qlx::fabric::GadgetProfileOp>(profileName);
  } else {
    for (auto candidate : module.getOps<qlx::fabric::GadgetProfileOp>()) {
      if (selected) {
        module.emitError(
            "EDEM requires profile= when multiple gadget profiles are present");
        return failure();
      }
      selected = candidate;
    }
  }
  if (!selected) {
    module.emitError("EDEM requires a selected fabric.gadget_profile");
    return failure();
  }
  auto projection = projectClifford(module, selected.getGadget());
  if (failed(projection))
    return failure();
  return buildGadgetEDEM(selected, *projection, *faults);
}

} // namespace cudaq::logical
