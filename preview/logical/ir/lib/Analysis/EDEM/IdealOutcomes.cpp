/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Analysis/IdealOutcomes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace cudaq::logical {
namespace {

using Kind = ProjectionInstruction::Kind;
using Factor = ProjectionInstruction::PauliFactor;

llvm::Error invalid(const std::string &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.c_str());
}

bool bit(const std::vector<std::uint64_t> &words, std::size_t index) {
  return (words[index / 64] >> (index % 64)) & 1u;
}

void flip(std::vector<std::uint64_t> &words, std::size_t index) {
  words[index / 64] ^= std::uint64_t{1} << (index % 64);
}

void set(std::vector<std::uint64_t> &words, std::size_t index) {
  words[index / 64] |= std::uint64_t{1} << (index % 64);
}

void xorAffine(IdealAffineBit &lhs, const IdealAffineBit &rhs) {
  lhs.constant ^= rhs.constant;
  if (lhs.randomWords.size() < rhs.randomWords.size())
    lhs.randomWords.resize(rhs.randomWords.size());
  for (std::size_t i = 0; i < rhs.randomWords.size(); ++i)
    lhs.randomWords[i] ^= rhs.randomWords[i];
}

struct PauliRow {
  std::vector<std::uint64_t> x;
  std::vector<std::uint64_t> z;
  // i^phase X^x Z^z, where phase is modulo four. The independent affine sign
  // describes which joint eigenspace has eigenvalue +1.
  unsigned phase = 0;
  IdealAffineBit sign;

  explicit PauliRow(std::size_t words) : x(words), z(words) {}

  bool xAt(std::size_t carrier) const { return bit(x, carrier); }
  bool zAt(std::size_t carrier) const { return bit(z, carrier); }

  void toggleX(std::size_t carrier) { flip(x, carrier); }
  void toggleZ(std::size_t carrier) { flip(z, carrier); }

  /// Matrix multiplication in the i^phase X^x Z^z convention.
  void multiply(const PauliRow &rhs) {
    unsigned crossing = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      crossing ^= std::popcount(z[i] & rhs.x[i]) & 1u;
      x[i] ^= rhs.x[i];
      z[i] ^= rhs.z[i];
    }
    phase = (phase + rhs.phase + 2u * crossing) & 3u;
    xorAffine(sign, rhs.sign);
  }

  bool anticommutes(const PauliRow &rhs) const {
    unsigned parity = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
      parity ^=
          (std::popcount(x[i] & rhs.z[i]) ^ std::popcount(z[i] & rhs.x[i])) &
          1u;
    return parity != 0;
  }

  bool vectorBit(std::size_t index, std::size_t carriers) const {
    return index < carriers ? xAt(index) : zAt(index - carriers);
  }
};

unsigned rank(std::vector<std::vector<std::uint64_t>> rows,
              std::size_t columns);

unsigned pauliRank(llvm::ArrayRef<PauliRow> paulis, std::size_t carriers) {
  const std::size_t columns = 2 * carriers;
  std::vector<std::vector<std::uint64_t>> rows;
  rows.reserve(paulis.size());
  for (const PauliRow &pauli : paulis) {
    std::vector<std::uint64_t> row((columns + 63) / 64);
    for (std::size_t carrier = 0; carrier < carriers; ++carrier) {
      if (pauli.xAt(carrier))
        set(row, carrier);
      if (pauli.zAt(carrier))
        set(row, carriers + carrier);
    }
    rows.push_back(std::move(row));
  }
  return rank(std::move(rows), columns);
}

/// The generators describe a stabilizer density operator on physical carriers
/// and inaccessible reference carriers. All generator matrices commute and
/// are independent; their signs may depend on prior ideal random outcomes.
class Tableau {
public:
  Tableau(unsigned physicalCarriers, unsigned references)
      : physicalCarriers_(physicalCarriers),
        totalCarriers_(std::size_t{physicalCarriers} + references),
        words_((totalCarriers_ + 63) / 64), initialized_(physicalCarriers) {}

  void markInputCarrier(unsigned carrier) { initialized_[carrier] = true; }

  void addGenerator(PauliRow row) { generators_.push_back(std::move(row)); }

  IdealAffineBit inputSyndrome() { return freshRandom(); }

  unsigned referenceCarrier(unsigned reference) const {
    return physicalCarriers_ + reference;
  }

  llvm::Error validateInitialState(std::size_t initializedDataCarriers,
                                   std::size_t references) const {
    if (generators_.size() != initializedDataCarriers + references)
      return invalid("encoded input basis does not fix its physical data and "
                     "Choi reference state");
    for (std::size_t i = 0; i < generators_.size(); ++i) {
      if (generators_[i].phase & 1u)
        return invalid("encoded input generator is not Hermitian");
      for (std::size_t j = 0; j < i; ++j)
        if (generators_[i].anticommutes(generators_[j]))
          return invalid("encoded input generators do not commute");
    }
    if (pauliRank(generators_, totalCarriers_) != generators_.size())
      return invalid("encoded input generators are linearly dependent");
    return llvm::Error::success();
  }

  bool initialized(unsigned carrier) const { return initialized_[carrier]; }

  void gate(Kind kind, unsigned a, unsigned b = 0) {
    for (PauliRow &row : generators_) {
      const bool xa = row.xAt(a);
      const bool za = row.zAt(a);
      switch (kind) {
      case Kind::H:
        if (xa != za) {
          row.toggleX(a);
          row.toggleZ(a);
        }
        row.phase = (row.phase + 2u * (xa && za)) & 3u;
        break;
      case Kind::S:
        if (xa)
          row.toggleZ(a);
        row.phase = (row.phase + unsigned(xa)) & 3u;
        break;
      case Kind::SDag:
        if (xa)
          row.toggleZ(a);
        row.phase = (row.phase + 4u - unsigned(xa)) & 3u;
        break;
      case Kind::X:
        row.phase = (row.phase + 2u * unsigned(za)) & 3u;
        break;
      case Kind::Z:
        row.phase = (row.phase + 2u * unsigned(xa)) & 3u;
        break;
      case Kind::CX:
        if (xa)
          row.toggleX(b);
        if (row.zAt(b))
          row.toggleZ(a);
        break;
      case Kind::CZ: {
        const bool xb = row.xAt(b);
        if (xb)
          row.toggleZ(a);
        if (xa)
          row.toggleZ(b);
        row.phase = (row.phase + 2u * unsigned(xa && xb)) & 3u;
        break;
      }
      case Kind::Swap: {
        const bool xb = row.xAt(b);
        const bool zb = row.zAt(b);
        if (xa != xb) {
          row.toggleX(a);
          row.toggleX(b);
        }
        if (za != zb) {
          row.toggleZ(a);
          row.toggleZ(b);
        }
        break;
      }
      default:
        break;
      }
    }
  }

  /// Measure a physical Pauli. The reported bit is the sign of the new
  /// stabilizer; it may be random even if the measured Pauli commutes with the
  /// current generators, because the density operator can be mixed.
  IdealAffineBit measure(PauliRow measured) {
    for (std::size_t pivot = 0; pivot < generators_.size(); ++pivot) {
      if (!generators_[pivot].anticommutes(measured))
        continue;
      PauliRow oldPivot = generators_[pivot];
      for (std::size_t i = 0; i < generators_.size(); ++i)
        if (i != pivot && generators_[i].anticommutes(measured))
          generators_[i].multiply(oldPivot);
      measured.sign = freshRandom();
      generators_[pivot] = std::move(measured);
      return generators_[pivot].sign;
    }

    // Eliminate the measured Pauli against a basis of the generator span.
    // A commuting Pauli outside that span is also an ideal random outcome.
    std::vector<PauliRow> basis;
    basis.reserve(generators_.size());
    std::vector<int> pivotOf(2 * totalCarriers_, -1);
    for (const PauliRow &generator : generators_) {
      PauliRow row = generator;
      for (std::size_t column = 0; column < pivotOf.size(); ++column) {
        if (!row.vectorBit(column, totalCarriers_))
          continue;
        if (pivotOf[column] < 0) {
          pivotOf[column] = static_cast<int>(basis.size());
          basis.push_back(std::move(row));
          break;
        }
        row.multiply(basis[pivotOf[column]]);
      }
    }
    PauliRow remainder = measured;
    for (std::size_t column = 0; column < pivotOf.size(); ++column) {
      if (!remainder.vectorBit(column, totalCarriers_))
        continue;
      if (pivotOf[column] < 0) {
        measured.sign = freshRandom();
        IdealAffineBit result = measured.sign;
        generators_.push_back(std::move(measured));
        return result;
      }
      remainder.multiply(basis[pivotOf[column]]);
    }
    // P times a matching stabilizer product must be +I or -I. Any odd
    // phase indicates malformed, non-Hermitian input; construction excludes it.
    remainder.sign.constant ^= (remainder.phase == 2);
    return remainder.sign;
  }

  /// Value of a Pauli observable in the current state as an affine bit
  /// (1 means eigenvalue -1), or nullopt when the observable is not, up to
  /// sign, in the stabilizer group. Does not modify the state. A residual
  /// odd phase means the observable was not Hermitian and is an error.
  llvm::Expected<std::optional<IdealAffineBit>>
  eigenvalue(PauliRow pauli) const {
    std::vector<PauliRow> basis;
    basis.reserve(generators_.size());
    std::vector<int> pivotOf(2 * totalCarriers_, -1);
    for (const PauliRow &generator : generators_) {
      PauliRow row = generator;
      for (std::size_t column = 0; column < pivotOf.size(); ++column) {
        if (!row.vectorBit(column, totalCarriers_))
          continue;
        if (pivotOf[column] < 0) {
          pivotOf[column] = static_cast<int>(basis.size());
          basis.push_back(std::move(row));
          break;
        }
        row.multiply(basis[pivotOf[column]]);
      }
    }
    for (std::size_t column = 0; column < pivotOf.size(); ++column) {
      if (!pauli.vectorBit(column, totalCarriers_))
        continue;
      if (pivotOf[column] < 0)
        return std::optional<IdealAffineBit>();
      pauli.multiply(basis[pivotOf[column]]);
    }
    if (pauli.phase & 1u)
      return invalid("expected logical stabilizer is not Hermitian");
    pauli.sign.constant ^= (pauli.phase == 2);
    return std::optional<IdealAffineBit>(std::move(pauli.sign));
  }

  void reset(unsigned carrier, bool xBasis) {
    PauliRow observable(words_);
    if (xBasis)
      set(observable.x, carrier);
    else
      set(observable.z, carrier);
    IdealAffineBit outcome = measure(std::move(observable));
    // Conditional Z after X measurement, or X after Z measurement, turns the
    // measured eigenspace into the +1 eigenspace. The hidden outcome remains
    // in other generators when the old carrier was entangled with them.
    for (PauliRow &row : generators_)
      if (xBasis ? row.xAt(carrier) : row.zAt(carrier))
        xorAffine(row.sign, outcome);
    initialized_[carrier] = true;
  }

  unsigned randomBits() const { return randomBits_; }
  std::size_t words() const { return words_; }

private:
  IdealAffineBit freshRandom() {
    IdealAffineBit value;
    value.randomWords.resize((std::size_t{randomBits_} + 64) / 64);
    set(value.randomWords, randomBits_++);
    return value;
  }

  unsigned physicalCarriers_;
  std::size_t totalCarriers_;
  std::size_t words_;
  std::vector<bool> initialized_;
  std::vector<PauliRow> generators_;
  unsigned randomBits_ = 0;
};

unsigned rank(std::vector<std::vector<std::uint64_t>> rows,
              std::size_t columns) {
  unsigned pivotRow = 0;
  for (std::size_t column = 0; column < columns && pivotRow < rows.size();
       ++column) {
    std::size_t candidate = pivotRow;
    while (candidate < rows.size() && !bit(rows[candidate], column))
      ++candidate;
    if (candidate == rows.size())
      continue;
    std::swap(rows[pivotRow], rows[candidate]);
    for (std::size_t other = pivotRow + 1; other < rows.size(); ++other)
      if (bit(rows[other], column))
        for (std::size_t word = 0; word < rows[other].size(); ++word)
          rows[other][word] ^= rows[pivotRow][word];
    ++pivotRow;
  }
  return pivotRow;
}

llvm::Expected<PauliRow> embed(llvm::SmallBitVector const &physical,
                               llvm::ArrayRef<unsigned> carriers,
                               unsigned carrierCount, std::size_t words) {
  const std::size_t n = carriers.size();
  if (physical.size() != 2 * n)
    return invalid("boundary Pauli row has the wrong physical width");
  PauliRow result(words);
  bool nonidentity = false;
  for (std::size_t i = 0; i < n; ++i) {
    const unsigned carrier = carriers[i];
    if (carrier >= carrierCount)
      return invalid("boundary Pauli row has an out-of-range carrier");
    if (physical[i]) {
      set(result.x, carrier);
      nonidentity = true;
    }
    if (physical[n + i]) {
      set(result.z, carrier);
      nonidentity = true;
    }
    if (physical[i] && physical[n + i])
      result.phase = (result.phase + 1) & 3u;
  }
  if (!nonidentity)
    return invalid("boundary Pauli row must not be identity");
  return result;
}

unsigned deterministicDimension(llvm::ArrayRef<IdealAffineBit> expressions,
                                unsigned randomBits) {
  std::vector<std::vector<std::uint64_t>> rows;
  rows.reserve(expressions.size());
  const std::size_t words = (std::size_t{randomBits} + 63) / 64;
  for (const IdealAffineBit &expression : expressions) {
    std::vector<std::uint64_t> row(words);
    std::copy(expression.randomWords.begin(), expression.randomWords.end(),
              row.begin());
    rows.push_back(std::move(row));
  }
  return static_cast<unsigned>(expressions.size()) -
         rank(std::move(rows), randomBits);
}

} // namespace

unsigned IdealOutcomeAnalysis::outcomeDeterministicRelationDimension() const {
  return deterministicDimension(outcomes_, randomBits_);
}

llvm::Expected<IdealOutcomeAnalysis>
analyzeIdealOutcomes(const Projection &projection,
                     llvm::ArrayRef<IdealBoundaryPort> inputPorts,
                     llvm::ArrayRef<IdealBoundaryPort> outputPorts,
                     const IdealLogicalAction *action) {
  if (inputPorts.size() != projection.inputDataCarriersByPort.size() ||
      outputPorts.size() != projection.outputDataCarriersByPort.size())
    return invalid("encoded boundary ports do not cover the projection");
  std::size_t inputs = 0;
  std::size_t references = 0;
  for (std::size_t i = 0; i < inputPorts.size(); ++i) {
    const IdealBoundaryPort &port = inputPorts[i];
    if (port.carriers != projection.inputDataCarriersByPort[i])
      return invalid("input code carriers disagree with projection order");
    if (port.logicalX.size() != port.logicalZ.size() ||
        port.keptStabilizers.size() + port.logicalX.size() !=
            port.carriers.size())
      return invalid("input code basis does not span its physical carriers");
    inputs += port.carriers.size();
    references += port.logicalX.size();
  }
  if (references > std::numeric_limits<unsigned>::max() -
                       std::size_t{projection.carrierCount})
    return invalid("projection has too many physical and Choi carriers");

  Tableau tableau(projection.carrierCount, static_cast<unsigned>(references));
  std::vector<bool> seenInput(projection.carrierCount);
  unsigned reference = 0;
  IdealOutcomeAnalysis result;
  for (const IdealBoundaryPort &port : inputPorts) {
    for (unsigned carrier : port.carriers) {
      if (carrier >= projection.carrierCount || seenInput[carrier])
        return invalid("input data carrier is out of range or repeated");
      seenInput[carrier] = true;
      tableau.markInputCarrier(carrier);
    }
    for (const auto &stabilizer : port.keptStabilizers) {
      auto row = embed(stabilizer, port.carriers, projection.carrierCount,
                       tableau.words());
      if (!row)
        return row.takeError();
      row->sign = tableau.inputSyndrome();
      result.outcomes_.push_back(row->sign);
      tableau.addGenerator(std::move(*row));
    }
    for (std::size_t logical = 0; logical < port.logicalX.size(); ++logical) {
      auto x = embed(port.logicalX[logical], port.carriers,
                     projection.carrierCount, tableau.words());
      if (!x)
        return x.takeError();
      auto z = embed(port.logicalZ[logical], port.carriers,
                     projection.carrierCount, tableau.words());
      if (!z)
        return z.takeError();
      set(x->x, tableau.referenceCarrier(reference));
      set(z->z, tableau.referenceCarrier(reference));
      ++reference;
      tableau.addGenerator(std::move(*x));
      tableau.addGenerator(std::move(*z));
    }
  }
  if (auto error = tableau.validateInitialState(inputs, references))
    return error;

  result.inputCount_ = static_cast<unsigned>(result.outcomes_.size());
  result.outcomes_.reserve(result.outcomes_.size() +
                           projection.measurementCount);
  for (const ProjectionInstruction &instruction : projection.instructions) {
    const auto fail = [&](llvm::StringRef reason) {
      return invalid("projection instruction '" + instruction.siteId +
                     "': " + reason.str());
    };
    const Kind kind = instruction.kind;
    const bool isMeasure =
        kind == Kind::M || kind == Kind::MX || kind == Kind::MPP;
    const bool isReset = kind == Kind::R || kind == Kind::RX;
    const bool isBinary =
        kind == Kind::CX || kind == Kind::CZ || kind == Kind::Swap;
    if (!isMeasure && instruction.inverted)
      return fail("inversion is legal only for measurements");
    if (kind == Kind::MPP) {
      if (!instruction.carriers.empty() || instruction.paulis.empty())
        return fail("MPP requires a nonempty Pauli product only");
    } else if (!instruction.paulis.empty()) {
      return fail("Pauli factors are legal only for MPP");
    }
    if (kind != Kind::MPP && (instruction.carriers.empty() ||
                              (isBinary && instruction.carriers.size() % 2)))
      return fail("wrong number of carrier operands");
    if (isMeasure && instruction.measurementStart != result.recordCount_)
      return fail("measurement indices are not contiguous");
    for (unsigned carrier : instruction.carriers) {
      if (carrier >= projection.carrierCount)
        return fail("carrier index is out of range");
      if (!isReset && !tableau.initialized(carrier))
        return fail("carrier was used before preparation or reset");
    }
    for (const Factor &factor : instruction.paulis)
      if (factor.carrier >= projection.carrierCount ||
          !tableau.initialized(factor.carrier))
        return fail("MPP carrier is out of range or uninitialized");

    if (isReset) {
      for (unsigned carrier : instruction.carriers)
        tableau.reset(carrier, kind == Kind::RX);
    } else if (kind == Kind::MPP) {
      PauliRow pauli(tableau.words());
      llvm::SmallDenseSet<unsigned, 8> seen;
      for (const Factor &factor : instruction.paulis) {
        if (!seen.insert(factor.carrier).second)
          return fail("MPP repeats a physical carrier");
        switch (factor.pauli) {
        case ProjectionInstruction::Pauli::X:
          set(pauli.x, factor.carrier);
          break;
        case ProjectionInstruction::Pauli::Y:
          set(pauli.x, factor.carrier);
          set(pauli.z, factor.carrier);
          pauli.phase = (pauli.phase + 1) & 3u;
          break;
        case ProjectionInstruction::Pauli::Z:
          set(pauli.z, factor.carrier);
          break;
        }
      }
      IdealAffineBit outcome = tableau.measure(std::move(pauli));
      outcome.constant ^= instruction.inverted;
      result.outcomes_.push_back(std::move(outcome));
      ++result.recordCount_;
    } else if (kind == Kind::M || kind == Kind::MX) {
      for (unsigned carrier : instruction.carriers) {
        PauliRow pauli(tableau.words());
        if (kind == Kind::MX)
          set(pauli.x, carrier);
        else
          set(pauli.z, carrier);
        IdealAffineBit outcome = tableau.measure(std::move(pauli));
        outcome.constant ^= instruction.inverted;
        result.outcomes_.push_back(std::move(outcome));
        ++result.recordCount_;
      }
    } else if (isBinary) {
      for (std::size_t i = 0; i < instruction.carriers.size(); i += 2) {
        unsigned a = instruction.carriers[i];
        unsigned b = instruction.carriers[i + 1];
        if (a == b)
          return fail("binary gate repeats the same carrier");
        tableau.gate(kind, a, b);
      }
    } else {
      switch (kind) {
      case Kind::H:
      case Kind::S:
      case Kind::SDag:
      case Kind::X:
      case Kind::Z:
        for (unsigned carrier : instruction.carriers)
          tableau.gate(kind, carrier);
        break;
      default:
        return fail("unsupported Clifford instruction");
      }
    }
  }
  if (result.recordCount_ != projection.measurementCount ||
      projection.recordOrder.size() != projection.measurementCount)
    return invalid("projection measurement count or record order disagrees "
                   "with typed instructions");
  for (const ProjectionRecord &record : projection.records)
    if (record.measurementIndex >= result.recordCount_)
      return invalid("record alias references a nonexistent measurement");

  std::vector<PauliRow> outputChecks;
  std::vector<PauliRow> outputLogicalX, outputLogicalZ;
  std::vector<bool> seenOutput(projection.carrierCount);
  for (std::size_t portIndex = 0; portIndex < outputPorts.size(); ++portIndex) {
    const IdealBoundaryPort &port = outputPorts[portIndex];
    if (port.carriers != projection.outputDataCarriersByPort[portIndex])
      return invalid("output code carriers disagree with projection order");
    for (unsigned carrier : port.carriers) {
      if (carrier >= projection.carrierCount || !tableau.initialized(carrier) ||
          seenOutput[carrier])
        return invalid("output code has an uninitialized, invalid, or "
                       "duplicate carrier");
      seenOutput[carrier] = true;
    }
    for (const auto &stabilizer : port.keptStabilizers) {
      auto row = embed(stabilizer, port.carriers, projection.carrierCount,
                       tableau.words());
      if (!row)
        return row.takeError();
      for (const PauliRow &previous : outputChecks)
        if (row->anticommutes(previous))
          return invalid("output stabilizer checks do not commute");
      outputChecks.push_back(std::move(*row));
    }
    if (action) {
      if (port.logicalX.size() != port.logicalZ.size())
        return invalid("output logical X and Z bases have different sizes");
      for (std::size_t logical = 0; logical < port.logicalX.size(); ++logical) {
        auto x = embed(port.logicalX[logical], port.carriers,
                       projection.carrierCount, tableau.words());
        if (!x)
          return x.takeError();
        auto z = embed(port.logicalZ[logical], port.carriers,
                       projection.carrierCount, tableau.words());
        if (!z)
          return z.takeError();
        outputLogicalX.push_back(std::move(*x));
        outputLogicalZ.push_back(std::move(*z));
      }
    }
  }
  if (pauliRank(outputChecks, projection.carrierCount) != outputChecks.size())
    return invalid("output kept-stabilizer checks are linearly dependent");
  result.outcomes_.reserve(result.outcomes_.size() + outputChecks.size());
  for (PauliRow &check : outputChecks)
    result.outcomes_.push_back(tableau.measure(std::move(check)));
  if (action) {
    result.action_.status = IdealActionReport::Status::Unsupported;
    auto compareAction = [&]() -> llvm::Error {
      auto identityBindings = [](unsigned count) {
        std::vector<unsigned> bindings;
        bindings.reserve(count);
        for (unsigned index = 0; index < count; ++index)
          bindings.push_back(index);
        return bindings;
      };
      auto validateBindings = [&](llvm::ArrayRef<unsigned> bindings,
                                  std::size_t count,
                                  llvm::StringRef side) -> llvm::Error {
        if (bindings.size() != count)
          return invalid("declared action " + side.str() +
                         " bindings do not cover the logical boundary");
        std::vector<bool> seen(count);
        for (unsigned index : bindings) {
          if (index >= count || seen[index])
            return invalid("declared action " + side.str() +
                           " binding is out of range or repeated");
          seen[index] = true;
        }
        return llvm::Error::success();
      };
      auto referenceLabel = [&](bool xBasis, unsigned index) {
        std::string label = xBasis ? "X_ref" : "Z_ref";
        if (references != 1)
          label += "[" + std::to_string(index) + "]";
        return label;
      };
      auto outputLabel = [&](char basis, unsigned index) {
        std::string label(1, basis);
        label += "\u0304";
        if (outputLogicalX.size() != 1)
          label += "[" + std::to_string(index) + "]";
        return label;
      };
      auto check = [&](PauliRow expected, IdealAffineBit expectedSign,
                       std::string label) -> llvm::Error {
        result.action_.labels.push_back(std::move(label));
        auto value = tableau.eigenvalue(std::move(expected));
        if (!value)
          return value.takeError();
        if (!*value) {
          result.action_.rows.push_back(IdealActionRowStatus::NotStabilized);
          return llvm::Error::success();
        }
        IdealAffineBit difference = std::move(**value);
        xorAffine(difference, expectedSign);
        if (llvm::any_of(difference.randomWords,
                         [](std::uint64_t word) { return word; }))
          result.action_.rows.push_back(IdealActionRowStatus::RandomSign);
        else if (difference.constant)
          result.action_.rows.push_back(IdealActionRowStatus::SignMismatch);
        else
          result.action_.rows.push_back(IdealActionRowStatus::Verified);
        return llvm::Error::success();
      };

      using ActionKind = IdealLogicalAction::Kind;
      using ActionBasis = IdealLogicalAction::Basis;
      if (action->kind == ActionKind::Clifford) {
        const unsigned arity = action->arity;
        if (!arity) {
          result.action_.reason =
              "an empty quantum objective requires classical outcome analysis";
          return llvm::Error::success();
        }
        if (action->images.size() != 2 * std::size_t{arity} ||
            action->phases.size() != 2 * std::size_t{arity} ||
            llvm::any_of(action->images, [&](const llvm::SmallBitVector &row) {
              return row.size() != 2 * std::size_t{arity};
            }))
          return invalid("declared logical action has an inconsistent shape");
        if (arity != references || outputLogicalX.size() != references) {
          result.action_.reason =
              "the declared action acts on " + std::to_string(arity) +
              " logical qubit(s), but the gadget's input boundary carries " +
              std::to_string(references) + " and its output boundary carries " +
              std::to_string(outputLogicalX.size());
          return llvm::Error::success();
        }
        const auto inputBindings = action->inputBindings.empty()
                                       ? identityBindings(arity)
                                       : action->inputBindings;
        const auto outputBindings = action->outputBindings.empty()
                                        ? identityBindings(arity)
                                        : action->outputBindings;
        if (auto error = validateBindings(inputBindings, references, "input"))
          return error;
        if (auto error = validateBindings(outputBindings, outputLogicalX.size(),
                                          "output"))
          return error;
        result.action_.status = IdealActionReport::Status::Checked;
        for (unsigned row = 0; row < 2 * arity; ++row) {
          // Hermitian Y_j = i X_j Z_j. Explicit bindings are applied to
          // both the reference and each declared output Pauli factor.
          PauliRow expected(tableau.words());
          std::string output;
          for (unsigned qubit = 0; qubit < arity; ++qubit) {
            const bool x = action->images[row][qubit];
            const bool z = action->images[row][arity + qubit];
            const unsigned bound = outputBindings[qubit];
            if (x)
              expected.multiply(outputLogicalX[bound]);
            if (z)
              expected.multiply(outputLogicalZ[bound]);
            if (x && z)
              expected.phase = (expected.phase + 1) & 3u;
            if (x || z) {
              if (!output.empty())
                output += " ";
              output += outputLabel(x && z ? 'Y' : x ? 'X' : 'Z', bound);
            }
          }
          const bool xBasis = row < arity;
          const unsigned input = inputBindings[row % arity];
          set(xBasis ? expected.x : expected.z,
              tableau.referenceCarrier(input));
          IdealAffineBit sign;
          sign.constant = action->phases[row];
          std::string label = referenceLabel(xBasis, input) + " " +
                              (sign.constant ? "-" : "") + output;
          if (auto error =
                  check(std::move(expected), std::move(sign), std::move(label)))
            return error;
        }
        return llvm::Error::success();
      }

      if (action->basis != ActionBasis::X && action->basis != ActionBasis::Z)
        return invalid("declared instrument has an unsupported Pauli basis");
      const bool xBasis = action->basis == ActionBasis::X;
      const char basis = xBasis ? 'X' : 'Z';
      const auto &outputBasis = xBasis ? outputLogicalX : outputLogicalZ;
      if (action->kind == ActionKind::Preparation) {
        if (references != 0 || outputLogicalX.size() != 1 ||
            !action->inputBindings.empty() ||
            action->outputBindings.size() != 1) {
          result.action_.reason =
              "signed X/Z preparation requires no logical input and exactly "
              "one bound logical output";
          return llvm::Error::success();
        }
        if (auto error = validateBindings(action->outputBindings, 1, "output"))
          return error;
        result.action_.status = IdealActionReport::Status::Checked;
        IdealAffineBit sign;
        sign.constant = action->negative;
        return check(outputBasis[action->outputBindings[0]], std::move(sign),
                     (action->negative ? "-" : "") +
                         outputLabel(basis, action->outputBindings[0]));
      }
      if (action->kind != ActionKind::Measurement)
        return invalid("declared instrument has an unsupported kind");
      if (!references || action->inputBindings.size() != references ||
          (action->destructive
               ? !outputLogicalX.empty() || !action->outputBindings.empty()
               : outputLogicalX.size() != references ||
                     action->outputBindings.size() != references)) {
        result.action_.reason =
            "X/Z measurement bindings must cover every logical input and "
            "either discard every input or preserve every logical output";
        return llvm::Error::success();
      }
      if (auto error =
              validateBindings(action->inputBindings, references, "input"))
        return error;
      if (!action->destructive)
        if (auto error = validateBindings(action->outputBindings,
                                          outputLogicalX.size(), "output"))
          return error;
      if (action->target >= action->inputBindings.size())
        return invalid("declared measurement target is out of range");
      IdealAffineBit measuredSign;
      measuredSign.constant = action->outcomeConstant ^ action->negative;
      for (unsigned record : action->outcomeRecords) {
        if (record >= result.recordCount_)
          return invalid("declared measurement outcome references an invalid "
                         "realization record");
        xorAffine(measuredSign, result.outcomes_[result.inputCount_ + record]);
      }
      for (unsigned syndrome : action->outcomeInputSyndromes) {
        if (syndrome >= result.inputCount_)
          return invalid("declared measurement outcome references an invalid "
                         "input syndrome");
        xorAffine(measuredSign, result.outcomes_[syndrome]);
      }

      result.action_.status = IdealActionReport::Status::Checked;
      PauliRow measuredReference(tableau.words());
      const unsigned input = action->inputBindings[action->target];
      set(xBasis ? measuredReference.x : measuredReference.z,
          tableau.referenceCarrier(input));
      const std::string signLabel = action->negative ? "-" : "";
      if (auto error =
              check(std::move(measuredReference), measuredSign,
                    signLabel + referenceLabel(xBasis, input) + " = result[0]"))
        return error;
      if (!action->destructive) {
        const unsigned output = action->outputBindings[action->target];
        if (auto error =
                check(outputBasis[output], measuredSign,
                      signLabel + outputLabel(basis, output) + " = result[0]"))
          return error;
        for (unsigned formal = 0; formal < action->inputBindings.size();
             ++formal) {
          if (formal == action->target)
            continue;
          const unsigned boundInput = action->inputBindings[formal];
          const unsigned boundOutput = action->outputBindings[formal];
          for (bool untouchedX : {true, false}) {
            PauliRow preserved = untouchedX ? outputLogicalX[boundOutput]
                                            : outputLogicalZ[boundOutput];
            set(untouchedX ? preserved.x : preserved.z,
                tableau.referenceCarrier(boundInput));
            if (auto error =
                    check(std::move(preserved), IdealAffineBit{},
                          referenceLabel(untouchedX, boundInput) + " " +
                              outputLabel(untouchedX ? 'X' : 'Z', boundOutput)))
              return error;
          }
        }
      }
      return llvm::Error::success();
    };
    if (auto error = compareAction())
      return error;
  }
  result.randomBits_ = tableau.randomBits();
  const std::size_t randomWords = (std::size_t{result.randomBits_} + 63) / 64;
  for (IdealAffineBit &outcome : result.outcomes_)
    outcome.randomWords.resize(randomWords);
  return result;
}

} // namespace cudaq::logical
