/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Analysis/IdealOutcomes.h"

#include "llvm/Support/Error.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace cudaq::logical;
using Gate = ProjectionInstruction::Kind;
using Action = IdealLogicalAction;
using Basis = Action::Basis;
unsigned checks = 0;
llvm::SmallBitVector bits(const char *text) {
  llvm::SmallBitVector result(std::char_traits<char>::length(text));
  for (unsigned i = 0; text[i]; ++i)
    if (text[i] == '1')
      result.set(i);
  return result;
}
void require(bool condition, const char *label) {
  if (!condition) {
    std::cerr << "FAILED: " << label << '\n';
    std::exit(1);
  }
  ++checks;
}
IdealBoundaryPort bare(unsigned carrier = 0) {
  return {{carrier}, {}, {bits("10")}, {bits("01")}};
}
IdealBoundaryPort pair() {
  return {
      {0, 1}, {}, {bits("1000"), bits("0100")}, {bits("0010"), bits("0001")}};
}
void add(Projection &p, Gate gate, std::vector<unsigned> carriers,
         bool inverted = false) {
  ProjectionInstruction instruction{};
  instruction.kind = gate;
  instruction.carriers = std::move(carriers);
  instruction.inverted = inverted;
  instruction.siteId = "site" + std::to_string(p.instructions.size());
  if (gate == Gate::M || gate == Gate::MX) {
    instruction.measurementStart = p.measurementCount;
    for (unsigned lane = 0; lane < instruction.carriers.size(); ++lane)
      p.recordOrder.push_back("m" + std::to_string(p.measurementCount++));
  }
  p.instructions.push_back(std::move(instruction));
}
IdealOutcomeAnalysis run(const Projection &p, std::vector<IdealBoundaryPort> in,
                         std::vector<IdealBoundaryPort> out, const Action &a) {
  auto result = analyzeIdealOutcomes(p, in, out, &a);
  if (!result) {
    std::cerr << llvm::toString(result.takeError()) << '\n';
    std::exit(2);
  }
  return std::move(*result);
}
Action measurement(bool preserving = false) {
  Action result;
  result.kind = Action::Kind::Measurement;
  result.inputBindings = {0};
  result.outcomeRecords = {0};
  result.destructive = !preserving;
  if (preserving)
    result.outputBindings = {0};
  return result;
}
} // namespace

int main() {
  {
    auto result = run(Projection{}, {}, {}, Action{});
    require(result.action().status == IdealActionReport::Status::Unsupported &&
                !result.action().verified(),
            "empty objective cannot certify an empty projection");
    IdealActionReport report;
    report.status = IdealActionReport::Status::Checked;
    require(!report.verified(), "empty checked report cannot pass vacuously");
  }
  for (auto basis : {Basis::X, Basis::Z})
    for (bool negative : {false, true}) {
      Projection p;
      p.carrierCount = 1;
      p.outputDataCarriersByPort = {{0}};
      add(p, basis == Basis::X ? Gate::RX : Gate::R, {0});
      if (negative)
        add(p, basis == Basis::X ? Gate::Z : Gate::X, {0});
      Action a;
      a.kind = Action::Kind::Preparation;
      a.basis = basis;
      a.negative = negative;
      a.outputBindings = {0};
      require(run(p, {}, {bare()}, a).action().verified(),
              "signed X/Z preparation");
      a.negative = !negative;
      require(!run(p, {}, {bare()}, a).action().verified(),
              "wrong preparation sign");
      a.negative = negative;
      a.basis = basis == Basis::X ? Basis::Z : Basis::X;
      require(!run(p, {}, {bare()}, a).action().verified(),
              "wrong preparation basis");
    }
  {
    Projection p;
    p.carrierCount = 2;
    p.outputDataCarriersByPort = {{0}};
    add(p, Gate::R, {0, 1});
    add(p, Gate::H, {0});
    add(p, Gate::CX, {0, 1});
    Action a;
    a.kind = Action::Kind::Preparation;
    a.outputBindings = {0};
    require(!run(p, {}, {bare()}, a).action().verified(),
            "discarded Bell partner does not prepare pure state");
  }
  for (auto basis : {Basis::X, Basis::Z})
    for (bool preserving : {false, true}) {
      Projection p;
      p.carrierCount = 1;
      p.inputDataCarriersByPort = {{0}};
      if (preserving)
        p.outputDataCarriersByPort = {{0}};
      add(p, basis == Basis::X ? Gate::MX : Gate::M, {0});
      Action a = measurement(preserving);
      a.basis = basis;
      std::vector<IdealBoundaryPort> out =
          preserving ? std::vector{bare()} : std::vector<IdealBoundaryPort>{};
      require(run(p, {bare()}, out, a).action().verified(),
              "destructive/preserving X/Z measurement");
      a.outcomeConstant = true;
      require(!run(p, {bare()}, out, a).action().verified(),
              "wrong affine complement rejected");
      p.instructions[0].inverted = true;
      require(run(p, {bare()}, out, a).action().verified(),
              "matching affine complement accepted");
      a.outcomeConstant = false;
      a.negative = true;
      require(run(p, {bare()}, out, a).action().verified(),
              "signed measurement accepted");
      if (preserving) {
        add(p, basis == Basis::X ? Gate::RX : Gate::R, {0});
        require(!run(p, {bare()}, out, a).action().verified(),
                "measurement followed by reset is not preserving");
      }
    }
  {
    Projection p;
    p.carrierCount = 2;
    p.inputDataCarriersByPort = {{0}};
    add(p, Gate::RX, {1});
    add(p, Gate::M, {1});
    add(p, Gate::M, {0});
    Action a = measurement();
    require(!run(p, {bare()}, {}, a).action().verified(),
            "independent random bit cannot replace readout");
  }
  {
    Projection p;
    p.carrierCount = 2;
    p.inputDataCarriersByPort = {{0, 1}};
    p.outputDataCarriersByPort = {{0, 1}};
    add(p, Gate::M, {0});
    add(p, Gate::Swap, {0, 1});
    Action a = measurement(true);
    a.inputBindings = {0, 1};
    a.outputBindings = {1, 0};
    require(run(p, {pair()}, {pair()}, a).action().verified(),
            "preserving measurement follows permuted outputs");
    a.outputBindings = {0, 1};
    require(!run(p, {pair()}, {pair()}, a).action().verified(),
            "wrong preserving output binding rejected");
    a.outputBindings = {1, 0};
    add(p, Gate::M, {0});
    require(!run(p, {pair()}, {pair()}, a).action().verified(),
            "hidden measurement of untouched logical rejected");
  }
  {
    Projection p;
    p.carrierCount = 2;
    p.inputDataCarriersByPort = {{0, 1}};
    add(p, Gate::M, {1});
    Action a = measurement();
    a.inputBindings = {0, 1};
    a.target = 1;
    require(run(p, {pair()}, {}, a).action().verified(),
            "destructive selected logical with other inputs discarded");
  }
  {
    Projection p;
    p.carrierCount = 2;
    p.inputDataCarriersByPort = {{0, 1}};
    p.outputDataCarriersByPort = {{0, 1}};
    add(p, Gate::X, {1});
    Action a{2,
             {bits("1000"), bits("0100"), bits("0010"), bits("0001")},
             {false, false, false, true}};
    require(run(p, {pair()}, {pair()}, a).action().verified(),
            "embedded selected-logical Pauli preserves its spectator");
    add(p, Gate::Z, {0});
    require(!run(p, {pair()}, {pair()}, a).action().verified(),
            "embedded selected-logical action rejects spectator changes");
  }
  {
    Projection p;
    p.carrierCount = 2;
    p.inputDataCarriersByPort = {{0, 1}};
    p.outputDataCarriersByPort = {{0, 1}};
    add(p, Gate::CX, {1, 0});
    Action a{2,
             {bits("1100"), bits("0100"), bits("0010"), bits("0011")},
             {false, false, false, false}};
    a.inputBindings = {1, 0};
    a.outputBindings = {1, 0};
    auto verified = run(p, {pair()}, {pair()}, a);
    require(verified.action().verified(),
            "Clifford explicit input/output bindings");
    require(verified.action().labels.size() == verified.action().rows.size() &&
                verified.action().labels[0] == "X_ref[1] X\u0304[1] X\u0304[0]",
            "diagnostic labels follow declared bindings");
    a.inputBindings = {0, 1};
    a.outputBindings = {0, 1};
    require(!run(p, {pair()}, {pair()}, a).action().verified(),
            "reversed Clifford binding rejected");
  }
  {
    IdealBoundaryPort repetition{
        {0, 1}, {bits("0011")}, {bits("1100")}, {bits("0010")}};
    Projection p;
    p.carrierCount = 2;
    p.inputDataCarriersByPort = {{0, 1}};
    add(p, Gate::M, {1});
    Action a = measurement();
    require(!run(p, {repetition}, {}, a).action().verified(),
            "wrong readout representative fails outside zero syndrome");
    a.outcomeInputSyndromes = {0};
    require(run(p, {repetition}, {}, a).action().verified(),
            "kept syndrome affine expression compared exactly");
  }
  {
    Projection p;
    p.carrierCount = 1;
    p.inputDataCarriersByPort = {{0}};
    add(p, Gate::M, {0});
    Action a = measurement();
    a.inputBindings = {1};
    auto result = analyzeIdealOutcomes(p, {bare()}, {}, &a);
    require(!result, "out-of-range binding rejected");
    if (!result)
      llvm::consumeError(result.takeError());
    a = measurement();
    a.outcomeRecords = {3};
    result = analyzeIdealOutcomes(p, {bare()}, {}, &a);
    require(!result, "invalid result record rejected");
    if (!result)
      llvm::consumeError(result.takeError());
  }
  std::cout << checks << " native instrument checks passed\n";
}
