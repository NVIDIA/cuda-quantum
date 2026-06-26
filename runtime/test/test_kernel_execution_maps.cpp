/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This test is compiled inside the runtime directory tree. We include it as a
// regression test and use FileCheck to verify the output.

// RUN: test_kernel_execution_maps | FileCheck %s

#include "common/KernelExecution.h"
#include "common/ResultReconstruction.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq/Target/CompileTarget.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cudaq_internal::compiler;

static mlir::OwningOpRef<mlir::ModuleOp>
parseModuleOrFail(mlir::MLIRContext *context, llvm::StringRef source) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, context);
  if (!module)
    throw std::runtime_error("failed to parse test module");
  return module;
}

static void printResultOutputMap(const cudaq::KernelExecution &execution) {
  llvm::outs() << execution.name;
  auto resultMap =
      cudaq::makeResultOutputMapFromEnrichedOutputNames(execution.output_names);
  for (const auto &entry : resultMap.outputs)
    llvm::outs() << " " << entry.resultIndex << "/" << entry.deviceQubit << ":"
                 << entry.outputName << ":" << entry.outputPosition;
  llvm::outs() << "\n";
}

static void printActiveDeviceQubits(const cudaq::KernelExecution &execution) {
  llvm::outs() << execution.name << " active";
  for (auto deviceQubit : execution.activeDeviceQubits)
    llvm::outs() << " " << deviceQubit;
  llvm::outs() << "\n";
}

// Print a register's reconstructed counts in a deterministic order so FileCheck
// can match exact bitstrings and tallies. std::map sorts the bitstrings.
static void printRegisterCounts(llvm::StringRef label,
                                const cudaq::sample_result &result,
                                const std::string &registerName) {
  std::map<std::string, std::size_t> sorted;
  for (const auto &[bits, count] : result.to_map(registerName))
    sorted.emplace(bits, count);
  llvm::outs() << label;
  for (const auto &[bits, count] : sorted)
    llvm::outs() << " " << bits << ":" << count;
  llvm::outs() << "\n";
}

// Reconstruct the global (user-order) register from a counts dictionary using
// the map built from a real emitted execution, then print it deterministically.
static void reconstructAndPrintGlobal(llvm::StringRef label,
                                      const cudaq::KernelExecution &execution,
                                      const cudaq::CountsDictionary &counts) {
  auto resultMap =
      cudaq::makeResultOutputMapFromEnrichedOutputNames(execution.output_names);
  auto result = cudaq::reconstructSampleResultFromDeviceIndexedMeasurements(
      counts, resultMap);
  printRegisterCounts(label, result, cudaq::GlobalRegisterName);
}

static const cudaq::KernelExecution &
findExecution(const std::vector<cudaq::KernelExecution> &codes,
              llvm::StringRef name) {
  for (const auto &code : codes)
    if (code.name == name)
      return code;
  throw std::runtime_error("emitted execution not found: " + name.str());
}

static std::vector<cudaq::KernelExecution>
emit(std::vector<CompiledModuleHelper::NamedCompiledArtifact> artifacts) {
  auto compiled = CompiledModuleHelper::createCompiledModule(
      "kernel", cudaq::ResultInfo{}, std::move(artifacts));
  auto target = std::make_unique<cudaq::CompileTarget>();
  target->emitTargetCode = false;
  target->pipelineConfig.codegenTranslation = "qasm2";
  Compiler compiler(std::move(target));
  auto codes = compiler.emitKernelExecutions(compiled);
  std::sort(codes.begin(), codes.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.name < rhs.name;
  });
  return codes;
}

static void
test_emit_kernel_executions_builds_result_map(mlir::MLIRContext *context) {
  auto firstModule = parseModuleOrFail(context, R"mlir(
    module {
      func.func @first() attributes {"cudaq-entrypoint", output_names = "[[[0,[2,\"alpha\"]]]]" } {
        return
      }
    }
  )mlir");
  auto secondModule = parseModuleOrFail(context, R"mlir(
    module {
      func.func @second() attributes {"cudaq-entrypoint", output_names = "[[[0,[8,\"gamma\",1]],[1,[7,\"delta\",0]]]]" } {
        return
      }
    }
  )mlir");
  auto mappedModule = parseModuleOrFail(context, R"mlir(
    module {
      quake.wire_set @mapped_wireset[100]
      func.func @mapped() attributes {"cudaq-entrypoint", output_names = "[[[0,[0,\"right\",1]],[1,[1,\"left\",0]]]]" } {
        %q0 = quake.borrow_wire @mapped_wireset[99] : !quake.wire
        %q1 = quake.borrow_wire @mapped_wireset[95] : !quake.wire
        quake.return_wire %q0 : !quake.wire
        quake.return_wire %q1 : !quake.wire
        return
      }
    }
  )mlir");

  auto sharedContext =
      std::shared_ptr<mlir::MLIRContext>(context, [](mlir::MLIRContext *) {});
  std::vector<CompiledModuleHelper::NamedCompiledArtifact> artifacts;
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "first", firstModule.get(), sharedContext));
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "second", secondModule.get(), sharedContext));
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "mapped", mappedModule.get(), sharedContext));
  auto codes = emit(std::move(artifacts));

  for (const auto &code : codes) {
    if (code.name == "mapped")
      printActiveDeviceQubits(code);
    printResultOutputMap(code);
  }
}

// CHECK-DAG: first 0/2:alpha:0
// CHECK-DAG: second 0/8:gamma:1 1/7:delta:0
// CHECK-DAG: mapped active 95 99
// CHECK-DAG: mapped 0/0:right:1 1/1:left:0

// A sparse projection on a mapped layout: result 0 lands on device qubit 1
// (user position 0) and result 1 lands on device qubit 4 (user position 1).
// Device qubits 2 and 3 are active ancillas that are never measured, so the
// returned bitstring has measured bits at indices 1 and 4 with an unmeasured
// gap between them. Reconstruction must project the two measured bits into the
// compact user order, dropping the gap.
static void test_sparse_projection_reconstructs(mlir::MLIRContext *context) {
  auto sparseModule = parseModuleOrFail(context, R"mlir(
    module {
      quake.wire_set @mapped_wireset[100]
      func.func @sparse() attributes {"cudaq-entrypoint", output_names = "[[[0,[1,\"__global__\",0]],[1,[4,\"__global__\",1]]]]" } {
        %q0 = quake.borrow_wire @mapped_wireset[1] : !quake.wire
        %q1 = quake.borrow_wire @mapped_wireset[2] : !quake.wire
        %q2 = quake.borrow_wire @mapped_wireset[3] : !quake.wire
        %q3 = quake.borrow_wire @mapped_wireset[4] : !quake.wire
        quake.return_wire %q0 : !quake.wire
        quake.return_wire %q1 : !quake.wire
        quake.return_wire %q2 : !quake.wire
        quake.return_wire %q3 : !quake.wire
        return
      }
    }
  )mlir");

  auto sharedContext =
      std::shared_ptr<mlir::MLIRContext>(context, [](mlir::MLIRContext *) {});
  std::vector<CompiledModuleHelper::NamedCompiledArtifact> artifacts;
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "sparse", sparseModule.get(), sharedContext));
  auto codes = emit(std::move(artifacts));
  const auto &sparse = findExecution(codes, "sparse");
  printActiveDeviceQubits(sparse);

  // Returned bitstrings index device qubits 0..4. Bit 1 is result 0 (position
  // 0), bit 4 is result 1 (position 1). Bits 0/2/3 are gaps.
  //   "01001" -> bit1='1', bit4='1' -> user "11".
  //   "00000" -> bit1='0', bit4='0' -> user "00".
  cudaq::CountsDictionary counts{{"01001", 6}, {"00000", 4}};
  reconstructAndPrintGlobal("sparse-global", sparse, counts);
}

// CHECK: sparse active 1 2 3 4
// CHECK: sparse-global 00:4 11:6

// Named-register split across two registers with multi-bit ordering inside one
// register. Register "edge" carries two results whose user positions (1 then 0)
// invert the order of their device bit indices, and register "mid" carries one
// result. The reconstruction must place each register's bits by user position.
static void test_named_register_split_reconstructs(mlir::MLIRContext *context) {
  auto namedModule = parseModuleOrFail(context, R"mlir(
    module {
      func.func @named() attributes {"cudaq-entrypoint", output_names = "[[[0,[0,\"edge\",2]],[1,[1,\"mid\",1]],[2,[2,\"edge\",0]]]]" } {
        return
      }
    }
  )mlir");

  auto sharedContext =
      std::shared_ptr<mlir::MLIRContext>(context, [](mlir::MLIRContext *) {});
  std::vector<CompiledModuleHelper::NamedCompiledArtifact> artifacts;
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "named", namedModule.get(), sharedContext));
  auto codes = emit(std::move(artifacts));
  const auto &named = findExecution(codes, "named");

  auto resultMap =
      cudaq::makeResultOutputMapFromEnrichedOutputNames(named.output_names);
  // Bit 0 -> edge position 2 (global), bit 1 -> mid, bit 2 -> edge position 0.
  //   "101": bit0='1', bit1='0', bit2='1'.
  //     global by position: pos0=bit2='1', pos1=bit1='0', pos2=bit0='1' ->
  //     "101". edge by local position: pos0=bit2='1', pos1=bit0='1' -> "11".
  //     mid: bit1='0' -> "0".
  cudaq::CountsDictionary counts{{"101", 5}, {"010", 3}};
  auto result = cudaq::reconstructSampleResultFromDeviceIndexedMeasurements(
      counts, resultMap);
  // "010": pos0=bit2='0', pos1=bit1='1', pos2=bit0='0' -> global "010".
  //   edge: pos0=bit2='0', pos1=bit0='0' -> "00". mid: bit1='1' -> "1".
  printRegisterCounts("named-global", result, cudaq::GlobalRegisterName);
  printRegisterCounts("named-edge", result, "edge");
  printRegisterCounts("named-mid", result, "mid");
}

// CHECK: named-global 010:3 101:5
// CHECK: named-edge 00:3 11:5
// CHECK: named-mid 0:5 1:3

// Per-shot sequential data via the flat-bitstring shots API. Two results map to
// bit indices 1 and 0 with user positions 0 and 1, so each shot's two bits are
// projected and reordered. The reconstruction must preserve per-shot ordering
// in the returned sequential_data, not just aggregate counts.
static void test_per_shot_sequential_reconstructs(mlir::MLIRContext *context) {
  auto shotsModule = parseModuleOrFail(context, R"mlir(
    module {
      func.func @shots() attributes {"cudaq-entrypoint", output_names = "[[[0,[1,\"__global__\",0]],[1,[0,\"__global__\",1]]]]" } {
        return
      }
    }
  )mlir");

  auto sharedContext =
      std::shared_ptr<mlir::MLIRContext>(context, [](mlir::MLIRContext *) {});
  std::vector<CompiledModuleHelper::NamedCompiledArtifact> artifacts;
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "shots", shotsModule.get(), sharedContext));
  auto codes = emit(std::move(artifacts));
  const auto &shots = findExecution(codes, "shots");

  auto resultMap =
      cudaq::makeResultOutputMapFromEnrichedOutputNames(shots.output_names);
  // Each shot "ba": bit1=a (position 0), bit0=b (position 1) -> user "ab".
  //   "10" -> bit1='0' pos0, bit0='1' pos1 -> "01".
  //   "01" -> bit1='1' pos0, bit0='0' pos1 -> "10".
  std::vector<std::string> shotData{"10", "01", "01"};
  auto result = cudaq::reconstructSampleResultFromDeviceIndexedBitstringShots(
      shotData, resultMap);
  printRegisterCounts("shots-global", result, cudaq::GlobalRegisterName);
  llvm::outs() << "shots-sequential";
  for (const auto &entry : result.sequential_data())
    llvm::outs() << " " << entry;
  llvm::outs() << "\n";
}

// CHECK: shots-global 01:1 10:2
// CHECK: shots-sequential 01 10 10

// Observe splitting emits one execution per Pauli term. The compiled module
// carries two entry points, each with its own enriched output_names and its own
// user positions. Each emitted execution must reconstruct independently from
// its own returned bitstrings.
static void test_observe_split_reconstructs(mlir::MLIRContext *context) {
  auto termXModule = parseModuleOrFail(context, R"mlir(
    module {
      func.func @term_x() attributes {"cudaq-entrypoint", output_names = "[[[0,[1,\"__global__\",0]],[1,[0,\"__global__\",1]]]]" } {
        return
      }
    }
  )mlir");
  auto termZModule = parseModuleOrFail(context, R"mlir(
    module {
      func.func @term_z() attributes {"cudaq-entrypoint", output_names = "[[[0,[2,\"__global__\",0]]]]" } {
        return
      }
    }
  )mlir");

  auto sharedContext =
      std::shared_ptr<mlir::MLIRContext>(context, [](mlir::MLIRContext *) {});
  std::vector<CompiledModuleHelper::NamedCompiledArtifact> artifacts;
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "term_x", termXModule.get(), sharedContext));
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "term_z", termZModule.get(), sharedContext));
  auto codes = emit(std::move(artifacts));

  // Term X: bit1 at position 0, bit0 at position 1 over a two-qubit string.
  //   "10" -> bit1='0' pos0, bit0='1' pos1 -> "01".
  const auto &termX = findExecution(codes, "term_x");
  reconstructAndPrintGlobal("term_x-global", termX,
                            cudaq::CountsDictionary{{"10", 4}});

  // Term Z: a single measured qubit at device bit 2 (position 0).
  //   "001" -> bit2='1' -> "1".
  const auto &termZ = findExecution(codes, "term_z");
  reconstructAndPrintGlobal("term_z-global", termZ,
                            cudaq::CountsDictionary{{"001", 6}});
}

// CHECK: term_x-global 01:4
// CHECK: term_z-global 1:6

// Legacy back-compat: an old compiler emits two-element output-location tuples
// with no user position. The reader falls back to result-index order, so the
// global order follows the result index even though the device bit indices are
// permuted. This guards reading an old artifact with a new runtime.
static void test_legacy_two_tuple_reconstructs(mlir::MLIRContext *context) {
  auto legacyModule = parseModuleOrFail(context, R"mlir(
    module {
      func.func @legacy() attributes {"cudaq-entrypoint", output_names = "[[[0,[2,\"__global__\"]],[1,[0,\"__global__\"]]]]" } {
        return
      }
    }
  )mlir");

  auto sharedContext =
      std::shared_ptr<mlir::MLIRContext>(context, [](mlir::MLIRContext *) {});
  std::vector<CompiledModuleHelper::NamedCompiledArtifact> artifacts;
  artifacts.push_back(CompiledModuleHelper::createMlirArtifact(
      "legacy", legacyModule.get(), sharedContext));
  auto codes = emit(std::move(artifacts));
  const auto &legacy = findExecution(codes, "legacy");

  // No third tuple element, so result 0 -> position 0 (bit index 2), result 1
  // -> position 1 (bit index 0).
  //   "101": bit2='1' pos0, bit0='1' pos1 -> "11".
  //   "100": bit2='0' pos0, bit0='1' pos1 -> "01".
  cudaq::CountsDictionary counts{{"101", 8}, {"100", 2}};
  reconstructAndPrintGlobal("legacy-global", legacy, counts);
}

// CHECK: legacy-global 01:2 11:8

int main() {
  auto context = getOwningMLIRContext();
  test_emit_kernel_executions_builds_result_map(context.get());
  test_sparse_projection_reconstructs(context.get());
  test_named_register_split_reconstructs(context.get());
  test_per_shot_sequential_reconstructs(context.get());
  test_observe_split_reconstructs(context.get());
  test_legacy_two_tuple_reconstructs(context.get());
  return 0;
}
