/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/RuntimeMLIR.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/algorithm.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace mlir;

namespace cudaq {
/// Typedef the KernelArgs Creator Function
typedef std::size_t (*Creator)(void **, void **);

/// Retrieve the kernel args creator function for the kernel name
Creator getArgsCreator(const std::string &);

/// @brief Utility function for mapping variadic args to required void*,
/// size_t. Note clients of this function own the allocated rawArgs.
template <typename... Args>
std::pair<void *, std::size_t> mapToRawArgs(const std::string &kernelName,
                                            Args &&...args) {
  void *rawArgs = nullptr;
  auto argsCreator = getArgsCreator(kernelName);
  void *argPointers[sizeof...(Args)] = {&args...};
  auto argsSize = argsCreator(argPointers, &rawArgs);
  return std::make_pair(rawArgs, argsSize);
}
} // namespace cudaq

/// @brief Run the Quake Synth pass on the given kernel with provided runtime
/// args.
LogicalResult runQuakeSynth(std::string_view kernelName, void *rawArgs,
                            OwningOpRef<mlir::ModuleOp> &module) {
  PassManager pm(module->getContext());
  module->getContext()->disableMultithreading();
  pm.enableIRPrinting();
  pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, rawArgs, 0, true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLoopNormalize());
  pm.addPass(cudaq::opt::createLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  return pm.run(*module);
}

/// @brief Lower the module to LLVM
LogicalResult lowerToLLVMDialect(ModuleOp module) {
  PassManager pm(module->getContext());
  module->getContext()->disableMultithreading();
  pm.enableIRPrinting();
  pm.addPass(createCanonicalizerPass());
  OpPassManager &optPM = pm.nest<func::FuncOp>();
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  optPM.addPass(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  optPM.addPass(cudaq::opt::createQuakeAddDeallocs());
  optPM.addPass(cudaq::opt::createQuakeAddMetadata());
  optPM.addPass(cudaq::opt::createLowerToCFGPass());
  optPM.addPass(cudaq::opt::createCombineQuantumAllocations());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(cudaq::opt::createConvertToQIR());
  return pm.run(module);
}

/// @brief Run sampling on the JIT compiled kernel function
cudaq::sample_result sampleJitCode(ExecutionEngine *jit,
                                   const std::string &kernelName) {
  auto &p = cudaq::get_platform();
  return cudaq::details::runSampling(
             [&]() {
               auto err = jit->invokePacked(cudaq::runtime::cudaqGenPrefixName +
                                            kernelName);
               ASSERT_TRUE(!err);
             },
             p, kernelName, 1000)
      .value();
}

/// @brief Run observation on the JIT compiled kernel function
cudaq::observe_result observeJitCode(ExecutionEngine *jit, cudaq::spin_op &h,
                                     const std::string &kernelName) {
  auto &p = cudaq::get_platform();
  return cudaq::details::runObservation(
             [&]() {
               auto err = jit->invokePacked(cudaq::runtime::cudaqGenPrefixName +
                                            kernelName);
               ASSERT_TRUE(!err);
             },
             h, p, /*shots=*/-1, "")
      .value();
}

TEST(QuakeSynthTests, checkSimpleIntegerInput) {

  // Create the Kernel, takes an int as input
  auto [kernel, nQubits] = cudaq::make_kernel<int>();
  auto qubits = kernel.qalloc(nQubits);
  kernel.h(qubits);
  kernel.mz(qubits);
  std::cout << kernel.to_quake() << '\n';

  // Set the proper name for the kernel
  auto properName = cudaq::runtime::cudaqGenPrefixName + kernel.name();

  // Should get a uniform distribution of all bit strings
  auto counts = cudaq::sample(kernel, 5);
  EXPECT_EQ(counts.size(), 32);

  // Map the kernel_builder to_quake output to MLIR
  auto context = cudaq::initializeMLIR();
  auto module = parseSourceString<ModuleOp>(kernel.to_quake(), context.get());

  // Create a struct defining the runtime args for the kernel
  auto [args, offset] = cudaq::mapToRawArgs(kernel.name(), 5);

  // Run quake-synth
  EXPECT_TRUE(succeeded(runQuakeSynth(kernel.name(), args, module)));

  // Get the function, make sure that it has no arguments
  auto func = module->lookupSymbol<func::FuncOp>(properName);
  EXPECT_TRUE(func);
  EXPECT_TRUE(func.getArguments().empty());

  // Lower to LLVM and create the JIT execution engine
  EXPECT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  EXPECT_TRUE(!!jitOrError);
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());

  // Sample this new kernel processed with quake synth
  counts = sampleJitCode(jit.get(), kernel.name());
  // Should see the same thing as before.
  EXPECT_EQ(counts.size(), 32);
}

TEST(QuakeSynthTests, checkDoubleInput) {

  // Create the Kernel, takes an int as input
  auto [kernel, theta, phi] = cudaq::make_kernel<double, double>();

  auto q = kernel.qalloc(3);
  kernel.x(q[0]);
  kernel.ry(theta, q[1]);
  kernel.ry(phi, q[2]);
  kernel.x<cudaq::ctrl>(q[2], q[0]);
  kernel.x<cudaq::ctrl>(q[0], q[1]);
  kernel.ry(-theta, q[1]);
  kernel.x<cudaq::ctrl>(q[0], q[1]);
  kernel.x<cudaq::ctrl>(q[1], q[0]);

  std::cout << kernel.to_quake() << '\n';

  // Set the proper name for the kernel
  auto properName = cudaq::runtime::cudaqGenPrefixName + kernel.name();

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  cudaq::spin_op h3 = h + 9.625 - 9.625 * z(2) - 3.913119 * x(1) * x(2) -
                      3.913119 * y(1) * y(2);

  double energy = cudaq::observe(kernel, h3, .3591, .2569);
  EXPECT_NEAR(energy, -2.045375, 1e-3);

  // Map the kernel_builder to_quake output  to MLIR
  auto context = cudaq::initializeMLIR();
  auto module = parseSourceString<ModuleOp>(kernel.to_quake(), context.get());

  // Create a struct defining the runtime args for the kernel
  auto [args, offset] = cudaq::mapToRawArgs(kernel.name(), .3591, .2569);

  // Run quake-synth
  EXPECT_TRUE(succeeded(runQuakeSynth(kernel.name(), args, module)));

  // Get the function, make sure that it has no arguments
  auto func = module->lookupSymbol<func::FuncOp>(properName);
  EXPECT_TRUE(func);
  EXPECT_TRUE(func.getArguments().empty());

  func.dump();

  // Lower to LLVM and create the JIT execution engine
  EXPECT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  EXPECT_TRUE(!!jitOrError);
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());

  // // Sample this new kernel processed with quake synth
  energy = observeJitCode(jit.get(), h3, kernel.name());
  // Should see the same thing as before.
  EXPECT_NEAR(energy, -2.045375, 1e-3);
}

TEST(QuakeSynthTests, checkVectorOfDouble) {
  auto [kernel, thetas] = cudaq::make_kernel<std::vector<double>>();
  auto theta = thetas[0];
  auto phi = thetas[1];
  auto q = kernel.qalloc(3);
  kernel.x(q[0]);
  kernel.ry(theta, q[1]);
  kernel.ry(phi, q[2]);
  kernel.x<cudaq::ctrl>(q[2], q[0]);
  kernel.x<cudaq::ctrl>(q[0], q[1]);
  kernel.ry(-theta, q[1]);
  kernel.x<cudaq::ctrl>(q[0], q[1]);
  kernel.x<cudaq::ctrl>(q[1], q[0]);

  std::cout << kernel.to_quake() << '\n';

  // Set the proper name for the kernel
  auto properName = cudaq::runtime::cudaqGenPrefixName + kernel.name();

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  cudaq::spin_op h3 = h + 9.625 - 9.625 * z(2) - 3.913119 * x(1) * x(2) -
                      3.913119 * y(1) * y(2);

  double energy = cudaq::observe(kernel, h3, std::vector<double>{.3591, .2569});
  EXPECT_NEAR(energy, -2.045375, 1e-3);

  // Map the kernel_builder to_quake output  to MLIR
  auto context = cudaq::initializeMLIR();
  auto module = parseSourceString<ModuleOp>(kernel.to_quake(), context.get());

  // Create a struct defining the runtime args for the kernel
  auto [args, offset] =
      cudaq::mapToRawArgs(kernel.name(), std::vector<double>{.3591, .2569});

  // Run quake-synth
  EXPECT_TRUE(succeeded(runQuakeSynth(kernel.name(), args, module)));

  // Get the function, make sure that it has no arguments
  auto func = module->lookupSymbol<func::FuncOp>(properName);
  EXPECT_TRUE(func);
  EXPECT_TRUE(func.getArguments().empty());

  func.dump();

  // Lower to LLVM and create the JIT execution engine
  EXPECT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  EXPECT_TRUE(!!jitOrError);
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());

  // // Sample this new kernel processed with quake synth
  energy = observeJitCode(jit.get(), h3, kernel.name());
  // Should see the same thing as before.
  EXPECT_NEAR(energy, -2.045375, 1e-3);
}

TEST(QuakeSynthTests, checkVectorOfInt) {
  auto [kernel, hiddenBits] = cudaq::make_kernel<std::vector<int>>();

  auto q = kernel.qalloc(hiddenBits.size());
  auto aq = kernel.qalloc();

  kernel.h(aq);
  kernel.z(aq);
  kernel.h(q);
  for (std::size_t i = 0; i < *q.constantSize(); ++i) {
    kernel.c_if(hiddenBits[i], [&]() { kernel.x<cudaq::ctrl>(aq, q[i]); });
  }
  kernel.h(q);
  kernel.mz(q);

  // Dump the kernel to stdout.
  std::cout << kernel.to_quake() << '\n';

  // Set the proper name for the kernel
  auto properName = cudaq::runtime::cudaqGenPrefixName + kernel.name();

  // Should get a uniform distribution of all bit strings
  std::vector<int> ghostBits = {0, 1, 1, 0, 0};
  auto counts = cudaq::sample(kernel, ghostBits);
  EXPECT_EQ(counts.size(), 1);

  // Map the kernel_builder to_quake output to MLIR
  auto context = cudaq::initializeMLIR();
  auto module = parseSourceString<ModuleOp>(kernel.to_quake(), context.get());

  // Create a struct defining the runtime args for the kernel
  auto [args, offset] = cudaq::mapToRawArgs(kernel.name(), ghostBits);

  // Run quake-synth
  EXPECT_TRUE(succeeded(runQuakeSynth(kernel.name(), args, module)));

  // Get the function, make sure that it has no arguments
  auto func = module->lookupSymbol<func::FuncOp>(properName);
  EXPECT_TRUE(func);
  EXPECT_TRUE(func.getArguments().empty());

  func.dump();

  // Lower to LLVM and create the JIT execution engine
  EXPECT_TRUE(succeeded(lowerToLLVMDialect(*module)));
  auto jitOrError = ExecutionEngine::create(*module);
  EXPECT_TRUE(!!jitOrError);
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());

  // Sample this new kernel processed with quake synth
  auto countz = sampleJitCode(jit.get(), kernel.name());
  EXPECT_EQ(countz.size(), 1);
}

TEST(QuakeSynthTests, checkCallable) {
  auto [ansatz, thetas] = cudaq::make_kernel<std::vector<double>>();
  auto q = ansatz.qalloc(2);
  ansatz.x(q[0]);
  ansatz.ry(thetas[0], q[1]);
  ansatz.x<cudaq::ctrl>(q[1], q[0]);

  auto [kernel, angles] = cudaq::make_kernel<std::vector<double>>();
  kernel.call(ansatz, angles);

  // Set the proper name for the kernel
  auto properName = cudaq::runtime::cudaqGenPrefixName + kernel.name();

  std::vector<double> argsValue = {0.0};
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  double energy = cudaq::observe(kernel, h, argsValue);
  std::cout << "Energy = " << energy << "\n";
  // Map the kernel_builder to_quake output to MLIR
  auto context = cudaq::initializeMLIR();
  std::cout << "Quake Code:\n" << kernel.to_quake() << "\n";
  auto module = parseSourceString<ModuleOp>(kernel.to_quake(), context.get());

  // Create a struct defining the runtime args for the kernel
  auto [args, offset] = cudaq::mapToRawArgs(kernel.name(), argsValue);

  // Run quake-synth
  EXPECT_TRUE(succeeded(runQuakeSynth(kernel.name(), args, module)));

  // Get the function, make sure that it has no arguments
  auto func = module->lookupSymbol<func::FuncOp>(properName);
  func.dump();
  EXPECT_TRUE(func);
  EXPECT_TRUE(func.getArguments().empty());
}

TEST(QuakeSynthTests, checkVectorOfComplex) {
  auto [colonel, stateVec] =
      cudaq::make_kernel<std::vector<std::complex<double>>>();
  auto qubits = colonel.qalloc(stateVec);
  colonel.h(qubits);
  colonel.mz(qubits);
  std::cout << colonel.to_quake() << '\n';

  // Generate name of the kernel
  auto colonelName = cudaq::runtime::cudaqGenPrefixName + colonel.name();
  std::vector<std::complex<double>> initialState = {1.0, 2.0, 3.0, 4.0};

  [[maybe_unused]] auto counts = cudaq::sample(colonel, initialState);
  counts.dump();

  auto context = cudaq::initializeMLIR();
  auto module = parseSourceString<ModuleOp>(colonel.to_quake(), context.get());

  auto [args, offset] = cudaq::mapToRawArgs(colonel.name(), initialState);

  EXPECT_TRUE(succeeded(runQuakeSynth(colonel.name(), args, module)));

  auto func = module->lookupSymbol<func::FuncOp>(colonelName);
  EXPECT_TRUE(func);
  EXPECT_TRUE(func.getArguments().empty());
  func.dump();
}

TEST(QuakeSynthTests, checkVectorOfPauliWord) {
  auto [colonel, stateVec] =
      cudaq::make_kernel<std::vector<cudaq::pauli_word>>();
  auto qubit = colonel.qalloc();
  colonel.h(qubit);
  colonel.y(qubit);
  colonel.z(qubit);
  colonel.mz(qubit);
  std::cout << colonel.to_quake() << '\n';

  // Generate name of the kernel
  auto colonelName = cudaq::runtime::cudaqGenPrefixName + colonel.name();
  std::vector<cudaq::pauli_word> peterPauli = {cudaq::pauli_word("IXII"),
                                               cudaq::pauli_word("IIIZ")};

  [[maybe_unused]] auto counts = cudaq::sample(colonel, peterPauli);
  counts.dump();

  auto context = cudaq::initializeMLIR();
  auto module = parseSourceString<ModuleOp>(colonel.to_quake(), context.get());

  auto [args, offset] = cudaq::mapToRawArgs(colonel.name(), peterPauli);

  EXPECT_TRUE(succeeded(runQuakeSynth(colonel.name(), args, module)));

  auto func = module->lookupSymbol<func::FuncOp>(colonelName);
  EXPECT_TRUE(func);
  EXPECT_TRUE(func.getArguments().empty());
  func.dump();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
