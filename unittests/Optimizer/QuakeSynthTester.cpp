/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, rawArgs));
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
  pm.addPass(cudaq::opt::createConvertToQIRPass());
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
  printf("%s\n", kernel.to_quake().c_str());

  // Set the proper name for the kernel
  auto properName = cudaq::runtime::cudaqGenPrefixName + kernel.name();

  // Should get a uniform distribution of all bit strings
  auto counts = cudaq::sample(kernel, 5);
  EXPECT_EQ(counts.size(), 32);

  // Map the kernel_builder to_quake output  to MLIR
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

  printf("%s\n", kernel.to_quake().c_str());

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

TEST(QuakeSynthTester, checkVector) {
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

  printf("%s\n", kernel.to_quake().c_str());

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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
