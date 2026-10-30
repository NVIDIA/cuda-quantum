/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/CliffordTPipelines.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TargetPrepPipelineOptions
    : public PassPipelineOptions<TargetPrepPipelineOptions> {
  PassOptions::Option<bool> eraseNoise{
      *this, "erase-noise", llvm::cl::desc("Erase apply noise calls."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> applyConstProp{
      *this, "apply-const-prop",
      llvm::cl::desc("Enable constant propagation in apply specialization."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> allowEarlyExit{
      *this, "allow-early-exit",
      llvm::cl::desc(
          "Enable loop unrolling on loops with early exit conditions."),
      llvm::cl::init(false)};
  PassOptions::Option<bool> disableLoopUnrolling{
      *this, "no-loop-unroll",
      llvm::cl::desc("Disable loop unrolling and preserve cc.loop operations."),
      llvm::cl::init(false)};
};

struct TargetDeployPipelineOptions
    : public PassPipelineOptions<TargetDeployPipelineOptions> {
  PassOptions::Option<bool> disableLoopUnrolling{
      *this, "no-loop-unroll",
      llvm::cl::desc("Disable loop unrolling and preserve cc.loop operations."),
      llvm::cl::init(false)};
};

struct TargetFinalizationPipelineOptions
    : public PassPipelineOptions<TargetFinalizationPipelineOptions> {
  PassOptions::Option<bool> allowBreaksInLoops{
      *this, "loops-may-have-break",
      llvm::cl::desc("Enable break statements in loops."),
      llvm::cl::init(true)};
};

struct PythonAOTOptions : public PassPipelineOptions<PythonAOTOptions> {
  PassOptions::Option<bool> autoGenRunStack{
      *this, "gen-run-stack",
      llvm::cl::desc("Autogenerate the cudaq::run dispatch stack."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> deferGKEToJIT{
      *this, "defer-gke-to-jit",
      llvm::cl::desc("Defer running GKE until JIT time."),
      llvm::cl::init(true)};
  PassOptions::Option<std::size_t> codegenKind{
      *this, "codegen-kind", llvm::cl::desc("GKE launch codegen kind."),
      llvm::cl::init(0)};
};
struct TargetFinalizationJitPipelineOptions
    : public PassPipelineOptions<TargetFinalizationJitPipelineOptions> {
  PassOptions::Option<bool> lowerDeviceCalls{
      *this, "lower-device-calls",
      llvm::cl::desc(
          "Lower device calls (to normal function calls) in JIT pipeline."),
      llvm::cl::init(true)};
};

struct CliffordTOptimizationPipelineOptions
    : public PassPipelineOptions<CliffordTOptimizationPipelineOptions> {
  PassOptions::Option<unsigned> level{
      *this, "level",
      llvm::cl::desc("Optimization level. Level 0 does nothing, level 1 "
                     "rewrites adjacent and commuting gates, and level 2 "
                     "adds analyses across the kernel that reduce the T "
                     "count. Level 3 is reserved for search, and higher "
                     "levels currently run level 2."),
      llvm::cl::init(2)};
  PassOptions::Option<double> approximationTolerance{
      *this, "approximation-tolerance",
      llvm::cl::desc("Operator-norm tolerance for erasing small rotations and "
                     "snapping angles to Clifford+T values."),
      llvm::cl::init(0)};
};

struct TwoQubitGateCountOptimizationPipelineOptions
    : public PassPipelineOptions<TwoQubitGateCountOptimizationPipelineOptions> {
  PassOptions::Option<unsigned> level{
      *this, "level",
      llvm::cl::desc("Optimization level. Level 0 does nothing, level 1 "
                     "rewrites adjacent and commuting gates, and level 2 "
                     "adds analyses across the kernel. Level 3 is reserved "
                     "for search, and higher levels currently run level 2."),
      llvm::cl::init(2)};
};

struct CliffordTSynthesisPipelineOptions
    : public PassPipelineOptions<CliffordTSynthesisPipelineOptions> {
  PassOptions::ListOption<std::string> basis{
      *this, "basis",
      llvm::cl::desc("Gate basis of the output. If decomposition can lower "
                     "any of r1, rx, ry, or rz to it exactly, the pipeline "
                     "adds no passes and rotations stay exact. Empty always "
                     "synthesizes.")};
  PassOptions::Option<unsigned> level{
      *this, "level",
      llvm::cl::desc("Optimization level. From level 2, the pipeline also "
                     "cleans up the Clifford+T gates that synthesis emits."),
      llvm::cl::init(2)};
  PassOptions::Option<double> approximationTolerance{
      *this, "approximation-tolerance",
      llvm::cl::desc("Approximation tolerance for Clifford+T synthesis."),
      llvm::cl::init(1e-10)};
  PassOptions::Option<uint64_t> seed{
      *this, "seed",
      llvm::cl::desc("Seed for Clifford+T synthesis. 0 leaves it unseeded."),
      llvm::cl::init(0)};
};
} // namespace

void cudaq::opt::addConvertToLinearValues(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createExpandVeqUses());
  pm.addNestedPass<func::FuncOp>(createFactorQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createExpandControlVeqs());
  pm.addNestedPass<func::FuncOp>(createCableRoughIn());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createLinearCtrlRelations());
}

void cudaq::opt::registerConvertToLinearValuesPipeline() {
  PassPipelineRegistration<>(
      "convert-to-linear-values",
      "Convert supported Quake IR to explicit linear values.",
      addConvertToLinearValues);
}

static void createTargetPrepPipeline(OpPassManager &pm,
                                     const TargetPrepPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createInjectImplicitOutput());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createAddDeallocs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddMetadata());
  pm.addPass(cudaq::opt::createQuakePropagateMetadata());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  cudaq::opt::createClassicalOptimizationPipeline(
      pm, std::nullopt, {options.allowEarlyExit}, std::nullopt,
      {options.disableLoopUnrolling});
  pm.addPass(cudaq::opt::createGlobalizeArrayValues());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Synthesis creates ApplyOps, which activate only when every control is |1>.
  // Expand negated controls into X conjugation before creating those calls.
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandControlNegations());
  pm.addPass(cudaq::opt::createUnitarySynthesis());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopInductionFusion());
  pm.addPass(cudaq::opt::createApplySpecialization(
      {.constantPropagation = options.applyConstProp}));
  cudaq::opt::addAggressiveInlining(pm);
}

static void
createHardwareTargetPrepPipeline(OpPassManager &pm,
                                 const TargetPrepPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createEraseNoise());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createEraseQEC());
  createTargetPrepPipeline(pm, options);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createStatePreparation());
}

/// Register the standard initial pipeline run for ALL target machines when
/// lowering to hardware. Either this preparation pipeline or the emulation
/// target preparation pipeline is always run between the JIT high-level and JIT
/// mid-level target-specific pipelines from the `.yml` file.
static void registerHardwareTargetPrepPipeline() {
  PassPipelineRegistration<TargetPrepPipelineOptions>(
      "hw-jit-prep-pipeline", "Prep for any hardware target.",
      [](OpPassManager &pm, const TargetPrepPipelineOptions &options) {
        createHardwareTargetPrepPipeline(pm, options);
      });
}

static void
createEmulationTargetPrepPipeline(OpPassManager &pm,
                                  const TargetPrepPipelineOptions &options) {
  if (options.eraseNoise)
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createEraseNoise());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createEraseQEC());
  createTargetPrepPipeline(pm, options);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createStatePreparation());
}

/// Register the standard initial pipeline run for ALL target machines when
/// emulation is enabled.
static void registerEmulationTargetPrepPipeline() {
  PassPipelineRegistration<TargetPrepPipelineOptions>(
      "emul-jit-prep-pipeline", "Prep for any emulation target.",
      [](OpPassManager &pm, const TargetPrepPipelineOptions &options) {
        createEmulationTargetPrepPipeline(pm, options);
      });
}

void cudaq::opt::addDecomposition(OpPassManager &pm,
                                  ArrayRef<std::string> enabledPats,
                                  ArrayRef<std::string> disabledPats) {
  // NB: Both of these ListOption *must* be set here or they may contain garbage
  // and the compiler may crash.
  cudaq::opt::DecompositionOptions opts;
  opts.disabledPatterns.assign(disabledPats.begin(), disabledPats.end());
  opts.enabledPatterns.assign(enabledPats.begin(), enabledPats.end());
  pm.addPass(cudaq::opt::createDecomposition(opts));
}

/// The cleanup that every cost model's optimization pipeline runs from level 1.
/// It erases no rotation larger than \p approximationTolerance.
static void addSharedCleanup(OpPassManager &funcPM,
                             double approximationTolerance) {
  cudaq::opt::QuakeSimplifyOptions cleanupOpts;
  cleanupOpts.threshold = approximationTolerance;
  funcPM.addPass(createCanonicalizerPass());
  funcPM.addPass(cudaq::opt::createQuakeSimplify(cleanupOpts));
  funcPM.addPass(cudaq::opt::createDeadQuantumElimination());
}

/// Reduce the T count and non-Clifford rotations before Clifford+T synthesis.
/// Level 1 only cleans up. Level 2 also fuses one-qubit chains, folds phases,
/// and snaps angles within `approximation-tolerance` of a Clifford+T value. No
/// pass erases a rotation larger than that tolerance.
static void
addCliffordTOptimization(OpPassManager &pm,
                         const CliffordTOptimizationPipelineOptions &options) {
  if (options.level == 0)
    return;
  auto &funcPM = pm.nest<func::FuncOp>();
  addSharedCleanup(funcPM, options.approximationTolerance);
  if (options.level == 1)
    return;

  cudaq::opt::FuseSingleQubitChainsForCliffordTOptions chainOpts;
  chainOpts.epsilon = options.approximationTolerance;
  funcPM.addPass(
      cudaq::opt::createFuseSingleQubitChainsForCliffordT(chainOpts));
  // Merge the constants that fusion creates.
  funcPM.addPass(createCanonicalizerPass());
  // Fold phases before snapping to expose shared rotations, and again at the
  // end to capture the exact phase terms that Clifford+T normalization exposes.
  funcPM.addPass(cudaq::opt::createPhaseFolding());
  funcPM.addPass(createCanonicalizerPass());
  cudaq::opt::QuakeSimplifyOptions snapOpts;
  snapOpts.threshold = options.approximationTolerance;
  snapOpts.rotationsToCliffordT = true;
  snapOpts.cliffordTEpsilon = options.approximationTolerance;
  funcPM.addPass(cudaq::opt::createQuakeSimplify(snapOpts));
  funcPM.addPass(cudaq::opt::createNormalizePhasePlacement());
  funcPM.addPass(cudaq::opt::createLowerPhase());
  funcPM.addPass(cudaq::opt::createNormalizeSingleQubitCliffordT());
  funcPM.addPass(createCanonicalizerPass());
  funcPM.addPass(cudaq::opt::createDeadQuantumElimination());
  funcPM.addPass(cudaq::opt::createPhaseFolding());
}

/// Reduce the two-qubit gate count before mapping. Level 1 runs the shared
/// cleanup, which erases no rotation, and level 2 also folds phases.
static void addTwoQubitGateCountOptimization(
    OpPassManager &pm,
    const TwoQubitGateCountOptimizationPipelineOptions &options) {
  if (options.level == 0)
    return;
  auto &funcPM = pm.nest<func::FuncOp>();
  addSharedCleanup(funcPM, /*approximationTolerance=*/0.0);
  if (options.level == 1)
    return;
  funcPM.addPass(cudaq::opt::createPhaseFolding());
  funcPM.addPass(createCanonicalizerPass());
}

void cudaq::opt::registerTwoQubitGateCountOptimizationPipeline() {
  PassPipelineRegistration<TwoQubitGateCountOptimizationPipelineOptions>(
      "two-qubit-gate-count-optimization-pipeline",
      "Reduce the two-qubit gate count before mapping.",
      addTwoQubitGateCountOptimization);
}

bool cudaq::opt::canOptimizeTwoQubitGateCount(ArrayRef<std::string> basis) {
  return basis.empty() || cudaq::canDecomposeTo("x(1)", basis);
}

/// Simplify and normalize the Clifford+T words that synthesis emits.
/// Clifford+T normalization emits a global phase, so lower it again.
static void addCliffordTCleanup(OpPassManager &funcPM,
                                double approximationTolerance) {
  cudaq::opt::QuakeSimplifyOptions cleanupOpts;
  cleanupOpts.threshold = approximationTolerance;
  funcPM.addPass(cudaq::opt::createQuakeSimplify(cleanupOpts));
  funcPM.addPass(cudaq::opt::createNormalizeSingleQubitCliffordT());
  funcPM.addPass(cudaq::opt::createPhaseFolding());
  funcPM.addPass(cudaq::opt::createNormalizePhasePlacement());
  funcPM.addPass(cudaq::opt::createLowerPhase());
  funcPM.addPass(createCanonicalizerPass());
  funcPM.addPass(cudaq::opt::createDeadQuantumElimination());
}

/// The rotations that `clifford-t-optimization-pipeline` merges and snaps and
/// `clifford-t-synthesis-pipeline` replaces with Clifford+T gates, when they
/// are uncontrolled and have constant angles.
static constexpr llvm::StringLiteral cliffordTRotations[] = {"r1", "rx", "ry",
                                                             "rz"};

std::vector<std::string>
cudaq::opt::cliffordTDecompositionBasis(std::span<const std::string> basis) {
  std::vector<std::string> withRotations(basis.begin(), basis.end());
  for (StringRef rotation : cliffordTRotations)
    if (!llvm::is_contained(withRotations, rotation))
      withRotations.emplace_back(rotation);
  return withRotations;
}

bool cudaq::opt::needsCliffordTSynthesis(std::span<const std::string> basis) {
  return basis.empty() ||
         llvm::none_of(cliffordTRotations, [&](StringRef rotation) {
           return cudaq::canDecomposeTo(rotation, {basis.data(), basis.size()});
         });
}

/// Rewrite every uncontrolled `rx`, `ry`, `r1`, and `rz` with a constant angle
/// as Clifford+T gates within `approximation-tolerance`, on references or
/// wires. From level 2,
/// also simplify and normalize the emitted Clifford+T words.
static void
addCliffordTSynthesis(OpPassManager &pm,
                      const CliffordTSynthesisPipelineOptions &options) {
  // A target basis that expresses any of these rotations exactly keeps that
  // rotation exact rather than approximating it. For example, `r1` in a basis
  // without `rz` stays exact, although synthesis would lower it to `rz` first.
  // Another rotation that such a basis cannot take fails in `basis-conversion`
  // either way.
  ArrayRef<std::string> basis = options.basis;
  if (!cudaq::opt::needsCliffordTSynthesis(basis))
    return;
  cudaq::opt::addDecomposition(pm, {"RxToRz", "RyToRz", "R1ToRz"});
  cudaq::opt::CliffordTSynthesisOptions synthesisOpts;
  synthesisOpts.epsilon = options.approximationTolerance;
  synthesisOpts.seed = options.seed;
  // The optimizer erases small rotations, bounded by the same tolerance.
  synthesisOpts.skipBelow = 0;
  // TODO: Synthesize controlled rotations once controlled Clifford+T synthesis
  // preserves the synthesis global phase.
  synthesisOpts.failOnControlledRotation = true;
  pm.addPass(cudaq::opt::createCliffordTSynthesis(synthesisOpts));
  // Synthesis outlines one helper per distinct angle.
  pm.addPass(createInlinerPass());
  // Erase the uncontrolled omega phase of each Clifford+T word.
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLowerPhase());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  if (options.level >= 2)
    addCliffordTCleanup(pm.nest<func::FuncOp>(),
                        options.approximationTolerance);
}

void cudaq::opt::registerCliffordTPipelines() {
  PassPipelineRegistration<CliffordTOptimizationPipelineOptions>(
      "clifford-t-optimization-pipeline",
      "Reduce the T count and non-Clifford rotations before synthesis.",
      addCliffordTOptimization);
  PassPipelineRegistration<CliffordTSynthesisPipelineOptions>(
      "clifford-t-synthesis-pipeline",
      "Rewrite constant-angle rotations as Clifford+T gates.",
      addCliffordTSynthesis);
}

void cudaq::opt::addPhaseLifecycle(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createNormalizePhasePlacement());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLowerPhase());
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
}

static void
createTargetDeployPipeline(OpPassManager &pm,
                           const TargetDeployPipelineOptions &options) {
  cudaq::opt::createClassicalOptimizationPipeline(
      pm, std::nullopt, std::nullopt, std::nullopt,
      {options.disableLoopUnrolling});
  cudaq::opt::addDecomposition(pm, {std::string("U3ToRotations")});
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createMultiControlDecomposition());
  cudaq::opt::addPhaseLifecycle(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandControlNegations());
}

/// Register the standard deployment pipeline run for ALL target machines. This
/// pipeline is run between the mid-level and low-level target-specific
/// pipelines.
static void registerTargetDeployPipeline() {
  PassPipelineRegistration<TargetDeployPipelineOptions>(
      "jit-deploy-pipeline", "Standard deployment pipeline for all targets.",
      [](OpPassManager &pm, const TargetDeployPipelineOptions &options) {
        ::createTargetDeployPipeline(pm, options);
      });
}

void cudaq::opt::createTargetFinalizePipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addPass(createSymbolDCEPass());
}

static void createJITTargetFinalizePipeline(
    OpPassManager &pm, const TargetFinalizationJitPipelineOptions &options) {
  if (options.lowerDeviceCalls)
    pm.addPass(cudaq::opt::createQIRDeviceCall());
  cudaq::opt::addAggressiveInlining(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandControlNegations());
  cudaq::opt::createTargetFinalizePipeline(pm);
}

/// Register the standard finalization pipeline run for ALL target machines.
/// This pipeline is run after the low-level target-specific pipelines.
static void registerTargetFinalizePipeline() {
  PassPipelineRegistration<TargetFinalizationJitPipelineOptions>(
      "jit-finalize-pipeline",
      "Standard JIT finalization pipeline for all targets.",
      [](OpPassManager &pm,
         const TargetFinalizationJitPipelineOptions &options) {
        createJITTargetFinalizePipeline(pm, options);
      });
}

void cudaq::opt::registerJITPipelines() {
  registerHardwareTargetPrepPipeline();
  registerEmulationTargetPrepPipeline();
  registerTargetDeployPipeline();
  registerTargetFinalizePipeline();
}

/// This pipeline is defined to mirror the nvq++ driver's pipeline. It can be
/// used post front-end bridge to lower the Quake IR to a target agnostic
/// version of Quake code.
static void createPythonAOTPipeline(OpPassManager &pm,
                                    const PythonAOTOptions &options) {
  // NB: This pipeline should be kept in synch with the pipeline in nvq++.
  pm.addPass(cudaq::opt::createVerifyAtomicQuantumRegions());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createVariableCoalesce());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createShrinkWrap());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createInjectImplicitOutput());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createAddDeallocs());
  pm.addPass(cudaq::opt::createLambdaLifting());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopInductionFusion());
  pm.addPass(cudaq::opt::createApplySpecialization());
  cudaq::opt::GenerateKernelExecutionOptions gkeOpts;
  gkeOpts.genRunStack = options.autoGenRunStack;
  gkeOpts.deferToJIT = options.deferGKEToJIT;
  gkeOpts.codegenKind = options.codegenKind;
  pm.addPass(cudaq::opt::createGenerateKernelExecution(gkeOpts));
  if (options.autoGenRunStack)
    pm.addPass(cudaq::opt::createRunSemanticsHackery());
  cudaq::opt::addAggressiveInlining(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddMetadata());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createConstantPropagation());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLiftArrayAlloc());
  pm.addPass(cudaq::opt::createQuakePropagateMetadata());
  pm.addPass(cudaq::opt::createGlobalizeArrayValues());
  // Python-only cleanup of front-end (AST bridge) residue. Runs after
  // constant-propagation (so statically-empty slices have already had their
  // cc.if folded away) and immediately before the canonicalize below, which
  // performs the actual veq_size folds this pass exposes.
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createPyFrontendCleanup());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createGetConcreteMatrix());
}

void cudaq::opt::createPythonAOTPipeline(OpPassManager &pm,
                                         bool autoGenRunStack) {
  PythonAOTOptions opts;
  opts.autoGenRunStack = autoGenRunStack;
  ::createPythonAOTPipeline(pm, opts);
}

static void registerPythonAOTPipeline() {
  PassPipelineRegistration<PythonAOTOptions>(
      "aot-prep-pipeline",
      "Pipeline to lower code for simulation or JIT compilation.",
      [](OpPassManager &pm, const PythonAOTOptions &options) {
        ::createPythonAOTPipeline(pm, options);
      });
}

void cudaq::opt::registerAOTPipelines() { registerPythonAOTPipeline(); }
