/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "nvqpp_driver.h"
#include "cudaq/Frontend/nvqpp/ASTBridge.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Support/Verifier.h"
#include "frontend_actions.h"
#include "nvqpp_flag_configs.h"
#include "nvqpp_options.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

extern "C" {
void getThisExecutablePath() { return; }
}

namespace cudaq {
Driver::Driver(ArgvStorageBase &cmdArgs)
    : driverPath(llvm::sys::fs::getMainExecutable(
          cmdArgs[0], (void *)(intptr_t)getThisExecutablePath)),
      diag(cmdArgs, driverPath),
      clangDriver(driverPath, llvm::sys::getDefaultTargetTriple(), diag.engine,
                  "nvq++ compiler") {
  clangDriver.ResourceDir = std::string(CLANG_RESOURCE_DIR);
  setInstallDir(cmdArgs);
  std::tie(clOptions, hostCompilerArgs) = preProcessCudaQArguments(cmdArgs);
  if (clOptions.hasArg(cudaq::nvqpp::options::OPT_target)) {
    const std::string clTargetName =
        clOptions.getLastArgValue(cudaq::nvqpp::options::OPT_target, "").str();
    auto target = TargetRegistry::lookupTarget(clTargetName);
    if (target.has_value()) {
      targetName = clTargetName;
      targetPlatformExtraArgs =
          target->generateConfig(clOptions, cudaqTargetsPath);

      // Insert backend config to the list of files to compile
      const std::string backendConfigCppFile =
          cudaqTargetsPath + "/backendConfig.cpp";
      hostCompilerArgs.insert(hostCompilerArgs.end(),
                              makeArgStringRef(backendConfigCppFile));
      const std::string targetConfigDef =
          targetName + ";emulate;" +
          (clOptions.hasArg(cudaq::nvqpp::options::OPT_emulate) ? "true"
                                                                : "false") +
          targetPlatformExtraArgs.platformExtraArgs;
      const std::string defArg =
          std::string("-DNVQPP_TARGET_BACKEND_CONFIG=\"") + targetConfigDef +
          "\"";
      hostCompilerArgs.insert(hostCompilerArgs.end(), makeArgStringRef(defArg));
    } else
      throw std::runtime_error("Unknown target: " + clTargetName);
  }
  // Add -std=c++20
  hostCompilerArgs.insert(hostCompilerArgs.end(), "-std=c++20");
  // Add CUDAQ include paths
  for (const char *include_flag : CUDAQ_INCLUDES_FLAGS)
    hostCompilerArgs.insert(hostCompilerArgs.end(), include_flag);
}

const char *Driver::makeArgStringRef(llvm::StringRef argStr) {
  synthesizedArgStrings.push_back(std::string(argStr));
  return synthesizedArgStrings.back().c_str();
}

std::pair<llvm::opt::InputArgList, llvm::opt::ArgStringList>
Driver::preProcessCudaQArguments(ArgvStorageBase &args) {
  unsigned missingArgIndex, missingArgCount;
  llvm::opt::InputArgList parsedArgs =
      cudaq::nvqpp::options::getDriverOptTable().ParseArgs(
          args, missingArgIndex, missingArgCount);
  // Check for missing argument error.
  if (missingArgCount)
    throw std::invalid_argument(
        std::string(parsedArgs.getArgString(missingArgIndex)) + " missed " +
        std::to_string(missingArgCount) + " arguments.");

  llvm::opt::ArgStringList clangArgs;
  for (const auto &arg : parsedArgs) {
    const bool isCudaqArgs =
        arg->getOption().isValid() &&
        arg->getOption().hasFlag(
            cudaq::nvqpp::options::NvqppFlags::NvqppOption);
    // Forward those unhandled args to Clang.
    if (!isCudaqArgs)
      arg->renderAsInput(parsedArgs, clangArgs);
  }

  return std::make_pair(std::move(parsedArgs), std::move(clangArgs));
}

std::string
Driver::constructCudaqOptPipeline(const llvm::opt::InputArgList &clOptions) {
  const bool doLink = clOptions.hasArg(cudaq::nvqpp::options::OPT_do_link);

  // Default options
  struct PipelineOpt {
    bool ENABLE_DEVICE_CODE_LOADERS = true;
    bool ENABLE_KERNEL_EXECUTION = true;
    bool ENABLE_AGGRESSIVE_EARLY_INLINE = true;
    bool ENABLE_LOWER_TO_CFG = true;
    bool ENABLE_APPLY_SPECIALIZATION = true;
    bool ENABLE_LAMBDA_LIFTING = true;
    // Run opt if any of the pass enabled.
    bool runOpt() const {
      return ENABLE_DEVICE_CODE_LOADERS || ENABLE_KERNEL_EXECUTION ||
             ENABLE_AGGRESSIVE_EARLY_INLINE || ENABLE_LOWER_TO_CFG ||
             ENABLE_APPLY_SPECIALIZATION || ENABLE_LAMBDA_LIFTING;
    }
  };

#define CHECK_OPTION(MEMBER_VAR, OPTION_FLAG_ON)                               \
  {                                                                            \
    if (clOptions.hasArg(cudaq::nvqpp::options::OPT_##OPTION_FLAG_ON))         \
      opt.MEMBER_VAR = true;                                                   \
    if (clOptions.hasArg(cudaq::nvqpp::options::OPT_no_##OPTION_FLAG_ON))      \
      opt.MEMBER_VAR = false;                                                  \
  }
  PipelineOpt opt;
  CHECK_OPTION(ENABLE_DEVICE_CODE_LOADERS, device_code_loading);
  CHECK_OPTION(ENABLE_KERNEL_EXECUTION, kernel_execution);
  CHECK_OPTION(ENABLE_AGGRESSIVE_EARLY_INLINE, aggressive_early_inline);
  CHECK_OPTION(ENABLE_APPLY_SPECIALIZATION, quake_apply_specialization);
  CHECK_OPTION(ENABLE_LAMBDA_LIFTING, lambda_lifting);

  if (!opt.runOpt())
    return "";
  std::string optPasses;
  const auto addPassToPipeline = [&optPasses](const std::string &passes) {
    if (optPasses.empty())
      optPasses = passes;
    else
      optPasses += (std::string(",") + passes);
  };
  if (opt.ENABLE_LAMBDA_LIFTING)
    addPassToPipeline("canonicalize,lambda-lifting");
  if (opt.ENABLE_APPLY_SPECIALIZATION)
    addPassToPipeline("func.func(memtoreg{quantum=0}),canonicalize,apply-op-"
                      "specialization");

  if (opt.ENABLE_KERNEL_EXECUTION)
    addPassToPipeline("kernel-execution");
  if (opt.ENABLE_AGGRESSIVE_EARLY_INLINE)
    addPassToPipeline(doLink ? "canonicalize,lambda-lifting"
                             : "func.func(indirect-to-direct-calls),inline");
  if (opt.ENABLE_DEVICE_CODE_LOADERS)
    addPassToPipeline(
        "func.func(quake-add-metadata),device-code-loader{use-quake=1}");

  if (opt.ENABLE_LOWER_TO_CFG)
    addPassToPipeline("func.func(unwind-lowering),expand-measurements,func."
                      "func(lower-to-cfg)");

  addPassToPipeline("canonicalize,cse");
  return optPasses;
}

int Driver::executeCC1Tool(ArgvStorageBase &cmdArgs) {
  llvm::cl::ResetAllOptionOccurrences();
  llvm::BumpPtrAllocator pointerAllocator;
  llvm::StringSaver saver(pointerAllocator);
  llvm::cl::ExpandResponseFiles(saver, &llvm::cl::TokenizeGNUCommandLine,
                                cmdArgs);

  llvm::StringRef tool = cmdArgs[1];
  auto [cudaqArgs, ccargs] = Driver::preProcessCudaQArguments(cmdArgs);

  if (tool == "-cc1") {
    auto ccargsRef = llvm::ArrayRef(ccargs).slice(2);
    void *getExecutablePathPtr = (void *)(intptr_t)getThisExecutablePath;

    return Driver::cc1Main(cudaqArgs, ccargsRef, cmdArgs[0],
                           getExecutablePathPtr);
  }

  llvm::errs() << "error: unknown integrated tool '" << tool << "'. "
               << "Valid tools include '-cc1'.\n";
  return 1;
}

static void errorHandler(void *userData, const char *msg, bool getCrashDiag) {
  auto &diags = *static_cast<clang::DiagnosticsEngine *>(userData);
  diags.Report(clang::diag::err_drv_command_failure) << msg;
  llvm::sys::RunInterruptHandlers();
  llvm::sys::Process::Exit(getCrashDiag ? 70 : 1);
}

int Driver::cc1Main(const CudaqArgs &cudaqArgs, ArgvT ccargs, ArgT tool,
                    void *mainAddr) {
  auto comp = std::make_unique<clang::CompilerInstance>();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  cudaq::BufferedDiagnostics diags(ccargs);
  bool success = clang::CompilerInvocation::CreateFromArgs(
      comp->getInvocation(), ccargs, diags.engine, tool);

  auto &frontendOpts = comp->getFrontendOpts();
  auto &headerOpts = comp->getHeaderSearchOpts();

  if (frontendOpts.TimeTrace || !frontendOpts.TimeTracePath.empty()) {
    frontendOpts.TimeTrace = 1;
    llvm::timeTraceProfilerInitialize(frontendOpts.TimeTraceGranularity, tool);
  }

  // Infer the builtin include path if unspecified.
  if (headerOpts.UseBuiltinIncludes && headerOpts.ResourceDir.empty())
    headerOpts.ResourceDir =
        clang::CompilerInvocation::GetResourcesPath(tool, mainAddr);

  // Create the actual diagnostics engine.
  if (comp->createDiagnostics(); !comp->hasDiagnostics())
    return 1;

  // Set an error handler, so that any LLVM backend diagnostics go through our
  // error handler.
  llvm::install_fatal_error_handler(
      errorHandler, static_cast<void *>(&comp->getDiagnostics()));

  diags.flush();
  if (!success) {
    comp->getDiagnosticClient().finish();
    return 1;
  }

  // Execute the frontend actions.
  try {
    llvm::TimeTraceScope TimeScope("ExecuteCompiler");
    success = executeCompilerInvocation(comp.get(), cudaqArgs);
  } catch (...) {
    comp->setSema(nullptr);
    comp->setASTConsumer(nullptr);
    comp->clearOutputFiles(true);
    throw;
  }

  // If any timers were active but haven't been destroyed yet, print their
  // results now.
  llvm::TimerGroup::printAll(llvm::errs());
  llvm::TimerGroup::clearAll();

  using small_string = llvm::SmallString<128>;

  if (llvm::timeTraceProfilerEnabled()) {
    small_string path(frontendOpts.OutputFile);
    llvm::sys::path::replace_extension(path, "json");

    if (!frontendOpts.TimeTracePath.empty()) {
      small_string tracePath(frontendOpts.TimeTracePath);
      if (llvm::sys::fs::is_directory(tracePath))
        llvm::sys::path::append(tracePath, llvm::sys::path::filename(path));

      path.assign(tracePath);
    }

    if (auto profilerOutput = comp->createOutputFile(
            path.str(), /*Binary=*/false, /*RemoveFileOnSignal=*/false,
            /*useTemporary=*/false)) {
      llvm::timeTraceProfilerWrite(*profilerOutput);
      profilerOutput.reset();
      llvm::timeTraceProfilerCleanup();
      comp->clearOutputFiles(false);
    }
  }
  llvm::remove_fatal_error_handler();

  // When running with -disable-free, don't do any destruction or shutdown.
  if (frontendOpts.DisableFree) {
    llvm::BuryPointer(std::move(comp));
    return !success;
  }

  return !success;
}

static std::unique_ptr<clang::FrontendAction> createFrontendAction(
    clang::CompilerInstance &ci, const CudaqArgs &cudaqArgs,
    mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::MLIRContext *context,
    cudaq::ASTBridgeAction::MangledKernelNamesMap &cxxMangled) {
  auto &opts = ci.getFrontendOpts();
  auto act = opts.ProgramAction;
  const bool mlirMode =
      cudaqArgs.hasArg(cudaq::nvqpp::options::OPT_enable_mlir);
  const std::string cudaqOptPipeline =
      Driver::constructCudaqOptPipeline(cudaqArgs);
  // Check if user just request quake emit action
  // Note: This is unique to NVQ++ since clang::frontend only support emit bc or
  // llvm.
  const bool emitQuakeAction =
      cudaqArgs.hasArg(cudaq::nvqpp::options::OPT_emit_quake);
  if (emitQuakeAction)
    return std::make_unique<CudaQAction<clang::EmitLLVMAction>>(
        ci, module, context, cxxMangled, cudaq::nvqpp::OutputGen::EmitMlir,
        cudaqOptPipeline);

  switch (act) {
  case (clang::frontend::ActionKind::ASTDump):
    // AST action: no need to invoke CUDAQ
    return std::make_unique<clang::ASTDumpAction>();
  case (clang::frontend::ActionKind::ASTPrint):
    // AST action: no need to invoke CUDAQ
    return std::make_unique<clang::ASTPrintAction>();
  case (clang::frontend::ActionKind::EmitBC):
    return mlirMode ? std::make_unique<CudaQAction<clang::EmitBCAction>>(
                          ci, module, context, cxxMangled,
                          cudaq::nvqpp::OutputGen::EmitLlvm, cudaqOptPipeline)
                    : std::make_unique<clang::EmitBCAction>();
  case (clang::frontend::ActionKind::EmitLLVM):
    return mlirMode ? std::make_unique<CudaQAction<clang::EmitLLVMAction>>(
                          ci, module, context, cxxMangled,
                          cudaq::nvqpp::OutputGen::EmitLlvm, cudaqOptPipeline)
                    : std::make_unique<clang::EmitLLVMAction>();
  case (clang::frontend::ActionKind::EmitObj):
    return mlirMode ? std::make_unique<CudaQAction<clang::EmitObjAction>>(
                          ci, module, context, cxxMangled,
                          cudaq::nvqpp::OutputGen::EmitObject, cudaqOptPipeline)
                    : std::make_unique<clang::EmitObjAction>();
  default:
    throw std::runtime_error("Not supported!!!");
  }

  return nullptr;
}

bool Driver::handleImmediateArgs() {
  if (clOptions.hasArg(cudaq::nvqpp::options::OPT_help)) {
    cudaq::nvqpp::options::getDriverOptTable().printHelp(
        llvm::outs(),
        /*Usage=*/"nvq++ [options] [host-compiler-options] file...",
        /*Title*/ "NVQ++ Compiler: https://github.com/NVIDIA/cuda-quantum",
        /*Include=*/0,
        /*Exclude=*/0, /*ShowAllAliases=*/false);
    return true;
  }
  return false;
}

bool Driver::executeCompilerInvocation(clang::CompilerInstance *ci,
                                       const CudaqArgs &cudaqArgs) {
  auto &opts = ci->getFrontendOpts();
  // -version.
  //
  // FIXME: Use a better -version message?
  if (opts.ShowVersion) {
    llvm::cl::PrintVersionMessage();
    return true;
  }

  ci->LoadRequestedPlugins();

  // If there were errors in processing arguments, don't do anything else.
  if (ci->getDiagnostics().hasErrorOccurred())
    return false;
  cudaq::ASTBridgeAction::MangledKernelNamesMap mangledKernelNameMap;
  mlir::registerAllPasses();
  cudaq::opt::registerOptCodeGenPasses();
  cudaq::opt::registerOptTransformsPasses();
  cudaq::opt::registerAggressiveEarlyInlining();
  cudaq::opt::registerUnrollingPipeline();
  cudaq::opt::registerBaseProfilePipeline();
  cudaq::opt::registerTargetPipelines();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<cudaq::cc::CCDialect, quake::QuakeDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OpBuilder builder(&context);
  auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module(moduleOp);
  // Create and execute the frontend action.
  auto action = createFrontendAction(*ci, cudaqArgs, module, &context,
                                     mangledKernelNameMap);
  if (!action)
    return false;

  bool success = ci->ExecuteAction(*action);

  if (opts.DisableFree)
    llvm::BuryPointer(std::move(action));

  return success;
}

std::unique_ptr<clang::driver::Compilation> Driver::makeCompilation() {
  clangDriver.CC1Main = Driver::executeCC1Tool;
  return std::unique_ptr<clang::driver::Compilation>(
      clangDriver.BuildCompilation(hostCompilerArgs));
}

std::optional<clang::driver::Driver::ReproLevel> Driver::getClangReproLevel(
    const std::unique_ptr<clang::driver::Compilation> &comp) const {
  std::optional<clang::driver::Driver::ReproLevel> level =
      clang::driver::Driver::ReproLevel::OnCrash;

  if (auto *arg = comp->getArgs().getLastArg(
          clang::driver::options::OPT_gen_reproducer_eq)) {
    level =
        llvm::StringSwitch<std::optional<clang::driver::Driver::ReproLevel>>(
            arg->getValue())
            .Case("off", clang::driver::Driver::ReproLevel::Off)
            .Case("crash", clang::driver::Driver::ReproLevel::OnCrash)
            .Case("error", clang::driver::Driver::ReproLevel::OnError)
            .Case("always", clang::driver::Driver::ReproLevel::Always)
            .Default(std::nullopt);

    if (!level) {
      llvm::errs() << "Unknown value for " << arg->getSpelling() << ": '"
                   << arg->getValue() << "'\n";
      return level;
    }
  }

  if (!!::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH"))
    level = clang::driver::Driver::ReproLevel::Always;

  return level;
}

int Driver::execute() {
  if (handleImmediateArgs())
    return 0;

  auto comp = makeCompilation();
  auto level = getClangReproLevel(comp);
  if (!level)
    return 1;

  int result = 1;
  bool isCrash = false;
  clang::driver::Driver::CommandStatus commandStatus =
      clang::driver::Driver::CommandStatus::Ok;

  const clang::driver::Command *failingCommand = nullptr;
  if (!comp->getJobs().empty())
    failingCommand = &(*comp->getJobs().begin());

  const bool doLink = [&]() {
    for (auto &Job : comp->getJobs()) {
      if (Job.getSource().getKind() ==
          clang::driver::Action::ActionClass::LinkJobClass)
        return true;
    }
    return false;
  }();

  if (comp && !comp->containsError()) {
    llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>
        failing;
    std::vector<std::string> objFilesToMerge;
    std::vector<std::string> sourceFiles;
    for (auto &Job : comp->getJobs()) {
      // Collect the list of input source files (i.e., .cpp files)
      // We use this to detect object files generated by quake/MLIR pipeline.
      if (Job.getSource().getKind() ==
          clang::driver::Action::ActionClass::AssembleJobClass) {
        for (const auto &input : Job.getInputInfos()) {
          if (input.isFilename()) {
            llvm::SmallString<128> inputFile =
                llvm::sys::path::filename(input.getFilename());
            sourceFiles.emplace_back(inputFile);
          }
        }
      }

      if (Job.getSource().getKind() ==
          clang::driver::Action::ActionClass::LinkJobClass) {
        std::vector<std::string> objFileNames;
        for (const auto &input : Job.getInputInfos()) {
          if (input.isFilename())
            objFileNames.emplace_back(input.getFilename());
        }
        // Collect all the ".qke.o" output files
        for (const auto &sourceFile : sourceFiles) {
          // Some source files may not have kernel code => no quake output.
          const auto quakeObjFile = sourceFile + ".qke.o";
          if (llvm::sys::fs::exists(quakeObjFile))
            objFilesToMerge.emplace_back(quakeObjFile);
        }
        // Strategy: inject out qke object file in front of the list
        llvm::opt::ArgStringList newLinkArgs;
        bool inserted = false;
        for (const auto &arg : Job.getArguments()) {
          if (std::find(objFileNames.begin(), objFileNames.end(), arg) !=
                  objFileNames.end() &&
              !inserted) {
            // Insert other object files, e.g., backend config and quake.
            for (const auto &objFile : objFilesToMerge)
              newLinkArgs.insert(newLinkArgs.end(), makeArgStringRef(objFile));
            inserted = true;
          }
          newLinkArgs.insert(newLinkArgs.end(), arg);
        }

        const std::string linkDir = std::string("-L") + cudaqLibPath;
        newLinkArgs.insert(newLinkArgs.end(), makeArgStringRef(linkDir));
        const std::array<const char *, 9> CUDAQ_LINK_LIBS{
            "-lcudaq",         "-lcudaq-common",    "-lcudaq-mlir-runtime",
            "-lcudaq-builder", "-lcudaq-ensmallen", "-lcudaq-nlopt",
            "-lcudaq-spin",    "-lnvqir",           "-lcudaq-em-qir",
        };

        for (const auto &linkLib : CUDAQ_LINK_LIBS)
          newLinkArgs.insert(newLinkArgs.end(), linkLib);
        const std::string nvqirBackend =
            (!targetName.empty() &&
             !targetPlatformExtraArgs.nvqirSimulationBackend.empty())
                ? targetPlatformExtraArgs.nvqirSimulationBackend
                : "qpp";
        const std::string backendLink = std::string("-lnvqir-") + nvqirBackend;
        newLinkArgs.insert(newLinkArgs.end(), makeArgStringRef(backendLink));

        const std::string platformName =
            (!targetName.empty() &&
             !targetPlatformExtraArgs.nvqirPlatform.empty())
                ? targetPlatformExtraArgs.nvqirPlatform
                : "default";
        const std::string platformLink =
            std::string("-lcudaq-platform-") + platformName;
        newLinkArgs.insert(newLinkArgs.end(), makeArgStringRef(platformLink));

        if (!targetName.empty() && targetPlatformExtraArgs.genTargetBackend)
          for (const auto &linkFlag : targetPlatformExtraArgs.linkFlags)
            newLinkArgs.insert(newLinkArgs.end(), makeArgStringRef(linkFlag));

        const std::string rpathDir = std::string("-rpath=") + cudaqLibPath;
        newLinkArgs.insert(newLinkArgs.end(), makeArgStringRef(rpathDir));
        Job.replaceArguments(newLinkArgs);
      } else {
        // Not a link job, i.e., frontend (cc1) job
        // Prepare args for the dual visiting frontend in MLIR mode or library
        // mode!
        auto currentArgs = Job.getArguments();
        const bool libMode = [&]() {
          if (clOptions.hasArg(cudaq::nvqpp::options::OPT_emit_quake))
            return false;
          // With target
          if (clOptions.hasArg(cudaq::nvqpp::options::OPT_target)) {
            // Use target setting (override CL options)
            return targetPlatformExtraArgs.libraryMode;
          } else {
            // Use command line (with or without `--enable-mlir`)
            return !clOptions.hasArg(cudaq::nvqpp::options::OPT_enable_mlir);
          }
        }();

        if (libMode)
          currentArgs.insert(currentArgs.end(), "-DCUDAQ_LIBRARY_MODE");
        else
          currentArgs.insert(currentArgs.end(), "--enable-mlir");

        // Forward CUDAQ front-end args (CC1) to be consumed by the front end
        // action (e.g., Quake optimization)
        for (const auto &arg : clOptions) {
          const bool isCudaqCC1Args =
              arg->getOption().isValid() &&
              arg->getOption().hasFlag(
                  cudaq::nvqpp::options::NvqppFlags::NvqppCC1Option);
          if (isCudaqCC1Args) {
            arg->print(llvm::outs());
            arg->renderAsInput(clOptions, currentArgs);
          }
        }
        // Notify the frontend that we'll do the linking (affecting certain
        // Quake optimization passes)
        if (doLink)
          currentArgs.insert(currentArgs.end(), "--opt-do-link");

        Job.replaceArguments(currentArgs);
      }
      const clang::driver::Command *failingCommand = nullptr;
      if (int Res = comp->ExecuteCommand(Job, failingCommand)) {
        failing.push_back(std::make_pair(Res, failingCommand));
        // bail out
        break;
      }
    }

    for (const auto &[cmdResult, cmd] : failing) {
      failingCommand = cmd;
      if (!result)
        result = cmdResult;

      isCrash = cmdResult < 0 || cmdResult == 70;
      commandStatus = isCrash ? clang::driver::Driver::CommandStatus::Crash
                              : clang::driver::Driver::CommandStatus::Error;

      if (isCrash)
        break;
    }
  }

  if (::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH"))
    llvm::dbgs() << llvm::getBugReportMsg();

  const auto maybeGenerateCompilationDiagnostics = [&] {
    return clangDriver.maybeGenerateCompilationDiagnostics(
        commandStatus, *level, *comp, *failingCommand);
  }();

  if (failingCommand != nullptr && maybeGenerateCompilationDiagnostics)
    result = 1;

  diag.finish();

  if (isCrash) {
    llvm::BuryPointer(llvm::TimerGroup::aquireDefaultGroup());
  } else {
    llvm::TimerGroup::printAll(llvm::errs());
    llvm::TimerGroup::clearAll();
  }

  // If we have multiple failing commands, we return the result of the first
  // failing command.
  return result;
}

void Driver::setInstallDir(ArgvStorageBase &argv) {
  // Attempt to find the original path used to invoke the driver, to determine
  // the installed path. We do this manually, because we want to support that
  // path being a symlink.
  llvm::SmallString<128> installedPath(argv[0]);

  // Do a PATH lookup, if there are no directory components.
  if (llvm::sys::path::filename(installedPath) == installedPath) {
    if (auto tmp = llvm::sys::findProgramByName(
            llvm::sys::path::filename(installedPath.str())))
      installedPath = *tmp;
  }

  llvm::sys::fs::make_absolute(installedPath);

  llvm::StringRef installedPathParent(
      llvm::sys::path::parent_path(installedPath));
  if (llvm::sys::fs::exists(installedPathParent)) {
    clangDriver.setInstalledDir(installedPathParent);
    {
      llvm::SmallString<128> libPath =
          llvm::sys::path::parent_path(llvm::sys::path::parent_path(
              llvm::sys::path::parent_path(installedPath)));
      llvm::sys::path::append(libPath, "lib");
      if (!llvm::sys::fs::exists(libPath)) {
        llvm::errs() << "nvq++ error: Directory not found: " << libPath << "\n";
        exit(1);
      }
      cudaqLibPath = libPath.str();
      llvm::SmallString<128> targetsPath =
          llvm::sys::path::parent_path(libPath);
      llvm::sys::path::append(targetsPath, "targets");
      cudaqTargetsPath = targetsPath.str();
    }
  }
}
} // namespace cudaq