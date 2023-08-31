/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "nvqpp_driver.h"
#include "nvqpp_flag_configs.h"
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
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"

namespace cudaq {

Driver::Driver(const std::string &path, ArgvStorageBase &cmdArgs,
               ExecCompileFuncT cc1)
    : cmdArgs(cmdArgs), diag(cmdArgs, path),
      drv(path, llvm::sys::getDefaultTargetTriple(), diag.engine,
          "nvq++ compiler"),
      cc1EntryPoint(cc1) {
  drv.ResourceDir = std::string(CLANG_RESOURCE_DIR);
  setInstallDir(cmdArgs);
  // Add -std=c++20
  cmdArgs.insert(cmdArgs.end(), "-std=c++20");
  for (const char *include_flag : CUDAQ_INCLUDES_FLAGS)
    cmdArgs.insert(cmdArgs.end(), include_flag);
  preProcessCudaQArguments(cmdArgs);
}

void Driver::preProcessCudaQArguments(ArgvStorageBase &cmdArgs) {
  std::tie(cudaqArgs, std::ignore) = CudaqArgs::filterArgs(cmdArgs);
  if (cudaqArgs.hasOption("target")) {
    if (auto targetOpt = cudaqArgs.getOption("target"); targetOpt.has_value()) {
      llvm::StringRef targetName = cudaqArgs.getOption("target").value();
      targetConfig = targetName.str();
      auto targetArgsHandler = cudaq::getTargetPlatformArgs(targetConfig);
      if (targetArgsHandler)
        targetPlatformExtraArgs = targetArgsHandler->parsePlatformArgs(cmdArgs);
    } else {
      llvm::errs() << "Invalid target option: must be in the form "
                      "'-cudaq-target=<name>'";
      exit(1);
    }
  }
}
std::string Driver::processOptPipeline(ArgvStorageBase &args, bool doLink) {
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

#define CHECK_OPTION(ARG_IT, MEMBER_VAR, TRUE_OPTION, FALSE_OPTION)            \
  {                                                                            \
    auto arg = llvm::StringRef(*ARG_IT);                                       \
    if (arg.equals(TRUE_OPTION)) {                                             \
      opt.MEMBER_VAR = true;                                                   \
      ARG_IT = args.erase(ARG_IT);                                             \
    }                                                                          \
    if (arg.equals(FALSE_OPTION)) {                                            \
      opt.MEMBER_VAR = false;                                                  \
      ARG_IT = args.erase(ARG_IT);                                             \
    }                                                                          \
  }
  PipelineOpt opt;
  // Note: erase args within the loop
  for (auto it = args.begin(); it != args.end(); ++it) {
    CHECK_OPTION(it, ENABLE_DEVICE_CODE_LOADERS, "--device-code-loading",
                 "--no-device-code-loading");
    CHECK_OPTION(it, ENABLE_KERNEL_EXECUTION, "--kernel-execution",
                 "--no-kernel-execution");
    CHECK_OPTION(it, ENABLE_AGGRESSIVE_EARLY_INLINE,
                 "--aggressive-early-inline", "--no-aggressive-early-inline");
    CHECK_OPTION(it, ENABLE_APPLY_SPECIALIZATION,
                 "--quake-apply-specialization",
                 "--no-quake-apply-specialization");
    CHECK_OPTION(it, ENABLE_LAMBDA_LIFTING, "--lambda-lifting",
                 "--no-lambda-lifting");
  }

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
  return std::string("--pass-pipeline=builtin.module(") + optPasses + ")";
}

std::unique_ptr<clang::driver::Compilation> Driver::makeCompilation() {
  drv.CC1Main = cc1EntryPoint;
  return std::unique_ptr<clang::driver::Compilation>(
      drv.BuildCompilation(cmdArgs));
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

  if (!!::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH")) {
    level = clang::driver::Driver::ReproLevel::Always;
  }

  return level;
}

int Driver::execute() {
  auto comp = makeCompilation();
  auto level = getClangReproLevel(comp);
  if (!level) {
    return 1;
  }
  int result = 1;
  bool isCrash = false;
  clang::driver::Driver::CommandStatus commandStatus =
      clang::driver::Driver::CommandStatus::Ok;

  const clang::driver::Command *failingCommand = nullptr;
  if (!comp->getJobs().empty()) {
    failingCommand = &(*comp->getJobs().begin());
  }

  const bool doLink = [&]() {
    for (auto &Job : comp->getJobs()) {
      if (Job.getSource().getKind() ==
          clang::driver::Action::ActionClass::LinkJobClass)
        return true;
    }
    return false;
  }();

  const std::string cudaqOptPipeline = processOptPipeline(cmdArgs, doLink);
  const auto sourceInputFileName = [&]() -> std::string {
    for (const auto &job : comp->getJobs()) {
      for (const auto &input : job.getInputInfos()) {
        if (input.isFilename()) {
          llvm::SmallString<128> inputFile =
              llvm::sys::path::filename(input.getFilename());
          return inputFile.c_str();
        }
      }
    }
    return "";
  }();
  const std::string quakeFile = sourceInputFileName + ".qke";
  const std::string quakeFileOpt =
      drv.CreateTempFile(*comp, sourceInputFileName, "opt");
  const std::string quakeFileLl =
      drv.CreateTempFile(*comp, sourceInputFileName, "ll");
  const std::string quakeFileObj =
      drv.CreateTempFile(*comp, sourceInputFileName + "-qke", "o");
  if (comp && !comp->containsError()) {
    llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>
        failing;
    bool quakeRun = false;
    std::vector<std::string> objFilesToMerge;

    for (auto &Job : comp->getJobs()) {
      const clang::driver::Command *failingCommand = nullptr;
      if (Job.getSource().getKind() ==
          clang::driver::Action::ActionClass::LinkJobClass) {
        std::vector<std::string> objFileNames;
        for (const auto &input : Job.getInputInfos()) {
          if (input.isFilename()) {
            objFileNames.emplace_back(input.getFilename());
          }
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
              newLinkArgs.insert(newLinkArgs.end(), strdup(objFile.c_str()));
            inserted = true;
          }
          newLinkArgs.insert(newLinkArgs.end(), arg);
        }

        const std::string linkDir = std::string("-L") + cudaqLibPath;
        // FIXME: leak
        newLinkArgs.insert(newLinkArgs.end(), strdup(linkDir.c_str()));
        const std::array<const char *, 11> CUDAQ_LINK_LIBS{
            "-lcudaq",
            "-lcudaq-common",
            "-lcudaq-mlir-runtime",
            "-lcudaq-builder",
            "-lcudaq-ensmallen",
            "-lcudaq-nlopt",
            "-lcudaq-spin",
            "-lnvqir",
            "-lcudaq-em-qir",
            "-lcudaq-platform-default",
            "-lnvqir-qpp"};
        for (const auto &linkLib : CUDAQ_LINK_LIBS)
          newLinkArgs.insert(newLinkArgs.end(), linkLib);
        if (!targetConfig.empty() && targetPlatformExtraArgs.genTargetBackend)
          for (const auto &linkFlag : targetPlatformExtraArgs.linkFlags)
            newLinkArgs.insert(newLinkArgs.end(), strdup(linkFlag.c_str()));

        const std::string rpathDir = std::string("-rpath=") + cudaqLibPath;
        // FIXME: leak
        newLinkArgs.insert(newLinkArgs.end(), strdup(rpathDir.c_str()));
        Job.replaceArguments(newLinkArgs);
      }
      if (Job.getSource().getKind() ==
              clang::driver::Action::ActionClass::AssembleJobClass &&
          !targetConfig.empty() && targetPlatformExtraArgs.genTargetBackend) {
        // If this is an `Assemble` job, i.e., compile .o file,
        // and there is a target config, compile backendConfig.cpp as well
        clang::driver::Command compileBackendConfigCmd(Job);
        llvm::opt::ArgStringList newArgs;
        const std::string outputFileName = Job.getOutputFilenames().front();
        // $backendConfig-<target>-%%%%%%.o
        const std::string prefix = std::string("backendConfig-") + targetConfig;
        const char *backendConfigObjFile =
            drv.CreateTempFile(*comp, prefix, "o");
        objFilesToMerge.emplace_back(backendConfigObjFile);
        for (const auto &arg : compileBackendConfigCmd.getArguments()) {
          if (std::equal(sourceInputFileName.rbegin(),
                         sourceInputFileName.rend(),
                         std::string(arg).rbegin())) {
            const std::string backendConfigCppFile =
                cudaqTargetsPath + "/backendConfig.cpp";
            newArgs.insert(newArgs.end(), strdup(backendConfigCppFile.c_str()));
          } else if (std::string(arg) == outputFileName) {
            newArgs.insert(newArgs.end(), backendConfigObjFile);
          } else {
            newArgs.insert(newArgs.end(), arg);
          }
        }
        const std::string targetConfigDef =
            targetConfig + ";emulate;" +
            (cudaqArgs.hasOption("emulate") ? "true" : "false") +
            targetPlatformExtraArgs.platformExtraArgs;
        const std::string defArg =
            std::string("-DNVQPP_TARGET_BACKEND_CONFIG=\"") + targetConfigDef +
            "\"";
        newArgs.insert(newArgs.end(), strdup(defArg.c_str()));

        compileBackendConfigCmd.replaceArguments(newArgs);
        if (int Res =
                comp->ExecuteCommand(compileBackendConfigCmd, failingCommand)) {
          failing.push_back(std::make_pair(Res, failingCommand));
          // bail out
          break;
        }
      }

      if (int Res = comp->ExecuteCommand(Job, failingCommand)) {
        failing.push_back(std::make_pair(Res, failingCommand));
        // bail out
        break;
      }

      if (llvm::sys::fs::exists(quakeFile) && !quakeRun) {
        // Track quake temp file to delete
        // FIXME: don't use a separate file stream for quake output, use the
        // driver::Compilation temp file system.
        llvm::SmallString<128> quakeTmpFile(quakeFile);
        llvm::sys::fs::make_absolute(quakeTmpFile);
        comp->addTempFile(strdup(quakeTmpFile.c_str()));
        quakeRun = true;
        // Run quake-opt
        // TODO: need to check the action requested (LLVM/Obj)
        if (!cudaqOptPipeline.empty()) {
          clang::driver::InputInfoList inputInfos;
          llvm::opt::ArgStringList cmdArgs;
          cmdArgs.insert(cmdArgs.end(), strdup(cudaqOptPipeline.c_str()));
          cmdArgs.insert(cmdArgs.end(), strdup(quakeFile.c_str()));
          cmdArgs.insert(cmdArgs.end(), "-o");
          cmdArgs.insert(cmdArgs.end(), strdup(quakeFileOpt.c_str()));
          auto quakeOptCmd = std::make_unique<clang::driver::Command>(
              Job.getSource(), Job.getCreator(),
              clang::driver::ResponseFileSupport::None(), cudaqOptExe.c_str(),
              cmdArgs, inputInfos);
          if (int Res = comp->ExecuteCommand(*quakeOptCmd, failingCommand)) {
            failing.push_back(std::make_pair(Res, failingCommand));
            // bail out
            break;
          }
        }
        {
          // Run quake-translate
          clang::driver::InputInfoList inputInfos;
          llvm::opt::ArgStringList cmdArgs;
          cmdArgs.insert(cmdArgs.end(), "--convert-to=qir");
          // If run opt -> chain the output file from cudaq-opt,
          // otherwise, take the output file from quake.
          if (!cudaqOptPipeline.empty())
            cmdArgs.insert(cmdArgs.end(), strdup(quakeFileOpt.c_str()));
          else
            cmdArgs.insert(cmdArgs.end(), strdup(quakeFile.c_str()));

          cmdArgs.insert(cmdArgs.end(), "-o");
          cmdArgs.insert(cmdArgs.end(), strdup(quakeFileLl.c_str()));
          auto quakeTranslateCmd = std::make_unique<clang::driver::Command>(
              Job.getSource(), Job.getCreator(),
              clang::driver::ResponseFileSupport::None(),
              cudaqTranslateExe.c_str(), cmdArgs, inputInfos);

          if (int Res =
                  comp->ExecuteCommand(*quakeTranslateCmd, failingCommand)) {
            failing.push_back(std::make_pair(Res, failingCommand));
            // bail out
            break;
          }
        }
        {
          // Run llc
          clang::driver::InputInfoList inputInfos;
          llvm::opt::ArgStringList cmdArgs;
          cmdArgs.insert(cmdArgs.end(), "--relocation-model=pic");
          cmdArgs.insert(cmdArgs.end(), "--filetype=obj");
          cmdArgs.insert(cmdArgs.end(), "-O2");
          cmdArgs.insert(cmdArgs.end(), strdup(quakeFileLl.c_str()));
          cmdArgs.insert(cmdArgs.end(), "-o");
          cmdArgs.insert(cmdArgs.end(), strdup(quakeFileObj.c_str()));
          const std::string llcPath = std::string(LLVM_BIN_DIR) + "/llc";
          auto llcCmd = std::make_unique<clang::driver::Command>(
              Job.getSource(), Job.getCreator(),
              clang::driver::ResponseFileSupport::None(),
              strdup(llcPath.c_str()), cmdArgs, inputInfos);

          if (int Res = comp->ExecuteCommand(*llcCmd, failingCommand)) {
            failing.push_back(std::make_pair(Res, failingCommand));
            // bail out
            break;
          }
          // LLC succeed, add quake obj file
          objFilesToMerge.emplace_back(quakeFileObj);
        }
      }
    }

    for (const auto &[cmdResult, cmd] : failing) {
      failingCommand = cmd;
      if (!result) {
        result = cmdResult;
      }

      isCrash = cmdResult < 0 || cmdResult == 70;
      commandStatus = isCrash ? clang::driver::Driver::CommandStatus::Crash
                              : clang::driver::Driver::CommandStatus::Error;

      if (isCrash) {
        break;
      }
    }
  }

  if (::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH"))
    llvm::dbgs() << llvm::getBugReportMsg();

  const auto maybeGenerateCompilationDiagnostics = [&] {
    return drv.maybeGenerateCompilationDiagnostics(commandStatus, *level, *comp,
                                                   *failingCommand);
  }();

  if (failingCommand != nullptr && maybeGenerateCompilationDiagnostics) {
    result = 1;
  }

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
            llvm::sys::path::filename(installedPath.str()))) {
      installedPath = *tmp;
    }
  }

  llvm::sys::fs::make_absolute(installedPath);

  llvm::StringRef installedPathParent(
      llvm::sys::path::parent_path(installedPath));
  if (llvm::sys::fs::exists(installedPathParent)) {
    drv.setInstalledDir(installedPathParent);

    {
      llvm::SmallString<128> binPath =
          llvm::sys::path::parent_path(installedPath);
      llvm::sys::path::append(binPath, "cudaq-opt");
      if (!llvm::sys::fs::exists(binPath)) {
        llvm::errs() << "nvq++ error: File not found: " << binPath << "\n";
        exit(1);
      }
      cudaqOptExe = binPath.str();
    }
    {
      llvm::SmallString<128> binPath =
          llvm::sys::path::parent_path(installedPath);
      llvm::sys::path::append(binPath, "cudaq-translate");
      if (!llvm::sys::fs::exists(binPath)) {
        llvm::errs() << "nvq++ error: File not found: " << binPath << "\n";
        exit(1);
      }
      cudaqTranslateExe = binPath.str();
    }
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