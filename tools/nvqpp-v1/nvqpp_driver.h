/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "nvqpp_args.h"
#include "nvqpp_diagnostics.h"
#include "nvqpp_flag_configs.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
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
#include <iostream>
namespace cudaq {
struct driver {
  argv_storage_base &cmd_args;
  errs_diagnostics diag;
  clang::driver::Driver drv;
  exec_compile_t cc1_entry_point;
  std::string cudaq_opt_exe;
  std::string cudaq_translate_exe;
  std::string cudaq_lib_path;

  driver(const std::string &path, argv_storage_base &cmd_args,
         exec_compile_t cc1)
      : cmd_args(cmd_args), diag(cmd_args, path),
        drv(path, llvm::sys::getDefaultTargetTriple(), diag.engine,
            "nvq++ compiler"),
        cc1_entry_point(cc1) {
    drv.ResourceDir = std::string(CLANG_RESOURCE_DIR);
    set_install_dir(cmd_args);
    // Add -std=c++20
    cmd_args.insert(cmd_args.end(), "-std=c++20");
    for (const char *include_flag : CUDAQ_INCLUDES_FLAGS)
      cmd_args.insert(cmd_args.end(), include_flag);
  }

  std::unique_ptr<clang::driver::Compilation> make_compilation() {
    drv.CC1Main = cc1_entry_point;
    return std::unique_ptr<clang::driver::Compilation>(
        drv.BuildCompilation(cmd_args));
  }

  std::optional<clang::driver::Driver::ReproLevel> get_clang_repro_level(
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

  const clang::driver::Command *
  first_job(const std::unique_ptr<clang::driver::Compilation> &comp) {
    return &(*comp->getJobs().begin());
  }

  using failing_commands =
      llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4>;

  int execute() {
    auto comp = make_compilation();
    auto level = get_clang_repro_level(comp);
    if (!level) {
      return 1;
    }
    int result = 1;
    bool is_crash = false;
    clang::driver::Driver::CommandStatus command_status =
        clang::driver::Driver::CommandStatus::Ok;

    const clang::driver::Command *failing_command = nullptr;
    if (!comp->getJobs().empty()) {
      failing_command = first_job(comp);
    }
    // drv.PrintHelp(true);
    // drv.PrintActions(*comp);
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
    const std::string quakeFileOpt = quakeFile + ".opt";
    const std::string quakeFileLl = sourceInputFileName + ".qir.ll";
    const std::string quakeFileObj = sourceInputFileName + ".qke.o";
    if (comp && !comp->containsError()) {
      failing_commands failing;
      bool quake_run = false;
      for (auto &Job : comp->getJobs()) {
        const clang::driver::Command *FailingCommand = nullptr;

        if (Job.getSource().getKind() ==
            clang::driver::Action::ActionClass::LinkJobClass) {
          std::vector<std::string> obj_file_names;
          for (const auto &input : Job.getInputInfos()) {
            if (input.isFilename()) {
              obj_file_names.emplace_back(input.getFilename());
            }
          }
          // Strategy: inject out qke object file in front of the list
          llvm::opt::ArgStringList new_link_args;
          bool inserted = false;
          for (const auto &arg : Job.getArguments()) {
            if (std::find(obj_file_names.begin(), obj_file_names.end(), arg) !=
                    obj_file_names.end() &&
                !inserted) {
              new_link_args.insert(new_link_args.end(),
                                   strdup(quakeFileObj.c_str()));
              inserted = true;
            }
            new_link_args.insert(new_link_args.end(), arg);
          }

          const std::string link_dir = std::string("-L") + cudaq_lib_path;
          // FIXME: leak
          new_link_args.insert(new_link_args.end(), strdup(link_dir.c_str()));
          // TODO: handle target selection, just use qpp for now!
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
            new_link_args.insert(new_link_args.end(), linkLib);

          const std::string rpath_dir = std::string("-rpath=") + cudaq_lib_path;
          // FIXME: leak
          new_link_args.insert(new_link_args.end(), strdup(rpath_dir.c_str()));
          Job.replaceArguments(new_link_args);
        }

        if (int Res = comp->ExecuteCommand(Job, FailingCommand)) {
          failing.push_back(std::make_pair(Res, FailingCommand));
          // bail out
          break;
        }

        if (llvm::sys::fs::exists(quakeFile) && !quake_run) {
          quake_run = true;
          // Run quake-opt
          // TODO: need to check the action requested (LLVM/Obj)
          {
            clang::driver::InputInfoList InputInfos;
            llvm::opt::ArgStringList CmdArgs;
            CmdArgs.insert(CmdArgs.end(), "--canonicalize");
            CmdArgs.insert(CmdArgs.end(), "--kernel-execution");
            CmdArgs.insert(CmdArgs.end(), strdup(quakeFile.c_str()));
            CmdArgs.insert(CmdArgs.end(), "-o");
            CmdArgs.insert(CmdArgs.end(), strdup(quakeFileOpt.c_str()));
            auto quake_opt_cmd = std::make_unique<clang::driver::Command>(
                Job.getSource(), Job.getCreator(),
                clang::driver::ResponseFileSupport::None(),
                cudaq_opt_exe.c_str(), CmdArgs, InputInfos);
            // if (comp->getArgs().hasArg(clang::driver::options::OPT_v))
            //   quake_opt_cmd->Print(llvm::errs(), "\n", true);
            if (int Res =
                    comp->ExecuteCommand(*quake_opt_cmd, FailingCommand)) {
              failing.push_back(std::make_pair(Res, FailingCommand));
              // bail out
              break;
            }
          }
          {
            // Run quake-translate
            clang::driver::InputInfoList InputInfos;
            llvm::opt::ArgStringList CmdArgs;
            CmdArgs.insert(CmdArgs.end(), "--convert-to=qir");
            CmdArgs.insert(CmdArgs.end(), strdup(quakeFileOpt.c_str()));
            CmdArgs.insert(CmdArgs.end(), "-o");
            CmdArgs.insert(CmdArgs.end(), strdup(quakeFileLl.c_str()));
            auto quake_opt_cmd = std::make_unique<clang::driver::Command>(
                Job.getSource(), Job.getCreator(),
                clang::driver::ResponseFileSupport::None(),
                cudaq_translate_exe.c_str(), CmdArgs, InputInfos);

            if (int Res =
                    comp->ExecuteCommand(*quake_opt_cmd, FailingCommand)) {
              failing.push_back(std::make_pair(Res, FailingCommand));
              // bail out
              break;
            }
          }
          {
            // Run llc
            clang::driver::InputInfoList InputInfos;
            llvm::opt::ArgStringList CmdArgs;
            CmdArgs.insert(CmdArgs.end(), "--relocation-model=pic");
            CmdArgs.insert(CmdArgs.end(), "--filetype=obj");
            CmdArgs.insert(CmdArgs.end(), "-O2");
            CmdArgs.insert(CmdArgs.end(), strdup(quakeFileLl.c_str()));
            CmdArgs.insert(CmdArgs.end(), "-o");
            CmdArgs.insert(CmdArgs.end(), strdup(quakeFileObj.c_str()));
            const std::string llcPath = std::string(LLVM_BIN_DIR) + "/llc";
            auto quake_opt_cmd = std::make_unique<clang::driver::Command>(
                Job.getSource(), Job.getCreator(),
                clang::driver::ResponseFileSupport::None(),
                strdup(llcPath.c_str()), CmdArgs, InputInfos);

            if (int Res =
                    comp->ExecuteCommand(*quake_opt_cmd, FailingCommand)) {
              failing.push_back(std::make_pair(Res, FailingCommand));
              // bail out
              break;
            }
          }
        }
      }

      for (const auto &[cmd_result, cmd] : failing) {
        failing_command = cmd;
        if (!result) {
          result = cmd_result;
        }

        is_crash = cmd_result < 0 || cmd_result == 70;
        command_status = is_crash ? clang::driver::Driver::CommandStatus::Crash
                                  : clang::driver::Driver::CommandStatus::Error;

        if (is_crash) {
          break;
        }
      }
    }

    if (::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH"))
      llvm::dbgs() << llvm::getBugReportMsg();

    auto maybe_generate_compilation_diagnostics = [&] {
      return drv.maybeGenerateCompilationDiagnostics(command_status, *level,
                                                     *comp, *failing_command);
    };

    if (failing_command != nullptr &&
        maybe_generate_compilation_diagnostics()) {
      result = 1;
    }

    diag.finish();

    if (is_crash) {
      llvm::BuryPointer(llvm::TimerGroup::aquireDefaultGroup());
    } else {
      llvm::TimerGroup::printAll(llvm::errs());
      llvm::TimerGroup::clearAll();
    }

    // If we have multiple failing commands, we return the result of the first
    // failing command.
    return result;
  }

  void set_install_dir(argv_storage_base &argv) {
    // Attempt to find the original path used to invoke the driver, to determine
    // the installed path. We do this manually, because we want to support that
    // path being a symlink.
    llvm::SmallString<128> installed_path(argv[0]);

    // Do a PATH lookup, if there are no directory components.
    if (llvm::sys::path::filename(installed_path) == installed_path) {
      if (auto tmp = llvm::sys::findProgramByName(
              llvm::sys::path::filename(installed_path.str()))) {
        installed_path = *tmp;
      }
    }

    llvm::sys::fs::make_absolute(installed_path);

    llvm::StringRef installed_path_parent(
        llvm::sys::path::parent_path(installed_path));
    if (llvm::sys::fs::exists(installed_path_parent)) {
      drv.setInstalledDir(installed_path_parent);

      {
        llvm::SmallString<128> binPath =
            llvm::sys::path::parent_path(installed_path);
        llvm::sys::path::append(binPath, "cudaq-opt");
        if (!llvm::sys::fs::exists(binPath)) {
          llvm::errs() << "nvq++ error: File not found: " << binPath << "\n";
          exit(1);
        }
        cudaq_opt_exe = binPath.str();
      }
      {
        llvm::SmallString<128> binPath =
            llvm::sys::path::parent_path(installed_path);
        llvm::sys::path::append(binPath, "cudaq-translate");
        if (!llvm::sys::fs::exists(binPath)) {
          llvm::errs() << "nvq++ error: File not found: " << binPath << "\n";
          exit(1);
        }
        cudaq_translate_exe = binPath.str();
      }
      {
        llvm::SmallString<128> libPath =
            llvm::sys::path::parent_path(llvm::sys::path::parent_path(
                llvm::sys::path::parent_path(installed_path)));
        llvm::sys::path::append(libPath, "lib");
        if (!llvm::sys::fs::exists(libPath)) {
          llvm::errs() << "nvq++ error: Directory not found: " << libPath
                       << "\n";
          exit(1);
        }
        cudaq_lib_path = libPath.str();
      }
    }
  }
};
} // namespace cudaq