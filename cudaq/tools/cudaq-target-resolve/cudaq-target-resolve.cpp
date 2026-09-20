/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Resolves a target by name via TargetRegistry plus CLI target arguments into
// a flat file of nvq++-compatible bash KEY=value assignments.

#include "cudaq/Target/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>

using namespace llvm;

std::string decodeBase64IfPrefixed(llvm::StringRef input) {
  if (!input.starts_with("base64_"))
    return input.str();

  if (input.size() <= 7)
    return "";

  auto encodedStr = input.substr(7);
  std::vector<char> decodedStr;
  if (auto err = llvm::decodeBase64(encodedStr, decodedStr)) {
    llvm::errs() << "DecodeBase64 error for '" << encodedStr << "' string.\n";
    abort();
  }
  return std::string(decodedStr.data(), decodedStr.size());
}

//===----------------------------------------------------------------------===//
// Command line options.
//===----------------------------------------------------------------------===//

static cl::opt<std::string> targetName(cl::Positional,
                                       cl::desc("<target name or YAML path>"),
                                       cl::init("-"), cl::value_desc("name"));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("filename"));

static cl::opt<std::string> targetArgs("arg",
                                       cl::desc("Specify target CLI arguments"),
                                       cl::value_desc("string"));

static cl::opt<bool> listTargets("list-targets",
                                 cl::desc("List available target names"));

static cl::opt<bool>
    includeUnavailable("include-unavailable",
                       cl::desc("Include targets that are not available on "
                                "this host when listing"));

static cl::opt<bool>
    listSimulators("list-simulators",
                   cl::desc("List available simulator target names"));

static cl::list<std::string>
    pluginRoots("plugin-root", cl::desc("External plugin package root"),
                cl::value_desc("path"));

static cl::opt<std::string> installDir("install-dir",
                                       cl::desc("CUDA-Q install prefix"),
                                       cl::value_desc("path"));

static cl::list<std::string> libDirs("lib-dir",
                                     cl::desc("Directory to search for "
                                              "simulator/platform libraries"),
                                     cl::value_desc("path"));

static cl::opt<unsigned> gpuCount("gpu-count", cl::desc("GPUs on this host"),
                                  cl::init(0));

static cl::opt<std::string>
    cudaqVersion("cudaq-version", cl::desc("Current CUDA-Q version string"),
                 cl::init(""));

static cl::opt<bool>
    enableYamlParsing("enable-yaml-parsing",
                      cl::desc("Allow loading target configurations from YAML "
                               "files in plugin roots (disabled by default)"));

static constexpr const char BOLD[] = "\033[1m";
static constexpr const char RED[] = "\033[91m";
static constexpr const char CLEAR[] = "\033[0m";

static void addPluginScope(cudaq::config::TargetRegistry &registry,
                           const std::filesystem::path &scope) {
  if (!std::filesystem::is_directory(scope))
    return;
  for (const auto &entry : std::filesystem::directory_iterator{scope}) {
    if (entry.is_directory())
      registry.addPluginRoot(entry.path());
  }
}

static std::filesystem::path resolveInstallDir(const char *argv0) {
  if (!installDir.empty())
    return std::filesystem::path(installDir.getValue());
  const auto self = sys::fs::getMainExecutable(argv0, (void *)&addPluginScope);
  if (self.empty())
    return {};
  return std::filesystem::path(self).parent_path().parent_path();
}

static cudaq::config::HostEnvironment
makeHostEnv(const std::filesystem::path &prefix) {
  cudaq::config::HostEnvironment env;
  env.gpuCount = gpuCount;
  env.cudaqVersion = cudaqVersion;
  for (const auto &dir : libDirs)
    env.libraryPaths.emplace_back(dir);
  if (env.libraryPaths.empty() && !prefix.empty())
    env.libraryPaths.emplace_back(prefix / "lib");
  return env;
}

static void populateRegistry(cudaq::config::TargetRegistry &registry,
                             const std::filesystem::path &prefix) {
  // User / extra plugin roots take precedence over system plugins
  for (const auto &root : pluginRoots)
    registry.addPluginRoot(root);
  if (!prefix.empty())
    addPluginScope(registry, prefix / "plugins");
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "CUDA-Q Target Build Configuration Resolver\n");

  cudaq::config::TargetRegistry registry;

  // Support loading YAML explicitly passed as argument
  std::string name = targetName.getValue();
  if (name != "-" && !name.empty()) {
    if (const std::filesystem::path path(name);
        std::filesystem::is_regular_file(path)) {
      if (!registry.addTargetConfigFile(path))
        return 1;
      name = path.stem().string();
    }
  }
  // From here on lock it down: only consider pre-compiled plugin libraries.
  if (!enableYamlParsing)
    cudaq::config::disableYAMLTargetConfigParsing();

  const auto prefix = resolveInstallDir(argv[0]);
  populateRegistry(registry, prefix);
  auto env = makeHostEnv(prefix);

  if (listTargets) {
    for (const auto &resolved : registry.resolveAll(env)) {
      if (!includeUnavailable && !resolved.status.isAvailable())
        continue;
      llvm::outs() << resolved.entry->name << "\n";
    }
    return 0;
  }

  if (listSimulators) {
    for (const auto &resolved : registry.resolveAll(env)) {
      if (!resolved.status.isAvailable())
        continue;
      const auto &cfg = *resolved.entry->config;
      const auto *backend = cfg.BackendConfig ? &*cfg.BackendConfig : nullptr;
      for (const auto &entry : cfg.ConfigMap)
        if (entry.Default.has_value() && entry.Default.value())
          backend = &entry.Config;
      if (!backend)
        continue;
      if (!backend->PlatformQpu.empty() ||
          !backend->LibraryModeExecutionManager.empty())
        continue;
      llvm::outs() << resolved.entry->name << "\n";
    }
    return 0;
  }

  if (name == "-" || name.empty()) {
    llvm::errs() << "error: a target name is required\n";
    return 1;
  }

  std::string targetArgsString = decodeBase64IfPrefixed(targetArgs);
  llvm::SmallVector<llvm::StringRef> args;
  llvm::StringRef(targetArgsString).split(args, ' ', -1, false);
  std::map<std::string, std::string> argsMap;
  if (args.size() > 0) {
    for (std::size_t idx = 0; idx < args.size() - 1; idx += 2) {
      std::string argKey = decodeBase64IfPrefixed(args[idx]);
      std::string argVal = decodeBase64IfPrefixed(args[idx + 1]);
      argsMap.insert({argKey, argVal});
    }
  }

  auto resolved = registry.resolve(name, env, argsMap);
  if (!resolved) {
    llvm::errs() << "Invalid Target: (" << name << ")\n";
    return 1;
  }
  if (!resolved->status.isAvailable()) {
    llvm::errs() << resolved->status.diagnostic << "\n";
    return 1;
  }

  // Success! Dump the config (bash variable setters)
  const auto &entry = *resolved->entry;
  if (!entry.config->WarningMsg.empty())
    llvm::outs() << BOLD << RED << "Warning: " << CLEAR
                 << entry.config->WarningMsg << "\n";
  if (!resolved->status.diagnostic.empty())
    llvm::outs() << BOLD << RED << "Warning: " << CLEAR
                 << resolved->status.diagnostic << "\n";

  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);
  if (ec) {
    errs() << "Failed to open output file '" << outputFilename << "'\n";
    return ec.value();
  }
  out.os() << cudaq::config::emitNvqppConfig(entry, argsMap);
  if (!entry.configPath.empty())
    out.os() << "TARGET_CONFIG_PATH=\"" << entry.configPath.string() << "\"\n";
  if (!entry.pluginLibDir.empty())
    out.os() << "TARGET_PLUGIN_LIB_DIR=\"" << entry.pluginLibDir.string()
             << "\"\n";
  out.os() << "TARGET_IS_EXTERNAL="
           << (entry.origin == cudaq::config::detail::TargetOrigin::Builtin
                   ? "false"
                   : "true")
           << "\n";
  out.keep();
  return 0;
}
