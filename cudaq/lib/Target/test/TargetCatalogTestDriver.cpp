/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test driver for the target-catalog LIT suite; prints parsed configs
// deterministically for FileCheck.

#include "TargetConfigHelper.h"
#include "cudaq/Target/TargetCatalog.h"
#include "cudaq/Target/TargetConfig.h"
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace cudaq::config;

namespace {

/// Render embedded newlines/backslashes visibly: one config field, one line.
std::string escaped(const std::string &value) {
  std::string out;
  for (char c : value) {
    if (c == '\n')
      out += "\\n";
    else if (c == '\\')
      out += "\\\\";
    else if (c == '"')
      out += "\\\"";
    else
      out += c;
  }
  return out;
}

std::string boolStr(bool value) { return value ? "true" : "false"; }

std::string argTypeName(ArgumentType type) {
  static const char *names[] = {"string", "integer", "uuid", "option-flags",
                                "machine-config"};
  const auto index = static_cast<unsigned>(type);
  return index <= static_cast<unsigned>(ArgumentType::machine_config)
             ? names[index]
             : "unknown";
}

void printField(const char *indent, const char *key, const std::string &value) {
  if (!value.empty())
    std::cout << indent << key << ": " << escaped(value) << "\n";
}

// Elements are quoted so `["a,b"]` stays distinguishable from `["a", "b"]`.
std::string listStr(const std::vector<std::string> &values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i)
    out += (i ? ", \"" : "\"") + escaped(values[i]) + '"';
  return out + "]";
}

void printList(const char *indent, const char *key,
               const std::vector<std::string> &values) {
  if (!values.empty())
    std::cout << indent << key << ": " << listStr(values) << "\n";
}

void printBackend(const BackendEndConfigEntry &backend, const char *indent) {
  std::cout << indent
            << "gen-target-backend: " << boolStr(backend.GenTargetBackend)
            << "\n"
            << indent << "library-mode: " << boolStr(backend.LibraryMode)
            << "\n"
            << indent << "support-resource-counts: "
            << boolStr(backend.SupportResourceCounts) << "\n";
  printField(indent, "jit-high-level-pipeline", backend.JITHighLevelPipeline);
  printField(indent, "jit-mid-level-pipeline", backend.JITMidLevelPipeline);
  printField(indent, "jit-low-level-pipeline", backend.JITLowLevelPipeline);
  printField(indent, "target-pass-pipeline", backend.TargetPassPipeline);
  printField(indent, "codegen-emission", backend.CodegenEmission);
  printField(indent, "post-codegen-passes", backend.PostCodeGenPasses);
  printField(indent, "platform-library", backend.PlatformLibrary);
  printField(indent, "library-mode-execution-manager",
             backend.LibraryModeExecutionManager);
  printField(indent, "platform-qpu", backend.PlatformQpu);
  printList(indent, "preprocessor-defines", backend.PreprocessorDefines);
  printList(indent, "compiler-flags", backend.CompilerFlags);
  printList(indent, "link-libs", backend.LinkLibs);
  printList(indent, "plugin-libraries", backend.PluginLibraries);
  printList(indent, "linker-flags", backend.LinkerFlags);
  printList(indent, "nvqir-simulation-backend",
            backend.SimulationBackend.values);
  for (const auto &rule : backend.ConditionalBuildConfigs)
    std::cout << indent << "rule: if=" << escaped(rule.Condition)
              << " compiler-flag=" << escaped(rule.CompileFlag)
              << " link-flag=" << escaped(rule.LinkFlag) << "\n";
}

void printConfig(const TargetConfig &config) {
  std::cout << "name: " << config.Name << "\n"
            << "description: " << escaped(config.Description) << "\n";
  printField("", "cudaq-version", config.CudaqVersion);
  printField("", "warning", config.WarningMsg);
  std::cout << "gpu-requirements: " << (config.GpuRequired ? "true" : "false")
            << "\n";
  for (const auto &arg : config.TargetArguments) {
    std::cout << "argument: " << arg.KeyName
              << " type=" << argTypeName(arg.Type)
              << " required=" << (arg.IsRequired ? "true" : "false")
              << " platform-arg="
              << (arg.PlatformArgKey.empty() ? "-" : arg.PlatformArgKey)
              << "\n";
    if (!arg.HelpString.empty())
      std::cout << "  help-string: " << escaped(arg.HelpString) << "\n";
    for (const auto &arch : arg.MachineConfigs)
      std::cout << "  arch: " << arch.Name << " codegen-emission="
                << escaped(arch.Configuration.CodegenEmission)
                << " pattern=" << escaped(arch.MachinePattern)
                << " machines=" << listStr(arch.MachineNames) << "\n";
  }
  if (config.BackendConfig) {
    std::cout << "config:\n";
    printBackend(*config.BackendConfig, "  ");
  }
  for (const auto &entry : config.ConfigMap) {
    std::cout << "matrix-entry: " << entry.Name
              << " flags=" << static_cast<unsigned>(entry.Flags)
              << " default=" << boolStr(entry.Default) << "\n";
    printBackend(entry.Config, "  ");
  }
  printList("", "plugin-libraries", config.PluginLibraries);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: TargetCatalogTestDriver <parse|emit|codegen> "
                 "<config.yml> [--plugin-root <dir>] [key=value ...]\n";
    return 2;
  }
  const std::string mode = argv[1];
  const std::filesystem::path configPath = argv[2];
  std::filesystem::path pluginRoot;
  std::map<std::string, std::string> args;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--plugin-root" && i + 1 < argc) {
      pluginRoot = argv[++i];
      continue;
    }
    const auto eq = arg.find('=');
    if (eq == std::string::npos) {
      std::cerr << "error: expected key=value argument, got '" << arg << "'\n";
      return 2;
    }
    args[arg.substr(0, eq)] = arg.substr(eq + 1);
  }

  TargetConfig config;
  try {
    config = loadTargetConfig(configPath, pluginRoot);
  } catch (const std::exception &ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  if (mode == "parse") {
    printConfig(config);
    return 0;
  }
  if (mode == "emit") {
    std::cout << processRuntimeArgs(config, args);
    return 0;
  }
  if (mode == "codegen") {
    std::cout << "codegen-emission: " << escaped(config.getCodeGenSpec(args))
              << "\n";
    return 0;
  }
  std::cerr << "error: unknown mode '" << mode << "'\n";
  return 2;
}
