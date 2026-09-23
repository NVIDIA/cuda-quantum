/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Build-time tool that reads every in-tree target's `.yml` file (via the
// exact same parser used at runtime, `cudaq::config::loadTargetConfig`) and
// emits a single generated C++ translation unit containing a literal,
// statically-allocated table of the parsed `cudaq::config::TargetConfig`
// structs. This lets `lookupBuiltinTarget` (TargetDatabase.h) resolve every
// in-tree target with zero YAML parsing and zero disk I/O at runtime.
//
// Usage:
//   cudaq-target-db-gen -o <output.cpp> <name>=<path/to/name.yml> ...

#include "cudaq/Target/TargetCatalog.h"
#include "cudaq/Target/TargetConfig.h"
#include "cudaq/Target/TargetDatabase.h"
#include "cudaq/Target/TargetPluginLibrary.h"
#include <cctype>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// Freezes this tool's own CUDAQ_TARGET_DB_ABI_SYMBOL_NAME (from whichever
// TargetDatabase.h this tool itself was built against) into a literal
// string, emitted verbatim as the ABI marker symbol name in generated
// output - see TargetDatabase.h for why this must be a frozen value, not a
// macro invocation re-expanded when the generated file is later compiled.
#define CUDAQ_STRINGIFY_IMPL(x) #x
#define CUDAQ_STRINGIFY(x) CUDAQ_STRINGIFY_IMPL(x)

namespace {

void printHelp() {
  std::cout << "CUDA-Q Target Database Generator\n"
               "\n"
               "Usage: cudaq-target-db-gen [options] "
               "<name>=<path/to/name.yml> ...\n"
               "\n"
               "Options:\n"
               "  -o <filename>  Specify output filename ('-', or omitting "
               "-o, writes to stdout)\n"
               "  --plugin       Emit a single-target plugin translation unit "
               "(compiled into a\n"
               "                 shared library exporting "
               "cudaq::config::kTargetPluginSymbolName)\n"
               "                 instead of the in-tree, multi-target "
               "database table.\n"
               "                 Exactly one <name>=<path> input is "
               "required.\n"
               "  -h, --help     Show this help and exit\n";
}

// Write `content` to `outputFilename`, or to stdout when `outputFilename` is
// "-" or empty. Mapping an omitted -o (empty) to stdout is a deliberate new
// superset: llvm::ToolOutputFile recognized only the literal "-" and failed
// an empty filename with ENOENT. Any failure removes a partially written
// file so it cannot be mistaken for a successful run.
int writeOutputChecked(const std::string &outputFilename,
                       const std::string &content) {
  if (outputFilename.empty() || outputFilename == "-") {
    std::cout << content;
    std::cout.flush();
    if (!std::cout) {
      std::cerr << "Failed to write output to stdout\n";
      return 1;
    }
    return 0;
  }

  errno = 0;
  std::ofstream file(outputFilename,
                     std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file) {
    const int openError = errno;
    std::cerr << "Failed to open output file '" << outputFilename << "'\n";
    return openError != 0 ? openError : 1;
  }
  file << content;
  file.flush();
  file.close();
  if (file.fail()) {
    std::cerr << "Failed to write output file '" << outputFilename << "'\n";
    // Best-effort cleanup of the partial file: non-throwing overload, since a
    // removal failure (e.g. unwritable directory) must not escape as an
    // uncaught exception on top of the write failure.
    std::error_code removeError;
    std::filesystem::remove(outputFilename, removeError);
    if (removeError)
      std::cerr << "Also failed to remove the partial output file: "
                << removeError.message() << "\n";
    return 1;
  }
  return 0;
}

std::string cxxStringLiteral(std::string_view s) {
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    default:
      out += c;
    }
  }
  out += "\"";
  return out;
}

std::string cxxStringVector(const std::vector<std::string> &values) {
  std::string out = "{";
  for (const auto &v : values)
    out += cxxStringLiteral(v) + ", ";
  out += "}";
  return out;
}

std::string cxxBool(bool value) { return value ? "true" : "false"; }

std::string cxxArgumentType(cudaq::config::ArgumentType type) {
  using cudaq::config::ArgumentType;
  switch (type) {
  case ArgumentType::string:
    return "cudaq::config::ArgumentType::string";
  case ArgumentType::integer:
    return "cudaq::config::ArgumentType::integer";
  case ArgumentType::uuid:
    return "cudaq::config::ArgumentType::uuid";
  case ArgumentType::option_flags:
    return "cudaq::config::ArgumentType::option_flags";
  case ArgumentType::machine_config:
    return "cudaq::config::ArgumentType::machine_config";
  }
  return "cudaq::config::ArgumentType::string";
}

std::string
cxxArchitectureSettings(const cudaq::config::TargetArchitectureSettings &s) {
  std::ostringstream os;
  os << "cudaq::config::TargetArchitectureSettings{.CodegenEmission = "
     << cxxStringLiteral(s.CodegenEmission) << "}";
  return os.str();
}

std::string cxxMachineConfigs(
    const std::vector<cudaq::config::MachineArchitectureConfig> &configs) {
  std::ostringstream os;
  os << "{";
  for (const auto &c : configs) {
    os << "cudaq::config::MachineArchitectureConfig{"
       << ".Name = " << cxxStringLiteral(c.Name) << ", "
       << ".MachineNames = " << cxxStringVector(c.MachineNames) << ", "
       << ".MachinePattern = " << cxxStringLiteral(c.MachinePattern) << ", "
       << ".Configuration = " << cxxArchitectureSettings(c.Configuration)
       << "}, ";
  }
  os << "}";
  return os.str();
}

std::string
cxxTargetArguments(const std::vector<cudaq::config::TargetArgument> &args) {
  std::ostringstream os;
  os << "{";
  for (const auto &a : args) {
    os << "cudaq::config::TargetArgument{"
       << ".KeyName = " << cxxStringLiteral(a.KeyName) << ", "
       << ".IsRequired = " << (a.IsRequired ? "true" : "false") << ", "
       << ".PlatformArgKey = " << cxxStringLiteral(a.PlatformArgKey) << ", "
       << ".HelpString = " << cxxStringLiteral(a.HelpString) << ", "
       << ".Type = " << cxxArgumentType(a.Type) << ", "
       << ".MachineConfigs = " << cxxMachineConfigs(a.MachineConfigs) << "}, ";
  }
  os << "}";
  return os.str();
}

std::string cxxConditionalBuildConfigs(
    const std::vector<cudaq::config::ConditionalBuildConfig> &configs) {
  std::ostringstream os;
  os << "{";
  for (const auto &c : configs) {
    os << "cudaq::config::ConditionalBuildConfig{"
       << ".Condition = " << cxxStringLiteral(c.Condition) << ", "
       << ".CompileFlag = " << cxxStringLiteral(c.CompileFlag) << ", "
       << ".LinkFlag = " << cxxStringLiteral(c.LinkFlag) << "}, ";
  }
  os << "}";
  return os.str();
}

std::string
cxxSimulationBackend(const cudaq::config::SimulationBackendSetting &s) {
  return "cudaq::config::SimulationBackendSetting{.values = " +
         cxxStringVector(s.values) + "}";
}

std::string
cxxBackendConfigEntry(const cudaq::config::BackendEndConfigEntry &c) {
  std::ostringstream os;
  os << "cudaq::config::BackendEndConfigEntry{"
     << ".GenTargetBackend = " << cxxBool(c.GenTargetBackend) << ", "
     << ".LibraryMode = " << cxxBool(c.LibraryMode) << ", "
     << ".SupportResourceCounts = " << cxxBool(c.SupportResourceCounts) << ", "
     << ".JITHighLevelPipeline = " << cxxStringLiteral(c.JITHighLevelPipeline)
     << ", "
     << ".JITMidLevelPipeline = " << cxxStringLiteral(c.JITMidLevelPipeline)
     << ", "
     << ".JITLowLevelPipeline = " << cxxStringLiteral(c.JITLowLevelPipeline)
     << ", "
     << ".TargetPassPipeline = " << cxxStringLiteral(c.TargetPassPipeline)
     << ", "
     << ".CodegenEmission = " << cxxStringLiteral(c.CodegenEmission) << ", "
     << ".PostCodeGenPasses = " << cxxStringLiteral(c.PostCodeGenPasses) << ", "
     << ".PlatformLibrary = " << cxxStringLiteral(c.PlatformLibrary) << ", "
     << ".LibraryModeExecutionManager = "
     << cxxStringLiteral(c.LibraryModeExecutionManager) << ", "
     << ".PlatformQpu = " << cxxStringLiteral(c.PlatformQpu) << ", "
     << ".PreprocessorDefines = " << cxxStringVector(c.PreprocessorDefines)
     << ", "
     << ".CompilerFlags = " << cxxStringVector(c.CompilerFlags) << ", "
     << ".LinkLibs = " << cxxStringVector(c.LinkLibs) << ", "
     << ".PluginLibraries = " << cxxStringVector(c.PluginLibraries) << ", "
     << ".LinkerFlags = " << cxxStringVector(c.LinkerFlags) << ", "
     << ".SimulationBackend = " << cxxSimulationBackend(c.SimulationBackend)
     << ", "
     << ".ConditionalBuildConfigs = "
     << cxxConditionalBuildConfigs(c.ConditionalBuildConfigs) << "}";
  return os.str();
}

std::string cxxOptionalBackendConfig(
    const std::optional<cudaq::config::BackendEndConfigEntry> &c) {
  if (!c.has_value())
    return "std::nullopt";
  return "std::optional<cudaq::config::BackendEndConfigEntry>(" +
         cxxBackendConfigEntry(c.value()) + ")";
}

std::string
cxxConfigMap(const std::vector<cudaq::config::BackendFeatureMap> &map) {
  std::ostringstream os;
  os << "{";
  for (const auto &m : map) {
    os << "cudaq::config::BackendFeatureMap{"
       << ".Name = " << cxxStringLiteral(m.Name) << ", "
       << ".Flags = static_cast<cudaq::config::TargetFeatureFlag>("
       << static_cast<unsigned>(m.Flags) << "u), "
       << ".Default = " << cxxBool(m.Default) << ", "
       << ".Config = " << cxxBackendConfigEntry(m.Config) << "}, ";
  }
  os << "}";
  return os.str();
}

std::string cxxTargetConfig(const cudaq::config::TargetConfig &config) {
  std::ostringstream os;
  os << "cudaq::config::TargetConfig{"
     << ".CudaqVersion = " << cxxStringLiteral(config.CudaqVersion) << ", "
     << ".PluginLibraries = " << cxxStringVector(config.PluginLibraries) << ", "
     << ".Name = " << cxxStringLiteral(config.Name) << ", "
     << ".Description = " << cxxStringLiteral(config.Description) << ", "
     << ".WarningMsg = " << cxxStringLiteral(config.WarningMsg) << ", "
     << ".TargetArguments = " << cxxTargetArguments(config.TargetArguments)
     << ", "
     << ".GpuRequired = " << (config.GpuRequired ? "true" : "false") << ", "
     << ".BackendConfig = " << cxxOptionalBackendConfig(config.BackendConfig)
     << ", "
     << ".ConfigMap = " << cxxConfigMap(config.ConfigMap) << "}";
  return os.str();
}

std::string sanitizeIdentifier(std::string_view name) {
  std::string out;
  for (char c : name)
    out += (std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  return out;
}

} // namespace

int main(int argc, char **argv) {
  std::vector<std::string> inputs;
  std::string outputFilename;
  bool pluginMode = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      printHelp();
      return 0;
    }
    if (arg == "--plugin") {
      pluginMode = true;
      continue;
    }
    if (arg == "-o") {
      if (++i >= argc) {
        std::cerr << "Option '-o' requires a value\n";
        return 1;
      }
      outputFilename = argv[i];
      continue;
    }
    if (arg.starts_with("-o=")) {
      outputFilename = arg.substr(3);
      continue;
    }
    if (!arg.empty() && arg.front() == '-') {
      std::cerr << "Unknown command line argument '" << arg
                << "'; run with --help for usage\n";
      return 1;
    }
    inputs.emplace_back(arg);
  }

  struct Entry {
    std::string name;
    cudaq::config::TargetConfig config;
  };
  std::vector<Entry> entries;
  for (const auto &input : inputs) {
    auto eq = input.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Malformed input '" << input
                << "', expected <name>=<path>\n";
      return 1;
    }
    std::string name = input.substr(0, eq);
    std::string path = input.substr(eq + 1);
    entries.push_back({name, cudaq::config::loadTargetConfig(path)});
  }

  if (pluginMode && entries.size() != 1) {
    std::cerr << "--plugin requires exactly one <name>=<path> input, got "
              << entries.size() << "\n";
    return 1;
  }

  // Build the whole translation unit in memory first: it cannot fail, and
  // it lets writeOutputChecked report output errors without leaving a
  // partial file behind.
  std::ostringstream os;
  if (pluginMode) {
    const auto &entry = entries.front();
    os << "// Generated by cudaq-target-db-gen --plugin. Do not edit.\n"
       << "#include \"cudaq/Target/TargetPluginLibrary.h\"\n\n"
       << "// Symbol name must match cudaq::config::kTargetPluginSymbolName.\n"
       << "extern \"C\" const cudaq::config::TargetConfig *"
       // Frozen at generation time to whatever kTargetPluginSymbolName *this*
       // binary was built against, like the database ABI marker below.
       << CUDAQ_TARGET_PLUGIN_SYMBOL_NAME_STR << "() {\n"
       << "  static const cudaq::config::TargetConfig kPluginTarget_"
       << sanitizeIdentifier(entry.name) << " = "
       << cxxTargetConfig(entry.config) << ";\n"
       << "  return &kPluginTarget_" << sanitizeIdentifier(entry.name) << ";\n"
       << "}\n";
    return writeOutputChecked(outputFilename, os.str());
  }

  os << "// Generated by cudaq-target-db-gen. Do not edit.\n"
     << "#include \"cudaq/Target/TargetDatabase.h\"\n\n"
     // ABI marker (see TargetDatabase.h): frozen at generation time to
     // whatever CUDAQ_TARGET_DB_ABI_VERSION *this* cudaq-target-db-gen
     // binary was built against. A stale generated file left over from
     // before a schema-changing rebuild will still carry the *old* name
     // here, so TargetDatabase.cpp's reference to the *current* name fails
     // to link instead of silently reading mismatched data.
     << "extern \"C\" void " << CUDAQ_STRINGIFY(CUDAQ_TARGET_DB_ABI_SYMBOL_NAME)
     << "() {}\n\n"
     << "namespace cudaq::config {\n"
     << "namespace {\n";
  for (const auto &entry : entries)
    os << "const TargetConfig kBuiltinTarget_" << sanitizeIdentifier(entry.name)
       << " = " << cxxTargetConfig(entry.config) << ";\n";
  os << "} // namespace\n\n"
     << "namespace detail {\n"
     << "const std::vector<std::pair<std::string_view, const TargetConfig "
        "*>> &builtinTargetTable() {\n"
     << "  static const std::vector<std::pair<std::string_view, const "
        "TargetConfig *>> table = {\n";
  for (const auto &entry : entries)
    os << "    {" << cxxStringLiteral(entry.name) << ", &kBuiltinTarget_"
       << sanitizeIdentifier(entry.name) << "},\n";
  os << "  };\n"
     << "  return table;\n"
     << "}\n"
     << "} // namespace detail\n"
     << "} // namespace cudaq::config\n";

  return writeOutputChecked(outputFilename, os.str());
}
